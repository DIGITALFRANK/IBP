# Databricks notebook source
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql import Row
import itertools
import re

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

#storing the parameters in variables
source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
dpdPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")

# COMMAND ----------

tgtPath

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

holiday_deltaTable = DeltaTable.forPath(spark, srcPath)
holiday_latest_version = holiday_deltaTable.history().select(max(col('version'))).collect()[0][0]
display(holiday_deltaTable.history())

# COMMAND ----------

# raw_holidays = spark.read.format("csv").option("inferSchema", "false").option("header", "true")\
#                           .option("sep", ",").load("/FileStore/tables/PEP_Volume_Forecasting/Iberian_Holidays_vF.csv")
#raw_holidays = spark.read.format("csv").option("inferSchema", "false").option("header", "true")\
#                           .option("with-encoding", "UTF-8").load(srcPath)

# COMMAND ----------

hol_df = spark.read.format("delta").option("versionAsOf", holiday_latest_version).load(srcPath)
print(hol_df.count())
display(hol_df)

# COMMAND ----------

max_value = hol_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
hol_df_new = hol_df.filter(col("PROCESS_DATE")==max_value).drop("DELTA_DATE")
display(hol_df_new)

# COMMAND ----------

silver_df = hol_df_new.select(col('Name').alias('HOL_NM'),
               col('Date').alias('DT').cast("timestamp"),
               col('Counties').alias('CNTY'),
               col('CountryCode').alias('MU'),
               col('LocalName').alias('HOL_LNM'),              
               col('Type').alias('HOL_TYP'))
              

# COMMAND ----------

df_shipments = spark.read.format("delta").load(dpdPath)

# COMMAND ----------

df_shipments = df_shipments.withColumn("DMNDFCST_MKT_UNIT_CDV",col("DMDGROUP").substr(1, 2))\
                           .where((col("DMNDFCST_MKT_UNIT_CDV").isin('ES','PT')))\
                           .select("LOC","DMNDFCST_MKT_UNIT_CDV").distinct()

# COMMAND ----------

silver_df = silver_df.join(df_shipments, silver_df.MU == df_shipments.DMNDFCST_MKT_UNIT_CDV, how = 'right')

# COMMAND ----------

silver_df = silver_df.select("HOL_NM","HOL_LNM","DT","CNTY","MU","HOL_TYP",col("LOC").alias("LOC") )

# COMMAND ----------

silver_df.groupby("HOL_LNM","DT","MU","LOC").count().filter("count > 1").display()

# COMMAND ----------

silver_df.display()

# COMMAND ----------

#silver_df.groupby("HOL_NM","MU","DT","CNTY","LOC").count().filter("count > 1").display()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.public_holiday") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.public_holiday

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))