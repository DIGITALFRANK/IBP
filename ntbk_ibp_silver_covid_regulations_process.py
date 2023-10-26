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

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

cov_rg_deltaTable = DeltaTable.forPath(spark, srcPath)
cov_rg_latest_version = cov_rg_deltaTable.history().select(max(col('version'))).collect()[0][0]
display(cov_rg_deltaTable.history())

# COMMAND ----------

cov_rg_df = spark.read.format("delta").option("versionAsOf", cov_rg_latest_version).load(srcPath)
print(cov_rg_df.count())
display(cov_rg_df)

# COMMAND ----------

max_value = cov_rg_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
cov_rg_df_new = cov_rg_df.filter(col("PROCESS_DATE")==max_value).drop("DELTA_DATE")
display(cov_rg_df_new)

# COMMAND ----------

silver_df = cov_rg_df_new.select("CountryName", "CountryCode", "Date", "ConfirmedCases", "ConfirmedDeaths", "StringencyIndex", "StringencyLegacyIndex", "GovernmentResponseIndex", "ContainmentHealthIndex", "EconomicSupportIndex").withColumn("Date", F.to_date(F.col("Date"), "yyyyMMdd"))
              

# COMMAND ----------

df_shipments = spark.read.format("delta").load(dpdPath)

# COMMAND ----------

df_shipments = df_shipments.withColumn("DMNDFCST_MKT_UNIT_CDV",col("DMDGROUP").substr(1, 2))\
                           .where((col("DMNDFCST_MKT_UNIT_CDV").isin('ES','PT')))\
                           .select("LOC","DMNDFCST_MKT_UNIT_CDV").distinct()
df_shipments = df_shipments.withColumn("country_name", F.when(F.col("DMNDFCST_MKT_UNIT_CDV")=="PT","Portugal").otherwise("Spain"))

# COMMAND ----------

silver_df = silver_df.join(df_shipments, silver_df.CountryName == df_shipments.country_name, how = 'inner')

# COMMAND ----------

silver_df =silver_df.withColumnRenamed("CountryName","CTRY_NM")\
.withColumnRenamed("LOC","LOC")\
.withColumnRenamed("DMNDFCST_MKT_UNIT_CDV","MU")\
.withColumnRenamed("Date","DT")\
.withColumnRenamed("ConfirmedCases","CNFRMD_CASES")\
.withColumnRenamed("ConfirmedDeaths","CNFRMD_DEATHS")\
.withColumnRenamed("StringencyIndex","STRNGNCY_INDX")\
.withColumnRenamed("StringencyLegacyIndex","STRNGNCY_LGCY_INDX")\
.withColumnRenamed("GovernmentResponseIndex","GOVT_RSPNS_INDX")\
.withColumnRenamed("ContainmentHealthIndex","CNTNMT_HLTH_INDX")\
.withColumnRenamed("EconomicSupportIndex","ECNMC_SPRT_INDX")

# COMMAND ----------

silver_df = silver_df.select("CTRY_NM","LOC","MU",col("DT").cast("timestamp"),col("CNFRMD_CASES").cast("int"),col("CNFRMD_DEATHS").cast("int"),col("STRNGNCY_INDX").cast("float"),col("STRNGNCY_LGCY_INDX").cast("float"),col("GOVT_RSPNS_INDX").cast("float"),col("CNTNMT_HLTH_INDX").cast("float"),col("ECNMC_SPRT_INDX").cast("float"))

# COMMAND ----------

silver_df.display()

# COMMAND ----------

silver_df.groupby("CTRY_NM","LOC","DT").count().filter("count > 1").display()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  #deltaTable = DeltaTable.forPath(spark, tgtPath)
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
    .tableName("sc_ibp_silver.covid_regulations") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.covid_regulations

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))