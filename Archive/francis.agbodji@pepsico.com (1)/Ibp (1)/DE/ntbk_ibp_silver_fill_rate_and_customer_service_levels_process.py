# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql import functions as f

# COMMAND ----------

dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
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

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#Reading the data from the bronze path of DFU table
fillRateCSL_deltaTable = DeltaTable.forPath(spark, srcPath)
fillRateCSL_latest_version = fillRateCSL_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(fillRateCSL_latest_version)
display(fillRateCSL_deltaTable.history())

# COMMAND ----------

#Reading the DFU source data from bonze layer
fillRateCSL_df = spark.read.format("delta").option("versionAsOf", fillRateCSL_latest_version).load(srcPath)

# COMMAND ----------

display(fillRateCSL_df)
fillRateCSL_df.printSchema()

# COMMAND ----------

silver_df = fillRateCSL_df.withColumn("SHPPED",regexp_replace(col("Metric_Numerator"), ",", ""))\
                          .withColumn("ORDERED",regexp_replace(col("Metric_Denominator"), ",",""))\
                .select(fillRateCSL_df.KPI_ID.alias("KPI_ID")
               ,fillRateCSL_df.Time_Bucket.alias("TIME_BUCKET")
               ,fillRateCSL_df.Year_ID.alias("YR_ID").cast('integer')
               ,fillRateCSL_df.Month_ID.alias("MNTH_ID").cast('integer')
               ,fillRateCSL_df.Week_ID.alias("WEEK_ID").cast('integer')
               ,fillRateCSL_df.Market.alias("MU")
               ,fillRateCSL_df.Category.alias("CTGY")
               ,fillRateCSL_df.Customer_ID.alias("CUST_GRP")             
               ,col("SHPPED").cast(FloatType())
               ,col("ORDERED").cast(FloatType()))


# COMMAND ----------

silver_df.printSchema()
display(silver_df)

# COMMAND ----------

#Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df.alias("updates"),
    condition = cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df.write\
  .format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))