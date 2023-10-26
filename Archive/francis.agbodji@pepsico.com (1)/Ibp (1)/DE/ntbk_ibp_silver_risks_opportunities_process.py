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
RO_df = spark.read.format("delta").option("versionAsOf", fillRateCSL_latest_version).load(srcPath)

# COMMAND ----------

RO_df.printSchema()
display(RO_df)

# COMMAND ----------

silver_df = RO_df.select(RO_df.Initiative_Type.alias("INITV_TYPE")
               ,RO_df.Initiative.alias("INITV")
               ,RO_df.Channel.alias("CHNL")
               ,RO_df.Customer.alias("CUST")
               ,RO_df.Category.alias("CTGY")
               ,RO_df.Brand.alias("BRND")
               ,RO_df.Format.alias("FRMT")
               ,RO_df.Demand_Group.alias("CUST_GRP")
               ,RO_df.Initiative_Owner.alias("INITV_OWNR")
               ,RO_df.Volume_in_TonnesKL.alias("VOL").cast(FloatType())
               ,RO_df.Net_Revenue.alias("NR").cast(FloatType())
               ,RO_df.Marginal_Contribution.alias("MC").cast(FloatType())
               ,RO_df.Probability.alias("PRBLTY")
               ,RO_df.Include.alias("INCLD")
               ,RO_df.Date_Entered.alias("DT_ENTRD").cast('date')
               ,RO_df.Start_Date.alias("STRT_DT").cast('date')
               ,RO_df.End_Date.alias("END_DT").cast('date')
               ,RO_df.Comment_on_probability.alias("CMNT_ON_PRBLTY")
               ,RO_df.Initiative_Status.alias("INIT_STTS"))

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