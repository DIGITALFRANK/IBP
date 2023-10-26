# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *

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

srcPath

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#Reading the delta history from the bronze path of Customer Master
src_deltaTable = DeltaTable.forPath(spark, srcPath)
src_latest_version = src_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(src_latest_version)
display(src_deltaTable.history())

# COMMAND ----------

#Reading the Customer Master source data from bronze layer
src_df = spark.read.format("delta").option("versionAsOf", src_latest_version).load(srcPath)
print(src_df.count())
display(src_df)

# COMMAND ----------

#creating the silver layer dataframe for Customer Master
silver_df = src_df.select(col('PLANG_CUST_GRP_VAL').alias('CUST_GRP'),
               col('PLANG_CUST_GRP_NM').alias('CUST_GRP_NM'),
               col('PLANG_CUST_GRP_TYP_NM').alias('LVL'),
               col('PLANG_CUST_SCTR_NM').alias('SCTR') ,
               col('PLANG_CUST_BU_NM').alias('BU'),
               col('HRCHY_LVL_3_NM').alias('MU'),
               col('HRCHY_LVL_2_NM').alias('CHNL'),
               col('HRCHY_LVL_1_NM').alias('CLNT'),
               col('HRCHY_LVL_2_ID').alias('HRCHY_LVL_2_ID'))

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

#Writing data into delta lake
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

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))