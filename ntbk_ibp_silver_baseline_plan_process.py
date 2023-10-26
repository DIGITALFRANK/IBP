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
source_container = dbutils.widgets.get("sourceContainer")
target_container = dbutils.widgets.get("targetContainer")
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
spath = dbutils.widgets.get("sourcePath")
tpath = dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#Defining Source and Target Path 
srcPath = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+spath
tgtPath = "abfss://"+target_container+"@"+target_stgAccnt+".dfs.core.windows.net/"+tpath

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#Reading the data from the bronze path of DFU table
base_deltaTable = DeltaTable.forPath(spark, srcPath)
base_latest_version = base_deltaTable.history().select(max(col('version'))).collect()[0][0]

print("Base delta table latest version ",base_latest_version)
display(base_deltaTable.history())

# COMMAND ----------

#Reading the Baseline plan(Rescource Details) from bronze layer
baselineDF = spark.read.format("delta").option("versionAsOf", base_latest_version).load(srcPath)

print("Baseline DF record count: ",baselineDF.count() )
baselineDF.printSchema()
baselineDF.display()

# COMMAND ----------

max_date = baselineDF.agg({"PROCESS_DATE" : "MAX"}).collect()[0][0]
baselineDF1 = baselineDF.filter(col("PROCESS_DATE") == max_date)

print("Latest Date in the Bronze Layer: ",max_date)
display(baselineDF1)

# COMMAND ----------

silver_df = baselineDF1.select("MTRL_ID",
                               col("DW_MTRL_ID").cast("bigint"),
                               "ASST_ID",
                               col("DW_ASST_ID").cast("bigint"),
                               "ASST_OWNR_LOC_ID",
                               col("ASST_OWNR_DW_LOC_ID").cast("bigint"),
                               col("SCP_LOAD_TYP_CDV").cast("float"),
                               "PRODTN_MTHD_VAL",
                               col("PRODTN_MTHD_STEP_NUM").cast("float"),
                               col("ASST_UTLZTN_STRT_DTM").cast("timestamp"),                                                            
                               col("ASST_PRODTN_MTHD_STEP_LOAD_VAL").cast("float"),
                               "ASST_STRG_LOC_ID",
                               col("ASST_STRG_DW_LOC_ID").cast("bigint"),
                               col("CUST_ORDR_LOAD_VAL").cast("float"),
                               col("FCST_ORDR_LOAD_VAL").cast("float"),
                               col("SAFSTK_ORDR_LOAD_VAL").cast("float"),
                               col("SCP_QTY").cast("float"),
                               col("SCP_ORDR_NUM").cast("int"),
                               col("DMND_MET_DT").cast("timestamp"),
                               col("PRPLN_ORDR_NUM").cast("int"),
                               "MTRL_CTGY_NM",
                               "MU_CDV",
                               col("ASST_UTLZTN_GNRTN_DTM").cast("timestamp")
                              )

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

#Writing data innto delta lake
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
    .tableName("sc_ibp_silver.baseline_plan") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.baseline_plan

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))