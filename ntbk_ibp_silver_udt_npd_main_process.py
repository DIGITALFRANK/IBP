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
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")source_container = dbutils.widgets.get("sourceContainer")
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
ndpMain_deltaTable = DeltaTable.forPath(spark, srcPath)
ndpMain_latest_version = ndpMain_deltaTable.history().select(max(col('version'))).collect()[0][0]

print("UDT Ndp Main table latest version ",ndpMain_latest_version)
display(ndpMain_deltaTable.history())

# COMMAND ----------

#Reading the Baseline plan(Rescource Details) from bronze layer
ndpMainDF = spark.read.format("delta").option("versionAsOf", ndpMain_latest_version).load(srcPath)

print("UDT Ndp Main DF record count: ",ndpMainDF.count() )
ndpMainDF.printSchema()
ndpMainDF.display()

# COMMAND ----------

max_date = ndpMainDF.agg({"PROCESS_DATE" : "MAX"}).collect()[0][0]
ndpMain_Latest = ndpMainDF.filter(col("PROCESS_DATE") == max_date)

print("Latest Date in the Bronze Layer: ",max_date)
display(ndpMain_Latest)

# COMMAND ----------

ndpMain_Updated = ndpMain_Latest.withColumn("AUTO_FLG",col("AUTO_FLG").cast("boolean")) \
                                .withColumn("APPROVE",col("APPROVE").cast("boolean")) \
                                .withColumn("CONVERSION_DT",col("CONVERSION_DATE").cast("timestamp")) \
                                .withColumn("CREATE_DT",col("CREATE_DATE").cast("timestamp")) \
                                .withColumn("VALID_HRZN_STRT_DT",col("VALID_HORIZON_START_DATE").cast("timestamp")) \
                                .withColumn("FIRST_FCST_DT",col("FIRST_FCST_DATE").cast("timestamp")) \
                                .withColumn("TOCS",col("TOCS").cast("float")) \
                                .withColumn("TOKG",col("TOKG").cast("float")) \
                                .withColumn("TOL",col("TOL").cast("float")) \
                                .withColumn("TOCS2",col("TOCS2").cast("float")) \
                                .withColumn("TORDL",col("TORDL").cast("float")) \
                                .withColumn("SEQ",col("SEQ").cast("int")) \
                                .withColumn("INVALID_HOR_FLG",col("INVALID_HOR_FLG").cast("int")) \
                                .withColumn("ATTR_CHG_FLG",col("ATTR_CHG_FLG").cast("int")) \
                                .withColumn("HOR_ALERT_SENT_DATE",col("HOR_ALERT_SENT_DATE").cast("timestamp")) \
                                .withColumn("ATTR_ALERT_SENT_DATE",col("ATTR_ALERT_SENT_DATE").cast("timestamp")) \
                                .withColumn("DELETE_DATE",col("DELETE_DATE").cast("timestamp")) \
                                .withColumnRenamed("MARKET_UNIT","MU") \
                                .withColumnRenamed("CATEGORY","CAT") \
                                .withColumnRenamed("NPD_ITEM","NPD_Prod") \
                                .withColumnRenamed("REAL_ITEM","Prod_CD") \
                                .withColumnRenamed("NPD_DESCR","NPD_Prod_Descr") \
                                .withColumnRenamed("STATUS","STTS") \
                                .withColumnRenamed("FLAVOUR_SHORT","FLVR_SHRT") \
                                .withColumnRenamed("PACK_CONTAINER_SHORT","PCK_CNTNR_SHRT") \
                                .withColumnRenamed("PACK_SIZE_SHORT","SIZE_SHRT") \
                                .withColumnRenamed("BRAND_GROUP","BRND_GRP") \
                                .withColumnRenamed("MATTYPE","MTRL_TYP_CDV")

# COMMAND ----------

silver_df = ndpMain_Updated.select("AUTO_FLG",
                                   "APPROVE",
                                   "MU",
                                   "CAT",
                                   "NPD_Prod",
                                   "Prod_CD",
                                   "NPD_Prod_Descr",
                                   "STTS",
                                   "CONVERSION_DT",
                                   "CREATE_DT",
                                   "GPID",
                                   "SUBBRAND_SHORT",
                                   "FLVR_SHRT",
                                   "PCK_CNTNR_SHRT",
                                   "SIZE_SHRT",
                                   "VALID_HRZN_STRT_DT",
                                   "FIRST_FCST_DT",
                                   "SEQ",
                                   "BRND_GRP",
                                   "BUOM",
                                   "TOCS",
                                   "TOKG","TOL",
                                   "TOCS2",
                                   "TORDL",
                                   "INVALID_HOR_FLG",
                                   "ATTR_CHG_FLG",
                                   "HOR_ALERT_SENT_DATE",
                                   "ATTR_ALERT_SENT_DATE",
                                   "DELETE_DATE",
                                   "MTRL_TYP_CDV"
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
    .tableName("sc_ibp_silver.udt_npd_main") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.udt_npd_main

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))