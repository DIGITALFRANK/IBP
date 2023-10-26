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
proSubs_deltaTable = DeltaTable.forPath(spark, srcPath)
proSubs_latest_version = proSubs_deltaTable.history().select(max(col('version'))).collect()[0][0]

print("UDT Product Substitution table latest version ",proSubs_latest_version)
display(proSubs_deltaTable.history())

# COMMAND ----------

#Reading the Baseline plan(Rescource Details) from bronze layer
proSubsDF = spark.read.format("delta").option("versionAsOf", proSubs_latest_version).load(srcPath)

print("UDT Product Substitution DF record count: ",proSubsDF.count() )
proSubsDF.printSchema()
proSubsDF.display()

# COMMAND ----------

max_date = proSubsDF.agg({"PROCESS_DATE" : "MAX"}).collect()[0][0]
proSubs_latest_DF = proSubsDF.filter(col("PROCESS_DATE") == max_date)

print("Latest Date in the Bronze Layer: ",max_date)
display(proSubs_latest_DF)

# COMMAND ----------

proSubs_Updated = proSubs_latest_DF.withColumn("EFF",col("EFF").cast("timestamp")) \
                                   .withColumn("DISC",col("DISC").cast("timestamp")) \
                                   .withColumn("HIST_FACTOR",col("HISTORY_FACTOR").cast("float")) \
                                   .withColumn("FCST_FACTOR",col("FORECAST_FACTOR").cast("float")) \
                                   .withColumn("COPY_HIST",col("COPY_HIST").cast("boolean")) \
                                   .withColumn("COPY_FCST",col("COPY_FCST").cast("boolean")) \
                                   .withColumn("COPY_PRM",col("COPY_PARAM").cast("boolean")) \
                                   .withColumn("SEQ",col("SEQ").cast("int")) \
                                   .withColumn("PROCESS",col("SEQ").cast("boolean")) \
                                   .withColumnRenamed("OLDDMDUNIT","PREV_PROD_CD") \
                                   .withColumnRenamed("OLDDMDGROUP","PREV_CUST_GRP") \
                                   .withColumnRenamed("OLDLOC","PREV_LOC") \
                                   .withColumnRenamed("OLDMODEL","PREV_MODEL") \
                                   .withColumnRenamed("NEWDMDUNIT","NEW_PROD_CD") \
                                   .withColumnRenamed("NEWDMDGROUP","NEW_CUST_GRP") \
                                   .withColumnRenamed("NEWLOC","NEW_LOC") \
                                   .withColumnRenamed("NEWMODEL","NEW_MODEL")\
                                   .withColumn("MU",col('NEW_CUST_GRP').substr(1,2))

# COMMAND ----------

silver_df = proSubs_Updated.select("PREV_PROD_CD",
                                   "PREV_CUST_GRP",
                                   "PREV_LOC",
                                   "PREV_MODEL",
                                   "NEW_PROD_CD",
                                   "NEW_CUST_GRP",
                                   "NEW_LOC",
                                   "NEW_MODEL",
                                   "EFF",
                                   "DISC",
                                   "HIST_FACTOR",
                                   "FCST_FACTOR",
                                   "COPY_HIST",
                                   "COPY_FCST",
                                   "COPY_PRM",
                                   "SEQ",
                                   "PROCESS",
                                   "GPID",                                   
                                   "MU"
                                  )

silver_df.display()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

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
    .tableName("sc_ibp_silver.udt_product_substitution") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.udt_product_substitution

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))