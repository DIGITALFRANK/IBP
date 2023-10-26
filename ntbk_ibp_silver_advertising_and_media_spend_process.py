# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql import Row

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
dpndntdatapath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
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
adv_deltaTable = DeltaTable.forPath(spark, srcPath)
adv_latest_version = adv_deltaTable.history().select(max(col('version'))).collect()[0][0]
print("The latest version from bronze layer: ",adv_latest_version)
display(adv_deltaTable.history())

# COMMAND ----------

#Reading the adv&media source data from bonze layer
adv_med_DF = spark.read.format("delta").option("versionAsOf", adv_latest_version).load(srcPath)

display(adv_med_DF)

# COMMAND ----------

col_names = ["WC_DT","CTRY_ISO","PEP_CTGY","BRND","SUB_BRND","SPNDNG_IN_USD","MEDA_CMPGN_SPNDNG_LCLCCY_AMT",
	   "MDMIX","CHNL","MEDA_CMPGN_PO_NUM_VAL","CMPGN_ID","CMPGN_GNRTN","ACTL_PLAND","FRMT","BUYNG_TRGT",
	   "BUY_TYP","BUYNG_MODL","CHNL_DTL","CREATIVE","UNIT_SIZE","PLCMT_NAME","MEDA_TYP_NM",
	   "ACTVTY_CMNTS","ACTVTY_STRT_DT","ACTVTY_END_DT","BRND_ORG"]

adv_df = adv_med_DF.withColumn("BRND_ORG",col("BRND")) \
                   .select(col_names)

# COMMAND ----------

#Replicating rows for Brand KAS into "KAS REFRESCOS" and "KASFRUIT"
adv_df_kas = adv_df.where(col('BRND')== "KAS")
adv_df_kas_v1 = adv_df_kas.withColumn('BRND',lit("KAS REFRESCOS"))
adv_df_kas_v2 = adv_df_kas.withColumn('BRND',lit("KASFRUIT"))
adv_df_v3 = adv_df.where(col('BRND') != "KAS").union(adv_df_kas_v1.union(adv_df_kas_v2))


adv_df_v3.display()

# COMMAND ----------

#Substituting Brand name with new Value and renaming column
adv_df_formated = adv_df_v3.withColumn("BRND",when(col("BRND") == "PEPSI ZERO/MAX",lit("PEPSI")) \
                                          .when(col("BRND")== "DIET 7UP",lit("7UP")).otherwise(col("BRND"))) \
                           .withColumn("SPND_TYP", when(col("MDMIX")==lit("DIGITAL"), lit("DIGITAL")).otherwise(col("CHNL"))) \
                           .withColumn("SPNDNG_IN_USD",col("SPNDNG_IN_USD").cast("float")) \
                           .withColumn("MEDA_CMPGN_SPNDNG_LCLCCY_AMT",col("MEDA_CMPGN_SPNDNG_LCLCCY_AMT").cast("float")) \
                           .withColumnRenamed("CTRY_ISO","MU") \
                           .withColumnRenamed("SUB_BRND","SUBBRND") \
                           .withColumnRenamed("CMPGN_GNRTN","CMPGN_DESCR") \
                           .withColumnRenamed("ACTL_PLAND","STTS") \
                           .withColumnRenamed("CREATIVE","CRTVE") \
                           .withColumnRenamed("PEP_CTGY","CTGY") 

# COMMAND ----------

adv_df_formated.printSchema()
adv_df_formated.display()

# COMMAND ----------

silver_df = adv_df_formated.groupBy("WC_DT","MU","CTGY","BRND","BRND_ORG","SUBBRND","SPND_TYP","MDMIX","CHNL","MEDA_CMPGN_PO_NUM_VAL",
                                    "CMPGN_ID","CMPGN_DESCR","STTS","FRMT","BUYNG_TRGT","BUY_TYP","BUYNG_MODL","CHNL_DTL","CRTVE",
                                    "UNIT_SIZE","PLCMT_NAME","MEDA_TYP_NM","ACTVTY_CMNTS","ACTVTY_STRT_DT","ACTVTY_END_DT"
                                   ) \
                          .agg(sum("SPNDNG_IN_USD").cast("float").alias("SPNDNG_IN_USD"),sum("MEDA_CMPGN_SPNDNG_LCLCCY_AMT").cast("float").alias("SPNDNG_LCLCCY_AMT"))

silver_df.display()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

#Writing data innto silver delta lake
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

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.advertising_and_media_spend") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# # remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.advertising_and_media_spend

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))