# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

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

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#Reading the data from the bronze path of DFU table
stft_deltaTable = DeltaTable.forPath(spark, srcPath)
stft_latest_version = stft_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(stft_latest_version)
display(stft_deltaTable.history())

# COMMAND ----------

#Reading the DFU source data from bonze layer
stft_df = spark.read.format("delta").option("versionAsOf", stft_latest_version).load(srcPath)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = stft_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
# stft_df2 =stft_df
stft_df2 = stft_df.filter(col("PROCESS_DATE")==max_value)
#display(stft_df2)

# COMMAND ----------

## DQ Check
pklist2 = ['store_id']
pklist2 = ','.join(str(e) for e in pklist2)
if len(pklist2.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in pklist2.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+pklist2+"').isNull()"

stft_dq = stft_df2.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pklist2.split(','))
                    .orderBy(desc("zip_code")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))
stft_dq_pass = stft_dq.where(col("Corrupt_Record").isNull()).drop("DUP_CHECK","Corrupt_Record")

# COMMAND ----------

print(stft_df2.count())
print(stft_dq.count())
print(stft_dq_pass.count())

# COMMAND ----------

silver_df = stft_dq_pass.select(stft_dq_pass.country.alias("CTRY")
               ,stft_dq_pass.store_name.alias("STOR_NM")
               ,stft_dq_pass.full_address.alias("FULL_ADDR")
               ,stft_dq_pass.zip_code.alias("ZIP_CD")
               ,stft_dq_pass.city.alias("CITY")
               ,stft_dq_pass.street.alias("STR")
               ,stft_dq_pass.latitude.alias("LAT")
               ,stft_dq_pass.longitude.alias("LNGTD")
               ,stft_dq_pass.rating.alias("RTNG")
               ,stft_dq_pass.reviews.alias("RVWS")
               ,stft_dq_pass.hours.alias("HRS")
               ,stft_dq_pass.price.alias("PRC")
               ,stft_dq_pass.does_shop_do_delivery.alias("SHOP_DLVRY")
               ,stft_dq_pass.store_id.alias("STOR_ID")
               ,stft_dq_pass.TOTAL_VISITORS_SUM_7_DAYS.alias("TOT_VISTORS_7_DAYS")
               ,stft_dq_pass.TOTAL_VISITORS_SUM_14_DAYS.alias("TOT_VISTORS_14_DAYS")
               ,stft_dq_pass.AWAKENING_SCORE.alias("AWAKENING_SCR")
               ,stft_dq_pass.PERCENT_CHANGE_FROM_BENCHMARK.alias("PCT_CHNG_FROM_BNCHMRK")
               ,stft_dq_pass.PERCENT_CHANGE_FROM_BENCHMARK_AVG_7_DAYS.alias("PCT_CHNG_FROM_BNCHMRK_AVG_7_DAYS")
               ,stft_dq_pass.type.alias("STOR_TYP")
               ,stft_dq_pass.phone.alias("phone")
               ,stft_dq_pass.website.alias("website")
               ,stft_dq_pass.google_maps_url.alias("google_maps_url"))

# COMMAND ----------

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