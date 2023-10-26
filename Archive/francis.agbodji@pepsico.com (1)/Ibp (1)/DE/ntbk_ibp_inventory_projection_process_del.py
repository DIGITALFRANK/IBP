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
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
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

#Reading the data from the bronze path of Pricing table
invPrjc_deltaTable = DeltaTable.forPath(spark, srcPath)
invPrjc_deltaTable_version = invPrjc_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(invPrjc_deltaTable)
display(invPrjc_deltaTable.history())

# COMMAND ----------

#Reading the Inventory Projection source data from bronze layer
invPrjc_df = spark.read.format("delta").option("versionAsOf", invPrjc_deltaTable_version).load(srcPath)

# COMMAND ----------

print(invPrjc_df.count())
display(invPrjc_df)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = invPrjc_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
invPrjc_df = invPrjc_df.filter(col("PROCESS_DATE")==max_value)

# COMMAND ----------

invPrjc_df = invPrjc_df.withColumn('SCNR',when(to_date(col("SPLY_PLANG_PRJCTN_GNRTN_DT"))==col("SPLY_PLANG_PRJCTN_STRT_DT"),"Actuals").otherwise("Projected"))

invPrjc_df = invPrjc_df.withColumn('GNRTN_DT',to_date(col("SPLY_PLANG_PRJCTN_GNRTN_DT")))\
                       .withColumn('STRT_DT',to_date(col("SPLY_PLANG_PRJCTN_STRT_DT")))

invPrjc_df = invPrjc_df.withColumn('WEEK_ID',concat(year(col("STRT_DT")),(weekofyear(col("STRT_DT")))))\
                       .withColumn('MNTH_ID',concat(year(col("STRT_DT")),month(col("STRT_DT"))))

invPrjc_df = invPrjc_df.withColumn('WEEK_ID',when(length(col("WEEK_ID"))<6,concat(year(col("STRT_DT")),lit("0"),month(col("STRT_DT")))))\
                       .withColumn('MNTH_ID',when(length(col("MNTH_ID"))<6,concat(year(col("STRT_DT")),lit("0"),month(col("STRT_DT")))))

# COMMAND ----------

invPrjc_df_filtered = invPrjc_df.where(invPrjc_df["MTRL_ID"].rlike("^[a-zA-Z]")== False)
invPrjc_df_final=invPrjc_df_filtered.select("SCNR","MTRL_ID","DW_MTRL_ID","LOC_ID","DW_LOC_ID","GNRTN_DT","STRT_DT","WEEK_ID","MNTH_ID","PRJCTD_INVEN_QTY","PRJCTD_OH_SHRT_DMND_QTY")
silver_df = invPrjc_df_final.withColumnRenamed("MTRL_ID","PROD_CD").withColumnRenamed("LOC_ID","LOC")

# COMMAND ----------

print(silver_df.count())

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