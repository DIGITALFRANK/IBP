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

#Reading the delta history from the bronze path of Distribution Center
src_deltaTable = DeltaTable.forPath(spark, srcPath)
src_latest_version = src_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(src_latest_version)
display(src_deltaTable.history())

# COMMAND ----------

#Reading the Distribution Center source data from bronze layer
src_df = spark.read.format("delta").option("versionAsOf", src_latest_version).load(srcPath)
print(src_df.count())
display(src_df)

# COMMAND ----------

#creating the silver layer dataframe for Distribution Center Master
silver_df = src_df.select(col('PLANG_LOC_GRP_VAL').alias('LOC'),
               col('PLANG_LOC_GRP_TYP_NM').alias('LVL'),
               col('PLANG_LOC_STTS_NM').alias('STTS'),
               col('PLANG_LOC_SCTR_NM').alias('SCTR') ,
               col('HRCHY_LVL_2_NM').alias('RGN'),
               col('PLANG_LOC_DIVSN_NM').alias('DIVSN'),
               col('PLANG_LOC_GRP_NM').alias('DSC'),
               col('PLANG_LOC_CTRY_ISO_CDV').alias('CTRY'),
               col('PLANG_LOC_CITY_NM').alias('CITY'),
               col('PLANG_LOC_BU_NM').alias('BU'),
               col('PLANG_LOC_UNIT_TYP_SHRT_NM').alias('ABRVTN_TYP'),
               col('PLANG_LOC_UNIT_TYP_CDV').alias('LOC_TYP'),
               col('PLANG_LOC_WRKNG_CALDR_ID').alias('WRKNG_CALDR'),
               col('PLANG_LOC_PSTL_AREA_VAL').alias('PSTLCD'),
               col('CRNCY_CDV').alias('CRNCY'),
               col('INVEN_PURCH_BORWNG_RT').alias('BORWNGPCT'),
               col('PLANG_LOC_GRP_TYP_NM').alias('HRCHYLVL'),
               col('AUTMTC_DMNDFCST_UNIT_CREATN_FLG').alias('CRTDFU'),
               col('PLANG_LOC_PCKGNG_RSRC_FLG').alias('PCKGNG'),
               col('PLANG_LOC_SHPMT_SCHEDR_PARNT_NM').alias('SSPARNT'),
               col('PLANG_LOC_SHPMT_SCHEDR_CHLD_NM').alias('SSCHLD_NM'))
              

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