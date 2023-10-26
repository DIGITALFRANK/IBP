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

#Reading the delta history from the bronze path of Demand Forecast
src_deltaTable = DeltaTable.forPath(spark, srcPath)
src_latest_version = src_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(src_latest_version)
display(src_deltaTable.history())

# COMMAND ----------

#Reading the demand forecast source data from bonze layer
src_df = spark.read.format("delta").option("versionAsOf", src_latest_version).load(srcPath)
print(src_df.count())
display(src_df)

# COMMAND ----------

#Getting the max process date from the bronze data
max_value = src_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
#this dataset is weekly append so we need to filter the latest process date
src_df2 = src_df.filter(col("PROCESS_DATE")==max_value)
display(src_df2)

# COMMAND ----------

print(src_df.count())
print(src_df2.count())

# COMMAND ----------

#creating the silver layer dataframe for Demand Forecast
silver_df = src_df.select(col('DMNDFCST_GNRTN_DTM').alias('FCST_EXCTN_DT'),
               col('DMNDFCST_WK_STRT_DT').alias('FCST_DT'),
               col('MTRL_UNIQ_ID_VAL').alias('PROD_CD'),
               col('DMNDFCST_MTRL_GRP_UNIQ_ID_VAL').alias('PROD_NM') ,
               col('DMNDFCST_LOC_GRP_UNIQ_ID_VAL').alias('LOC'),
               col('DMND_GRP_UNIQ_ID_VAL').alias('CUST_GRP'),
               col('DMNDFCST_UNIT_CTRY_ISO_CDV').alias('MKT_UNIT'),
               col('DMNDFCST_TMFRM_NM').alias('PRD'),
               #col('DMNDFCST_QTY').alias('FCST_QTY'),
               col('DMNDFCST_TOT_UNIT_QTY').alias('FCST_QTY'),
               col('FCST_QTY_UOM_CDV').alias('FCST_QTY_UOM'),
               col('LAG_WK_QTY').alias('FCST_LAG'))

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

print(len(silver_df.columns))

# COMMAND ----------

#Additional aggregation requested by DS Team - for removing PROD_NM
silver_df = silver_df.groupBy('FCST_EXCTN_DT', 'FCST_DT', 'PROD_CD', 'LOC', 'CUST_GRP', 'MKT_UNIT', 'PRD', 'FCST_QTY_UOM', 'FCST_LAG').agg(sum('FCST_QTY').alias('FCST_QTY'))
print(len(silver_df.columns))


#Adding additional WEEK_OF_YEAR Column
#silver_df = silver_df.withColumn("WEEK_OF_YEAR", concat(year(silver_df.FCST_DT),weekofyear(silver_df.FCST_DT)))
silver_df = silver_df.withColumn("WEEK_OF_YEAR", concat(year(silver_df.FCST_DT),(when(weekofyear(silver_df.FCST_DT)<10, concat(lit(0),weekofyear(silver_df.FCST_DT))).otherwise(weekofyear(silver_df.FCST_DT)))))
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

# COMMAND ----------

