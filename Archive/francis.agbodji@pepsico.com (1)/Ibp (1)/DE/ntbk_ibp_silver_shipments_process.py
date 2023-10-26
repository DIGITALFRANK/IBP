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
dfuPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

tgtPath

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

#Reading the delta history from the bronze path of Shipments table
ship_deltaTable = DeltaTable.forPath(spark, srcPath)
ship_latest_version = ship_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(ship_latest_version)
display(ship_deltaTable.history())

# COMMAND ----------

#Reading the shipments source data from bonze layer
ship_df = spark.read.format("delta").option("versionAsOf", ship_latest_version).load(srcPath)
print(ship_df.count())
display(ship_df)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = ship_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
ship_df2 = ship_df.filter(col("PROCESS_DATE")==max_value).drop("DELTA_DATE")
display(ship_df2)

# COMMAND ----------

#comparing the counts
print(ship_df.count())
print(ship_df2.count())

# COMMAND ----------

#reading the bronze data for DFU for joining to shipments
join_deltaTable = DeltaTable.forPath(spark, dfuPath)
join_latest_version = join_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(join_latest_version)
display(join_deltaTable.history())

# COMMAND ----------

#reading the bronze DFU data in a dataframe
join_df = spark.read.format("delta").option("versionAsOf", join_latest_version).load(dfuPath)
print(join_df.count())
display(join_df)

# COMMAND ----------

#creating the silver dataframe
cond = [ship_df2.PLANG_MTRL_GRP_VAL == join_df.PLANG_MTRL_GRP_VAL, ship_df2.PLANG_CUST_GRP_VAL == join_df.PLANG_CUST_GRP_VAL, ship_df2.PLANG_LOC_GRP_VAL == join_df.PLANG_LOC_GRP_VAL]
temp_silver_df = ship_df2.join(join_df, cond).filter(join_df.DMNDFCST_UNIT_LVL_VAL=='SB-S-FL-ITEM_CLIENT_DC').select(ship_df2.PLANG_MTRL_GRP_VAL.alias('PROD_CD'),
               ship_df2.PLANG_CUST_GRP_VAL.alias('CUST_GRP'),
               ship_df2.PLANG_LOC_GRP_VAL.alias('LOC'),
               ship_df2.DMNDFCST_MKT_UNIT_CDV.alias('MKT_UNIT') ,
               ship_df2.MTRL_CTGY_NM.alias('CTGY'),
               ship_df2.HSTRY_TMFRM_STRT_DT.alias('STRTDT'),  ##change request: change STRTDATE TO STRTDT
               ship_df2.DMND_HSTRY_QTY.alias('QTY').cast('float'),
               ship_df2.DMND_HSTRY_SRC_UOM_QTY.alias('UOM'),
               ship_df2.HSTRY_DURTN_MIN_QTY.alias('DURTN'),
               ship_df2.DMND_HSTRY_STREM_NM.alias('HISTSTREM'),
               ship_df2.DMND_HSTRY_TYP_CDV.alias('TYP'))

# COMMAND ----------

#Adding the WEEK_OF_YEAR column as requested by DS Team
#silver_df = temp_silver_df.withColumn("WEEK_OF_YEAR", concat(year(temp_silver_df.STRTDT),weekofyear(temp_silver_df.STRTDT)))
silver_df = temp_silver_df.withColumn("WEEK_OF_YEAR", concat(year(temp_silver_df.STRTDT),(when(weekofyear(temp_silver_df.STRTDT)<10, concat(lit(0),weekofyear(temp_silver_df.STRTDT))).otherwise(weekofyear(temp_silver_df.STRTDT)))))
display(silver_df)

# COMMAND ----------

print("Count: "+str(silver_df.count()))
display(silver_df)

# COMMAND ----------

'''
##General Data Quality checks on Primary Key columns of Silver dataframe
##Primary Key null check
#creating the null condition statement
if len(pkList.split(','))>1:
  ls = [attr+" is NULL" for attr in pkList.split(',')]
  null_cond = " or ".join(ls)
else :
  null_cond = pkList+" is NULL"
print(null_cond)
#filtering the silver dataframe for null values on PK columns
null_df = silver_df.filter(null_cond)
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")
#removing the null rows from silver dataframe
silver_df = silver_df.exceptAll(null_df)

#Primary Key duplicate check
dup_df = silver_df.groupBy(pkList.split(',')).count().filter("count > 1")
display(dup_df)
'''

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

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

