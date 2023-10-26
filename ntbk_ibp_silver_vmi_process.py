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
configPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

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

#Reading the delta history from the bronze path of VMI table
vmi_deltaTable = DeltaTable.forPath(spark, srcPath)
vmi_latest_version = vmi_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(vmi_latest_version)
display(vmi_deltaTable.history())

# COMMAND ----------

#Reading the vmi edw source data from bronze layer
vmi_df = spark.read.format("delta").option("versionAsOf", vmi_latest_version).load(srcPath)
print(vmi_df.count())
display(vmi_df)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = vmi_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
vmi_df2 = vmi_df#.filter(col("PROCESS_DATE")==max_value)
display(vmi_df2)

# COMMAND ----------

#comparing the counts
print(vmi_df.count())
print(vmi_df2.count())

# COMMAND ----------

#reading the customer-location mapping from config folder in bronze layer
config_deltaTable = DeltaTable.forPath(spark, configPath)
config_latest_version = config_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(config_latest_version)
display(config_deltaTable.history())

# COMMAND ----------

#reading the customer-location mapping in a dataframe
config_df = spark.read.format("delta").option("versionAsOf", config_latest_version).load(configPath)
print(config_df.count())
display(config_df)

# COMMAND ----------

print(vmi_df2.count())
print(vmi_df2.filter(vmi_df2.LOC_ID == 'PL_PT_CAR_01').count())

# COMMAND ----------

#selecting the required columns from VMI bronze layer and joining to customer maping
join_cond = [vmi_df2.LOC_ID == config_df.location_code]

vmi_silver_df = vmi_df2.join(config_df, join_cond, 'left').select(col('MTRL_ID').alias('PROD_CD'),
                               col('customer_group').alias('CUST_GRP'),
                               col('LOC_ID').alias('LOC'),
                               #col('INVEN_BAL_DT').alias('AVAILDT'),
                               col('INVEN_GNRTN_DT').alias('AVAILDT'),                                  
                               col('MTRL_EXPR_DT').alias('EXPDT'),
                               col('SPLY_PLANG_PRJCT_ID').alias('PRJCT'),
                               col('MTRL_QRNTN_FLG').alias('QUARANTINE'),
                               col('SRC_MTRL_UOM_CDV').alias('UOM'),
                               col('SRC_INVEN_QTY').alias('QTY'),
                               col('CTRY_CDV').alias('SRC_SYS_CNTRY'),
                               col('MU_CDV').alias('MU'),
                               col('INVEN_BAL_DT').alias('INVEN_BAL_DT'))
                               #col('MTRL_CTGY_NM').alias('CTGY')
                                                                 

display(vmi_silver_df)

# COMMAND ----------

#Counts in silver layer
print(vmi_silver_df.count())
print(vmi_silver_df.filter(vmi_silver_df.LOC == 'PL_PT_CAR_01').count())

# COMMAND ----------

#Adding vmi_flg column
silver_df = vmi_silver_df.withColumn('VMI_FLG', when(vmi_silver_df.PRJCT.like('%VMI%'), 'Y').otherwise('N')).withColumnRenamed("QTY","QTY_CS")
display(silver_df)

# COMMAND ----------

#Filtering only the records where VMI_FLG is Y
print(silver_df.count())
silver_df = silver_df.filter(col('vmi_flg') == 'Y')
print(silver_df.count())

# COMMAND ----------

#Replacing null in CUST_GRP with a dummy value 'NA'
silver_df = silver_df.na.fill('NA', ['CUST_GRP'])
display(silver_df)

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
    .tableName("sc_ibp_silver.vmi") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.vmi

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))