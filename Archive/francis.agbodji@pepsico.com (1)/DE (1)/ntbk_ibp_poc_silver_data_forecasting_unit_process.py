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
dfuPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#splitting the dependentDatasetPath in different variables
dependentPath = dbutils.widgets.get("dependentDatasetPath")

print(dependentPath)

dependentPath_list = dependentPath.split(';')

for path in dependentPath_list:
  if '/dfu-view' in path:
    dfuPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path  
  if '/product-master' in path:
    productmaster = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/distribution-master' in path:
    distributionmaster = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path

# COMMAND ----------

print(dfuPath)
print(productmaster)
print(distributionmaster)

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
dfu_deltaTable = DeltaTable.forPath(spark, srcPath)
dfu_latest_version = dfu_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(dfu_latest_version)
display(dfu_deltaTable.history())

# COMMAND ----------

#Reading the DFU source data from bonze layer
dfu_df = spark.read.format("delta").option("versionAsOf", dfu_latest_version).load(srcPath)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
#max_value = dfu_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
#print(max_value)
#dfu_df2 = dfu_df.filter(col("PROCESS_DATE")==max_value)
#display(dfu_df2)

# COMMAND ----------

#comparing the counts
#print(dfu_df.count())
#print(dfu_df2.count())

# COMMAND ----------

## DQ Check
pklist2 = ['PLANG_CUST_GRP_VAL','PLANG_MTRL_GRP_VAL','PLANG_LOC_GRP_VAL','DMND_CALC_MDL_NM']
pklist2 = ','.join(str(e) for e in pklist2)
if len(pklist2.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in pklist2.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+pklist2+"').isNull()"

dfu_dq = dfu_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pklist2.split(','))
                    .orderBy(desc("DW_LAST_UPDT_DTM")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key"))
                                                           .when(to_timestamp(substring(col("DMND_POST_DT"),1,10),'yyyy-mm-dd').isNull(),lit("NoFollowedDateformat")))

dfu_dq_pass = dfu_dq.where(col("Corrupt_Record").isNull()).drop("DUP_CHECK","Corrupt_Record")

# COMMAND ----------

#reading the bronze data for DFU_View for joining to shipments
#joinPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp-poc/dfu-view"
join_deltaTable = DeltaTable.forPath(spark, dfuPath)
join_latest_version = join_deltaTable.history().select(max(col('version'))).collect()[0][0]

# COMMAND ----------

#reading the bronze DFU data in a dataframe
join_df = spark.read.format("delta").option("versionAsOf", join_latest_version).load(dfuPath)

# COMMAND ----------

###Reading the dependent datasets into dataframes
productmaster_df = spark.read.format('delta').load(productmaster)
distributionmaster_df = spark.read.format('delta').load(distributionmaster)

# COMMAND ----------

#creating the silver dataframe
join_cond = [dfu_dq_pass.PLANG_MTRL_GRP_VAL == join_df.PLANG_MTRL_GRP_VAL, dfu_dq_pass.PLANG_CUST_GRP_VAL == join_df.PLANG_CUST_GRP_VAL, dfu_dq_pass.PLANG_LOC_GRP_VAL == join_df.PLANG_LOC_GRP_VAL]
productmaster_cond=[dfu_dq_pass.PLANG_MTRL_GRP_VAL == productmaster_df.PLANG_MTRL_GRP_VAL]
distributionmaster_cond=[dfu_dq_pass.PLANG_LOC_GRP_VAL==distributionmaster_df.PLANG_LOC_GRP_VAL]
silver_df = dfu_dq_pass.join(join_df, join_cond).join(productmaster_df,productmaster_cond).join(distributionmaster_df,distributionmaster_cond).filter(dfu_dq_pass.DMNDFCST_UNIT_LVL_VAL=='SB-S-FL-ITEM_CLIENT_DC').select(dfu_dq_pass.PLANG_MTRL_GRP_VAL.alias('PROD_CD'),
               dfu_dq_pass.PLANG_LOC_GRP_VAL.alias('LOC'),
               dfu_dq_pass.PLANG_CUST_GRP_VAL.alias('CUST_GRP'),
               dfu_dq_pass.DMND_CALC_MDL_NM.alias('MODL') ,
               dfu_dq_pass.DMNDFCST_UNIT_LVL_VAL.alias('DFU_LVL_VAL'),
               #dfu_dq_pass.DMND_POST_DT.alias('DMNDPOSTDT'),
               join_df.DMNDPLN_FCST_UNIT_UOM_CDV.alias("FCST_UNIT_UOM"),
               join_df.DMNDFCST_UNIT_TO_8OZ_CNVRSN_FCTR.alias("TO8OZ"),
               join_df.DMNDFCST_UNIT_TO_KG_CNVRSN_FCTR.alias("TOKG"),
               join_df.DMNDFCST_UNIT_TO_LT_CNVRSN_FCTR.alias("TOLT"),
               join_df.DMNDFCST_UNIT_TO_EA_CNVRSN_FCTR.alias("UNIT_TO_EA_CNVRSN_FCTR"),
               join_df.DMNDFCST_UNIT_TO_CASE_CNVRSN_FCTR.alias("TOCASE"),
               join_df.DMNDFCST_UNIT_TO_EQVLNT_CASE_CNVRSN_FCTR.alias("UNIT_TO_EQVLNT_CASE_CNVRSN_FCTR"),
               productmaster_df.BRND_NM.alias("BRND"),
               distributionmaster_df.PLANG_LOC_BU_NM.alias("BU")
               )

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

# COMMAND ----------

#checking the silver layer count
#print(tgtPath)
#silver_deltaTable = DeltaTable.forPath(spark, tgtPath)
#silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
#print(silver_latest_version)
#display(silver_deltaTable.history())