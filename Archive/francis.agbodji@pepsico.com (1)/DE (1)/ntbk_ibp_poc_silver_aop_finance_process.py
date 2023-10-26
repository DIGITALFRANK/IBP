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
dependentPath = dbutils.widgets.get("dependentDatasetPath")

# COMMAND ----------

bronze/iberia/mosaic-edw/ibp-poc/planning/

# COMMAND ----------



# COMMAND ----------

#splitting the dependentDatasetPath in different variables
print(dependentPath)

dependentPath_list = dependentPath.split(';')

for path in dependentPath_list:
  if '/channel-master' in path:
    aop_channel = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/financial-accounts-master' in path:
    aop_financial = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/product-master' in path:
    product = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/customer-master' in path:
    customer = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path

# COMMAND ----------

print(aop_channel)
print(aop_financial)
print(product)
print(customer)

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#Reading the delta history from the bronze path of AOP planning
aop_planning_deltaTable = DeltaTable.forPath(spark, srcPath)
aop_planning_latest_version = aop_planning_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(aop_planning_latest_version)
display(aop_planning_deltaTable.history())

# COMMAND ----------

#Reading the AOP Planning source data from bonze layer
aop_planning_df = spark.read.format("delta").option("versionAsOf", aop_planning_latest_version).load(srcPath)
display(aop_planning_df)

# COMMAND ----------

print(aop_planning_df.count())

# COMMAND ----------

display(aop_planning_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = aop_planning_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)

aop_planning_df_filtered = aop_planning_df.filter(col("PROCESS_DATE")==max_process_date)
display(aop_planning_df_filtered)

# COMMAND ----------

print("Overall Count of AOP Planning in Bronze Layer: "+str(aop_planning_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(aop_planning_df_filtered.count()))

# COMMAND ----------

###Reading the dependent datasets into dataframes
aop_channel_df = spark.read.format('delta').load(aop_channel)
aop_financial_df = spark.read.format('delta').load(aop_financial)
product_df = spark.read.format('delta').load(product)
customer_df = spark.read.format('delta').load(customer)

# COMMAND ----------

##Creating the silver dataframe
prod_cond = [aop_planning_df_filtered.DW_PLANG_MTRL_UNIT_ID == product_df.DW_PLANG_MTRL_UNIT_ID]
cust_cond = [aop_planning_df_filtered.PLANG_CUST_GRP_VAL == customer_df.PLANG_CUST_GRP_VAL]
fin_cond = [aop_planning_df_filtered.FINC_ACCT_ID == aop_financial_df.FINC_ACCT_ID]
chan_cond = [aop_planning_df_filtered.SLS_CHNL_LVL_1_CDV == aop_channel_df.SLS_CHNL_LVL_1_CDV, aop_planning_df_filtered.SLS_CHNL_LVL_2_CDV == aop_channel_df.SLS_CHNL_LVL_2_CDV, aop_planning_df_filtered.SLS_CHNL_LVL_3_CDV == aop_channel_df.SLS_CHNL_LVL_3_CDV, aop_planning_df_filtered.SLS_CHNL_LVL_4_CDV == aop_channel_df.SLS_CHNL_LVL_4_CDV, aop_planning_df_filtered.SLS_CHNL_LVL_5_CDV == aop_channel_df.SLS_CHNL_LVL_5_CDV, aop_planning_df_filtered.SHOPR_CHNL_CDV == aop_channel_df.SHOPR_CHNL_CDV]

silver_df = aop_planning_df_filtered
.join(product_df, prod_cond, "inner")
.join(customer_df, cust_cond, "inner")
.join(aop_financial_df, fin_cond, "inner")
.join(aop_channel_df, chan_cond, "inner")
.select(
product_df.BRND_CDV.alias('BRND_CDV'),
product_df.BRND_NM.alias('BRND'),
aop_planning_df_filtered.BU_CRNCY_AMT.alias('BU_CRNCY_AMT'),
aop_planning_df_filtered.BU_CRNCY_CDV.alias('BU_CRNCY_CDV'),
aop_planning_df_filtered.BUSN_MODL_TYP_VAL.alias('BUSN_MODL_TYP_VAL'),
aop_planning_df_filtered.CNSTNT_CRNCY_AMT.alias('CNSTNT_CRNCY_AMT'),
aop_planning_df_filtered.CTRY_ISO_CDV.alias('CTRY_ISO_CDV'),
aop_planning_df_filtered.ENTTY_LBL_CDV.alias('ENTTY_LBL_CDV'),
aop_planning_df_filtered.ENTTY_TYP_CDV.alias('ENTTY_TYP_CDV'),
aop_planning_df_filtered.FINC_ACCT_ID.alias('FINC_ACCT_ID'),
aop_financial_df.FINC_ACCT_TYP_NM.alias('FINC_ACCT_TYP_NM'),
aop_financial_df.FINC_ACCT_UOM_CDV.alias('FINC_ACCT_UOM_CDV'),
aop_planning_df_filtered.GTM_LOCL_LVL_0_CDV.alias('GTM_LOCL_LVL_0_CDV'),
aop_planning_df_filtered.GTM_LOCL_LVL_0_NM.alias('GTM_LOCL_LVL_0_NM'),
aop_planning_df_filtered.GTM_LOCL_LVL_1_CDV.alias('GTM_LOCL_LVL_1_CDV'),
aop_planning_df_filtered.GTM_LOCL_LVL_1_NM.alias('GTM_LOCL_LVL_1_NM'),
aop_planning_df_filtered.GTM_LOCL_LVL_2_CDV.alias('GTM_LOCL_LVL_2_CDV'),
aop_planning_df_filtered.GTM_LOCL_LVL_2_NM.alias('GTM_LOCL_LVL_2_NM'),
aop_planning_df_filtered.MNTH_NUM.alias('MNTH_NUM'),
customer_df.PLANG_CUST_GRP_NM.alias('CUST_GRP_NM'),
aop_planning_df_filtered.PLANG_CUST_GRP_VAL.alias('CUST_GRP'),
product_df.PLANG_MTRL_GRP_NM.alias('PROD_NM'),
aop_planning_df_filtered.PLANG_PROD_GRP_VAL.alias('PROD_CD'),
aop_planning_df_filtered.SCENRO_CDV.alias('SCENRO_CDV'),
aop_planning_df_filtered.SHOPR_CHNL_CDV.alias('SHOPR_CHNL_CDV'),
aop_planning_df_filtered.SLS_CHNL_CDV.alias('SLS_CHNL_CDV'),
aop_planning_df_filtered.SLS_CHNL_NM.alias('SLS_CHNL_NM'),
aop_planning_df_filtered.USD_AMT.alias('USD_AMT'),
aop_planning_df_filtered.VOL_QTY.alias('VOL_QTY'),
aop_planning_df_filtered.YR_NUM.alias('YR_NUM'),
aop_planning_df_filtered.SLS_CHNL_LVL_1_CDV.alias('SLS_CHNL_LVL_1_CDV'),
aop_channel_df.SLS_CHNL_LVL_1_NM.alias('SLS_CHNL_LVL_1_NM'),
aop_planning_df_filtered.SLS_CHNL_LVL_2_CDV.alias('SLS_CHNL_LVL_2_CDV'),
aop_channel_df.SLS_CHNL_LVL_2_NM.alias('SLS_CHNL_LVL_2_NM'),
aop_planning_df_filtered.SLS_CHNL_LVL_3_CDV.alias('SLS_CHNL_LVL_3_CDV'),
aop_channel_df.SLS_CHNL_LVL_3_NM.alias('SLS_CHNL_LVL_3_NM'),
aop_planning_df_filtered.SLS_CHNL_LVL_4_CDV.alias('SLS_CHNL_LVL_4_CDV'),
aop_channel_df.SLS_CHNL_LVL_4_NM.alias('SLS_CHNL_LVL_4_NM'),
aop_planning_df_filtered.SLS_CHNL_LVL_5_CDV.alias('SLS_CHNL_LVL_5_CDV'),
aop_channel_df.SLS_CHNL_LVL_5_NM.alias('SLS_CHNL_LVL_5_NM'),
aop_planning_df_filtered.DW_CMPT_PROD_ID.alias('DW_CMPT_PROD_ID'),
aop_planning_df_filtered.LOCL_ORG_L0_CDV.alias('LOCL_ORG_L0_CDV'),
aop_planning_df_filtered.LOCL_ORG_L0_NM.alias('LOCL_ORG_L0_NM'),
aop_planning_df_filtered.DW_PLANG_MTRL_UNIT_ID.alias('DW_PLANG_MTRL_UNIT_ID'))

# COMMAND ----------

display(silver_df)

# COMMAND ----------

print("Count: "+str(silver_df.count()))

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

