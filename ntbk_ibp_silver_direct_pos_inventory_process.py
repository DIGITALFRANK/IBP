# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.types import *
from datetime import datetime
from pyspark.sql import functions as f

# COMMAND ----------

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
srcPath = dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
dependentPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
print(dependentPath)

# COMMAND ----------

#splitting the dependentDatasetPath in different variables
print(srcPath)

srcPath_list = srcPath.split(';')

for path in srcPath_list:
  if '/direct-pos-sonae-stock' in path:
    direct_pos_sonae_stock = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/direct-pos-sonae-sales' in path:
    direct_pos_sonae_sales = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/direct-pos-pingo-stock' in path:
    direct_pos_pingo_stock = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/direct-pos-pingo-sales' in path:
    direct_pos_pingo_sales = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  


# COMMAND ----------

print(direct_pos_sonae_stock)
print(direct_pos_sonae_sales)
print(direct_pos_pingo_stock)
print(direct_pos_pingo_sales)

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

#Reading the delta history from the bronze path of direct-pos-sonae-stock
sonae_stock_deltaTable = DeltaTable.forPath(spark, direct_pos_sonae_stock)
sonae_stock_latest_version = sonae_stock_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(sonae_stock_latest_version)
display(sonae_stock_deltaTable.history())

#Reading the delta history from the bronze path of direct-pos-sonae-sales
sonae_sales_deltaTable = DeltaTable.forPath(spark, direct_pos_sonae_sales)
sonae_sales_latest_version = sonae_sales_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(sonae_sales_latest_version)
display(sonae_sales_deltaTable.history())

client_prod_mapping_deltaTable = DeltaTable.forPath(spark, dependentPath)
client_prod_mapping_latest_version = client_prod_mapping_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(client_prod_mapping_latest_version)
display(client_prod_mapping_deltaTable.history())

#Reading the delta history from the bronze path of direct-pos-pingo-stock
pingo_stock_deltaTable = DeltaTable.forPath(spark, direct_pos_pingo_stock)
pingo_stock_latest_version = pingo_stock_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(pingo_stock_latest_version)
display(pingo_stock_deltaTable.history())

#Reading the delta history from the bronze path of direct-pos-pingo-sales
pingo_sales_deltaTable = DeltaTable.forPath(spark, direct_pos_pingo_sales)
pingo_sales_latest_version = pingo_sales_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(pingo_sales_latest_version)
display(pingo_sales_deltaTable.history())

# COMMAND ----------

sonae_stock_df = spark.read.format("delta").option("versionAsOf", sonae_stock_latest_version).load(direct_pos_sonae_stock)
print(sonae_stock_df.count())
display(sonae_stock_df)

sonae_sales_df = spark.read.format("delta").option("versionAsOf", sonae_sales_latest_version).load(direct_pos_sonae_sales)
print(sonae_sales_df.count())
display(sonae_sales_df)

client_prod_mapping_df = spark.read.format("delta").option("versionAsOf", sonae_sales_latest_version).load(dependentPath)
print(client_prod_mapping_df.count())
display(client_prod_mapping_df)

# # display(main_df)
pingo_stock_df = spark.read.format("delta").option("versionAsOf", pingo_stock_latest_version).load(direct_pos_pingo_stock)
print(pingo_stock_df.count())
display(pingo_stock_df)

pingo_sales_df = spark.read.format("delta").option("versionAsOf", pingo_sales_latest_version).load(direct_pos_pingo_sales)
print(pingo_sales_df.count())
display(pingo_sales_df)


# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = sonae_stock_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
sonae_stock_df_filtered = sonae_stock_df.filter(col("PROCESS_DATE")==max_process_date)
display(sonae_stock_df_filtered)

max_process_date = sonae_sales_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
sonae_sales_df_filtered = sonae_sales_df.filter(col("PROCESS_DATE")==max_process_date)
display(sonae_sales_df_filtered)

max_process_date = client_prod_mapping_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
client_prod_mapping_df_filtered = client_prod_mapping_df.filter(col("PROCESS_DATE")==max_process_date)
display(client_prod_mapping_df_filtered)

max_process_date = pingo_stock_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
pingo_stock_df_filtered_temp = pingo_stock_df.filter(col("PROCESS_DATE")==max_process_date)
pingo_stock_df_filtered = pingo_stock_df_filtered_temp.filter("Data is not null")
display(pingo_stock_df_filtered)

max_process_date = pingo_sales_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
pingo_sales_df_filtered_temp = pingo_sales_df.filter(col("PROCESS_DATE")==max_process_date)
pingo_sales_df_filtered = pingo_sales_df_filtered_temp.filter("Data is not null")
display(pingo_sales_df_filtered)

# COMMAND ----------

print("Overall Count of Sonae stock  in Bronze Layer: "+str(sonae_stock_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(sonae_stock_df_filtered.count()))

print("Overall Count of Sonae sales in Bronze Layer: "+str(sonae_sales_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(sonae_sales_df_filtered.count()))

print("Overall Count of Sonae sales in Bronze Layer: "+str(client_prod_mapping_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(client_prod_mapping_df_filtered.count()))

print("Overall Count of Pingo stock in Bronze Layer: "+str(pingo_stock_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(pingo_stock_df_filtered.count()))

print("Overall Count of Pingo sales in Bronze Layer: "+str(pingo_sales_df.count()))
print("Latest Process Date Count of AOP Planning in Bronze Layer: "+str(pingo_sales_df_filtered.count()))

# COMMAND ----------

### Selecting the required columns for Sonae & prod mapping
sonae_stock_sel_df = sonae_stock_df_filtered.select(col('DATA').alias('STRT_DT'),
                                                    col('SOH'),col('ARTIGO-PACK').alias('ARTIGO_PACK'),col('STORE'))

sonae_sales_sel_df = sonae_sales_df_filtered.select(col('data').alias('STRT_DT'),
                                                   col('ARTIGO-PACK').alias('ARTIGO_PACK'),col('QTD'),col('loja'))

client_prod_mapping_sel_df = client_prod_mapping_df_filtered.select(col('DMDUNIT').alias('PROD_CD'),                                                                   col('Sonae_code_that_we_use_in_the_data_files_of_sell_and_stocks').alias('Sonae_SKU'),            col('Pingo_Doce_code_that_we_use_in_the_data_files_of_sell_and_stocks').alias('PD_SKU'))

# display(client_prod_mapping_sel_df)


# COMMAND ----------

## Start Creating the silver dataframe - Sonae

##Join condition
sonae_stock_prod_cond = [sonae_stock_sel_df.ARTIGO_PACK == client_prod_mapping_sel_df.Sonae_SKU]
sonae_sales_prod_cond = [sonae_sales_sel_df.ARTIGO_PACK == client_prod_mapping_sel_df.Sonae_SKU]

sonae_stock_prod_ext_join = sonae_stock_sel_df.join(client_prod_mapping_sel_df,sonae_stock_prod_cond,"left").withColumn("LOC",when(sonae_stock_sel_df.STORE.contains('Madeira'),lit("FG_CU_PT_MAD_01")).otherwise("PL_PT_CAR_01")).select(client_prod_mapping_sel_df.PROD_CD,sonae_stock_sel_df.STRT_DT.alias('STRT_DT'),"LOC",sonae_stock_sel_df.SOH.alias('QTY').cast(DoubleType()))
sonae_stock_prod_ext_join.printSchema()

sonae_stock_grp = sonae_stock_prod_ext_join.groupBy("PROD_CD","LOC","STRT_DT").agg(sum("QTY").alias("QTY"))
display(sonae_stock_grp)

sonae_sales_prod_ext_join = sonae_sales_sel_df.join(client_prod_mapping_sel_df,sonae_sales_prod_cond,"left").withColumn("LOC",when(sonae_sales_sel_df.loja.contains('Madeira'),lit("FG_CU_PT_MAD_01")).otherwise("PL_PT_CAR_01")).select(client_prod_mapping_sel_df.PROD_CD.alias("PROD_CD_SONAESALES"),sonae_sales_sel_df.STRT_DT.alias('STRT_DT_SONAESALES'),col('LOC').alias('LOC_SONAESALES'),sonae_sales_sel_df.QTD.alias('SLS_OUT_UNITS').cast(DoubleType()))
sonae_stock_prod_ext_join.printSchema()

sonae_sales_grp = sonae_sales_prod_ext_join.groupBy("PROD_CD_SONAESALES","LOC_SONAESALES","STRT_DT_SONAESALES").agg(sum("SLS_OUT_UNITS").alias("SLS_OUT_UNITS"))
display(sonae_sales_grp)

##Join Sonae Stock and Vendas datasets
sonae_stock_sales_cond = [sonae_stock_grp.PROD_CD == sonae_sales_grp.PROD_CD_SONAESALES,sonae_stock_grp.STRT_DT == sonae_sales_grp.STRT_DT_SONAESALES,sonae_stock_grp.LOC == sonae_sales_grp.LOC_SONAESALES]

silver_res_df1 = sonae_stock_grp.join(sonae_sales_grp,sonae_stock_sales_cond,"full").withColumn("CUST_GRP",lit("PT_OT_Sonae")).select(coalesce(sonae_stock_grp.PROD_CD,sonae_sales_grp.PROD_CD_SONAESALES).alias('PROD_CD'),"CUST_GRP",coalesce(sonae_stock_grp.STRT_DT,sonae_sales_grp.STRT_DT_SONAESALES).alias('STRT_DT'),coalesce(sonae_stock_grp.LOC,sonae_sales_grp.LOC_SONAESALES).alias('LOC'),sonae_stock_grp.QTY.alias('QTY').cast(DoubleType()),sonae_sales_grp.SLS_OUT_UNITS.alias('SLS_OUT_UNITS').cast(DoubleType()))

display(silver_res_df1)


# COMMAND ----------

### Selecting the required columns for Pingo & prod mapping
pingo_stock_sel_df = pingo_stock_df_filtered.select(col('Data').alias('STRT_DT'),col('Cdigo_Interno_RTL'),col('Stock_pcs'),col('Loja'))
display(pingo_stock_sel_df)

pingo_sales_sel_df = pingo_sales_df_filtered.select(col('Data').alias('STRT_DT'),col('Cdigo_Interno_RTL'),col('Qtd_Pcs'),col('Loja'))
display(pingo_sales_sel_df)

# COMMAND ----------

## Start Creating the silver dataframe - Pingo

##Join condition
pingo_stock_prod_cond = [pingo_stock_sel_df.Cdigo_Interno_RTL == client_prod_mapping_sel_df.PD_SKU]
pingo_sales_prod_cond = [pingo_sales_sel_df.Cdigo_Interno_RTL == client_prod_mapping_sel_df.PD_SKU]

pingo_stock_prod_ext_join = pingo_stock_sel_df.join(client_prod_mapping_sel_df,pingo_stock_prod_cond,"left").withColumn("LOC",lit("PL_PT_CAR_01")).select(client_prod_mapping_sel_df.PROD_CD,pingo_stock_sel_df.STRT_DT.alias('STRT_DT'),"LOC",pingo_stock_sel_df.Stock_pcs.alias('QTY').cast(DoubleType()))

pingo_stock_grp = pingo_stock_prod_ext_join.groupBy("PROD_CD","LOC","STRT_DT").agg(sum("QTY").alias("QTY"))
display(pingo_stock_grp)

pingo_sales_prod_ext_join = pingo_sales_sel_df.join(client_prod_mapping_sel_df,pingo_sales_prod_cond,"left").withColumn("LOC",lit("PL_PT_CAR_01")).select(client_prod_mapping_sel_df.PROD_CD.alias('PROD_CD_PINGOSALES'),pingo_sales_sel_df.STRT_DT.alias('STRT_DT_PINGOSALES'),col('LOC').alias('LOC_PINGOSALES'),pingo_sales_sel_df.Qtd_Pcs.alias('SLS_OUT_UNITS').cast(DoubleType()))

pingo_sales_grp = pingo_sales_prod_ext_join.groupBy("PROD_CD_PINGOSALES","LOC_PINGOSALES","STRT_DT_PINGOSALES").agg(sum("SLS_OUT_UNITS").alias("SLS_OUT_UNITS"))
display(pingo_sales_grp)

#Join Pingo Stock and Vendas datasets
pingo_stock_sales_cond = [pingo_stock_grp.PROD_CD == pingo_sales_grp.PROD_CD_PINGOSALES,pingo_stock_grp.LOC == pingo_sales_grp.LOC_PINGOSALES,pingo_stock_grp.STRT_DT == pingo_sales_grp.STRT_DT_PINGOSALES]

silver_res_df2 = pingo_stock_grp.join(pingo_sales_grp,pingo_stock_sales_cond,"full").withColumn("CUST_GRP",lit("PT_OT_PINGODOCE")).select(coalesce(pingo_stock_grp.PROD_CD,pingo_sales_grp.PROD_CD_PINGOSALES).alias('PROD_CD'),"CUST_GRP",coalesce(pingo_stock_grp.STRT_DT,pingo_sales_grp.STRT_DT_PINGOSALES).alias('STRT_DT'), coalesce(pingo_stock_grp.LOC,pingo_sales_grp.LOC_PINGOSALES).alias('LOC'),pingo_stock_grp.QTY.alias('QTY').cast(DoubleType()),pingo_sales_grp.SLS_OUT_UNITS.alias('SLS_OUT_UNITS').cast(DoubleType()))

display(silver_res_df2)


# COMMAND ----------

union_data_temp = silver_res_df1.unionAll(silver_res_df2)
union_data = union_data_temp.withColumn("STRT_DT", f.from_unixtime(f.unix_timestamp("STRT_DT",'M/d/yyyy hh:mm:ss a'),'yyyy-MM-dd').cast('date'))
display(union_data)

# COMMAND ----------

union_data = union_data.withColumn("MU",lit("PT")).withColumn("PROCESS_DATE",current_date())
display(union_data)

# COMMAND ----------

###Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = union_data.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  union_data.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  union_data.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  union_data.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.direct_pos_inventory") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.direct_pos_inventory

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(union_data.count()))

# COMMAND ----------

