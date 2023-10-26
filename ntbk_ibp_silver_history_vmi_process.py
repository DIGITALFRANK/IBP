# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.types import DoubleType

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

#dropping the columns not required from vmi history df
silver_vmi_df = vmi_df.drop("TIPO").drop("BDC").drop("selling").drop("Formato").drop("descricao")
#.drop("Unidcaixa")
display(silver_vmi_df)

# COMMAND ----------

#Renaming columns: Producto to PROD_CD; CATEGORIA to CTGY; Localidade to LOC
silver_vmi_df = silver_vmi_df.withColumnRenamed('Producto', 'PROD_CD').withColumnRenamed('CATEGORIA', 'CTGY').withColumnRenamed('Localidade', 'LOC')
display(silver_vmi_df)

# COMMAND ----------

#Adding "_05" to PROD_CD
silver_vmi_df = silver_vmi_df.withColumn('PROD_CD', concat(col('PROD_CD'), lit('_05')))
display(silver_vmi_df)

# COMMAND ----------

#Un-pivoting the dataframe

#capturing the columns in a list
cols_list = silver_vmi_df.columns
#print(cols_list)
print(cols_list[0:3])
print(cols_list[-1])

#removing the non-date columns from the list
del(cols_list[0:3])
del(cols_list[-1])
#print(cols_list)
n = len(cols_list)
print(n)

#creating the sql statement
seperator = ','
date_list = seperator.join(cols_list).split(",")
#print(date_list)

l=[]
for i in range(len(cols_list)):
  l.append("'{}'".format(cols_list[i])+",`"+date_list[i]+"`")

k = seperator.join(l)
#print(k)

#creating a temp view
silver_vmi_df.createOrReplaceTempView('silver_view')
#running the query to unpivot the dataframe
silver_df = spark.sql("select PROD_CD, LOC, stack({0},{1}) as (AVAILDT, QTY), CTGY,Unidcaixa from silver_view".format(n,k))

# COMMAND ----------

display(silver_df)

# COMMAND ----------

print(silver_df.count()) #should be bronze_count X 1339

# COMMAND ----------

#cast QTY column to numeric
silver_df = silver_df.withColumn("QTY", silver_df["QTY"].cast(DoubleType())).withColumn("AVAILDT", silver_df["AVAILDT"].cast('timestamp'))
display(silver_df)

# COMMAND ----------

#replacing nulls in QTY with 0
silver_df = silver_df.na.fill(0, ['QTY'])

#filtering out all records with AVAILDT August 1 2021 onwards
silver_df = silver_df.filter(silver_df['AVAILDT']<'2021-08-01T00:00:00.000+0000')

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

#Joining with Customer Location Mapping and building the final dataframe
join_cond = [silver_df.LOC == config_df.location_code]

final_df = silver_df.join(config_df, join_cond, 'left').select(col('PROD_CD'),
                                                               col('customer_group').alias('CUST_GRP'),
                                                               col('LOC'),
                                                               col('AVAILDT'),
                                                               col('QTY'),
                                                               col('Unidcaixa'))
final_df=final_df.withColumn("QTY_CS",col('QTY')/col('Unidcaixa')).drop('QTY').drop('Unidcaixa')
print(final_df.count())
display(final_df)

# COMMAND ----------

#Adding the missing columns as compared to EDW build 

#Adding EXPDT with 1970-01-01
final_df = final_df.withColumn('EXPDT', lit('1970-01-01').cast('timestamp'))

#Adding PRJCT with 'INVENTARIO VMI'
final_df = final_df.withColumn('PRJCT', lit('INVENTARIO VMI'))

#Adding QUARANTINE with NULL
final_df = final_df.withColumn('QUARANTINE', lit(None))

#Adding UOM with NULL
final_df = final_df.withColumn('UOM', lit(None))

#Adding SRC_SYS_CNTRY with PT
final_df = final_df.withColumn('SRC_SYS_CNTRY', lit('PT'))

#Adding MU with PT
final_df = final_df.withColumn('MU', lit('PT'))

#Adding VMI_FLG with Y
final_df = final_df.withColumn('VMI_FLG', lit('Y'))

#Adding INVEN_BAL_DT with NULL
final_df = final_df.withColumn('INVEN_BAL_DT',lit('9999-12-31').cast('timestamp'))

display(final_df)

# COMMAND ----------

##Aggregate the full row duplicates into single unique rows
print("Number of full row duplicates: "+str(final_df.count()-final_df.distinct().count()))
#Taking distinct to remove full row duplicates
silver_df = final_df.distinct()
print(silver_df.count())

# COMMAND ----------

#Re-arranging the columns as per preference
silver_df = silver_df.select(col('PROD_CD'),
                             col('CUST_GRP'),
                             col('LOC'),
                             col('AVAILDT'),
                             col('EXPDT'),
                             col('PRJCT'),
                             col('QUARANTINE').cast('string'),
                             col('UOM').cast('string'),
                             col('QTY_CS'),
                             col('SRC_SYS_CNTRY'),
                             col('MU'),
                             col('VMI_FLG'),
                             col('INVEN_BAL_DT'))

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