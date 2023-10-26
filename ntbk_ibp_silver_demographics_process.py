# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from functools import reduce
from pyspark.sql import DataFrame

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
#dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

#storing the parameters in variables
source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
#srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
sourcePath=dbutils.widgets.get("sourcePath")
dependentPath = dbutils.widgets.get("dependentDatasetPath")
demographics_raw_data = []

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

print(sourcePath)
list=sourcePath.split(';')
for path in list:
  filePath="abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  #print(filePath)
  #Reading the delta history
  deltaTable = DeltaTable.forPath(spark, filePath)
  latest_version = deltaTable.history().select(max(col('version'))).collect()[0][0]
  #Reading the data from bonze layer
  df = spark.read.format("delta").option("versionAsOf", latest_version).load(filePath)
  #Getting the max process date from the bronze data and then filtering on the max process date
  max_process_date = df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
  df_filtered = df.filter(col("PROCESS_DATE")==max_process_date)
  demographics_raw_data.append(df_filtered)

# COMMAND ----------

print(dependentPath)
list=dependentPath.split(';')
for path in list:
  if '/shipment-actuals' in path:
    shipment = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  else:    
    filePath="abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path  
    #Reading the delta history
    deltaTable = DeltaTable.forPath(spark, filePath)
    latest_version = deltaTable.history().select(max(col('version'))).collect()[0][0]
    #Reading the data from bonze layer
    df = spark.read.format("delta").option("versionAsOf", latest_version).load(filePath)
    #Getting the max process date from the bronze data and then filtering on the max process date
    max_process_date = df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
    df_filtered = df.filter(col("PROCESS_DATE")==max_process_date)
    demographics_raw_data.append(df_filtered)

# COMMAND ----------

demographics_df=reduce(DataFrame.unionByName, demographics_raw_data)

# COMMAND ----------

print("Overall Count of unemployment_es in Bronze Layer: "+str(demographics_df.count()))
print("Latest Process Date Count of unemployment_es in Bronze Layer: "+str(demographics_df.count()))

# COMMAND ----------

demographics_df_renamed=demographics_df.withColumnRenamed("country_id", "MU").withColumnRenamed("country_value", "CTRY_NM").withColumnRenamed("indicator_id", "IND_CD").withColumnRenamed("indicator_value", "IND_NM").withColumnRenamed("date", "YR").withColumnRenamed("value", "IND_VAL").withColumnRenamed("unit", "UNIT")

# COMMAND ----------

silver_df_raw=demographics_df_renamed.select("IND_NM","IND_CD","CTRY_NM","MU","IND_VAL","UNIT","YR")

# COMMAND ----------

silver_df=silver_df_raw.exceptAll(silver_df_raw.filter((silver_df_raw['IND_NM'].isNull()) & (silver_df_raw['IND_CD'].isNull()) & (silver_df_raw['CTRY_NM'].isNull()) & (silver_df_raw['MU'].isNull()) & (silver_df_raw['IND_VAL'].isNull()) & (silver_df_raw['UNIT'].isNull()) & (silver_df_raw['YR'].isNull())))

# COMMAND ----------

display(silver_df)

# COMMAND ----------

silver_df.filter((col("MU")=="PT") | (col("MU")=="ES")).select("IND_NM").distinct().display()

# COMMAND ----------

#Reading the delta history shipment
shipment_deltaTable = DeltaTable.forPath(spark, shipment)
shipment_latest_version = shipment_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(shipment_latest_version)
display(shipment_deltaTable.history())

#Reading the data from bonze layer shipment
shipment_df = spark.read.format("delta").option("versionAsOf", shipment_latest_version).load(shipment)
display(shipment_df)

# COMMAND ----------

print(shipment_df.count())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = shipment_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
shipment_df_filtered = shipment_df.filter(col("PROCESS_DATE")==max_process_date)
display(shipment_df_filtered)

print("Overall Count of shipment in Bronze Layer: "+str(shipment_df.count()))
print("Latest Process Date Count of shipment in Bronze Layer: "+str(shipment_df_filtered.count()))

# COMMAND ----------

shipment_df_filtered = shipment_df_filtered.withColumn("DMNDFCST_MKT_UNIT_CDV",col("DMDGROUP").substr(1, 2))\
                           .where((col("DMNDFCST_MKT_UNIT_CDV").isin('ES','PT')))\
                           .select("LOC","DMNDFCST_MKT_UNIT_CDV").distinct()

cond = [silver_df.MU == shipment_df_filtered.DMNDFCST_MKT_UNIT_CDV]

display(shipment_df_filtered)

silver_df = silver_df.join(shipment_df_filtered, cond).select(silver_df.IND_NM,
                                                                 silver_df.IND_CD,
                                                                 silver_df.CTRY_NM,
                                                                 silver_df.MU,
                                                                 silver_df.IND_VAL,
                                                                 silver_df.YR,
                                                                 silver_df.UNIT,
                                                                 shipment_df_filtered.LOC.alias('LOC'))

# COMMAND ----------



# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())\
                     .withColumn("YR",col("YR").cast("integer"))

# COMMAND ----------

dup=silver_df.groupBy("IND_CD","MU","YR","LOC").count().filter("count > 1")
dup.display()

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
    .tableName("sc_ibp_silver.demographics") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.demographics

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

