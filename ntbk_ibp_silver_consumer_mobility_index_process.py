# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql import functions as F

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

srcPath

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

#Reading the delta history from the bronze path of Customer Master
src_deltaTable = DeltaTable.forPath(spark, srcPath)
src_latest_version = src_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(src_latest_version)
display(src_deltaTable.history())

# COMMAND ----------

#Reading the  source data from bronze layer
src_df = spark.read.format("delta").option("versionAsOf", src_latest_version).load(srcPath)
print(src_df.count())
display(src_df)

# COMMAND ----------

#The location mapping file
loc_map = spark.read.csv(dpdPath, header="true", inferSchema="true", encoding="ISO-8859-1")
display(loc_map)

# COMMAND ----------

# src_df.filter((src_df.sub_region_1=="Andalusia")&(src_df.date=="2021-01-01")).display()

# COMMAND ----------

# src_df.filter((src_df.country_region=="Portugal")).select("country_region","sub_region_1","sub_region_2").distinct().display()

# COMMAND ----------

# winSpec = Window.partitionBy('sub_region_1','date').orderBy('date')

# pt_cols_to_remove = ["retail_and_recreation_percent_change_from_baseline","grocery_and_pharmacy_percent_change_from_baseline","parks_percent_change_from_baseline","transit_stations_percent_change_from_baseline","workplaces_percent_change_from_baseline","residential_percent_change_from_baseline"]

consumer_mobility_loc = src_df.join(loc_map, when(src_df.country_region_code=="ES", (src_df.sub_region_1==loc_map.External_Region) & (src_df.sub_region_2==loc_map.External_Province)).otherwise(src_df.sub_region_1==loc_map.External_Region),"right")


#consumer_mobility_loc = src_df.join(loc_map,(src_df.sub_region_1==loc_map.External_Region) & (src_df.sub_region_2==loc_map.External_Province),"right")
consumer_mobility_loc = consumer_mobility_loc.filter((consumer_mobility_loc.date).isNotNull())

# consumer_mobility_loc_PT = consumer_mobility_loc.filter(col("country_region_code")=="PT")
# consumer_mobility_loc_ES = consumer_mobility_loc.filter(col("country_region_code")=="ES")

# consumer_mobility_loc_PT = consumer_mobility_loc_PT\
# .withColumn("RTL_AND_RCRTN_PCNT_CHNG_FRM_BSLNE",F.avg('retail_and_recreation_percent_change_from_baseline').over(winSpec))\
# .withColumn("GRCRY_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE",F.avg('grocery_and_pharmacy_percent_change_from_baseline').over(winSpec))\
# .withColumn("PRKS_PCNT_CHNG_FRM_BSLNE",F.avg('parks_percent_change_from_baseline').over(winSpec))\
# .withColumn("TRNST_STTNS_PCNT_CHNG_FRM_BSLNE",F.avg('transit_stations_percent_change_from_baseline').over(winSpec))\
# .withColumn("WRKPLCS_PCNT_CHNG_FRM_BSLNE",F.avg('workplaces_percent_change_from_baseline').over(winSpec))\
# .withColumn("RSDNTL_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE",F.avg('residential_percent_change_from_baseline').over(winSpec))
# 
# consumer_mobility_loc_PT = consumer_mobility_loc_PT.drop(*pt_cols_to_remove)
# 
# consumer_mobility_loc_ES = consumer_mobility_loc_ES\
# .withColumnRenamed("retail_and_recreation_percent_change_from_baseline", "RTL_AND_RCRTN_PCNT_CHNG_FRM_BSLNE")\
# .withColumnRenamed("grocery_and_pharmacy_percent_change_from_baseline", "GRCRY_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE")\
# .withColumnRenamed("parks_percent_change_from_baseline", "PRKS_PCNT_CHNG_FRM_BSLNE")\
# .withColumnRenamed("transit_stations_percent_change_from_baseline", "TRNST_STTNS_PCNT_CHNG_FRM_BSLNE")\
# .withColumnRenamed("workplaces_percent_change_from_baseline", "WRKPLCS_PCNT_CHNG_FRM_BSLNE")\
# .withColumnRenamed("residential_percent_change_from_baseline", "RSDNTL_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE")

#creating the silver layer data frame

# silver_df = consumer_mobility_loc_PT.union(consumer_mobility_loc_ES)

silver_df = consumer_mobility_loc\
.withColumnRenamed("Channel", "MU_CHNL")\
.withColumnRenamed("Category", "CTGY")\
.withColumnRenamed("LOCATION", "LOC")\
.withColumnRenamed("country_region", "CNTRY")\
.withColumnRenamed("sub_region_1", "RGN")\
.withColumnRenamed("sub_region_2", "PRVNC")\
.withColumnRenamed("metro_area", "MTR_AREA")\
.withColumnRenamed("date", "DT")\
.withColumnRenamed("retail_and_recreation_percent_change_from_baseline", "RTL_AND_RCRTN_PCNT_CHNG_FRM_BSLNE")\
.withColumnRenamed("grocery_and_pharmacy_percent_change_from_baseline", "GRCRY_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE")\
.withColumnRenamed("parks_percent_change_from_baseline", "PRKS_PCNT_CHNG_FRM_BSLNE")\
.withColumnRenamed("transit_stations_percent_change_from_baseline", "TRNST_STTNS_PCNT_CHNG_FRM_BSLNE")\
.withColumnRenamed("workplaces_percent_change_from_baseline", "WRKPLCS_PCNT_CHNG_FRM_BSLNE")\
.withColumnRenamed("residential_percent_change_from_baseline", "RSDNTL_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE")

silver_df =silver_df.select("MU_CHNL","CTGY","LOC","MU","CNTRY","RGN","PRVNC","MTR_AREA",
                            col("RTL_AND_RCRTN_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("GRCRY_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("PRKS_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("TRNST_STTNS_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("WRKPLCS_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("RSDNTL_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE").cast("int"),
                            col("DT").cast("timestamp")
                           )
silver_df = silver_df.na.drop(subset=["PRVNC"])

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

silver_df.select("TRNST_STTNS_PCNT_CHNG_FRM_BSLNE","DT","PRVNC","RGN","MU").filter(col("TRNST_STTNS_PCNT_CHNG_FRM_BSLNE").isNull()).distinct().display()

# COMMAND ----------

src_df.filter((src_df.date=="2021-02-17")&(src_df.country_region=="Portugal")).display()

# COMMAND ----------

# src_df.filter((src_df.sub_region_1=="Braga")&(src_df.date=="2020-03-15")).display()

# COMMAND ----------

# (src_df.sub_region_2=="BEJA")
# consumer_mobility_loc_PT.filter((consumer_mobility_loc_PT.sub_region_1=="Braga")&(consumer_mobility_loc_PT.date=="2020-03-15")).display()

# COMMAND ----------

silver_df.groupby("MU","RGN","PRVNC","MU_CHNL","CTGY","LOC","DT").count().filter(("count > 1") ).display()

# COMMAND ----------

silver_df.filter(col("CTGY").isNull()).display()

# COMMAND ----------

silver_df.filter(col("RGN").isNull()).display()

# COMMAND ----------

silver_df.filter(col("RGN").isNull()).display()

# COMMAND ----------

silver_df.count()

# COMMAND ----------

silver_df.filter("PRVNC is null").select("MU").distinct().display()
# total count
#PRVNC is null count 75582
silver_df.filter("PRVNC is null").count()

# COMMAND ----------

null_df = silver_df.filter("MU is NULL or DT is NULL or RGN is NULL or MU_CHNL is NULL or CTGY is NULL or LOC is NULL or PRVNC is NULL")  
null_df.count()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

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

# spark.sql("CREATE DATABASE IF NOT EXISTS sc_ibp_silver")

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.consumer_mobility_index") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# if spark._jsparkSession.catalog().tableExists('sc_ibp_silver', 'consumer_mobility_index'):
#   spark.sql("OPTIMIZE sc_ibp_silver.consumer_mobility_index")
# else :
#   print("Table doesnt exist")

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.consumer_mobility_index

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

df = spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/consumer-mobility-index")
df.count()

# COMMAND ----------

df.display()

# COMMAND ----------

df.filter("PRVNC is null").select("MU").distinct().display()
# total count
#PRVNC is null count 75582
df.filter("PRVNC is null").count()

# COMMAND ----------

df.groupby("MU","RGN","PRVNC","MU_CHNL","CTGY","LOC","DT").count().filter("count > 1").display()