# Databricks notebook source
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql import Row
import itertools
import re

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

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

cov_cases_deltaTable = DeltaTable.forPath(spark, srcPath)
cov_cases_latest_version = cov_cases_deltaTable.history().select(max(col('version'))).collect()[0][0]
display(cov_cases_deltaTable.history())

# COMMAND ----------

cov_cases_df = spark.read.format("delta").option("versionAsOf", cov_cases_latest_version).load(srcPath)
print(cov_cases_df.count())
display(cov_cases_df)

# COMMAND ----------

max_value = cov_cases_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
cov_cases_df_new = cov_cases_df.filter(col("PROCESS_DATE")==max_value).drop("DELTA_DATE")
display(cov_cases_df_new)

# COMMAND ----------

#The location mapping file
loc_map = spark.read.csv("/FileStore/tables/temp/location_mapping_external.csv", header="true", inferSchema="true")
loc_map_PT = loc_map.filter(col("MU")=="PT")
loc_map_ES = loc_map.filter(col("MU")=="ES")

# COMMAND ----------

#Obtain location from Shipment. This will be used for Portugal as an interim solution as its at a country level
shipment_df = spark.read.format("delta").load(dpdPath)
display(shipment_df)
shipment_df_filtered = shipment_df.withColumn("DMNDFCST_MKT_UNIT_CDV",col("DMDGROUP").substr(1, 2))\
                           .where((col("DMNDFCST_MKT_UNIT_CDV").isin('ES','PT')))\
                           .select("LOC","DMNDFCST_MKT_UNIT_CDV").distinct()
shipment_df_filtered = shipment_df_filtered.withColumn("country_name", F.when(F.col("DMNDFCST_MKT_UNIT_CDV")=="PT","Portugal").otherwise("Spain"))

# COMMAND ----------

covid_ES = cov_cases_df_new.filter(F.col("Country_Region")=="Spain").withColumn("marketunit",lit("ES"))
covid_PT = cov_cases_df_new.filter(F.col("Country_Region")=="Portugal").withColumn("marketunit",lit("PT"))

#loc_map = loc_map.withColumn("country",F.when(F.col("MU")=="PT",F.lit("Portugal")).otherwise("Spain"))

covid_ES = covid_ES.join(loc_map_ES, (covid_ES.Province_State==loc_map_ES.External_Region_Covid), "right")
covid_ES = covid_ES.select("Province_State","PROVINCIA","Country_Region","Last_Update","Confirmed","Deaths","Recovered","Active","Incident_Rate","Case_Fatality_Ratio","Location","Channel","Category","MU").withColumnRenamed("Location","LOC")

covid_PT = covid_PT.join(shipment_df_filtered, covid_PT.marketunit==shipment_df_filtered.DMNDFCST_MKT_UNIT_CDV, "right").filter(col("DMNDFCST_MKT_UNIT_CDV")=="PT").distinct()
covid_PT = covid_PT.withColumn("PROVINCIA", F.lit("n/a")).withColumn("Province_State",F.lit("n/a")).withColumn("Category",F.lit("n/a")).withColumn("Channel",F.lit("n/a"))
covid_PT = covid_PT.select("Province_State","PROVINCIA","Country_Region","Last_Update","Confirmed","Deaths","Recovered","Active","Incident_Rate","Case_Fatality_Ratio","LOC","Channel","Category","DMNDFCST_MKT_UNIT_CDV").withColumnRenamed("LOC","LOC").withColumnRenamed("DMNDFCST_MKT_UNIT_CDV","MU")

covid_all = covid_ES.union(covid_PT)
covid_all = covid_all.filter(covid_all.Last_Update.isNotNull())

# COMMAND ----------

silver_df = covid_all\
.withColumnRenamed("Province_State","RGN")\
.withColumnRenamed("PROVINCIA","PRVNC")\
.withColumnRenamed("Country_Region","CTRY_NM")\
.withColumnRenamed("Last_Update","DT")\
.withColumnRenamed("Confirmed","CNFIRM_CS")\
.withColumnRenamed("Deaths","DEATH")\
.withColumnRenamed("Recovered","REC")\
.withColumnRenamed("Active","ACT")\
.withColumnRenamed("Incident_Rate","IR")\
.withColumnRenamed("Case_Fatality_Ratio","CFR")\
.withColumnRenamed("Location","LOC")\
.withColumnRenamed("Channel","MU_CHNL")\
.withColumnRenamed("Category","CTGY")

silver_df = silver_df.select("RGN","CTRY_NM",col("DT").cast("timestamp"),col("CNFIRM_CS").cast("int"),"DEATH",col("REC").cast("int"),col("ACT").cast("int"),col("IR").cast("float"),col("CFR").cast("float"),"LOC","MU_CHNL","CTGY","MU","PRVNC")

# COMMAND ----------

silver_df.display()

# COMMAND ----------

silver_df.groupby("RGN","MU","DT","LOC","MU_CHNL","CTGY","PRVNC").count().filter("count > 1").count()

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  #deltaTable = DeltaTable.forPath(spark, tgtPath)
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
    .tableName("sc_ibp_silver.covid_prevalence") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.covid_prevalence

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))