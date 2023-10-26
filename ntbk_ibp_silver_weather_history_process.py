# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.window import Window
import itertools
import re
import datetime

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
#srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
srcPath = dbutils.widgets.get("sourcePath")
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
print(srcPath)

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$targetStorageAccount

# COMMAND ----------

#Code to establish connection to weather source storage account
weather_tenant_id = dbutils.secrets.get(scope="cdo-ibp-kvinst-scope",key="cdo-ibp-weather-tenant-id")
weather_client_id = dbutils.secrets.get(scope="cdo-ibp-kvinst-scope",key="cdo-ibp-weather-client-id")
weather_client_endpoint = f'https://login.microsoftonline.com/{weather_tenant_id}/oauth2/token'
spark.conf.set("fs.azure.account.auth.type.edapadlsdevna.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.edapadlsdevna.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.edapadlsdevna.dfs.core.windows.net", weather_client_id)
spark.conf.set("fs.azure.account.oauth2.client.secret.edapadlsdevna.dfs.core.windows.net", dbutils.secrets.get(scope="cdodev_dbws_scope",key="sp-ibp-edapadlsdevna-secret"))
spark.conf.set("fs.azure.account.oauth2.client.endpoint.edapadlsdevna.dfs.core.windows.net", weather_client_endpoint)

# COMMAND ----------

#Reading the weather history data from source table - weather_history_onpoint
weather_hist = spark.sql("select * from "+srcPath)
display(weather_hist)

# COMMAND ----------

#Converting the column date_valid_std to date type
weather_hist = weather_hist.withColumn("date_valid_std",to_date("date_valid_std", 'yyyy-MM-dd'))
display(weather_hist)

# COMMAND ----------

#Max and Min date_valid_std in the source data
min_source_date = weather_hist.agg({"date_valid_std": "min"}).collect()[0][0]
max_source_date = weather_hist.agg({"date_valid_std": "max"}).collect()[0][0]
print(min_source_date)
print(max_source_date)

# COMMAND ----------

##Code for deciding full load or incremental load
##If data exists in the target path then INCR else FULL load
print(tgtPath)
if DeltaTable.isDeltaTable(spark, tgtPath):
  #Incremental
  print("Data is present in silver layer and incremental load will take place.")
  ##Reading the silver layer data to get the max date
  temp_df = spark.read.format("delta").load(tgtPath)
  max_silver_date = temp_df.agg({"DT": "max"}).collect()[0][0]
  print("The maximum date value present in silver layer is: "+str(max_silver_date))
  weather_hist_fltrd = weather_hist.filter(weather_hist.date_valid_std > max_silver_date)
  print("Full Count: "+str(weather_hist.count()))
  print("Incremental Count: "+str(weather_hist_fltrd.count()))
  #source_df will be used in all subsequent transformations
  source_df = weather_hist_fltrd
  #if incremental count is 0
  if source_df.count() == 0:
    print("No incremental data was pulled.")
    dbutils.notebook.exit("recordCount : "+str(source_df.count()))
else:
  #Full Load
  print("Data is not present in silver layer and full load will take place.")
  print("Full Count: "+str(weather_hist.count()))
  source_df = weather_hist

# COMMAND ----------

#Reading the loc and population mapping files
dpndntdatapath_list = dpndntdatapath.split(";")
for path in dpndntdatapath_list:
  srcPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/"+path
  print(srcPath)
  if '/location-mapping-external/' in path:
    print("Reading Location Mapping")
    loc_map = spark.read.csv(srcPath, header="true", inferSchema="true", encoding="ISO-8859-1")
  if '/regional-population-pt-es/' in path:
    print("Reading Population Mapping")
    population = spark.read.csv(srcPath, header="true", inferSchema="true", encoding="ISO-8859-1")

# COMMAND ----------

#Weather and Population join
population = population.withColumnRenamed("MU","MarketUnit").withColumnRenamed("Region","pop_region")
weather_pop = source_df.join(population, source_df.region==population.Weather_Region, "left").withColumnRenamed("region","RGN")

# COMMAND ----------

#creating the final dataframe for silver layer

winSpec = Window.partitionBy('RGN','date_valid_std').orderBy('date_valid_std')

weather_final = weather_pop.withColumn("WGHTD_AVG_TMP_AIR_2M_F",F.avg('avg_temperature_air_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_TMP_WTBLB_2M_F",F.avg('avg_temperature_wetbulb_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_TMP_DWPNT_2M_F",F.avg('avg_temperature_dewpoint_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_TMP_FLSLK_2M_F",F.avg('avg_temperature_feelslike_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_TMP_WNDCHLL_2M_F",F.avg('avg_temperature_windchill_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_TMP_HTNDX_2M_F",F.avg('avg_temperature_heatindex_2m_f').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_HMDTY_RLTV_2M_PCT",F.avg('avg_humidity_relative_2m_pct').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_HMDTY_SPEC_2M_GPKG",F.avg('avg_humidity_specific_2m_gpkg').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_PRSSR_2M_MB",F.avg('avg_pressure_2m_mb').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_PRSSR_MEAN_SEA_LEVEL_MB",F.avg('avg_pressure_mean_sea_level_mb').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_SPD_10M_MPH",F.avg('avg_wind_speed_10m_mph').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_DRCTN_10M_DEG",F.avg('avg_wind_direction_10m_deg').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_SPD_80M_MPH",F.avg('avg_wind_speed_80m_mph').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_DRCTN_80M_DEG",F.avg('avg_wind_direction_80m_deg').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_SPD_100M_MPH",F.avg('avg_wind_speed_100m_mph').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_WIND_DRCTN_100M_DEG",F.avg('avg_wind_direction_100m_deg').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_TOT_PRECIPITATION_IN",F.avg('tot_precipitation_in').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_TOT_SNWFALL_IN",F.avg('tot_snowfall_in').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_CLD_CVR_TOT_PCT",F.avg('avg_cloud_cover_tot_pct').over(winSpec)*F.col("Ratio"))\
.withColumn("WGHTD_AVG_RAD_SOLAR_TOTAL_WPM2",F.avg('avg_radiation_solar_total_wpm2').over(winSpec)*F.col("Ratio"))

#joining with location mapping
silver_df = loc_map.join(weather_final, loc_map.Weather_Region==weather_final.Weather_Region,"left")

silver_df = silver_df.withColumnRenamed("date_valid_std","DT")\
.withColumnRenamed("doy_std","D_YR")\
.withColumnRenamed("LOCATION","LOC")\
.withColumnRenamed("Channel","MU_CHNL")\
.withColumnRenamed("Category","CTGY")\
.withColumnRenamed("PROVINCIA","PRVN")
col("age").cast('int').alias("age")

# silver_df_final = silver_df.select("RGN","PRVN","MU","DT",col("D_YR").cast('int').alias("D_YR"),"LOC","MU_CHNL","CTGY", "WGHTD_AVG_TMP_AIR_2M_F","WGHTD_AVG_TMP_WTBLB_2M_F",'WGHTD_AVG_TMP_DWPNT_2M_F','WGHTD_AVG_TMP_FLSLK_2M_F','WGHTD_AVG_TMP_WNDCHLL_2M_F','WGHTD_AVG_TMP_HTNDX_2M_F','WGHTD_AVG_HMDTY_RLTV_2M_PCT','WGHTD_AVG_HMDTY_SPEC_2M_GPKG','WGHTD_AVG_PRSSR_2M_MB','WGHTD_AVG_PRSSR_MEAN_SEA_LEVEL_MB','WGHTD_AVG_WIND_SPD_10M_MPH','WGHTD_AVG_WIND_DRCTN_10M_DEG','WGHTD_AVG_WIND_SPD_80M_MPH','WGHTD_AVG_WIND_DRCTN_80M_DEG','WGHTD_AVG_WIND_SPD_100M_MPH','WGHTD_AVG_WIND_DRCTN_100M_DEG','WGHTD_TOT_PRECIPITATION_IN','WGHTD_TOT_SNWFALL_IN','WGHTD_AVG_CLD_CVR_TOT_PCT','WGHTD_AVG_RAD_SOLAR_TOTAL_WPM2').distinct()

# COMMAND ----------



# COMMAND ----------

silver_df_final = silver_df.select("RGN",
                                   "PRVN",
                                   "MU",
                                   col("DT").cast("timestamp"),
                                   col("D_YR").cast("int").alias("D_YR"),
                                   "LOC",
                                   "MU_CHNL",
                                   "CTGY",
                                   col("WGHTD_AVG_TMP_AIR_2M_F").cast("float"),                                 
                                   col("WGHTD_AVG_TMP_WTBLB_2M_F").cast("float"),
                                   col("WGHTD_AVG_TMP_DWPNT_2M_F").cast("float"),
                                   col("WGHTD_AVG_TMP_FLSLK_2M_F").cast("float"),
                                   col("WGHTD_AVG_TMP_WNDCHLL_2M_F").cast("float"),
                                   col("WGHTD_AVG_TMP_HTNDX_2M_F").cast("float"),
                                   col("WGHTD_AVG_HMDTY_RLTV_2M_PCT").cast("float"),
                                   col("WGHTD_AVG_HMDTY_SPEC_2M_GPKG").cast("float"),
                                   col("WGHTD_AVG_PRSSR_2M_MB").cast("float"),
                                   col("WGHTD_AVG_PRSSR_MEAN_SEA_LEVEL_MB").cast("float"),
                                   col("WGHTD_AVG_WIND_SPD_10M_MPH").cast("float"),
                                   col("WGHTD_AVG_WIND_DRCTN_10M_DEG").cast("float"),
                                   col("WGHTD_AVG_WIND_SPD_80M_MPH").cast("float"),
                                   col("WGHTD_AVG_WIND_DRCTN_80M_DEG").cast("float"),
                                   col("WGHTD_AVG_WIND_SPD_100M_MPH").cast("float"),
                                   col("WGHTD_AVG_WIND_DRCTN_100M_DEG").cast("float"),
                                   col("WGHTD_TOT_PRECIPITATION_IN").cast("float"),
                                   col("WGHTD_TOT_SNWFALL_IN").cast("float"),
                                   col("WGHTD_AVG_CLD_CVR_TOT_PCT").cast("float"),
                                   col("WGHTD_AVG_RAD_SOLAR_TOTAL_WPM2").cast("float")).distinct()

# COMMAND ----------

display(silver_df_final)

# COMMAND ----------

#Adding PROCESS_DATE
silver_df_final = silver_df_final.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

#Filtering out records having NULL values for date columns - DT
silver_df_final = silver_df_final.filter("DT is not NULL")

# COMMAND ----------

#Count of the data to be written to silver layer
silver_count = silver_df_final.count()
print(silver_count)

# COMMAND ----------

#Writing data into delta lake - silver layer
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df_final.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df_final.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df_final.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df_final.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.weather_history") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.weather_history

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_count))