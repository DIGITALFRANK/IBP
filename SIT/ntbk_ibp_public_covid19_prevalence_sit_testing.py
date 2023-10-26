# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Product Master dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from pyspark.sql.types import *
from copy import deepcopy
from collections import Counter
from delta.tables import *
from pyspark.sql import SparkSession
from pyspark import SparkContext

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

source_path_hist = 'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/covid-19-prevalence-history/datepart=2021-10-20/'

# COMMAND ----------

import re

def cleanColumn(tmpdf):
  cols_list = tmpdf.schema.names
  regex = r"[A-Za-z0-9\s/_-]"
  new_cols = []
  for col in cols_list:
    matches = re.finditer(regex, col, re.MULTILINE)
    name = []
    for matchNum, match in enumerate(matches, start=1):
      name.append(match.group())
      nn = "".join(name).replace(" ","_").replace("/","_")
      nn = nn.replace("__","_")
    tmpdf = tmpdf.withColumnRenamed(col, nn)
  return tmpdf

# COMMAND ----------

srcPath_list =[x.path for x in dbutils.fs.ls(source_path_hist)]
for path in srcPath_list:
  print(path)
  if '_01-22-2020' in path:
    df_batch0 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
    #df_batch0.withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit(""))
  if '_01-23-2020' in path:
    df_batch1 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_01-31-2020' in path:
    df_batch2 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_02-02-2020' in path:
    df_batch3 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_03-01-2020' in path:
    df_batch4 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_03-22-2020' in path:
    df_batch5 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_03-23-2020' in path:
    df_batch6 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_03-28-2020' in path:
    df_batch7 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_03-31-2020' in path:
    df_batch8 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_04-02-2020' in path:
    df_batch9 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_04-03-2020' in path:
    df_batch8 = df_batch8.union(cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path)))
  if '_04-04-2020' in path:
    df_batch9 = df_batch9.union(cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path)))
  if '_04-05-2020' in path:
    df_batch8 = df_batch8.union(cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path)))
  if '_04-06-2020' in path:
    df_batch9 = df_batch9.union(cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path)))
  if '_04-07-2020' in path:
    df_batch10 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))
  if '_05-29-2020' in path:
    df_batch11 = cleanColumn(spark.read.option("header","true").option("delimiter",",").csv(path))

# COMMAND ----------

df_batch0 = df_batch0.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yyyy H:mm")," yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch1 = df_batch1.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch2 = df_batch2.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yyyy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch3 = df_batch3.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"yyyy-MM-dd'T'HH:mm:ss"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch4 = df_batch4.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"yyyy-MM-dd'T'HH:mm:ss"),"yyyy-MM-dd HH:mm:ss"))\
.withColumnRenamed("Latitude","Lat")\
.withColumnRenamed("Longitude","Long_")\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch5 = df_batch5.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch6 = df_batch6.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch7 = df_batch7.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch8 = df_batch8.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch9 = df_batch9.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch10 = df_batch10.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

# COMMAND ----------

source_path_inc = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/covid-19-prevalence/datepart=2021-10-20/"
bronze_path_hist = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/covid-19-prevalence/"

# COMMAND ----------

source_df_hist = df_batch0.union(df_batch1.union(df_batch2.union(df_batch3.union(df_batch4.union(df_batch5.union(df_batch6.union(df_batch7.union(df_batch8.union(df_batch9.union(df_batch10.union(df_batch11)))))))))))
source_df_inc = spark.read.format("CSV").option("header",True).load(source_path_inc)
source_df = source_df_hist.union(source_df_inc)
bronze_df = spark.read.format("delta").load(bronze_path_hist)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDW
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

#Source and Bronze layer column validation for EDW
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))

# COMMAND ----------

print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

print(len(source_df.columns))
print(len(bronze_df.columns))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for EDW
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDW Source Layer Primary Key Uniqueness check (Not Applicable for Covid19 Prevalence)
#print(source_df.count())
#print(source_df.select("Province_State","Country_Region","Last_Update","Lat","Long_","FIPS","Admin2").distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check (Not Applicable for Covid19 Prevalence)
#print(bronze_df.count())
#print(bronze_df.select("Province_State","Country_Region","Last_Update","Lat","Long_","FIPS","Admin2").distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check (Not Applicable for Covid19 Prevalence)
#print(source_df.where((col("Province_State").isNull()) | (col("Last_Update").isNull()) | (col("Combined_Key").isNull())).count())
#EDW Bronze Layer PK Null check (Not Applicable for Covid19 Prevalence)
#print(bronze_df.where((col("Province_State").isNull()) | (col("Last_Update").isNull()) | (col("Combined_Key").isNull())).count())

# COMMAND ----------

#EDW Source Layer PK Duplicate check (Not Applicable for Covid19 Prevalence)
#source_df\
    #.groupby("Province_State","Last_Update","Combined_Key") \
    #.count() \
    #.where('count > 1') \
    #.sort('count', ascending=False) \
    #.show()

# COMMAND ----------

#EDW Bronze Layer PK Duplicate check (Not Applicable for Covid19 Prevalence)
#bronze_df \
    #.groupby("Date","CountryName")\
    #.count() \
    #.where('count > 1') \
    #.sort('count', ascending=False) \
    #.show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/covid-19-prevalence/"
silver_df = spark.read.format("delta").load(edw_silver_path)

# COMMAND ----------

loc_map = spark.read.csv("/FileStore/tables/temp/location_mapping_external.csv", header="true", inferSchema="true")

# COMMAND ----------

#Reading the data from bonze layer
ship_path = 'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/shipment-actuals'
shipment_df = spark.read.format("delta").load(ship_path)
shipment_df_filtered = shipment_df.select('PLANG_LOC_GRP_VAL','DMNDFCST_MKT_UNIT_CDV').distinct()
shipment_df_filtered = shipment_df_filtered.withColumn("country_name", when(col("DMNDFCST_MKT_UNIT_CDV")=="PT","Portugal").otherwise("Spain"))

# COMMAND ----------

covid_ES = bronze_df.filter(col("Country_Region")=="Spain")
covid_PT = bronze_df.filter(col("Country_Region")=="Portugal")

loc_map = loc_map.withColumn("country",when(col("MU")=="PT",lit("Portugal")).otherwise("Spain"))

covid_ES = covid_ES.join(loc_map, covid_ES.Province_State==loc_map.External_Region_Covid, "right")
covid_ES = covid_ES.select("Province_State","PROVINCIA","Country_Region","Last_Update","Confirmed","Deaths","Recovered","Active","Incident_Rate","Case_Fatality_Ratio","Location","Channel","Category","MU").withColumnRenamed("Location","LOC")

covid_PT = covid_PT.join(shipment_df_filtered, covid_PT.Country_Region==shipment_df_filtered.DMNDFCST_MKT_UNIT_CDV, "right").distinct()
covid_PT = covid_PT.withColumn("PROVINCIA",lit("n/a")).withColumn("Province_State",lit("n/a")).withColumn("Category",lit("n/a")).withColumn("Channel",lit("n/a"))
covid_PT = covid_PT.select("Province_State","PROVINCIA","Country_Region","Last_Update","Confirmed","Deaths","Recovered","Active","Incident_Rate","Case_Fatality_Ratio","PLANG_LOC_GRP_VAL","Channel","Category","DMNDFCST_MKT_UNIT_CDV").withColumnRenamed("PLANG_LOC_GRP_VAL","LOC").withColumnRenamed("DMNDFCST_MKT_UNIT_CDV","MU")

covid_all = covid_ES.union(covid_PT)
covid_all = covid_all.filter(covid_all.Last_Update.isNotNull())

# COMMAND ----------

print("Bronze Layer Count is "+str(covid_all.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['RGN','CTRY_NM','MU','DT','CNFIRM_CS','DEATH','REC','ACT','IR','CFR','MU_CHNL','CTGY','LOC','PRVNC']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select('RGN','MU','DT','MU_CHNL','CTGY','LOC','PRVNC').distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
print(silver_df.where((col("RGN").isNull())|(col("MU").isNull())|(col("DT").isNull())|(col("MU_CHNL").isNull())|(col("CTGY").isNull())|(col("LOC").isNull())|(col("PRVNC").isNull())).count())

# COMMAND ----------

#EDW Silver Layer PK Duplicate check
silver_df \
    .groupby('RGN','MU','DT','MU_CHNL','CTGY','LOC','PRVNC') \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

