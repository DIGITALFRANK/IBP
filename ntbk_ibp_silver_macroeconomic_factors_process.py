# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from functools import reduce

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

#splitting the dependentDatasetPath in different variables
print(dependentPath)

dependentPath_list = dependentPath.split(';')

for path in dependentPath_list:
  if '/q.es.ngdp_nsa_xdc' in path:
    gross_domestic_product_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/q.pt.ngdp_nsa_xdc' in path:
    gross_domestic_product_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.es.aotv_pe_num' in path:
    tourism_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.pt.aotv_pe_num' in path:
    tourism_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.es.pcpi_ix' in path:
    consumer_prices_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.pt.pcpi_ix' in path:
    consumer_prices_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.es.lwr_ix' in path:
    labor_market_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/q.pt.lwr_ix' in path:
    labor_market_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/q.es.llf_pe_num' in path:
    labor_force_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.pt.llf_pe_num' in path:
    labor_force_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.es.lur_pt' in path:
    unemployment_es = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/m.pt.lur_pt' in path:
    unemployment_pt = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/shipment-actuals' in path:
    shipment = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  

# COMMAND ----------

print(gross_domestic_product_es) 
print(gross_domestic_product_pt) 
print(tourism_es) 
print(tourism_pt) 
print(consumer_prices_es) 
print(consumer_prices_pt) 
print(labor_market_es)
print(labor_market_pt) 
print(labor_force_es)
print(labor_force_pt)
print(unemployment_es) 
print(unemployment_pt)
print(shipment)

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

#Reading the delta history gross_domestic_product_es
gross_domestic_product_es_deltaTable = DeltaTable.forPath(spark, gross_domestic_product_es)
gross_domestic_product_es_latest_version = gross_domestic_product_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(gross_domestic_product_es_latest_version)
display(gross_domestic_product_es_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer gross_domestic_product_es
gross_domestic_product_es_df = spark.read.format("delta").option("versionAsOf", gross_domestic_product_es_latest_version).load(gross_domestic_product_es)
display(gross_domestic_product_es_df)

# COMMAND ----------

print(gross_domestic_product_es_df.count())

# COMMAND ----------

display(gross_domestic_product_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = gross_domestic_product_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)

gross_domestic_product_es_df_filtered = gross_domestic_product_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(gross_domestic_product_es_df_filtered)

# COMMAND ----------

print("Overall Count of gross_domestic_product_es in Bronze Layer: "+str(gross_domestic_product_es_df.count()))
print("Latest Process Date Count of gross_domestic_product_es in Bronze Layer: "+str(gross_domestic_product_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history gross_domestic_product_pt
gross_domestic_product_pt_deltaTable = DeltaTable.forPath(spark, gross_domestic_product_pt)
gross_domestic_product_pt_latest_version = gross_domestic_product_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(gross_domestic_product_pt_latest_version)
display(gross_domestic_product_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer gross_domestic_product_pt
gross_domestic_product_pt_df = spark.read.format("delta").option("versionAsOf", gross_domestic_product_pt_latest_version).load(gross_domestic_product_pt)
display(gross_domestic_product_pt_df)

# COMMAND ----------

print(gross_domestic_product_pt_df.count())

# COMMAND ----------

display(gross_domestic_product_pt_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = gross_domestic_product_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
gross_domestic_product_pt_df_filtered = gross_domestic_product_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(gross_domestic_product_pt_df_filtered)

# COMMAND ----------

print("Overall Count of gross_domestic_product_pt in Bronze Layer: "+str(gross_domestic_product_pt_df.count()))
print("Latest Process Date Count of gross_domestic_product_pt in Bronze Layer: "+str(gross_domestic_product_pt_df_filtered.count()))

# COMMAND ----------

#Reading the delta history tourism_es
tourism_es_deltaTable = DeltaTable.forPath(spark, tourism_es)
tourism_es_latest_version = tourism_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(tourism_es_latest_version)
display(tourism_es_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer tourism_es
tourism_es_df = spark.read.format("delta").option("versionAsOf", tourism_es_latest_version).load(tourism_es)
display(tourism_es_df)

# COMMAND ----------

print(tourism_es_df.count())

# COMMAND ----------

display(tourism_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = tourism_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
tourism_es_df_filtered = tourism_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(tourism_es_df_filtered)

# COMMAND ----------

print("Overall Count of tourism_es in Bronze Layer: "+str(tourism_es_df.count()))
print("Latest Process Date Count of tourism_es in Bronze Layer: "+str(tourism_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history tourism_pt
tourism_pt_deltaTable = DeltaTable.forPath(spark, tourism_pt)
tourism_pt_latest_version = tourism_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(tourism_pt_latest_version)
display(tourism_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer tourism_pt
tourism_pt_df = spark.read.format("delta").option("versionAsOf", tourism_pt_latest_version).load(tourism_pt)
display(tourism_pt_df)


# COMMAND ----------

print(tourism_pt_df.count())

# COMMAND ----------

display(tourism_pt_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = tourism_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
tourism_pt_df_filtered = tourism_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(tourism_pt_df_filtered)

# COMMAND ----------

print("Overall Count of tourism_pt in Bronze Layer: "+str(tourism_pt_df.count()))
print("Latest Process Date Count of tourism_pt in Bronze Layer: "+str(tourism_pt_df_filtered.count()))

# COMMAND ----------

#Reading the delta history consumer_prices_es
consumer_prices_es_deltaTable = DeltaTable.forPath(spark, consumer_prices_es)
consumer_prices_es_latest_version = consumer_prices_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(consumer_prices_es_latest_version)
display(consumer_prices_es_deltaTable.history())


# COMMAND ----------

#Reading the data from bonze layer consumer_prices_es
consumer_prices_es_df = spark.read.format("delta").option("versionAsOf", consumer_prices_es_latest_version).load(consumer_prices_es)
display(consumer_prices_es_df)


# COMMAND ----------

print(consumer_prices_es_df.count())

# COMMAND ----------

display(consumer_prices_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = consumer_prices_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
consumer_prices_es_df_filtered = consumer_prices_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(consumer_prices_es_df_filtered)

# COMMAND ----------

print("Overall Count of consumer_prices_es in Bronze Layer: "+str(consumer_prices_es_df.count()))
print("Latest Process Date Count of consumer_prices_es in Bronze Layer: "+str(consumer_prices_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history consumer_prices_pt
consumer_prices_pt_deltaTable = DeltaTable.forPath(spark, consumer_prices_pt)
consumer_prices_pt_latest_version = consumer_prices_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(consumer_prices_pt_latest_version)
display(consumer_prices_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer consumer_prices_pt
consumer_prices_pt_df = spark.read.format("delta").option("versionAsOf", consumer_prices_pt_latest_version).load(consumer_prices_pt)
display(consumer_prices_pt_df)

# COMMAND ----------

print(consumer_prices_pt_df.count())

# COMMAND ----------

display(consumer_prices_pt_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = consumer_prices_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
consumer_prices_pt_df_filtered = consumer_prices_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(consumer_prices_pt_df_filtered)

# COMMAND ----------

print("Overall Count of consumer_prices_pt in Bronze Layer: "+str(consumer_prices_pt_df.count()))
print("Latest Process Date Count of consumer_prices_pt in Bronze Layer: "+str(consumer_prices_pt_df_filtered.count()))

# COMMAND ----------

#Reading the delta history labor_market_es
labor_market_es_deltaTable = DeltaTable.forPath(spark, labor_market_es)
labor_market_es_latest_version = labor_market_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(labor_market_es_latest_version)
display(labor_market_es_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer labor_market_es
labor_market_es_df = spark.read.format("delta").option("versionAsOf", labor_market_es_latest_version).load(labor_market_es)
display(labor_market_es_df)


# COMMAND ----------

print(labor_market_es_df.count())

# COMMAND ----------

display(labor_market_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = labor_market_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
labor_market_es_df_filtered = labor_market_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(labor_market_es_df_filtered)

# COMMAND ----------

print("Overall Count of labor_market_es in Bronze Layer: "+str(labor_market_es_df.count()))
print("Latest Process Date Count of labor_market_es in Bronze Layer: "+str(labor_market_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history labor_market_pt
labor_market_pt_deltaTable = DeltaTable.forPath(spark, labor_market_pt)
labor_market_pt_latest_version = labor_market_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(labor_market_pt_latest_version)
display(labor_market_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer labor_market_pt
labor_market_pt_df = spark.read.format("delta").option("versionAsOf", labor_market_pt_latest_version).load(labor_market_pt)
display(labor_market_pt_df)


# COMMAND ----------

print(labor_market_pt_df.count())

# COMMAND ----------

display(labor_market_pt_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = labor_market_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
labor_market_pt_df_filtered = labor_market_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(labor_market_pt_df_filtered)

# COMMAND ----------

print("Overall Count of labor_market_pt in Bronze Layer: "+str(labor_market_pt_df.count()))
print("Latest Process Date Count of labor_market_pt in Bronze Layer: "+str(labor_market_pt_df_filtered.count()))

# COMMAND ----------

#Reading the delta history labor_force_es
labor_force_es_deltaTable = DeltaTable.forPath(spark, labor_force_es)
labor_force_es_latest_version = labor_force_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(labor_force_es_latest_version)
display(labor_force_es_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer labor_force_es
labor_force_es_df = spark.read.format("delta").option("versionAsOf", labor_force_es_latest_version).load(labor_force_es)
display(labor_force_es_df)

# COMMAND ----------

print(labor_force_es_df.count())

# COMMAND ----------

display(labor_force_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = labor_force_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
labor_force_es_df_filtered = labor_force_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(labor_force_es_df_filtered)

# COMMAND ----------

print("Overall Count of labor_force_es in Bronze Layer: "+str(labor_force_es_df.count()))
print("Latest Process Date Count of labor_force_es in Bronze Layer: "+str(labor_force_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history labor_force_pt
labor_force_pt_deltaTable = DeltaTable.forPath(spark, labor_force_pt)
labor_force_pt_latest_version = labor_force_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(labor_force_pt_latest_version)
display(labor_force_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer labor_force_pt
labor_force_pt_df = spark.read.format("delta").option("versionAsOf", labor_force_pt_latest_version).load(labor_force_pt)
display(labor_force_pt_df)

# COMMAND ----------

print(labor_force_pt_df.count())

# COMMAND ----------

display(labor_force_pt_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = labor_force_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
labor_force_pt_df_filtered = labor_force_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(labor_force_pt_df_filtered)

# COMMAND ----------

print("Overall Count of labor_force_pt in Bronze Layer: "+str(labor_force_pt_df.count()))
print("Latest Process Date Count of labor_force_pt in Bronze Layer: "+str(labor_force_pt_df_filtered.count()))

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
display(shipment_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = shipment_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
shipment_df_filtered = shipment_df.filter(col("PROCESS_DATE")==max_process_date)
display(shipment_df_filtered)

print("Overall Count of shipment in Bronze Layer: "+str(shipment_df.count()))
print("Latest Process Date Count of shipment in Bronze Layer: "+str(shipment_df_filtered.count()))

# COMMAND ----------

#Reading the delta history unemployment_es
unemployment_es_deltaTable = DeltaTable.forPath(spark, unemployment_es)
unemployment_es_latest_version = unemployment_es_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(unemployment_es_latest_version)
display(unemployment_es_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer unemployment_es
unemployment_es_df = spark.read.format("delta").option("versionAsOf", unemployment_es_latest_version).load(unemployment_es)
display(unemployment_es_df)

# COMMAND ----------

print(unemployment_es_df.count())

# COMMAND ----------

display(unemployment_es_df.select("PROCESS_DATE").distinct())

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = unemployment_es_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
unemployment_es_df_filtered = unemployment_es_df.filter(col("PROCESS_DATE")==max_process_date)
display(unemployment_es_df_filtered)

# COMMAND ----------

print("Overall Count of unemployment_es in Bronze Layer: "+str(unemployment_es_df.count()))
print("Latest Process Date Count of unemployment_es in Bronze Layer: "+str(unemployment_es_df_filtered.count()))

# COMMAND ----------

#Reading the delta history unemployment_pt
unemployment_pt_deltaTable = DeltaTable.forPath(spark, unemployment_pt)
unemployment_pt_latest_version = unemployment_pt_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(unemployment_pt_latest_version)
display(unemployment_pt_deltaTable.history())

# COMMAND ----------

#Reading the data from bonze layer unemployment_pt
unemployment_pt_df = spark.read.format("delta").option("versionAsOf", unemployment_pt_latest_version).load(unemployment_pt)
display(unemployment_pt_df)

# COMMAND ----------

print(unemployment_pt_df.count())

# COMMAND ----------

display(unemployment_pt_df.select("PROCESS_DATE").distinct())


# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = unemployment_pt_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_process_date)
unemployment_pt_df_filtered = unemployment_pt_df.filter(col("PROCESS_DATE")==max_process_date)
display(unemployment_pt_df_filtered)

# COMMAND ----------

print("Overall Count of unemployment_pt in Bronze Layer: "+str(unemployment_pt_df.count()))
print("Latest Process Date Count of unemployment_pt in Bronze Layer: "+str(unemployment_pt_df_filtered.count()))

# COMMAND ----------

#gross_domestic_product_es
gross_domestic_product_es_df_final=gross_domestic_product_es_df.select(
gross_domestic_product_es_df.TIME_PERIOD.alias('DT'),
gross_domestic_product_es_df.FREQ.alias('PRD'),
gross_domestic_product_es_df.INDICATOR.alias('IND'),
gross_domestic_product_es_df.REF_AREA.alias('CNTRY_CD'),
gross_domestic_product_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Gross Domestic Product, Nominal, Seasonally Adjusted, Domestic Currency"))

# COMMAND ----------

#gross_domestic_product_pt
gross_domestic_product_pt_df_final=gross_domestic_product_pt_df.select(
gross_domestic_product_pt_df.TIME_PERIOD.alias('DT'),
gross_domestic_product_pt_df.FREQ.alias('PRD'),
gross_domestic_product_pt_df.INDICATOR.alias('IND'),
gross_domestic_product_pt_df.REF_AREA.alias('CNTRY_CD'),
gross_domestic_product_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Gross Domestic Product, Nominal, Seasonally Adjusted, Domestic Currency"))

# COMMAND ----------

#tourism_es_df
tourism_es_df_final=tourism_es_df.select(
tourism_es_df.TIME_PERIOD.alias('DT'),
tourism_es_df.FREQ.alias('PRD'),
tourism_es_df.INDICATOR.alias('IND'),
tourism_es_df.REF_AREA.alias('CNTRY_CD'),
tourism_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Tourism, Number of Visitors, Persons"))

# COMMAND ----------

#tourism_pt_df
tourism_pt_df_final=tourism_pt_df.select(
tourism_pt_df.TIME_PERIOD.alias('DT'),
tourism_pt_df.FREQ.alias('PRD'),
tourism_pt_df.INDICATOR.alias('IND'),
tourism_pt_df.REF_AREA.alias('CNTRY_CD'),
tourism_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Tourism, Number of Visitors, Persons"))

# COMMAND ----------

#consumer_prices_es_df
consumer_prices_es_df_final=consumer_prices_es_df.select(
consumer_prices_es_df.TIME_PERIOD.alias('DT'),
consumer_prices_es_df.FREQ.alias('PRD'),
consumer_prices_es_df.INDICATOR.alias('IND'),
consumer_prices_es_df.REF_AREA.alias('CNTRY_CD'),
consumer_prices_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Consumer prices, all items"))

# COMMAND ----------

#display(unemployment_pt_df.groupBy("INDICATOR","REF_AREA","TIME_PERIOD").count().filter("count > 1"))

# COMMAND ----------

#consumer_prices_pt_df
consumer_prices_pt_df_final=consumer_prices_pt_df.select(
consumer_prices_pt_df.TIME_PERIOD.alias('DT'),
consumer_prices_pt_df.FREQ.alias('PRD'),
consumer_prices_pt_df.INDICATOR.alias('IND'),
consumer_prices_pt_df.REF_AREA.alias('CNTRY_CD'),
consumer_prices_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Consumer prices, all items"))

# COMMAND ----------

#labor_market_es_df
labor_market_es_df_final=labor_market_es_df.select(
labor_market_es_df.TIME_PERIOD.alias('DT'),
labor_market_es_df.FREQ.alias('PRD'),
labor_market_es_df.INDICATOR.alias('IND'),
labor_market_es_df.REF_AREA.alias('CNTRY_CD'),
labor_market_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Labor market, wage rates, index"))

# COMMAND ----------

labor_market_pt_df
labor_market_pt_df_final=labor_market_pt_df.select(
labor_market_pt_df.TIME_PERIOD.alias('DT'),
labor_market_pt_df.FREQ.alias('PRD'),
labor_market_pt_df.INDICATOR.alias('IND'),
labor_market_pt_df.REF_AREA.alias('CNTRY_CD'),
labor_market_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Labor market, wage rates, index"))

# COMMAND ----------

labor_force_es_df
labor_force_es_df_final=labor_force_es_df.select(
labor_force_es_df.TIME_PERIOD.alias('DT'),
labor_force_es_df.FREQ.alias('PRD'),
labor_force_es_df.INDICATOR.alias('IND'),
labor_force_es_df.REF_AREA.alias('CNTRY_CD'),
labor_force_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Labor force, persons"))

# COMMAND ----------

#labor_force_pt_df
labor_force_pt_df_final=labor_force_pt_df.select(
labor_force_pt_df.TIME_PERIOD.alias('DT'),
labor_force_pt_df.FREQ.alias('PRD'),
labor_force_pt_df.INDICATOR.alias('IND'),
labor_force_pt_df.REF_AREA.alias('CNTRY_CD'),
labor_force_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Labor force, persons"))

# COMMAND ----------

#unemployment_es_df
unemployment_es_df_final=unemployment_es_df.select(
unemployment_es_df.TIME_PERIOD.alias('DT'),
unemployment_es_df.FREQ.alias('PRD'),
unemployment_es_df.INDICATOR.alias('IND'),
unemployment_es_df.REF_AREA.alias('CNTRY_CD'),
unemployment_es_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Unemployment, persons"))

# COMMAND ----------

#unemployment_pt_df
unemployment_pt_df_final=unemployment_pt_df.select(
unemployment_pt_df.TIME_PERIOD.alias('DT'),
unemployment_pt_df.FREQ.alias('PRD'),
unemployment_pt_df.INDICATOR.alias('IND'),
unemployment_pt_df.REF_AREA.alias('CNTRY_CD'),
unemployment_pt_df.OBS_VALUE.alias('VAL')
).withColumn("FCTR_NM",lit("Unemployment, persons"))

# COMMAND ----------

dfs = [gross_domestic_product_es_df_final,gross_domestic_product_pt_df_final,tourism_es_df_final,tourism_pt_df_final,consumer_prices_es_df_final,consumer_prices_pt_df_final,labor_market_es_df_final,labor_market_pt_df_final,labor_force_es_df_final,labor_force_pt_df_final,unemployment_es_df_final,unemployment_pt_df_final]
silver_df_temp = reduce(DataFrame.unionByName, dfs).withColumn("VAL",col("VAL").cast('float'))
silver_df_temp=silver_df_temp.withColumn("VAL",when(col("IND").isin('LUR_PT','LLF_PE_NUM','AOTV_PE_NUM'),col('VAL')*1000).otherwise(col('VAL')))

# COMMAND ----------

shipment_df_filtered = shipment_df_filtered.withColumn("DMNDFCST_MKT_UNIT_CDV",col("DMDGROUP").substr(1, 2))\
                           .where((col("DMNDFCST_MKT_UNIT_CDV").isin('ES','PT')))\
                           .select("LOC","DMNDFCST_MKT_UNIT_CDV").distinct()

display(shipment_df_filtered)

cond = [silver_df_temp.CNTRY_CD == shipment_df_filtered.DMNDFCST_MKT_UNIT_CDV]

silver_df = silver_df_temp.join(shipment_df_filtered, cond).select(silver_df_temp.DT.alias('DT'),
                                                                 silver_df_temp.PRD.alias('PRD'),
                                                                 silver_df_temp.IND.alias('IND'),
                                                                 silver_df_temp.CNTRY_CD.alias('CNTRY_CD'),
                                                                 silver_df_temp.VAL.alias('VAL'),
                                                                 silver_df_temp.FCTR_NM.alias('FCTR_NM'),
                                                                 shipment_df_filtered.LOC.alias('LOC'))

# COMMAND ----------

display(silver_df)

# COMMAND ----------

silver_df.printSchema()
display(silver_df.groupBy("DT","PRD","IND","CNTRY_CD","FCTR_NM","LOC").count().filter("count > 1"))

# COMMAND ----------

from pyspark.sql.functions import to_timestamp
silver_df = silver_df.withColumn("DT_DATE",when(substring(col("DT"),6,2)=='Q1',concat(substring(col("DT"),1,5),lit('01-01 00:00:00')))\
                                      .when(substring(col("DT"),6,2)=='Q2',concat(substring(col("DT"),1,5),lit('04-01 00:00:00')))\
                                      .when(substring(col("DT"),6,2)=='Q3',concat(substring(col("DT"),1,5),lit('07-01 00:00:00')))\
                                      .when(substring(col("DT"),6,2)=='Q4',concat(substring(col("DT"),1,5),lit('10-01 00:00:00'))).otherwise(concat(col('DT'),lit('-01 00:00:00'))))\
                    .withColumn("DT_DATE",to_timestamp(col("DT_DATE"), 'yyyy-MM-dd HH:mm:ss'))

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

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
    .tableName("sc_ibp_silver.macroeconomic_factors") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.macroeconomic_factors

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))