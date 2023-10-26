# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Product Master dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import col, explode_outer, current_date
from pyspark.sql.types import *
from copy import deepcopy
from collections import Counter
from delta.tables import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
from functools import reduce
import re
import json

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_json_common_classes

# COMMAND ----------

sourcePath = ['abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-growth/es/sp.pop.grow/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-growth/pt/sp.pop.grow/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population/es/sp.pop.totl/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population/pt/sp.pop.totl/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gdp-per-capita-growth/es/ny.gdp.pcap.pp.cd/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gdp-per-capita-growth/pt/ny.gdp.pcap.pp.cd/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/annual-consumer-prices-inflation/es/fp.cpi.totl.zg/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/annual-consumer-prices-inflation/pt/fp.cpi.totl.zg/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/urban-population/es/sp.urb.totl/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/urban-population/pt/sp.urb.totl/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gdp-per-capita/es/ny.gdp.pcap.cd/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gdp-per-capita/pt/ny.gdp.pcap.cd/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gross-savings/es/ny.gns.ictr.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/gross-savings/pt/ny.gns.ictr.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/net-migration/es/sm.pop.netm/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/net-migration/pt/sm.pop.netm/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/total-population-ages-15-64/es/sp.pop.1564.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/total-population-ages-15-64/pt/sp.pop.1564.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/prevalence-of-undernourishment/es/sn.itk.defc.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/prevalence-of-undernourishment/pt/sn.itk.defc.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-ages-0-14/es/sp.pop.0014.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-ages-0-14/pt/sp.pop.0014.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-ages-65-and-above/es/sp.pop.65up.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/population-ages-65-and-above/pt/sp.pop.65up.to.zs/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/male-life-expectancy-at-birth/es/sp.dyn.le00.ma.in/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/male-life-expectancy-at-birth/pt/sp.dyn.le00.ma.in/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/female-life-expectancy-at-birth/es/sp.dyn.le00.fe.in/datepart=2021-09-28',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/demographics/female-life-expectancy-at-birth/pt/sp.dyn.le00.fe.in/datepart=2021-09-28'
]

# COMMAND ----------

landing_final_df = []
for path in sourcePath:
  if '/demographics/' in path:
    df=spark.read.option("multiLine","false").json(path)
  else:
    df = spark.read.option("multiLine","true").json(path)
  json_schema = df.schema

  af = AutoFlatten(json_schema)
  af.compute()
  df1 = df
  
  visited = set([f'.{column}' for column in df1.columns])
  duplicate_target_counter = Counter(af.all_fields.values())
  cols_to_select = df1.columns
  for rest_col in af.rest:
    if rest_col not in visited:
      cols_to_select += [rest_col[1:]] if (duplicate_target_counter[af.all_fields[rest_col]]==1 and af.all_fields[rest_col] not in df1.columns) else [col(rest_col[1:]).alias(f"{rest_col[1:].replace('.', '>')}")]
      visited.add(rest_col)

  df1 = df1.select(cols_to_select)


  if af.order:
    for key in af.order:
      column = key.split('.')[-1]
      if af.bottom_to_top[key]:
        #########
        #values for the column in bottom_to_top dict exists if it is an array type
        #########
        df1 = df1.select('*', explode_outer(col(column)).alias(f"{column}_exploded")).drop(column)
        data_type = df1.select(f"{column}_exploded").schema.fields[0].dataType
        if not (isinstance(data_type, StructType) or isinstance(data_type, ArrayType)):
          df1 = df1.withColumnRenamed(f"{column}_exploded", column if duplicate_target_counter[af.all_fields[key]]<=1 else key[1:].replace('.', '>'))
          visited.add(key)
        else:
          #grabbing all paths to columns after explode
          cols_in_array_col = set(map(lambda x: f'{key}.{x}', df1.select(f'{column}_exploded.*').columns))
          #retrieving unvisited columns
          cols_to_select_set = cols_in_array_col.difference(visited)
          all_cols_to_select_set = set(af.bottom_to_top[key])
          #check done for duplicate column name & path
          cols_to_select_list = list(map(lambda x: f"{column}_exploded{'.'.join(x.split(key)[1:])}" if (duplicate_target_counter[af.all_fields[x]]<=1 and x.split('.')[-1] not in df1.columns) else col(f"{column}_exploded{'.'.join(x.split(key)[1:])}").alias(f"{x[1:].replace('.', '>')}"), list(all_cols_to_select_set)))
          #updating visited set
          visited.update(cols_to_select_set)
          rem = list(map(lambda x: f"{column}_exploded{'.'.join(x.split(key)[1:])}", list(cols_to_select_set.difference(all_cols_to_select_set))))
          df1 = df1.select(df1.columns + cols_to_select_list + rem).drop(f"{column}_exploded")      
      else:
        
        #########
        #values for the column in bottom_to_top dict do not exist if it is a struct type / array type containing a string type
        #########
        #grabbing all paths to columns after opening
        cols_in_array_col = set(map(lambda x: f'{key}.{x}', df1.selectExpr(f'{column}.*').columns))
        #retrieving unvisited columns
        cols_to_select_set = cols_in_array_col.difference(visited)
        #check done for duplicate column name & path
        cols_to_select_list = list(map(lambda x: f"{column}.{x.split('.')[-1]}" if (duplicate_target_counter[x.split('.')[-1]]<=1 and x.split('.')[-1] not in df1.columns) else col(f"{column}.{x.split('.')[-1]}").alias(f"{x[1:].replace('.', '>')}"), list(cols_to_select_set)))
        #updating visited set
        visited.update(cols_to_select_set)
        df1 = df1.select(df1.columns + cols_to_select_list).drop(f"{column}")
  final_df = df1.select([field[1:].replace('.', '>') if duplicate_target_counter[af.all_fields[field]]>1 else af.all_fields[field] for field in af.all_fields])
  final_df = final_df.toDF(*[re.sub('[>]','_',col) for col in final_df.columns])
  final_df = final_df.toDF(*[re.sub('[:]','-',col) for col in final_df.columns])
  final_df = final_df.toDF(*[re.sub('[@,#]','',col) for col in final_df.columns])
  landing_final_df.append(final_df)

# COMMAND ----------

from pyspark.sql import DataFrame
source_df =reduce(DataFrame.unionByName, landing_final_df)

# COMMAND ----------

def cleanColumn(tmpdf):
  cols_list = tmpdf.schema.names
  regex = r"[A-Za-z0-9\s_-]"
  new_cols = []
  for col in cols_list:
    matches = re.finditer(regex, col, re.MULTILINE)
    name = []
    for matchNum, match in enumerate(matches, start=1):
      name.append(match.group())
      nn = "".join(name).replace(" ","_")
      nn = nn.replace("__","_")
    tmpdf = tmpdf.withColumnRenamed(col, nn)
  return tmpdf

# COMMAND ----------

source_df = cleanColumn(source_df)

# COMMAND ----------

bronze_df_path = ['abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-growth/es/sp.pop.grow',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-growth/pt/sp.pop.grow',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population/es/sp.pop.totl',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population/pt/sp.pop.totl',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gdp-per-capita-growth/es/ny.gdp.pcap.pp.cd',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gdp-per-capita-growth/pt/ny.gdp.pcap.pp.cd',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/annual-consumer-prices-inflation/es/fp.cpi.totl.zg',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/annual-consumer-prices-inflation/pt/fp.cpi.totl.zg',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/urban-population/es/sp.urb.totl',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/urban-population/pt/sp.urb.totl',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gdp-per-capita/es/ny.gdp.pcap.cd',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gdp-per-capita/pt/ny.gdp.pcap.cd',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gross-savings/es/ny.gns.ictr.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/gross-savings/pt/ny.gns.ictr.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/net-migration/es/sm.pop.netm',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/net-migration/pt/sm.pop.netm',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/total-population-ages-15-64/es/sp.pop.1564.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/total-population-ages-15-64/pt/sp.pop.1564.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/prevalence-of-undernourishment/es/sn.itk.defc.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/prevalence-of-undernourishment/pt/sn.itk.defc.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-ages-0-14/es/sp.pop.0014.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-ages-0-14/pt/sp.pop.0014.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-ages-65-and-above/es/sp.pop.65up.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/population-ages-65-and-above/pt/sp.pop.65up.to.zs',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/male-life-expectancy-at-birth/es/sp.dyn.le00.ma.in',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/male-life-expectancy-at-birth/pt/sp.dyn.le00.ma.in',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/female-life-expectancy-at-birth/es/sp.dyn.le00.fe.in',
'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/demographics/female-life-expectancy-at-birth/pt/sp.dyn.le00.fe.in'
]

# COMMAND ----------

from pyspark.sql.functions import *
bronze_df_data = []
for path in bronze_df_path:
  #print(filePath)
  #Reading the delta history
  deltaTable = DeltaTable.forPath(spark, path)
  latest_version = deltaTable.history().select(max(col('version'))).collect()[0][0]
  #Reading the data from bonze layer
  df = spark.read.format("delta").option("versionAsOf", latest_version).load(path)
  #Getting the max process date from the bronze data and then filtering on the max process date
  max_process_date = df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
  df_filtered = df.filter(col("PROCESS_DATE")==max_process_date)
  bronze_df_data.append(df_filtered)

# COMMAND ----------

bronze_df =reduce(DataFrame.unionByName, bronze_df_data)

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

#EDW Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select("country_id","indicator_id","date").distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("country_id","indicator_id","date").distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check
print(source_df.where((col("indicator_id").isNull()) | (col("date").isNull()) | (col("indicator_id").isNull())).count())
#EDW Bronze Layer PK Null check
print(bronze_df.where((col("indicator_id").isNull()) | (col("date").isNull()) | (col("indicator_id").isNull())).count())

# COMMAND ----------

#EDW Source Layer PK Duplicate check
source_df \
    .groupby("indicator_id","date","country_id") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#EDW Bronze Layer PK Duplicate check
bronze_df \
    .groupby("indicator_id","date","country_id")\
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/demographics"
silver_df = spark.read.format("delta").load(edw_silver_path)

# COMMAND ----------

path = 'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/shipment-actuals'
deltaTable = DeltaTable.forPath(spark, path)
shipment_version = deltaTable.history().select(max(col('version'))).collect()[0][0]
#Reading the data from bonze layer
df = spark.read.format("delta").option("versionAsOf", shipment_version).load(path)
#Getting the max process date from the bronze data and then filtering on the max process date
max_process_date = df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
shipment = df.filter(col("PROCESS_DATE")==max_process_date)

# COMMAND ----------

cond = [bronze_df.country_id == shipment.DMNDFCST_MKT_UNIT_CDV]
shipment_df_filtered = shipment.select('PLANG_LOC_GRP_VAL','DMNDFCST_MKT_UNIT_CDV').distinct()
bronze_df_silver_transf = bronze_df.join(shipment_df_filtered, cond)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df_silver_transf.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['YR','CTRY_NM','MU','IND_NM','IND_CD','IND_VAL','UNIT','LOC']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("IND_CD","MU","YR","LOC").distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
print(silver_df.where((col("IND_CD").isNull())|(col("MU").isNull())|(col("YR").isNull())|(col("LOC").isNull())).count())

# COMMAND ----------

#EDW Silver Layer PK Duplicate check
silver_df \
    .groupby("IND_CD","MU","YR","LOC") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

