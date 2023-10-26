# Databricks notebook source
# MAGIC %md # Macroeconomic Factors SIT

# COMMAND ----------

from pyspark.sql.functions import col, explode_outer, current_date
from pyspark.sql.types import *
from copy import deepcopy
from collections import Counter
from delta.tables import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
import re
from functools import reduce
import json
from delta.tables import *
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

# MAGIC %md ## Source to Bronze Layer

# COMMAND ----------

#ADLS Paths - Archive
consumer_prices_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/consumer-prices/m.es.pcpi_ix/datepart=2021-10-08/"
consumer_prices_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/consumer-prices/m.pt.pcpi_ix/datepart=2021-10-08/"

gross_domestic_product_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/gross-domestic-product/q.es.ngdp_nsa_xdc/datepart=2021-10-08/"
gross_domestic_product_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/gross-domestic-product/q.pt.ngdp_nsa_xdc/datepart=2021-10-08/"

labor_force_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/labor-force/q.es.llf_pe_num/datepart=2021-10-08/"
labor_force_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/labor-force/m.pt.llf_pe_num/datepart=2021-10-08/"

labor_market_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/labor-market/m.es.lwr_ix/datepart=2021-10-08/"
labor_market_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/labor-market/q.pt.lwr_ix/datepart=2021-10-08/"

tourism_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/tourism/m.es.aotv_pe_num/datepart=2021-10-08/"
tourism_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/tourism/m.pt.aotv_pe_num/datepart=2021-10-08/"

unemployment_es_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/unemployment/m.es.lur_pt/datepart=2021-10-08/"
unemployment_pt_archive = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/macroeconomic-factors/unemployment/m.pt.lur_pt/datepart=2021-10-08/"

# COMMAND ----------

#ADLS Paths - Bronze
consumer_prices_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/consumer-prices/m.es.pcpi_ix/"
consumer_prices_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/consumer-prices/m.pt.pcpi_ix/"

gross_domestic_product_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/gross-domestic-product/q.es.ngdp_nsa_xdc/"
gross_domestic_product_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/gross-domestic-product/q.pt.ngdp_nsa_xdc/"

labor_force_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/labor-force/q.es.llf_pe_num/"
labor_force_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/labor-force/m.pt.llf_pe_num/"

labor_market_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/labor-market/m.es.lwr_ix/"
labor_market_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/labor-market/q.pt.lwr_ix/"

tourism_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/tourism/m.es.aotv_pe_num/"
tourism_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/tourism/m.pt.aotv_pe_num/"

unemployment_es_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/unemployment/m.es.lur_pt/"
unemployment_pt_bronze = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/macroeconomic-factors/unemployment/m.pt.lur_pt/"

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_json_common_classes

# COMMAND ----------

###Function to replicate the steps performed in the bronze json load process
def json_flatten(path):
    df = spark.read.option("multiLine","true").json(path)
    json_schema = df.schema
    #print(json_schema)
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
    
    final_df = cleanColumn(final_df)
    
    return final_df


# COMMAND ----------

#column cleanup function called in the json process
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

#### Source to Bronze layer check for - consumer_prices_es
print("Dataset: consumer_prices_es")
###Reading consumer_prices_es from archive layer in flattened format
consumer_prices_es_archive_flt_df = json_flatten(consumer_prices_es_archive)
#Archive Count
print("Source Count: "+str(consumer_prices_es_archive_flt_df.count()))

###Reading consumer_prices_es from bronze layer
consumer_prices_es_bronze_df = spark.read.format('delta').load(consumer_prices_es_bronze)
#Bronze Count
print("Bronze Count: "+str(consumer_prices_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  consumer_prices_es_archive_flt_df.columns
brnz_col = consumer_prices_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(consumer_prices_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(consumer_prices_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(consumer_prices_es_archive_flt_df.select(src_col).exceptAll(consumer_prices_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(consumer_prices_es_bronze_df.select(src_col).exceptAll(consumer_prices_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - consumer_prices_pt
print("Dataset: consumer_prices_pt")
###Reading consumer_prices_pt from archive layer in flattened format
consumer_prices_pt_archive_flt_df = json_flatten(consumer_prices_pt_archive)
#Archive Count
print("Source Count: "+str(consumer_prices_pt_archive_flt_df.count()))

###Reading consumer_prices_pt from bronze layer
consumer_prices_pt_bronze_df = spark.read.format('delta').load(consumer_prices_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(consumer_prices_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  consumer_prices_pt_archive_flt_df.columns
brnz_col = consumer_prices_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(consumer_prices_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(consumer_prices_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(consumer_prices_pt_archive_flt_df.select(src_col).exceptAll(consumer_prices_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(consumer_prices_pt_bronze_df.select(src_col).exceptAll(consumer_prices_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - gross_domestic_product_es_archive
print("Dataset: gross_domestic_product_es_archive")
###Reading gross_domestic_product_es from archive layer in flattened format
gross_domestic_product_es_archive_flt_df = json_flatten(gross_domestic_product_es_archive)
#Archive Count
print("Source Count: "+str(gross_domestic_product_es_archive_flt_df.count()))

###Reading gross_domestic_product_es from bronze layer
gross_domestic_product_es_bronze_df = spark.read.format('delta').load(gross_domestic_product_es_bronze)
#Bronze Count
print("Bronze Count: "+str(gross_domestic_product_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  gross_domestic_product_es_archive_flt_df.columns
brnz_col = gross_domestic_product_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(gross_domestic_product_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(gross_domestic_product_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(gross_domestic_product_es_archive_flt_df.select(src_col).exceptAll(gross_domestic_product_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(gross_domestic_product_es_bronze_df.select(src_col).exceptAll(gross_domestic_product_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - gross_domestic_product_pt
print("Dataset: gross_domestic_product_pt_pt")
###Reading gross_domestic_product_pt from archive layer in flattened format
gross_domestic_product_pt_archive_flt_df = json_flatten(gross_domestic_product_pt_archive)
#Archive Count
print("Source Count: "+str(gross_domestic_product_pt_archive_flt_df.count()))

###Reading consumer_prices_pt from bronze layer
gross_domestic_product_pt_bronze_df = spark.read.format('delta').load(gross_domestic_product_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(gross_domestic_product_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  gross_domestic_product_pt_archive_flt_df.columns
brnz_col = gross_domestic_product_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(gross_domestic_product_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(gross_domestic_product_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(gross_domestic_product_pt_archive_flt_df.select(src_col).exceptAll(gross_domestic_product_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(gross_domestic_product_pt_bronze_df.select(src_col).exceptAll(gross_domestic_product_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - labor_force_es_archive
print("Dataset: labor_force_es_archive")
###Reading labor_force_es from archive layer in flattened format
labor_force_es_archive_flt_df = json_flatten(labor_force_es_archive)
#Archive Count
print("Source Count: "+str(labor_force_es_archive_flt_df.count()))

###Reading labor_force_es from bronze layer
labor_force_es_bronze_df = spark.read.format('delta').load(labor_force_es_bronze)
#Bronze Count
print("Bronze Count: "+str(labor_force_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  labor_force_es_archive_flt_df.columns
brnz_col = labor_force_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(labor_force_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(labor_force_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(labor_force_es_archive_flt_df.select(src_col).exceptAll(labor_force_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(labor_force_es_bronze_df.select(src_col).exceptAll(labor_force_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - labor_force_pt_pt_archive
print("Dataset: labor_force_pt_archive")
###Reading labor_force_pt from archive layer in flattened format
labor_force_pt_archive_flt_df = json_flatten(labor_force_pt_archive)
#Archive Count
print("Source Count: "+str(labor_force_pt_archive_flt_df.count()))

###Reading labor_force_pt from bronze layer
labor_force_pt_bronze_df = spark.read.format('delta').load(labor_force_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(labor_force_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  labor_force_pt_archive_flt_df.columns
brnz_col = labor_force_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(labor_force_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(labor_force_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(labor_force_pt_archive_flt_df.select(src_col).exceptAll(labor_force_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(labor_force_pt_bronze_df.select(src_col).exceptAll(labor_force_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - labor_market_es_archive
print("Dataset: labor_market_es_archive")
###Reading labor_market_es from archive layer in flattened format
labor_market_es_archive_flt_df = json_flatten(labor_market_es_archive)
#Archive Count
print("Source Count: "+str(labor_market_es_archive_flt_df.count()))

###Reading labor_market_es from bronze layer
labor_market_es_bronze_df = spark.read.format('delta').load(labor_market_es_bronze)
#Bronze Count
print("Bronze Count: "+str(labor_market_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  labor_market_es_archive_flt_df.columns
brnz_col = labor_market_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(labor_market_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(labor_market_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(labor_market_es_archive_flt_df.select(src_col).exceptAll(labor_market_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(labor_market_es_bronze_df.select(src_col).exceptAll(labor_market_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - labor_market_pt_archive
print("Dataset: labor_market_pt_archive")
###Reading labor_market_pt from archive layer in flattened format
labor_market_pt_archive_flt_df = json_flatten(labor_market_pt_archive)
#Archive Count
print("Source Count: "+str(labor_market_pt_archive_flt_df.count()))

###Reading labor_market_pt from bronze layer
labor_market_pt_bronze_df = spark.read.format('delta').load(labor_market_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(labor_market_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  labor_market_pt_archive_flt_df.columns
brnz_col = labor_market_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(labor_market_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(labor_market_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(labor_market_pt_archive_flt_df.select(src_col).exceptAll(labor_market_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(labor_market_pt_bronze_df.select(src_col).exceptAll(labor_market_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - tourism_es_archive
print("Dataset: tourism_es_archive")
###Reading tourism_es from archive layer in flattened format
tourism_es_archive_flt_df = json_flatten(tourism_es_archive)
#Archive Count
print("Source Count: "+str(tourism_es_archive_flt_df.count()))

###Reading tourism_es from bronze layer
tourism_es_bronze_df = spark.read.format('delta').load(tourism_es_bronze)
#Bronze Count
print("Bronze Count: "+str(tourism_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  tourism_es_archive_flt_df.columns
brnz_col = tourism_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(tourism_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(tourism_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(tourism_es_archive_flt_df.select(src_col).exceptAll(tourism_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(tourism_es_bronze_df.select(src_col).exceptAll(tourism_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - tourism_pt_archive
print("Dataset: tourism_pt_archive")
###Reading tourism_pt from archive layer in flattened format
tourism_pt_archive_flt_df = json_flatten(tourism_pt_archive)
#Archive Count
print("Source Count: "+str(tourism_pt_archive_flt_df.count()))

###Reading tourism_pt from bronze layer
tourism_pt_bronze_df = spark.read.format('delta').load(tourism_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(tourism_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  tourism_pt_archive_flt_df.columns
brnz_col = tourism_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(tourism_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(tourism_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(tourism_pt_archive_flt_df.select(src_col).exceptAll(tourism_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(tourism_pt_bronze_df.select(src_col).exceptAll(tourism_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - unemployment_es_archive
print("Dataset: unemployment_es_archive")
###Reading unemployment_es from archive layer in flattened format
unemployment_es_archive_flt_df = json_flatten(unemployment_es_archive)
#Archive Count
print("Source Count: "+str(unemployment_es_archive_flt_df.count()))

###Reading unemployment_es from bronze layer
unemployment_es_bronze_df = spark.read.format('delta').load(unemployment_es_bronze)
#Bronze Count
print("Bronze Count: "+str(unemployment_es_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  unemployment_es_archive_flt_df.columns
brnz_col = unemployment_es_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(unemployment_es_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(unemployment_es_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(unemployment_es_archive_flt_df.select(src_col).exceptAll(unemployment_es_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(unemployment_es_bronze_df.select(src_col).exceptAll(unemployment_es_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

#### Source to Bronze layer check for - unemployment_pt_archive
print("Dataset: unemployment_pt_archive")
###Reading unemployment_pt from archive layer in flattened format
unemployment_pt_archive_flt_df = json_flatten(unemployment_pt_archive)
#Archive Count
print("Source Count: "+str(unemployment_pt_archive_flt_df.count()))

###Reading unemployment_pt from bronze layer
unemployment_pt_bronze_df = spark.read.format('delta').load(unemployment_pt_bronze)
#Bronze Count
print("Bronze Count: "+str(unemployment_pt_bronze_df.count()))

#Source and Bronze layer column validation
src_col =  unemployment_pt_archive_flt_df.columns
brnz_col = unemployment_pt_bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))
print("Number of columns in Source: "+str(len(unemployment_pt_archive_flt_df.columns)))
print("Number of columns in Bronze: "+str(len(unemployment_pt_bronze_df.columns)))

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa
print("Count of Missing Rows in Bronze are " + str(+(unemployment_pt_archive_flt_df.select(src_col).exceptAll(unemployment_pt_bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(unemployment_pt_bronze_df.select(src_col).exceptAll(unemployment_pt_archive_flt_df.select(src_col))).count()))

#PK checks

# COMMAND ----------

# consumer_prices_pt_archive_flt_df.printSchema()
consumer_prices_pt_bronze_df.printSchema()

# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
consumer_prices_pt_archive_flt_df.createOrReplaceTempView("consumer_prices_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from consumer_prices_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
consumer_prices_pt_bronze_df.createOrReplaceTempView("consumer_prices_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from consumer_prices_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

# COMMAND ----------

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",consumer_prices_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",consumer_prices_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",consumer_prices_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",consumer_prices_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",consumer_prices_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",consumer_prices_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# COMMAND ----------

# Source Layer Primary Key Uniqueness check
print(consumer_prices_pt_archive_flt_df.count())
print(consumer_prices_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(consumer_prices_pt_bronze_df.count())
print(consumer_prices_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())

# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
consumer_prices_es_archive_flt_df.createOrReplaceTempView("consumer_prices_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from consumer_prices_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
consumer_prices_es_bronze_df.createOrReplaceTempView("consumer_prices_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from consumer_prices_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",consumer_prices_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",consumer_prices_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",consumer_prices_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",consumer_prices_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",consumer_prices_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",consumer_prices_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(consumer_prices_es_archive_flt_df.count())
print(consumer_prices_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(consumer_prices_es_bronze_df.count())
print(consumer_prices_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
gross_domestic_product_es_archive_flt_df.createOrReplaceTempView("gross_domestic_product_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from gross_domestic_product_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
gross_domestic_product_es_bronze_df.createOrReplaceTempView("gross_domestic_product_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from gross_domestic_product_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",gross_domestic_product_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",gross_domestic_product_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",gross_domestic_product_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",gross_domestic_product_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",gross_domestic_product_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",gross_domestic_product_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(gross_domestic_product_es_archive_flt_df.count())
print(gross_domestic_product_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(gross_domestic_product_es_bronze_df.count())
print(gross_domestic_product_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())

# COMMAND ----------



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
gross_domestic_product_pt_archive_flt_df.createOrReplaceTempView("gross_domestic_product_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from gross_domestic_product_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
gross_domestic_product_pt_bronze_df.createOrReplaceTempView("gross_domestic_product_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from gross_domestic_product_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",gross_domestic_product_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",gross_domestic_product_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",gross_domestic_product_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",gross_domestic_product_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",gross_domestic_product_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",gross_domestic_product_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(gross_domestic_product_pt_archive_flt_df.count())
print(gross_domestic_product_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(gross_domestic_product_pt_bronze_df.count())
print(gross_domestic_product_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
labor_force_pt_archive_flt_df.createOrReplaceTempView("labor_force_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_force_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
labor_force_pt_bronze_df.createOrReplaceTempView("labor_force_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_force_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",labor_force_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",labor_force_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",labor_force_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",labor_force_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",labor_force_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",labor_force_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(labor_force_pt_archive_flt_df.count())
print(labor_force_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(labor_force_pt_bronze_df.count())
print(labor_force_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
labor_force_es_archive_flt_df.createOrReplaceTempView("labor_force_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_force_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
labor_force_es_bronze_df.createOrReplaceTempView("labor_force_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_force_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",labor_force_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",labor_force_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",labor_force_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",labor_force_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",labor_force_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",labor_force_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(labor_force_es_archive_flt_df.count())
print(labor_force_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(labor_force_es_bronze_df.count())
print(labor_force_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
labor_market_pt_archive_flt_df.createOrReplaceTempView("labor_market_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_market_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
labor_market_pt_bronze_df.createOrReplaceTempView("labor_market_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_market_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",labor_market_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",labor_market_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",labor_market_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",labor_market_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",labor_market_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",labor_market_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(labor_market_pt_archive_flt_df.count())
print(labor_market_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(labor_market_pt_bronze_df.count())
print(labor_market_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
labor_market_es_archive_flt_df.createOrReplaceTempView("labor_market_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_market_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
labor_market_es_bronze_df.createOrReplaceTempView("labor_market_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from labor_market_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",labor_market_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",labor_market_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",labor_market_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",labor_market_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",labor_market_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",labor_market_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(labor_market_es_archive_flt_df.count())
print(labor_market_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(labor_market_es_bronze_df.count())
print(labor_market_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
tourism_pt_archive_flt_df.createOrReplaceTempView("tourism_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from tourism_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
tourism_pt_bronze_df.createOrReplaceTempView("tourism_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from tourism_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",tourism_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",tourism_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",tourism_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",tourism_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",tourism_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",tourism_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(tourism_pt_archive_flt_df.count())
print(tourism_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(tourism_pt_bronze_df.count())
print(tourism_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
tourism_es_archive_flt_df.createOrReplaceTempView("tourism_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from tourism_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
tourism_es_bronze_df.createOrReplaceTempView("tourism_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from tourism_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",tourism_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",tourism_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",tourism_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",tourism_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",tourism_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",tourism_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(tourism_es_archive_flt_df.count())
print(tourism_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(tourism_es_bronze_df.count())
print(tourism_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
unemployment_pt_archive_flt_df.createOrReplaceTempView("unemployment_pt_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from unemployment_pt_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
unemployment_pt_bronze_df.createOrReplaceTempView("unemployment_pt_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from unemployment_pt_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",unemployment_pt_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",unemployment_pt_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",unemployment_pt_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",unemployment_pt_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",unemployment_pt_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",unemployment_pt_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(unemployment_pt_archive_flt_df.count())
print(unemployment_pt_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(unemployment_pt_bronze_df.count())
print(unemployment_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

##Primary Key check - Source to Bronze Layer 
unemployment_es_archive_flt_df.createOrReplaceTempView("unemployment_es_archive")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from unemployment_es_archive group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)
unemployment_es_bronze_df.createOrReplaceTempView("unemployment_es_bronze")
spark.sql("""select TIME_PERIOD,INDICATOR,REF_AREA,count(*) from unemployment_es_bronze group by TIME_PERIOD,INDICATOR,REF_AREA
                          having count(*) > 1 """).show(10)

#Source Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in TIME_PERIOD Column: ",unemployment_es_archive_flt_df.where(col("TIME_PERIOD").isNull()).count())
print("Source Layer Null Values in INDICATOR Column: ",unemployment_es_archive_flt_df.where(col("INDICATOR").isNull()).count())
print("Source Layer Null Values in REF_AREA Column: ",unemployment_es_archive_flt_df.where(col("REF_AREA").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in TIME_PERIOD Column: ",unemployment_es_bronze_df.where(col("TIME_PERIOD").isNull()).count())
print("Bronze Layer Null Values in INDICATOR Column: ",unemployment_es_bronze_df.where(col("INDICATOR").isNull()).count())
print("Bronze Layer Null Values in REF_AREA Column: ",unemployment_es_bronze_df.where(col("REF_AREA").isNull()).count())

# Source Layer Primary Key Uniqueness check
print(unemployment_es_archive_flt_df.count())
print(unemployment_es_archive_flt_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())
# Bronze Layer Primary Key Uniqueness check
print(unemployment_es_bronze_df.count())
print(unemployment_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA').distinct().count())



# COMMAND ----------

silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/macroeconomic-factors"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

silver_df.count()

# COMMAND ----------

print("Count of consumer_prices_es_bronze_df : " +str(consumer_prices_es_bronze_df.count()) )
print("Count of consumer_prices_pt_bronze_df : " +str(consumer_prices_pt_bronze_df.count()))
print("Count of gross_domestic_product_es_bronze_df : " +str(gross_domestic_product_es_bronze_df.count()))
print("Count of gross_domestic_product_pt_bronze_df : " +str(gross_domestic_product_pt_bronze_df.count()))
print("Count of labor_force_es_bronze_df : " +str(labor_force_es_bronze_df.count()))
print("Count of labor_force_pt_bronze_df : " +str(labor_force_pt_bronze_df.count()))
print("Count of labor_market_es_bronze_df : " +str(labor_market_es_bronze_df.count()))
print("Count of labor_market_pt_bronze_df : " +str(labor_market_pt_bronze_df.count()))
print("Count of tourism_es_bronze_df : " +str(tourism_es_bronze_df.count()))
print("Count of tourism_pt_bronze_df : " +str(tourism_pt_bronze_df.count()))
print("Count of unemployment_es_bronze_df : " +str(unemployment_es_bronze_df.count()))
print("Count of unemployment_pt_bronze_df : " +str(unemployment_pt_bronze_df.count()))
total_count = consumer_prices_es_bronze_df.count() + consumer_prices_pt_bronze_df.count() + gross_domestic_product_es_bronze_df.count() + gross_domestic_product_pt_bronze_df.count() + labor_force_es_bronze_df.count() + labor_force_pt_bronze_df.count() + labor_market_es_bronze_df.count() + labor_market_pt_bronze_df.count() + tourism_es_bronze_df.count() + tourism_pt_bronze_df.count() + unemployment_es_bronze_df.count() + unemployment_pt_bronze_df.count()
print("total_count : " +str(total_count))

# COMMAND ----------

consumer_prices_es_bronze_df = consumer_prices_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
consumer_prices_pt_bronze_df = consumer_prices_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
gross_domestic_product_es_bronze_df = gross_domestic_product_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
gross_domestic_product_pt_bronze_df= gross_domestic_product_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
labor_force_es_bronze_df = labor_force_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
labor_force_pt_bronze_df = labor_force_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
labor_market_es_bronze_df = labor_market_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
labor_market_pt_bronze_df = labor_market_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
tourism_es_bronze_df = tourism_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
tourism_pt_bronze_df = tourism_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
unemployment_es_bronze_df = unemployment_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')
unemployment_pt_bronze_df = unemployment_pt_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')

dfs = [consumer_prices_es_bronze_df,consumer_prices_pt_bronze_df,gross_domestic_product_es_bronze_df,gross_domestic_product_pt_bronze_df,labor_force_es_bronze_df,labor_force_pt_bronze_df,labor_market_es_bronze_df,labor_market_pt_bronze_df,tourism_es_bronze_df,tourism_pt_bronze_df,unemployment_es_bronze_df,unemployment_pt_bronze_df]

silver_df_temp = reduce(DataFrame.unionByName, dfs)
print("Total union count : " +str(silver_df_temp.count()))


# COMMAND ----------

shipment = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/shipment-actuals/"
#Reading the delta history shipment
shipment_deltaTable = DeltaTable.forPath(spark, shipment)
shipment_latest_version = shipment_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(shipment_latest_version)

#Reading the data from bonze layer shipment
shipment_df = spark.read.format("delta").option("versionAsOf", shipment_latest_version).load(shipment)

cond = [silver_df_temp.REF_AREA == shipment_df.DMNDFCST_MKT_UNIT_CDV]

shipment_df_filtered = shipment_df.select('PLANG_LOC_GRP_VAL','DMNDFCST_MKT_UNIT_CDV').distinct()
Bronze_final_df = silver_df_temp.join(shipment_df_filtered, cond)

print("Bronze Final Count : " +str(Bronze_final_df.count()))

silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/macroeconomic-factors"
silver_df = spark.read.format("delta").load(silver_path)
print("Silver Count : " +str(Bronze_final_df.count()))

# COMMAND ----------

consumer_prices_es_bronze_df = consumer_prices_es_bronze_df.select('TIME_PERIOD','INDICATOR','REF_AREA')

# COMMAND ----------

#Silver Layer PK Null check - Source to Bronze Layer 
print("Source Layer Null Values in DT Column: ",silver_df.where(col("DT").isNull()).count())
print("Source Layer Null Values in IND Column: ",silver_df.where(col("IND").isNull()).count())
print("Source Layer Null Values in CNTRY_CD Column: ",silver_df.where(col("CNTRY_CD").isNull()).count())
print("Source Layer Null Values in FCTR_NM Column: ",silver_df.where(col("FCTR_NM").isNull()).count())
print("Source Layer Null Values in LOC Column: ",silver_df.where(col("LOC").isNull()).count())
print("Source Layer Null Values in PRD Column: ",silver_df.where(col("PRD").isNull()).count())

# COMMAND ----------

###Silver Duplicate check
dup=silver_df.groupBy("DT","PRD","IND","CNTRY_CD","FCTR_NM","LOC").count().filter("count > 1")
dup.display() 

# COMMAND ----------

###Silver unique Count check
print("Silver count : "  +str(silver_df.count()))
print("Silver unique count "  +str(silver_df.select("DT","PRD","IND","CNTRY_CD","FCTR_NM","LOC").distinct().count()))


# COMMAND ----------

silver_col_mdl = ['DT','IND','CNTRY_CD','VAL','FCTR_NM','PRD','LOC']
silver_df_col = silver_df.columns

print("Missing columns in silver : ", (set(silver_col_mdl).difference(silver_df_col)))
print("Extra columns in Silver : ", set(silver_df_col).difference(silver_col_mdl) )

# COMMAND ----------

silver_df.printSchema()

# COMMAND ----------

display(silver_df.filter('IND = "AOTV_PE_NUM" '))

# COMMAND ----------

silver_df.select('IND','FCTR_NM').distinct().show(10,False)

# COMMAND ----------

