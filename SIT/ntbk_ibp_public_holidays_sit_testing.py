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

edw_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/public-holidays/datepart=2021-10-07/"
edw_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/public-holidays/"

# COMMAND ----------

source_df = spark.read.format("csv").option("header",True).load(edw_source_path)
bronze_df = spark.read.format("delta").load(edw_bronze_path)

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
print(source_df.select("Date","CountryCode","LocalName").distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("Date","CountryCode","LocalName").distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check
print(source_df.where((col("Date").isNull()) | (col("CountryCode").isNull()) | (col("LocalName").isNull())).count())
#EDW Bronze Layer PK Null check
print(bronze_df.where((col("Date").isNull()) | (col("CountryCode").isNull()) | (col("LocalName").isNull())).count())

# COMMAND ----------

#EDW Source Layer PK Duplicate check
source_df \
    .groupby("Date","CountryCode","LocalName") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#EDW Bronze Layer PK Duplicate check
bronze_df \
    .groupby("Date","CountryCode","LocalName")\
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/public-holidays"
silver_df = spark.read.format("delta").load(edw_silver_path)

# COMMAND ----------

path = 'abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/shipment-actuals'
#Reading the data from bonze layer
shipment_df = spark.read.format("delta").load(path).select("PLANG_LOC_GRP_VAL","DMNDFCST_MKT_UNIT_CDV").distinct()

# COMMAND ----------

bronze_df_transf = bronze_df.join(shipment_df, bronze_df.CountryCode == shipment_df.DMNDFCST_MKT_UNIT_CDV, how = 'left')

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df_transf.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['MU','DT','HOL_LNM','HOL_TYP','CNTY','LOC','HOL_NM']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("HOL_LNM","MU","DT","LOC").distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
print(silver_df.where((col("HOL_LNM").isNull())|(col("MU").isNull())|(col("DT").isNull())|(col("LOC").isNull())).count())

# COMMAND ----------

#EDW Silver Layer PK Duplicate check
silver_df \
    .groupby("HOL_LNM","MU","DT","LOC") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

