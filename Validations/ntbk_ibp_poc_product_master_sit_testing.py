# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Product Master dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#Defining the source and bronze path for EDW source of Product Master
edw_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/edw/ibp-poc/product-master/datepart=2021-07-29/"
edw_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/product-master"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edw_source_path)
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
print(source_df.select("PLANG_MTRL_GRP_VAL").distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("PLANG_MTRL_GRP_VAL").distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check
print(source_df.where(col("PLANG_MTRL_GRP_VAL").isNull()).count())
#EDW Bronze Layer PK Null check
print(bronze_df.where(col("PLANG_MTRL_GRP_VAL").isNull()).count())

# COMMAND ----------

#EDW Source Layer PK Duplicate check
source_df \
    .groupby("PLANG_MTRL_GRP_VAL") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#EDW Bronze Layer PK Duplicate check
bronze_df \
    .groupby("PLANG_MTRL_GRP_VAL") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/product-master"
silver_df = spark.read.format("delta").load(edw_silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['PROD_CD','PROD_NM','PCKSIZE','UNITSIZE','HRCHYLVL','SUBBRND','SIZE','FLVR','LVL','BRND','BRND_GRP','CTGY','FAT','KG','LITRES','PCK_CNTNR','PARNT','PROD_LN','DEL_DT','STTS','CS','CS2','CRTDFU','8OZ','SIZE_SHRT','PCK_CNTNR_SHRT','FLVR_SHRT','LOCL_DSC','CASE_TYP']

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("PROD_CD").distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
print(silver_df.where(col("PROD_CD").isNull()).count())

# COMMAND ----------

#EDW Silver Layer PK Duplicate check
silver_df \
    .groupby("PROD_CD") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

