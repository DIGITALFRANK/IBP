# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Sourcing
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= 'cdodevadls2' 

# COMMAND ----------

#Defining the source and bronze path for EDL source of Sourcing
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/sourcing/datepart=2021-10-04/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp/sourcing"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(source_path)
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Sourcing
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

#Source and Bronze layer column validation for Sourcing
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

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Sourcing
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))