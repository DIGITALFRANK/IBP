# Databricks notebook source
# MAGIC %md # Time Master SIT

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

# MAGIC %md ## Silver Layer

# COMMAND ----------

silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/time-master"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

display(silver_df)

# COMMAND ----------

#count
print("Silver Layer Count: "+str(silver_df.count()))

# COMMAND ----------

# MAGIC %md ### Silver Layer Primary Key check

# COMMAND ----------

#Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("Week_start_date","Month_Of_Year").distinct().count())

# COMMAND ----------

#Silver Layer PK Null check
print(silver_df.where(col("Week_start_date").isNull()).where(col("Month_Of_Year").isNull()).count())

# COMMAND ----------

#Silver Layer PK Duplicate check
silver_df \
    .groupby("Week_start_date", "Month_Of_Year") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

