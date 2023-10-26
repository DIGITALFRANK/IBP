# Databricks notebook source
"""
This notebook is used to perform validations on the Demand Forecast data present in Silver layer
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Users/schakra1@pepsico.com/Silver/NTBK_ADLS_CRED

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/dfu-to-sku-forecast"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

#checking for full row duplicates
distinct_df = silver_df.distinct()
print(silver_df.exceptAll(distinct_df).count())

# COMMAND ----------

display(silver_df)

# COMMAND ----------

