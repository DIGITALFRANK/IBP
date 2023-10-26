# Databricks notebook source
"""
This notebook is used to perform validations on the distribution master data present in Silver layer
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Users/schakra1@pepsico.com/Silver/NTBK_ADLS_CRED

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/distribution-master"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check
null_df = silver_df.filter("LOC is NULL")
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("LOC").count().filter("count > 1")
display(dup_df)

# COMMAND ----------

