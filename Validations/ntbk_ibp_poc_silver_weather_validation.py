# Databricks notebook source
"""
This notebook is used to perform validations on the Weather Forecast and History data present in Silver layer
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

# MAGIC %md # Weather History

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/weather-history"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check
print(silver_df.filter("RGN is NULL").count())
print(silver_df.filter("PRVN is NULL").count())
print(silver_df.filter("DT is NULL").count())
print(silver_df.filter("LOC is NULL").count())
print(silver_df.filter("MU_CHNL is NULL").count())
print(silver_df.filter("CTGY is NULL").count())

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("RGN", "PRVN", "DT", "LOC", "MU_CHNL", "CTGY").count().filter("count > 1")
display(dup_df)

# COMMAND ----------

display(silver_df)

# COMMAND ----------

silver_df.filter("PRVN is NULL").display()

# COMMAND ----------

silver_df.filter("DT is NULL").display()

# COMMAND ----------

# MAGIC %md # Weather Forecast

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/weather-forecast"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check - RGN;PRVN;TIME_INIT_UTC;DT;LOC;MU_CHNL;CTGY
print(silver_df.filter("RGN is NULL").count())
print(silver_df.filter("PRVN is NULL").count())
print(silver_df.filter("TIME_INIT_UTC is NULL").count())
print(silver_df.filter("DT is NULL").count())
print(silver_df.filter("LOC is NULL").count())
print(silver_df.filter("MU_CHNL is NULL").count())
print(silver_df.filter("CTGY is NULL").count())

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("RGN", "PRVN", "TIME_INIT_UTC", "DT", "LOC", "MU_CHNL", "CTGY").count().filter("count > 1")
display(dup_df)

# COMMAND ----------

