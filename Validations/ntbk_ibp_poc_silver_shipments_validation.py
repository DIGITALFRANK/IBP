# Databricks notebook source
"""
This notebook is used to perform validations on the shipment-actuals data present in Silver layer
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=cdodevadls2

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/shipment-actuals"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check
null_df = silver_df.filter("PROD_CD is NULL or CUST_GRP is NULL or LOC is NULL or STRTDT is NULL")
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("TYP").count().filter("count > 1")
display(dup_df)

# COMMAND ----------

display(silver_df)

# COMMAND ----------

display(silver_df.select('QTY'))

# COMMAND ----------

