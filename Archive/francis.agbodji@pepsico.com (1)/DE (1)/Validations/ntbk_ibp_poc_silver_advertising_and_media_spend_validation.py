# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Users/bishnumohan.tiwary.contractor@pepsico.com/Pipeline-Notebook/NTBK_ADLS_CRED

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/advertising-and-media-spend"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check
null_df = silver_df.filter("CMPGN_ID is NULL or STRT_DT is NULL or CHNL is NULL or PLCMT_NAME is NULL")
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("CMPGN_ID","STRT_DT","CHNL","PLCMT_NAME").count().filter("count > 1")
print("There are total of "+str(dup_df.count())+" rows with duplicate values on PK columns")

# COMMAND ----------

