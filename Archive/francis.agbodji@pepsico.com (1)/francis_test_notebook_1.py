# Databricks notebook source
#imports
import pyspark
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

#storing the parameters in variables
source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
dpndntdatapath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

print(dbutils.widgets.get("targetContainer"))

# COMMAND ----------

# saving silver target path for each dataset as a parameter 
dbutils.widgets.text("slv_advertising_target_path", "silver/iberia/ibp-poc/advertising-and-media-spend")
dbutils.widgets.text("slv_customer_target_path", "")
dbutils.widgets.text("slv_dfu_target_path", "")
dbutils.widgets.text("slv_demand_forcast_target_path", "")
dbutils.widgets.text("target_path", "")
dbutils.widgets.text("target_path", "")
dbutils.widgets.text("target_path", "")
dbutils.widgets.text("target_path", "")
dbutils.widgets.text("target_path", "")
dbutils.widgets.text("target_path", "")

# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install delta-spark

# COMMAND ----------

from delta import *

builder = pyspark.sql.SparkSession.builder.appName("MyApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# COMMAND ----------

from delta import *

spark = pyspark.sql.SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:0.8.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()


# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

slv_advertising_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/advertising-and-media-spend"

# COMMAND ----------

# Reading the DFU source data from bonze layer
adv_df = spark.read.format("delta").load(slv_advertising_delta_path)
print(adv_df.count())
adv_df.show(5)

# COMMAND ----------

slv_adv_delta_table = DeltaTable.forPath(spark, slv_advertising_delta_path)
print(slv_adv_delta_table.count())
slv_adv_delta_table.show(5)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

