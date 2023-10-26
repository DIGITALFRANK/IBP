# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run Shared/IBP/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetFilename", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")
dbutils.widgets.text("loadType","")
dbutils.widgets.text("deltaColumn","")
dbutils.widgets.text("isHistory","")

# COMMAND ----------

source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
tgtFname = dbutils.widgets.get("targetFilename")
loadType = dbutils.widgets.get("loadType")
incCol = dbutils.widgets.get("deltaColumn")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
isHistory = dbutils.widgets.get("isHistory")

# COMMAND ----------

if (loadType == 'Incremental' and isHistory != 'True'):
  df = spark.read.format("delta").load(srcPath)
  max_value = df.agg({incCol: "max"}).collect()[0][0]
  df = df.filter(col(incCol)==max_value)
else :
  df = spark.read.format("delta").load(srcPath)
df.display()

# COMMAND ----------

df = df.withColumn("created_by", lit("adb_silver_svc"))\
.withColumn("created_datetime",current_timestamp())

# COMMAND ----------

def file_rename(tgtPath, file_format):
  [dbutils.fs.rm(f.path,True) for  f in [file for file in dbutils.fs.ls(tgtPath) if file.name.startswith("_")]]
  partFile = "".join([file.name for file in dbutils.fs.ls(tgtPath) if file.name.startswith("part-")])
  dbutils.fs.mv(tgtPath+"/"+partFile,tgtPath+"/"+tgtFname+"."+file_format)

# COMMAND ----------

df.coalesce(1).write.mode("overwrite").parquet(tgtPath)
file_rename(tgtPath,"parquet")       

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(df.count()))