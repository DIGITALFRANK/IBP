# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependent_dataset_path", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("stgAccount", "")

# COMMAND ----------

#storing the parameters in variables
stgAccnt = dbutils.widgets.get("stgAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
dfuviewPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependent_dataset_path")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

# MAGIC %run /Users/bishnumohan.tiwary.contractor@pepsico.com/Pipeline-Notebook/NTBK_ADLS_CRED

# COMMAND ----------

#Reading the data from the bronze path of DFU table
dfu_deltaTable = DeltaTable.forPath(spark, tgtPath)
dfu_latest_version = dfu_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(dfu_latest_version)
display(dfu_deltaTable.history())

# COMMAND ----------

#Reading the DFU source data from bonze layer
dfu_df = spark.read.format("delta").option("versionAsOf", dfu_latest_version).load(tgtPath)

# COMMAND ----------

if len(pkList.split(';'))>1:
  ls = ["col('"+attr+"').isNull()" for attr in pkList.split(';')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+pkList+"').isNull()"
print(null_cond)

# COMMAND ----------

dfu_dq_passed = dfu_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pkList.split(';')).orderBy(desc("STRT_DT")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null")))

# COMMAND ----------

display(dfu_dq_passed.select(col("Corrupt_Record")).distinct())

# COMMAND ----------

##General Data Quality checks on Primary Key columns of Silver dataframe
##Primary Key null check
if len(pkList.split(';'))>1:
  ls = [attr+" is NULL" for attr in pkList.split(';')]
  null_cond = " or ".join(ls)
else :
  null_cond = pkList+" is NULL"
print(null_cond)
#filtering the silver dataframe for null values on PK columns
null_df = dfu_df.filter(null_cond)
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")
#removing the null rows from silver dataframe
dfu_df = dfu_df.exceptAll(null_df)

# COMMAND ----------

#Primary Key duplicate check and DateFormat check
dfu_dq_passed = dfu_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pkList.split(';'))
                                                         .orderBy(desc("STRT_DT")))).withColumn("Corrupt_Record",when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key"))).where(col("Corrupt_Record").isNull()).drop("Corrupt_Record","DUP_CHECK")

# COMMAND ----------

# COunt Should be same
print(dfu_df.count())
print(dfu_dq_passed.count())

# COMMAND ----------

