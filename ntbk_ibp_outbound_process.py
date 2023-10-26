# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetFilename", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")

# COMMAND ----------

tgtFname = dbutils.widgets.get("targetFilename")
wk_fname = tgtFname.split(";")[0]
m_fname = tgtFname.split(";")[1]

# COMMAND ----------

tgtPath = dbutils.widgets.get("targetPath")
wk_tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+tgtPath.split(";")[0]
m_tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+tgtPath.split(";")[1]

# COMMAND ----------

#srcPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/DBO_FORECAST_FUTURE_PERIOD"
dfu_df = spark.read.format("delta").load(srcPath)

# COMMAND ----------

dfu_df.display()

# COMMAND ----------

dfu_df.registerTempTable("DFU_VW")

# COMMAND ----------

manualweeklydf = spark.sql("Select PROD_CD AS DMDUNIT, CUST_GRP AS DMDGROUP, LOC, 'SHIPMENTS_LEW' AS MODEL, date_format(STRT_DT,'MM/dd/yyyy') AS STARTDATE, 'IBP' AS FCST_ID, '3' AS TYPE, fnl_aggrd_val as QTY, CONCAT(tm_durtn,'D') As DUR From DFU_VW \
Where STRT_DT > (select MAX(STRT_DT) from DFU_VW WHERE CASES_ORIGNL<>0 And dmnd_flg='Weekly') And dmnd_flg='Weekly'")

automatedweeklydf = spark.sql("Select PROD_CD AS DMDUNIT, CUST_GRP AS DMDGROUP, LOC, 'SHIPMENTS_LEW' AS MODEL, date_format(STRT_DT,'MM/dd/yyyy') AS STARTDATE, 'IBP' AS FCST_ID, '3' AS TYPE, fnl_aggrd_val as QTY, CONCAT(tm_durtn,'D') As DUR From DFU_VW \
Where STRT_DT > (select MAX(STRT_DT) from DFU_VW WHERE CASES_ORIGNL<>0 And dmnd_flg='Weekly') And dmnd_flg='Weekly'")

# COMMAND ----------

manualweeklydf.registerTempTable("manualweeklydf")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from manualweeklydf

# COMMAND ----------

spark.sql("select min(startdate), max(startdate) from manualweeklydf").display()
spark.sql("select count(1) from manualweeklydf").display()

# COMMAND ----------

monthlydf = spark.sql("Select PROD_CD AS DMDUNIT, CUST_GRP AS  DMDGROUP, LOC, 'SHIPMENTS_LEW' AS MODEL, STRT_DT AS STARTDATE, 'IBP' AS FCST_ID, '3' AS TYPE, fnl_aggrd_val as QTY, tm_durtn As DUR From DFU_VW \
Where STRT_DT > (select MAX(STRT_DT) from DFU_VW WHERE CASES_ORIGNL<>0 And dmnd_flg='Monthly') AND dmnd_flg='Monthly'")

# COMMAND ----------

monthlydf.registerTempTable("monthlydf")

# COMMAND ----------

spark.sql("select min(startdate), max(startdate) from monthlydf").display()
spark.sql("select count(1) from monthlydf").display()

# COMMAND ----------

def file_rename(tgtPath, file_format,tgtFname):
  [dbutils.fs.rm(f.path,True) for  f in [file for file in dbutils.fs.ls(tgtPath) if file.name.startswith("_")]]
  partFile = "".join([file.name for file in dbutils.fs.ls(tgtPath) if file.name.startswith("part-")])
  dbutils.fs.mv(tgtPath+"/"+partFile,tgtPath+"/"+tgtFname+"."+file_format)

# COMMAND ----------

manualweeklydf.display()

# COMMAND ----------

#tgtPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/processing/JDA-Integration/DemandBrainWeeklyManualForecast"
manualweeklydf.coalesce(1).write.mode("overwrite").option("header","true").csv(wk_tgtPath)
file_rename(wk_tgtPath,"csv",wk_fname)

# COMMAND ----------

#tgtPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/processing/jda-integration/demandbrain-monthly-forecast"
monthlydf.coalesce(1).write.mode("overwrite").option("header","true").csv(m_tgtPath)
file_rename(m_tgtPath,"csv",m_fname)
#demandbrain_weekly_manual_forecast;demandbrain_monthly_manual_forecast

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(manualweeklydf.count()+monthlydf.count()))