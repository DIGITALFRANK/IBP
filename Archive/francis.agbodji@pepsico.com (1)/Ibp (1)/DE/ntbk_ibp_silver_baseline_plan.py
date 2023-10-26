# Databricks notebook source
#imports
from pyspark.sql.functions import *

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
source_container = dbutils.widgets.get("sourceContainer")
srcPath = dbutils.widgets.get("sourcePath")
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

Path = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+srcPath

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

BaselineDF = spark.read.format("delta").load(Path)

BaselineDF.printSchema()
BaselineDF.display()

# COMMAND ----------

BaselineDF_Formated = BaselineDF.select("MTRL_ID","DW_MTRL_ID","ASST_ID","DW_ASST_ID","ASST_OWNR_LOC_ID","ASST_OWNR_DW_LOC_ID",
"SCP_LOAD_TYP_CDV","PRODTN_MTHD_VAL","PRODTN_MTHD_STEP_NUM","ASST_UTLZTN_STRT_DTM","ASST_PRODTN_MTHD_STEP_LOAD_VAL",
"ASST_STRG_LOC_ID","ASST_STRG_DW_LOC_ID","CUST_ORDR_LOAD_VAL","FCST_ORDR_LOAD_VAL","SAFSTK_ORDR_LOAD_VAL","SCP_QTY","SCP_ORDR_NUM",
"DMND_MET_DT","PRPLN_ORDR_NUM","MTRL_CTGY_NM","MU_CDV","ASST_UTLZTN_GNRTN_DTM")

# COMMAND ----------

BaselineDF_Formated.write.format("delta").option("overwriteschema",True).mode(loadType).save(tgtPath)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(BaselineDF_Formated.count()))