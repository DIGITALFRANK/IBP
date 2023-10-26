# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Baseline Plan dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred

# COMMAND ----------

#Defining the source and bronze path for EDW source of Product Master
edw_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp/baseline-plan/datepart=2021-10-05"
edw_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/baseline-plan"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edw_source_path)
bronze_df = spark.read.format("delta").load(edw_bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDW
print("Source Layer Count is "+str(source_df.count()))
print("Below Bronze Layer Count is ")
display(bronze_df.groupBy("process_date").count().filter("process_date = '2021-10-05' "))

# COMMAND ----------

#Source and Bronze layer column validation for EDW
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))

# COMMAND ----------

print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

print(len(source_df.columns))
print(len(bronze_df.columns))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for EDW
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.filter("process_date = '2021-10-05' ").select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.filter("process_date = '2021-10-05' ").select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDW Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check
source_df = source_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM')
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDW Source Layer PK Duplicate check
source_df.groupby('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDW Bronze Layer PK Duplicate check
bronze_df.groupby('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/baseline-plan"
silver_df = spark.read.format("delta").load(edw_silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.filter(" process_date = '2021-10-05' ").count()))
print("Silver Layer Count is "+str(silver_df.filter(" process_date = '2021-10-05' ").count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['MTRL_ID','DW_MTRL_ID','ASST_ID','DW_ASST_ID','ASST_OWNR_LOC_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','ASST_PRODTN_MTHD_STEP_LOAD_VAL','ASST_STRG_LOC_ID','ASST_STRG_DW_LOC_ID','CUST_ORDR_LOAD_VAL','FCST_ORDR_LOAD_VAL','SAFSTK_ORDR_LOAD_VAL','SCP_QTY','SCP_ORDR_NUM','DMND_MET_DT','PRPLN_ORDR_NUM','MTRL_CTGY_NM','MU_CDV','ASST_UTLZTN_GNRTN_DTM']

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Source Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').distinct().count())

print(silver_df.count())
print(silver_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
bronze_df = bronze_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)

silver_df = silver_df.select('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM')
silver_df_agg = silver_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in silver_df.columns])
display(silver_df_agg)

# COMMAND ----------

#EDW Source Layer PK Duplicate check
silver_df.groupby('DW_MTRL_ID','DW_ASST_ID','ASST_OWNR_DW_LOC_ID','SCP_LOAD_TYP_CDV','PRODTN_MTHD_VAL','PRODTN_MTHD_STEP_NUM','ASST_UTLZTN_STRT_DTM','SCP_ORDR_NUM','DMND_MET_DT','MU_CDV','ASST_UTLZTN_GNRTN_DTM').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

