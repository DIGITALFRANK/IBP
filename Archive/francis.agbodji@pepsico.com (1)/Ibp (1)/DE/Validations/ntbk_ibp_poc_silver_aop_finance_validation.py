# Databricks notebook source
"""
This notebook is used to perform validations on the AOP Finance data present in Silver layer
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#Capturning the count from Silver layer
path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/finance"
silver_deltaTable = DeltaTable.forPath(spark, path)
silver_latest_version = silver_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(silver_latest_version)
display(silver_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format("delta").option("versionAsOf", silver_latest_version).load(path)
print(silver_df.count())

# COMMAND ----------

##PK Null check - 
null_df = silver_df.filter("ENTTY_LBL_CDV is Null or ENTTY_TYP_CDV is Null or FINC_ACCT_ID is Null or GTM_LOCL_LVL_0_CDV is Null or MNTH_NUM is Null or CUST_GRP is Null or PROD_CD is Null or YR_NUM is Null or SLS_CHNL_LVL_1_CDV is Null or SLS_CHNL_LVL_2_CDV is Null or SLS_CHNL_LVL_3_CDV is Null or SLS_CHNL_LVL_4_CDV is Null or SLS_CHNL_LVL_5_CDV is Null or DW_CMPT_PROD_ID is Null or LOCL_ORG_L0_CDV is Null or DW_PLANG_MTRL_UNIT_ID is Null")
print("There are total of "+str(null_df.count())+" rows with NULL values on PK columns")

# COMMAND ----------

#PK Duplicate check
dup_df = silver_df.groupBy("ENTTY_LBL_CDV","ENTTY_TYP_CDV","FINC_ACCT_ID","GTM_LOCL_LVL_0_CDV","MNTH_NUM","CUST_GRP","PROD_CD","YR_NUM","SLS_CHNL_LVL_1_CDV","SLS_CHNL_LVL_2_CDV","SLS_CHNL_LVL_3_CDV","SLS_CHNL_LVL_4_CDV","SLS_CHNL_LVL_5_CDV","DW_CMPT_PROD_ID","LOCL_ORG_L0_CDV","DW_PLANG_MTRL_UNIT_ID","SHOPR_CHNL_CDV").count().filter("count > 1")
display(dup_df)

# COMMAND ----------

