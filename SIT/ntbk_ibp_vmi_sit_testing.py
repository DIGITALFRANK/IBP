# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred

# COMMAND ----------

# DBTITLE 1,Reading the source and Bronze path tables Update date part to date of pipeline execution
#Defining the source and bronze path for EDL source of Pricing
edl_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp/vmi/datepart=2021-10-08"
edl_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/vmi"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edl_source_path)
bronze_df = spark.read.format("delta").load(edl_bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDL
print("Source Layer Count is "+str(source_df.count()))
print("Below Bronze Layer Count is ")
display(bronze_df.groupBy("process_date").count().filter("process_date = '2021-10-08'"))

# COMMAND ----------

#Source and Bronze layer column validation for EDL
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
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.filter("process_date = '2021-10-08' ").select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.filter("process_date = '2021-10-08' ").select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDL Source Layer Primary Key Uniqueness check
print("source count")
print(source_df.count())
print("source grouby count on PK fields")
print(source_df.select('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL').distinct().count())
#EDL Bronze Layer Primary Key Uniqueness check
print("bronze count")
print(bronze_df.filter("process_date = '2021-10-08'").count())
print("bronze group by count on PK fields")
print(bronze_df.filter("process_date = '2021-10-08'").select('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL').distinct().count())

# COMMAND ----------

#EDL Source Layer PK Null check
source_df = source_df.select('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL')
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDL Source Layer PK Duplicate check
source_df.groupby('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDl Bronze Layer PK Duplicate check
bronze_df.groupby('MTRL_ID','LOC_ID','INVEN_BAL_DT','SPLY_PLANG_PRJCT_ID','MTRL_EXPR_DT','INVEN_GNRTN_DT','STRG_LOC_VAL').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/vmi_test"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.filter(" process_date = '2021-10-08' ").count()))
print("Silver Layer Count is "+str(silver_df.filter(" process_date = '2021-11-03' ").count()))
#print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["PROD_CD","CUST_GRP","LOC","AVAILDT","EXPDT","PRJCT","VMI_FLG","QUARANTINE","UOM","QTY_CS","SRC_SYS_CNTRY","MU","INVEN_BAL_DT"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))


# COMMAND ----------

#EDL silver Layer Primary Key Uniqueness check
print(silver_df.count())
#.filter(" process_date = '2021-10-08' ")
print(silver_df.select('PROD_CD','CUST_GRP','LOC','AVAILDT','EXPDT','PRJCT','MU','INVEN_BAL_DT').distinct().count())
#.filter(" process_date = '2021-10-12' ")
#EDl silver Layer PK Duplicate check
silver_df.groupby('PROD_CD','CUST_GRP','LOC','AVAILDT','EXPDT','PRJCT','MU','INVEN_BAL_DT').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

silver_df.where("PROD_CD  in ('11077301_05','11063901_05','11058501_05','11058701_05','11072901_05','11303501_05','11089001_05','595601_05','11303201_05') and CUST_GRP='NA' and LOC='CM_ES_BOR_01' and AVAILDT='2021-10-01 00:00:00' and EXPDT='2025-01-02 00:00:00' and PRJCT='CARGA DIOGENES' and MU='ES'").display()

# COMMAND ----------

#EDL Silver Layer PK Null check
silver_df = silver_df.filter(" process_date = '2021-10-08' ").select('PROD_CD','CUST_GRP','LOC','AVAILDT','EXPDT','PRJCT','MU','INVEN_BAL_DT')
silver_df_agg = silver_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in silver_df.columns])
display(silver_df_agg)