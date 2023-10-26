# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Inventory Projection dataset
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
edw_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp/inventory-projection/datepart=2021-10-13"
edw_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/inventory-projection"


# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edw_source_path)
bronze_df = spark.read.format("delta").load(edw_bronze_path)

# COMMAND ----------

bronze_df.printSchema()

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDW
print("Source Layer Count is "+str(source_df.count()))
print("Below Bronze Layer Count is ")
display(bronze_df.groupBy("process_date").count().filter("process_date = '2021-10-13' "))

# COMMAND ----------

#Source and Bronze layer column validation for EDW
src_col =  source_df.columns
brnz_col = bronze_df.columns
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for EDW
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.filter("process_date = '2021-10-13' ").select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.filter("process_date = '2021-10-13' ").select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDW Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT').distinct().count())
#EDW Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT').distinct().count())

# COMMAND ----------

#EDW Source Layer PK Null check
source_df = source_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT')
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDW Source Layer PK Duplicate check
source_df.groupby('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT').count().where('count > 1').sort('count', ascending=False).show()

#EDW Bronze Layer PK Duplicate check
bronze_df.groupby('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#Reading the data from the bronze path of Distribution master table
dpndntdatapath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/distribution-master"
distributionMaster_deltaTable = DeltaTable.forPath(spark, dpndntdatapath)
distributionMaster_deltaTable_version = distributionMaster_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(distributionMaster_deltaTable)
display(distributionMaster_deltaTable.history())

#Reading the Inventory Projection source data from bronze layer
distributionMaster_df = spark.read.format("delta").option("versionAsOf", distributionMaster_deltaTable_version).load(dpndntdatapath)

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
edw_silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/inventory-projection"
silver_df = spark.read.format("delta").load(edw_silver_path)

print("Bronze Layer Count is "+str(bronze_df.filter(" process_date = '2021-10-16' ").count()))
print("Silver Layer Count is "+str(silver_df.filter(" process_date = '2021-10-16' ").count()))

# COMMAND ----------

bronze_join_df = silver_df.join(distributionMaster_df, silver_df.LOC ==distributionMaster_df.PLANG_LOC_GRP_VAL, "left")
print("Bronze Count : " +str(bronze_join_df.count()))
print("Silver Count : " +str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ['PROD_CD','SCNR','DW_MTRL_ID','LOC','DW_LOC_ID','GNRTN_DT','STRT_DT','WEEK_ID','MNTH_ID','PRJCTD_INVEN_QTY','PRJCTD_OH_SHRT_DMND_QTY','MU']

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#EDW Source Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT').distinct().count())

print(silver_df.count())
print(silver_df.select('PROD_CD','SCNR','LOC','GNRTN_DT','STRT_DT','WEEK_ID','MNTH_ID','MU').distinct().count())

# COMMAND ----------

#EDW Silver Layer PK Null check
bronze_df = bronze_df.select('MTRL_ID','LOC_ID','SPLY_PLANG_PRJCTN_GNRTN_DT','SPLY_PLANG_PRJCTN_STRT_DT')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)

silver_df = silver_df.select('PROD_CD','SCNR','LOC','GNRTN_DT','STRT_DT','WEEK_ID','MNTH_ID','MU')
silver_df_agg = silver_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in silver_df.columns])
display(silver_df_agg)

# COMMAND ----------

#EDW Source Layer PK Duplicate check
silver_df.groupby('PROD_CD','SCNR','LOC','GNRTN_DT','STRT_DT','WEEK_ID','MNTH_ID','MU').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

# bronze_df.select('SPLY_PLANG_PRJCTN_STRT_DT').distinct().display(200)
display(silver_df)

# COMMAND ----------

