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
edl_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/udt-product-substitution/datepart=2021-10-22/"
edl_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp/udt-product-substitution/"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edl_source_path)
bronze_df = spark.read.format("delta").load(edl_bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDL
print("Source Layer Count is "+str(source_df.count()))
print("Below Bronze Layer Count is ")
display(bronze_df.groupBy("process_date").count().filter("process_date = '2021-10-22'"))

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
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.filter("process_date = '2021-10-22' ").select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.filter("process_date = '2021-10-22' ").select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDL Source Layer Primary Key Uniqueness check
print("source count")
print(source_df.count())
print("source grouby count on PK fields")
print(source_df.select("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC").distinct().count())
#EDL Bronze Layer Primary Key Uniqueness check
print("bronze count")
print(bronze_df.filter("process_date = '2021-10-22'").count())
print("bronze group by count on PK fields")
print(bronze_df.filter("process_date = '2021-10-22'").select("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC").distinct().count())

# COMMAND ----------

#EDL Source Layer PK Null check
source_df = source_df.select("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC")
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC")
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDL Source Layer PK Duplicate check
source_df.groupby("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC").count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDl Bronze Layer PK Duplicate check
bronze_df.groupby("OLDDMDUNIT","OLDDMDGROUP","OLDLOC","NEWDMDUNIT","NEWDMDGROUP","NEWLOC","EFF","DISC").count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/udt-product-substitution/"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.filter(" process_date = '2021-10-22' ").count()))
print("Silver Layer Count is "+str(silver_df.filter(" process_date = '2021-10-22' ").count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["PREV_PROD_CD","PREV_CUST_GRP","PREV_LOC","PREV_MODEL","NEW_PROD_CD","NEW_CUST_GRP","NEW_LOC","NEW_MODEL","EFF","DISC","HIST_FACTOR","FCST_FACTOR","COPY_HIST","COPY_FCST","COPY_PRM","SEQ","PROCESS","GPID","MU"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))


# COMMAND ----------

#EDL silver Layer Primary Key Uniqueness check
print(silver_df.filter(" process_date = '2021-10-22' ").count())
print(silver_df.filter(" process_date = '2021-10-22' ").select("PREV_PROD_CD","PREV_CUST_GRP","PREV_LOC","PREV_MODEL","NEW_PROD_CD","NEW_CUST_GRP","NEW_LOC","NEW_MODEL","EFF","DISC").distinct().count())

#EDl silver Layer PK Duplicate check
silver_df.groupby("PREV_PROD_CD","PREV_CUST_GRP","PREV_LOC","PREV_MODEL","NEW_PROD_CD","NEW_CUST_GRP","NEW_LOC","NEW_MODEL","EFF","DISC").count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDL Silver Layer PK Null check
silver_df = silver_df.filter(" process_date = '2021-10-22' ").select("PREV_PROD_CD","PREV_CUST_GRP","PREV_LOC","PREV_MODEL","NEW_PROD_CD","NEW_CUST_GRP","NEW_LOC","NEW_MODEL","EFF","DISC")
silver_df_agg = silver_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in silver_df.columns])
display(silver_df_agg)

# COMMAND ----------

