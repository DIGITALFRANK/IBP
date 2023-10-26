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
edw_source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/edw/ibp/sku-master/datepart=2021-09-29"
edw_bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp/sku-master/"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("parquet").load(edw_source_path)
bronze_df = spark.read.format("delta").load(edw_bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDL
print("Source Layer Count is "+str(source_df.count()))
print("Below Bronze Layer Count is ")
display(bronze_df.groupBy("process_date").count().filter("process_date = '2021-09-29'"))

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
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.filter("process_date = '2021-09-29' ").select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.filter("process_date = '2021-09-29' ").select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#EDL Source Layer Primary Key Uniqueness check
print("source count")
print(source_df.count())
print("source grouby count on PK fields")
print(source_df.select('DW_LOC_ID','DW_MTRL_ID').distinct().count())
#EDL Bronze Layer Primary Key Uniqueness check
print("bronze count")
print(bronze_df.filter("process_date = '2021-09-29'").count())
print("bronze group by count on PK fields")
print(bronze_df.filter("process_date = '2021-09-29'").select('DW_LOC_ID','DW_MTRL_ID').distinct().count())

# COMMAND ----------

#EDL Source Layer PK Null check
source_df = source_df.select('DW_LOC_ID','DW_MTRL_ID')
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select('DW_LOC_ID','DW_MTRL_ID')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDL Source Layer PK Duplicate check
source_df.groupby('DW_LOC_ID','DW_MTRL_ID').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDl Bronze Layer PK Duplicate check
bronze_df.groupby('DW_LOC_ID','DW_MTRL_ID').count().where('count > 1').sort('count', ascending=False).show()