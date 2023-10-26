# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Consumer Mobility Index dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= 'cdodevadls2' 

# COMMAND ----------

#Defining the source and bronze path for Consumer Mobility Index 
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/external/ibp/consumer-mobility-index/datepart=2021-10-20/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/external/ibp/consumer-mobility-index"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = (spark.read.format("csv").option("header",True).option("delimiter",",").load(source_path) )
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Consumer Mobility Index 
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

#Source and Bronze layer column validation for Consumer Mobility Index 
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

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Consumer Mobility Index 
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select("country_region_code","country_region","place_id","date","sub_region_1","sub_region_2").distinct().count())
#Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("country_region_code","country_region","place_id","date","sub_region_1","sub_region_2").distinct().count())

# COMMAND ----------

#Source Layer PK Null check
print("Source Layer Null Values in country_region_code Column: ",source_df.where(col("country_region_code").isNull()).count())
print("Source Layer Null Values in date Column: ",source_df.where(col("date").isNull()).count())
print("Source Layer Null Values in sub_region_1 Column: ",source_df.where(col("sub_region_1").isNull()).count())
print("Source Layer Null Values in country_region Column: ",source_df.where(col("country_region").isNull()).count())
print("Source Layer Null Values in place_id Column: ",source_df.where(col("place_id").isNull()).count())
print("Source Layer Null Values in sub_region_2 Column: ",source_df.where(col("sub_region_2").isNull()).count())

#Bronze Layer PK Null check
print("Bronze Layer Null Values in country_region_code Column: ",bronze_df.where(col("country_region_code").isNull()).count())
print("Bronze Layer Null Values in date Column: ",bronze_df.where(col("date").isNull()).count())
print("Bronze Layer Null Values in sub_region_1 Column: ",bronze_df.where(col("sub_region_1").isNull()).count())
print("Bronze Layer Null Values in country_region Column: ",bronze_df.where(col("country_region").isNull()).count())
print("Bronze Layer Null Values in place_id Column: ",bronze_df.where(col("place_id").isNull()).count())
print("Bronze Layer Null Values in sub_region_2 Column: ",bronze_df.where(col("sub_region_2").isNull()).count())

# COMMAND ----------

#Source Layer PK Duplicate check
source_df \
    .groupby("country_region_code","country_region","place_id","date","sub_region_1","country_region_code","sub_region_2") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze Layer PK Duplicate check
bronze_df \
    .groupby("country_region_code","country_region","place_id","date","sub_region_1","country_region_code","sub_region_2") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/consumer-mobility-index"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["MU","DT","CNTRY","RGN","MTR_AREA","RTL_AND_RCRTN_PCNT_CHNG_FRM_BSLNE","GRCRY_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE",
                   "PRKS_PCNT_CHNG_FRM_BSLNE","TRNST_STTNS_PCNT_CHNG_FRM_BSLNE","WRKPLCS_PCNT_CHNG_FRM_BSLNE",
                     "RSDNTL_AND_PHRMCY_PCNT_CHNG_FRM_BSLNE","MU_CHNL","CTGY","LOC","PRVNC"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("MU","DT","RGN","MU_CHNL","CTGY","LOC","PRVNC").distinct().count())
silver_df.display()

# COMMAND ----------

#Silver Layer PK Null check
print("Silver Layer Null Values in MU Column: ",silver_df.where(col("MU").isNull()).count())
print("Silver Layer Null Values in DT Column: ",silver_df.where(col("DT").isNull()).count())
print("Silver Layer Null Values in RGN Column: ",silver_df.where(col("RGN").isNull()).count())
print("Silver Layer Null Values in MU_CHNL Column: ",silver_df.where(col("MU_CHNL").isNull()).count())
print("Silver Layer Null Values in CTGY Column: ",silver_df.where(col("CTGY").isNull()).count())
print("Silver Layer Null Values in LOC Column: ",silver_df.where(col("LOC").isNull()).count())
print("Silver Layer Null Values in PRVNC Column: ",silver_df.where(col("PRVNC").isNull()).count())

# COMMAND ----------

#Silver Layer PK Duplicate check
silver_df \
    .groupby("MU","DT","RGN","MU_CHNL","CTGY","LOC","PRVNC") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

silver_df.printSchema()

# COMMAND ----------

