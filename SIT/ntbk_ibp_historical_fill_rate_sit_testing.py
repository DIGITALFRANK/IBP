# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Historical Fill Rate dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= 'cdodevadls2' 

# COMMAND ----------

#Defining the source and bronze path for Historical fill rate
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/excel/ibp/historical-fillrate-customer-service-levels/datepart=2021-10-12/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp/historical-fillrate-customer-service-levels"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("csv").option("header",True).load(source_path).withColumnRenamed("Time Bucket","Time_Bucket")
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Historical fill rate
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

bronze_df.display()

# COMMAND ----------

#Source and Bronze layer column validation for Historical fill rate
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

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Historical fill rate
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select("KPI_ID","Time_Bucket","Year_ID","Month_ID","Week_ID","Market","Category","Customer_ID").distinct().count())
#Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("KPI_ID","Time_Bucket","Year_ID","Month_ID","Week_ID","Market","Category","Customer_ID").distinct().count())

# COMMAND ----------

#Source Layer PK Null check
print("Source Layer Null Values in KPI_ID Column: ",source_df.where(col("KPI_ID").isNull()).count())
print("Source Layer Null Values in Time Bucket Column: ",source_df.where(col("Time Bucket").isNull()).count())
print("Source Layer Null Values in Year_ID Column: ",source_df.where(col("Year_ID").isNull()).count())
print("Source Layer Null Values in Month_ID Column: ",source_df.where(col("Month_ID").isNull()).count())
print("Source Layer Null Values in Week_ID Column: ",source_df.where(col("Week_ID").isNull()).count())
print("Source Layer Null Values in Market Column: ",source_df.where(col("Market").isNull()).count())
print("Source Layer Null Values in Category Column: ",source_df.where(col("Category").isNull()).count())
print("Source Layer Null Values in Customer_ID Column: ",source_df.where(col("Customer_ID").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in KPI_ID Column: ",bronze_df.where(col("KPI_ID").isNull()).count())
print("Bronze Layer Null Values in Time_Bucket Column: ",bronze_df.where(col("Time_Bucket").isNull()).count())
print("Bronze Layer Null Values in Year_ID Column: ",bronze_df.where(col("Year_ID").isNull()).count())
print("Bronze Layer Null Values in Month_ID Column: ",bronze_df.where(col("Month_ID").isNull()).count())
print("Bronze Layer Null Values in Week_ID Column: ",bronze_df.where(col("Week_ID").isNull()).count())
print("Bronze Layer Null Values in Market Column: ",bronze_df.where(col("Market").isNull()).count())
print("Bronze Layer Null Values in Category Column: ",bronze_df.where(col("Category").isNull()).count())
print("Bronze Layer Null Values in Customer_ID Column: ",bronze_df.where(col("Customer_ID").isNull()).count())

# COMMAND ----------

#Source Layer PK Duplicate check
source_df \
    .groupby("KPI_ID","Time_Bucket","Year_ID","Month_ID","Week_ID","Market","Category","Customer_ID") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze Layer PK Duplicate check
bronze_df \
    .groupby("KPI_ID","Time_Bucket","Year_ID","Month_ID","Week_ID","Market","Category","Customer_ID") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/historical-fillrate-customer-service-levels"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.filter('Market in ("ES","PT") and Metric_Numerator is not null and Metric_Numerator is not null').count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["KPI_ID","TIME_BUCKET","YR_ID","MNTH_ID","WEEK_ID","MU","CTGY","CUST_GRP","SHPPED","ORDERED"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("KPI_ID","TIME_BUCKET","YR_ID","MNTH_ID","WEEK_ID","MU","CTGY","CUST_GRP").distinct().count())

# COMMAND ----------

#Silver Layer PK Null check
print("Silver Layer Null Values in KPI_ID Column: ",silver_df.where(col("KPI_ID").isNull()).count())
print("Silver Layer Null Values in TIME_BUCKET Column: ",silver_df.where(col("TIME_BUCKET").isNull()).count())
print("Silver Layer Null Values in YR_ID Column: ",silver_df.where(col("YR_ID").isNull()).count())
print("Silver Layer Null Values in MNTH_ID Column: ",silver_df.where(col("MNTH_ID").isNull()).count())
print("Silver Layer Null Values in WEEK_ID Column: ",silver_df.where(col("WEEK_ID").isNull()).count())
print("Silver Layer Null Values in MU Column: ",silver_df.where(col("MU").isNull()).count())
print("Silver Layer Null Values in CTGY Column: ",silver_df.where(col("CTGY").isNull()).count())
print("Silver Layer Null Values in CUST_GRP Column: ",silver_df.where(col("CUST_GRP").isNull()).count())

# COMMAND ----------

#EDW Silver Layer PK Duplicate check
silver_df \
    .groupby("KPI_ID","TIME_BUCKET","YR_ID","MNTH_ID","WEEK_ID","MU","CTGY","CUST_GRP") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

silver_df.printSchema()