# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Risk & opportunities dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= 'cdodevadls2' 

# COMMAND ----------

#Defining the source and bronze path for Risk & Opportunities 
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/excel/ibp/risk-opportunities/datepart=2021-10-21/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp/risk-opportunities"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = (spark.read.format("csv").option("header",True).option("delimiter","~").load(source_path) 
                                    .withColumnRenamed("Initiative Type","Initiative_Type") 
                                    .withColumnRenamed("Demand Group*","Demand_Group")
                                    .withColumnRenamed("Start Date","Start_Date")
                                    .withColumnRenamed("End Date","End_Date")
                                    .withColumnRenamed("Date Entered","Date_Entered")
                                    .withColumnRenamed("Initiative Status","Initiative_Status")
                                    .withColumnRenamed("Comment on probability","Comment_on_probability")
                                    .withColumnRenamed("Initiative Owner","Initiative_Owner")
                                    .withColumnRenamed("Marginal Contribution","Marginal_Contribution")
                                    .withColumnRenamed("Net Revenue","Net_Revenue")
                                    .withColumnRenamed("Volume (in Tonnes/KL)","Volume_in_TonnesKL")
                                    .filter(col("Initiative Type").isNotNull() & col("Initiative").isNotNull())
            )
bronze_df = spark.read.format("delta").load(bronze_path).filter(col("Initiative_Type").isNotNull() & col("Initiative").isNotNull())

# COMMAND ----------

#Source and Bronze Layer Count Validation for Risk & Opportunities 
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

#Source and Bronze layer column validation for Risk & Opportunities
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

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Risk & Opportunities
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

#Source Layer Primary Key Uniqueness check
print(source_df.count())
print(source_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())
#Bronze Layer Primary Key Uniqueness check
print(bronze_df.count())
print(bronze_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())

# COMMAND ----------

#Source Layer PK Null check
print("Source Layer Null Values in Initiative Type Column: ",source_df.where(col("Initiative_Type").isNull()).count())
print("Source Layer Null Values in Initiative Column: ",source_df.where(col("Initiative").isNull()).count())
print("Source Layer Null Values in Channel Column: ",source_df.where(col("Channel").isNull()).count())
print("Source Layer Null Values in Customer Column: ",source_df.where(col("Customer").isNull()).count())
print("Source Layer Null Values in Brand Column: ",source_df.where(col("Brand").isNull()).count())
print("Source Layer Null Values in Demand Group Column: ",source_df.where(col("Demand_Group").isNull()).count())
print("Source Layer Null Values in Start Date Column: ",source_df.where(col("Start_Date").isNull()).count())
print("Source Layer Null Values in End Date Column: ",source_df.where(col("End_Date").isNull()).count())
# print("Source Layer Null Values in MU Column: ",source_df.where(col("MU").isNull()).count())
#Bronze Layer PK Null check
print("Bronze Layer Null Values in Initiative_Type Column: ",bronze_df.where(col("Initiative_Type").isNull()).count())
print("Bronze Layer Null Values in Initiative Column: ",bronze_df.where(col("Initiative").isNull()).count())
print("Bronze Layer Null Values in Channel Column: ",bronze_df.where(col("Channel").isNull()).count())
print("Bronze Layer Null Values in Customer Column: ",bronze_df.where(col("Customer").isNull()).count())
print("Bronze Layer Null Values in Brand Column: ",bronze_df.where(col("Brand").isNull()).count())
print("Bronze Layer Null Values in Demand_Group Column: ",bronze_df.where(col("Demand_Group").isNull()).count())
print("Bronze Layer Null Values in Start_Date Column: ",bronze_df.where(col("Start_Date").isNull()).count())
print("Bronze Layer Null Values in End_Date Column: ",bronze_df.where(col("End_Date").isNull()).count())
# print("Bronze Layer Null Values in MU Column: ",bronze_df.where(col("MU").isNull()).count())

# COMMAND ----------

#Source Layer PK Duplicate check
source_df \
    .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze Layer PK Duplicate check
bronze_df \
    .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/risk-opportunities"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["INITV_TYPE","INITV","CHNL","CUST","CTGY","BRND","FRMT","CUST_GRP","INITV_OWNR","VOL","NR",
                     "MC","PRBLTY","INCLD","DT_ENTRD","STRT_DT","END_DT","CMNT_ON_PRBLTY","INIT_STTS","MU"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("INITV_TYPE","INITV","CHNL","CUST","BRND","CUST_GRP","STRT_DT","END_DT").distinct().count())
silver_df.display()

# COMMAND ----------

#Silver Layer PK Null check
print("Silver Layer Null Values in INITV_TYPE Column: ",silver_df.where(col("INITV_TYPE").isNull()).count())
print("Silver Layer Null Values in INITV Column: ",silver_df.where(col("INITV").isNull()).count())
print("Silver Layer Null Values in CHNL Column: ",silver_df.where(col("CHNL").isNull()).count())
print("Silver Layer Null Values in CUST Column: ",silver_df.where(col("CUST").isNull()).count())
print("Silver Layer Null Values in BRND Column: ",silver_df.where(col("BRND").isNull()).count())
print("Silver Layer Null Values in CUST_GRP Column: ",silver_df.where(col("CUST_GRP").isNull()).count())
print("Silver Layer Null Values in STRT_DT Column: ",silver_df.where(col("STRT_DT").isNull()).count())
print("Silver Layer Null Values in END_DT Column: ",silver_df.where(col("END_DT").isNull()).count())
# print("Silver Layer Null Values in MU Column: ",silver_df.where(col("MU").isNull()).count())

# COMMAND ----------

#Silver Layer PK Duplicate check
silver_df \
    .groupby("INITV_TYPE","INITV","CHNL","CUST","BRND","CUST_GRP","STRT_DT","END_DT") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

silver_df.printSchema()