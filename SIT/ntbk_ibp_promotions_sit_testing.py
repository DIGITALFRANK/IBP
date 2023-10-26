# Databricks notebook source
"""
This notebook is for performing the SIT test cases of Promotions dataset
"""

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= 'cdodevadls2' 

# COMMAND ----------

# MAGIC %md
# MAGIC ##Promotions SFA

# COMMAND ----------

#Defining the source and bronze path for Promotions SFA
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/excel/ibp/promotions-sfa/datepart=2021-11-02/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp/promotions-sfa"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("csv").option("header",True).option("delimiter",";").load(source_path) \
                                    .withColumnRenamed("Promo Status","Promo_Status") \
                                    .withColumnRenamed("Promo Buy-in Finish","Promo_Buy-in_Finish") \
                                    .withColumnRenamed("Promo Sub Category","Promo_Sub_Category") \
                                    .withColumnRenamed("Promo Buy-in Start","Promo_Buy-in_Start") \
                                    .withColumnRenamed("Promo Display Type","Promo_Display_Type") \
                                    .withColumnRenamed("Gondola End Number","Gondola_End_Number") \
                                    .withColumnRenamed("Promo Start","Promo_Start") \
                                    .withColumnRenamed("Team Notes","Team_Notes") \
                                    .withColumnRenamed("Promo ProductPackName","Promo_ProductPackName") \
                                    .withColumnRenamed("Promo Product Pack ID","Promo_Product_Pack_ID") \
                                    .withColumnRenamed("Promo Updated By","Promo_Updated_By") \
                                    .withColumnRenamed("Promo Updated On","Promo_Updated_On") \
                                    .withColumnRenamed("Promo Description","Promo_Description") \
                                    .withColumnRenamed("Promo Category","Promo_Category") \
                                    .withColumnRenamed("Tipo Establecimiento","Tipo_Establecimiento") \
                                    .withColumnRenamed("Promo Finish","Promo_Finish") \
                                    .withColumnRenamed("Promo Cadena","Promo_Cadena") \
                                    .withColumnRenamed("Dia semana","Dia_semana") \
                                    .withColumnRenamed("Promotion Location","Promotion_Location") 
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Promotions SFA
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

sfa_bronze_count = bronze_df.count()

# COMMAND ----------

#Source and Bronze layer column validation for Promotions SFA
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

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Promotions SFA
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

# #Source Layer PK Null check
# print("Source Layer Null Values in Initiative Type Column: ",source_df.where(col("Initiative_Type").isNull()).count())
# print("Source Layer Null Values in Initiative Column: ",source_df.where(col("Initiative").isNull()).count())
# print("Source Layer Null Values in Channel Column: ",source_df.where(col("Channel").isNull()).count())
# print("Source Layer Null Values in Customer Column: ",source_df.where(col("Customer").isNull()).count())
# print("Source Layer Null Values in Brand Column: ",source_df.where(col("Brand").isNull()).count())
# print("Source Layer Null Values in Demand Group Column: ",source_df.where(col("Demand_Group").isNull()).count())
# print("Source Layer Null Values in Start Date Column: ",source_df.where(col("Start_Date").isNull()).count())
# print("Source Layer Null Values in End Date Column: ",source_df.where(col("End_Date").isNull()).count())
# # print("Source Layer Null Values in MU Column: ",source_df.where(col("MU").isNull()).count())
# #Bronze Layer PK Null check
# print("Bronze Layer Null Values in Initiative_Type Column: ",bronze_df.where(col("Initiative_Type").isNull()).count())
# print("Bronze Layer Null Values in Initiative Column: ",bronze_df.where(col("Initiative").isNull()).count())
# print("Bronze Layer Null Values in Channel Column: ",bronze_df.where(col("Channel").isNull()).count())
# print("Bronze Layer Null Values in Customer Column: ",bronze_df.where(col("Customer").isNull()).count())
# print("Bronze Layer Null Values in Brand Column: ",bronze_df.where(col("Brand").isNull()).count())
# print("Bronze Layer Null Values in Demand_Group Column: ",bronze_df.where(col("Demand_Group").isNull()).count())
# print("Bronze Layer Null Values in Start_Date Column: ",bronze_df.where(col("Start_Date").isNull()).count())
# print("Bronze Layer Null Values in End_Date Column: ",bronze_df.where(col("End_Date").isNull()).count())
# # print("Bronze Layer Null Values in MU Column: ",bronze_df.where(col("MU").isNull()).count())

# COMMAND ----------

# #Source Layer PK Duplicate check
# source_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()

# COMMAND ----------

# #Bronze Layer PK Duplicate check
# bronze_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Promotion Portugal 

# COMMAND ----------

#Defining the source and bronze path for Promotions Portugal
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/excel/ibp/portugal-promotions/datepart=2021-11-02/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp/portugal-promotions"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("csv").option("header",True).option("delimiter",";").load(source_path) \
                                    .withColumnRenamed("DATA DE CONFIRMAÇÃO","DATA_DE_CONFIRMAO") \
                                    .withColumnRenamed("DESCRIÇÃO","DESCRIO") \
                                    .withColumnRenamed("CODIGO LOGISTICO","CODIGO_LOGISTICO") \
                                    .withColumnRenamed("DURAÇÃO","DURAO") \
                                    .withColumnRenamed("BOLSAS/CAIXA","BOLSASCAIXA") \
                                    .withColumnRenamed("COMENTÁRIOS","COMENTRIOS") \
                                    .withColumnRenamed("DATA FIM","DATA_FIM") \
                                    .withColumnRenamed("SEMANA SERVIÇO","SEMANA_SERVIO") \
                                    .withColumnRenamed("PEDIDO S0","PEDIDO_S0") \
                                    .withColumnRenamed("DATA INICIO","DATA_INICIO") \
                                    .withColumnRenamed("ESTIMATIVA FINAL (CAIXAS)","ESTIMATIVA_FINAL_CAIXAS") \
                                    .withColumnRenamed("PEDIDO S+1","PEDIDO_S1") \
                                    .withColumnRenamed("ACÇÃO","ACO") \
                                    .withColumnRenamed("PEDIDO S-1","PEDIDO_S-1") \
                                    .withColumnRenamed("SEMANA IN-MARKET","SEMANA_IN-MARKET")
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Promotions Portugal
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))
PT_bronze_count = bronze_df.count()

# COMMAND ----------

#Source and Bronze layer column validation for Promotions Portugal
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))	
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

print(len(source_df.columns))
print(len(bronze_df.columns))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Promotions Portugal
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

# #Source Layer Primary Key Uniqueness check
# print(source_df.count())
# print(source_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())
# #Bronze Layer Primary Key Uniqueness check
# print(bronze_df.count())
# print(bronze_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())

# COMMAND ----------

# #Source Layer PK Null check
# print("Source Layer Null Values in Initiative Type Column: ",source_df.where(col("Initiative_Type").isNull()).count())
# print("Source Layer Null Values in Initiative Column: ",source_df.where(col("Initiative").isNull()).count())
# print("Source Layer Null Values in Channel Column: ",source_df.where(col("Channel").isNull()).count())
# print("Source Layer Null Values in Customer Column: ",source_df.where(col("Customer").isNull()).count())
# print("Source Layer Null Values in Brand Column: ",source_df.where(col("Brand").isNull()).count())
# print("Source Layer Null Values in Demand Group Column: ",source_df.where(col("Demand_Group").isNull()).count())
# print("Source Layer Null Values in Start Date Column: ",source_df.where(col("Start_Date").isNull()).count())
# print("Source Layer Null Values in End Date Column: ",source_df.where(col("End_Date").isNull()).count())
# # print("Source Layer Null Values in MU Column: ",source_df.where(col("MU").isNull()).count())
# #Bronze Layer PK Null check
# print("Bronze Layer Null Values in Initiative_Type Column: ",bronze_df.where(col("Initiative_Type").isNull()).count())
# print("Bronze Layer Null Values in Initiative Column: ",bronze_df.where(col("Initiative").isNull()).count())
# print("Bronze Layer Null Values in Channel Column: ",bronze_df.where(col("Channel").isNull()).count())
# print("Bronze Layer Null Values in Customer Column: ",bronze_df.where(col("Customer").isNull()).count())
# print("Bronze Layer Null Values in Brand Column: ",bronze_df.where(col("Brand").isNull()).count())
# print("Bronze Layer Null Values in Demand_Group Column: ",bronze_df.where(col("Demand_Group").isNull()).count())
# print("Bronze Layer Null Values in Start_Date Column: ",bronze_df.where(col("Start_Date").isNull()).count())
# print("Bronze Layer Null Values in End_Date Column: ",bronze_df.where(col("End_Date").isNull()).count())
# # print("Bronze Layer Null Values in MU Column: ",bronze_df.where(col("MU").isNull()).count())

# COMMAND ----------

# #Source Layer PK Duplicate check
# source_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()

# COMMAND ----------

# #Bronze Layer PK Duplicate check
# bronze_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Promotion SFA KAM

# COMMAND ----------

#Defining the source and bronze path for Promotions ALDI
source_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/excel/ibp/promotions-kam-sfa/datepart=2021-11-02/"
bronze_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp/promotions-kam-sfa"

# COMMAND ----------

#Creating the dataframe for each layer
source_df = spark.read.format("csv").option("header",True).option("delimiter",";").load(source_path) \
                                    .withColumnRenamed("Prod_Case (Item)","Prod_Case_Item") \
                                    .withColumnRenamed("Promo_Mechanic_Scope (%PoS)","Promo_Mechanic_Scope_PoS") 
bronze_df = spark.read.format("delta").load(bronze_path)

# COMMAND ----------

#Source and Bronze Layer Count Validation for Promotions ALDI
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))
kam_bronze_count = bronze_df.count()

# COMMAND ----------

#Source and Bronze layer column validation for Promotions ALDI
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))	
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

print(len(source_df.columns))
print(len(bronze_df.columns))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for Promotions ALDI
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))

# COMMAND ----------

# #Source Layer Primary Key Uniqueness check
# print(source_df.count())
# print(source_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())
# #Bronze Layer Primary Key Uniqueness check
# print(bronze_df.count())
# print(bronze_df.select("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date").distinct().count())


# COMMAND ----------

# #Source Layer PK Null check
# print("Source Layer Null Values in Initiative Type Column: ",source_df.where(col("Initiative_Type").isNull()).count())
# print("Source Layer Null Values in Initiative Column: ",source_df.where(col("Initiative").isNull()).count())
# print("Source Layer Null Values in Channel Column: ",source_df.where(col("Channel").isNull()).count())
# print("Source Layer Null Values in Customer Column: ",source_df.where(col("Customer").isNull()).count())
# print("Source Layer Null Values in Brand Column: ",source_df.where(col("Brand").isNull()).count())
# print("Source Layer Null Values in Demand Group Column: ",source_df.where(col("Demand_Group").isNull()).count())
# print("Source Layer Null Values in Start Date Column: ",source_df.where(col("Start_Date").isNull()).count())
# print("Source Layer Null Values in End Date Column: ",source_df.where(col("End_Date").isNull()).count())
# # print("Source Layer Null Values in MU Column: ",source_df.where(col("MU").isNull()).count())
# #Bronze Layer PK Null check
# print("Bronze Layer Null Values in Initiative_Type Column: ",bronze_df.where(col("Initiative_Type").isNull()).count())
# print("Bronze Layer Null Values in Initiative Column: ",bronze_df.where(col("Initiative").isNull()).count())
# print("Bronze Layer Null Values in Channel Column: ",bronze_df.where(col("Channel").isNull()).count())
# print("Bronze Layer Null Values in Customer Column: ",bronze_df.where(col("Customer").isNull()).count())
# print("Bronze Layer Null Values in Brand Column: ",bronze_df.where(col("Brand").isNull()).count())
# print("Bronze Layer Null Values in Demand_Group Column: ",bronze_df.where(col("Demand_Group").isNull()).count())
# print("Bronze Layer Null Values in Start_Date Column: ",bronze_df.where(col("Start_Date").isNull()).count())
# print("Bronze Layer Null Values in End_Date Column: ",bronze_df.where(col("End_Date").isNull()).count())
# # print("Bronze Layer Null Values in MU Column: ",bronze_df.where(col("MU").isNull()).count())



# COMMAND ----------

# #Source Layer PK Duplicate check
# source_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()
	

# COMMAND ----------

# #Bronze Layer PK Duplicate check
# bronze_df \
#     .groupby("Initiative_Type","Initiative","Channel","Customer","Brand","Demand_Group","Start_Date","End_Date") \
#     .count() \
#     .where('count > 1') \
#     .sort('count', ascending=False) \
#     .show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/promotions"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

print("Bronze Layer Count is "+str(sfa_bronze_count+PT_bronze_count+kam_bronze_count))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_TAC","PRMO_DESC","PRMO_DISP","STTS","BU","SRC","CONFIRMED_DT","EST_CS","ACT_CS_W-1","ACT_CS_W0","ACT_CS_W+1"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

#Silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select("CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_DESC","BU","CONFIRMED_DT","STTS").distinct().count())
silver_df.display()

# COMMAND ----------

#Silver Layer PK Null check
print("Silver Layer Null Values in CUST_GRP Column: ",silver_df.where(col("CUST_GRP").isNull()).count())
print("Silver Layer Null Values in PROD_CD Column: ",silver_df.where(col("PROD_CD").isNull()).count())
print("Silver Layer Null Values in STRT_DT Column: ",silver_df.where(col("STRT_DT").isNull()).count())
print("Silver Layer Null Values in END_DT Column: ",silver_df.where(col("END_DT").isNull()).count())
print("Silver Layer Null Values in PRMO_DESC Column: ",silver_df.where(col("PRMO_DESC").isNull()).count())
print("Silver Layer Null Values in BU Column: ",silver_df.where(col("BU").isNull()).count())
print("Silver Layer Null Values in CONFIRMED_DT Column: ",silver_df.where(col("CONFIRMED_DT").isNull()).count())
print("Silver Layer Null Values in STTS Column: ",silver_df.where(col("STTS").isNull()).count())

# COMMAND ----------

#Silver Layer PK Duplicate check
silver_df \
    .groupby("CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_DESC","BU","CONFIRMED_DT","STTS") \
    .count() \
    .where('count > 1') \
    .sort('count', ascending=False) \
    .show()

# COMMAND ----------

silver_df.printSchema()