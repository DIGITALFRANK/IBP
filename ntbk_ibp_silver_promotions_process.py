# Databricks notebook source
#imports
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from delta.tables import *
from pyspark.sql import Row
from pyspark.sql.window import Window
import itertools
from pyspark.sql.types import *
import re

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

#storing the parameters in variables
source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
source_container = dbutils.widgets.get("sourceContainer")
target_container = dbutils.widgets.get("targetContainer")
srcPath = dbutils.widgets.get("sourcePath")
tpath = dbutils.widgets.get("targetPath")
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#Defining user defined functions for spl character validation

def check_spl_char(df,name):
  if df.filter(col(name).contains("�")).count() > 0 :
    return 0
  else:
    return 1

# COMMAND ----------

## Defining the Source path for Portugal and Spain
srcPath_list = srcPath.split(";")

for path in srcPath_list:
  if 'promotions-sfa' in path:
    sfa_bronze = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if 'promotions-kam-sfa' in path:
    sfa_kam_bronze = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if 'portugal-promotions' in path:
    pt_promo_bronze = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path

# COMMAND ----------

#Defining target path for promotion
tgtPath = "abfss://"+target_container+"@"+target_stgAccnt+".dfs.core.windows.net/"+tpath

# COMMAND ----------

#Printing the source and target path details
print("Spain source path: ",sfa_bronze)
print("Spain KAM source path: ",sfa_kam_bronze)
print("Portugal source path: ",pt_promo_bronze)
print("Promotion target path: ",tgtPath)

# COMMAND ----------

# MAGIC 
# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

# Created a User definded function to convert the dataframe column into list
def from_column_to_list(df, colname):
  l = []
  list_col = df.select(colname).collect()
  for elem in list_col:
    l.append(elem[colname])
  
  return l

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy","LEGACY") 

# COMMAND ----------

#Creating a list of Promo Tactics for later verification  
tactics = [ "multibuy_BuyXforY",
            "multibuy_BOGO_withCategory",
            "multibuy_BOGO_CrossCategory",
            "multi_tiered_BuyMoreSaveMore",
            "purchase_with_purchase",
            "tpr_DiscountPrice",
            "tpr_DiscountAmount",
            "tpr_DiscountPerc",
            "loyalty_InStore",
            "loyalty_General",
            "freegoods_StoreOpening",
            "freegoods_Premium",
            "freegoods_Samples",
            "freegoods_Tastings",
            "freegoods_General",
            "display_Temp_4way",
            "display_Lobby",
            "display_Temp_End",
            "display_Temp_Countertop",
            "display_Seasonal",
            "display_Bunker",
            "display_General",
            "ad_flyer_Solo",
            "ad_flyer_FrontPage",
            "ad_flyer_BackPage",
            "ad_flyer_Inside",
            "ad_flyer_Loyalty",
            "ad_flyer_OnlineAd",
            "ad_flyer_General",
            "coupons_InStore",
            "coupons_General"
           ]

# COMMAND ----------

#Reading the delta history from the bronze path for Promotion SFA-KAM table
SFA_kam_deltaTable = DeltaTable.forPath(spark,sfa_kam_bronze)
sfa_kam_latest_version = SFA_kam_deltaTable.history().select(max(col("version"))).collect()[0][0]

print("SFA Latest Version: ", sfa_kam_latest_version)
display(SFA_kam_deltaTable.history())

# COMMAND ----------

#Reading the delta history from the bronze path of Promotion SFA table
SFA_deltaTable = DeltaTable.forPath(spark,sfa_bronze)
sfa_latest_version = SFA_deltaTable.history().select(max(col("version"))).collect()[0][0]

print("SFA Latest Version: ", sfa_latest_version)
display(SFA_deltaTable.history())

# COMMAND ----------

#Reading the delta history from the bronze path of Promotion Portugal Table
PT_DeltaTable = DeltaTable.forPath(spark,pt_promo_bronze)
PT_latest_version = PT_DeltaTable.history().select(max(col("version"))).collect()[0][0]

print("PT Latest Version: ", PT_latest_version)
display(PT_DeltaTable.history())

# COMMAND ----------

# MAGIC %md # Protugal Promotions

# COMMAND ----------

# Reading Portugal file into Dataframe

PT_Promo = (spark.read.format("delta").option("versionAsOf", PT_latest_version)
                                      .load(pt_promo_bronze)
                                      .filter(col("CLIENTE").isNotNull())
                                      .withColumn("STRT_DT", to_timestamp(col("DATA_INICIO")))
                                      .withColumn("END_DT", to_timestamp(col("DATA_FIM")))
                                      .withColumn("CONFIRMED_DT", when(col("DATA_DE_CONFIRMAO").isNull(),to_timestamp(date_sub(col("STRT_DT"),21))) 
                                                  .otherwise(to_timestamp(col("DATA_DE_CONFIRMAO"))))
           )

print("Record Count in Bronze Layer: ",PT_Promo.count())
PT_Promo.printSchema()
PT_Promo.display()

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = PT_Promo.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print("Max Process Date In Bronze Layer: ",max_value)

PT_Promo2 = PT_Promo.filter(col("PROCESS_DATE")==max_value) 

print("Record count in bronze layer after filter: ",PT_Promo2.count())
display(PT_Promo2)

# COMMAND ----------

#Populating the DMDGROUP column based on condition
PT_Promo_Formatted = PT_Promo2.withColumn("CUST_GRP",  when(col("CLIENTE") == "FROIZ", "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "LIDL", "PT_OT_LIDL") \
                                             .when(col("CLIENTE") == "ITM", "PT_OT_ITM") \
                                             .when(col("CLIENTE") == "LECLERC", "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "ESTEVÃO NEVES", "PT_DTS_MADEIRA") \
                                             .when(col("CLIENTE") == "LIDOSOL", "PT_DTS_MADEIRA") \
                                             .when(col("CLIENTE") == "ECI",  "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "AUCHAN", "PT_OT_AUCHAN") \
                                             .when(col("CLIENTE") == "ALDI", "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "BOLAMA", "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "SONAE", "PT_OT_SONAE") \
                                             .when(col("CLIENTE") == "DIA", "PT_OT_DIA") \
                                             .when(col("CLIENTE") == "MAKRO", "PT_OT_OTHERS") \
                                             .when(col("CLIENTE") == "PINGO DOCE", "PT_OT_PINGO_DOCE") \
                                             .otherwise("NULL"))

# COMMAND ----------

PT_Promo_Formatted = ( PT_Promo_Formatted.withColumnRenamed("ESTIMATIVA_FINAL_CAIXAS", "EST_CS")
                                       .withColumnRenamed("PEDIDO_S-1", "ACT_CS_W-1")
                                       .withColumnRenamed("PEDIDO_S0", "ACT_CS_W0")
                                       .withColumnRenamed("PEDIDO_S1", "ACT_CS_W+1")
                                       .withColumnRenamed("DMDUNIT", "PROD_CD")
									   .select("CUST_GRP", 
									   	 "PROD_CD", 
									   	 "STRT_DT", 
									   	 "END_DT", 
									   	 "MECANICA", 
                                         "ACO",
									   	 "STATUS", 
									   	 "CONFIRMED_DT", 
									   	 "EST_CS", 
									   	 "ACT_CS_W-1", 
									   	 "ACT_CS_W0", 
									   	 "ACT_CS_W+1", 
									   	 lit(None).alias("PRMO_DISP"),
									   	 lit("PT").alias("BU"), 
									   	 lit("PT_DP").alias("SRC")
									    ).dropDuplicates() 
                     )

PT_Promo_Formatted.display()

# COMMAND ----------

PT_Promo_Final = (PT_Promo_Formatted.withColumn("PRMO_TAC", 
									 when(PT_Promo_Formatted.MECANICA.rlike("^L[0-9]P[0-9]"), "multibuy_BuyXforY")
									.when(PT_Promo_Formatted.MECANICA.contains("TALÃO"), "loyalty_InStore")
									.when(PT_Promo_Formatted.MECANICA.contains("CARTÃO"), "loyalty_InStore")
									.when(PT_Promo_Formatted.MECANICA.contains("€"), "tpr_DiscountPrice")
									.when(PT_Promo_Formatted.MECANICA.contains("SEM PROMO"), "no_Promo")
									.when(PT_Promo_Formatted.MECANICA.contains("S/PROMO"), "no_Promo")
									.when(PT_Promo_Formatted.MECANICA.contains("PACK"), "freegoods_Premium")
									.when(PT_Promo_Formatted.MECANICA.contains("MENU"), "purchase_with_purchase")
									.when(PT_Promo_Formatted.MECANICA.contains("%"), "tpr_DiscountPerc")
									.when(PT_Promo_Formatted.MECANICA.contains("SP"), "tpr_DiscountPerc")
									.when(PT_Promo_Formatted.MECANICA.rlike("[0-9][\.,][0-9][0-9]|[0-9]+"), "tpr_DiscountPrice")
								)
                  .withColumn("PRMO_DESC", concat(col("ACO"),lit(" | "),col("MECANICA"))) 
                  .withColumnRenamed("STATUS", "STTS")
                 ).select("CUST_GRP", 
                          "PROD_CD", 
                          "STRT_DT", 
                          "END_DT", 
                          "PRMO_DESC", 
                          "PRMO_TAC", 
                          "PRMO_DISP", 
                          "STTS", 
                          "CONFIRMED_DT",  
                          "EST_CS", 
                          "ACT_CS_W-1", 
						  "ACT_CS_W0", 
						  "ACT_CS_W+1", 
                          "BU", 
                          "SRC"
                         )

# COMMAND ----------

PT_Promo_Final.display()

# COMMAND ----------

#PK Duplicate check for PT_DP
dup_df = PT_Promo_Final.groupBy("CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_DESC","BU","CONFIRMED_DT","STTS").count().filter("count > 1")
print("There are total of "+str(dup_df.count())+" rows with duplicate values on PK columns")
dup_df.display()

# COMMAND ----------

##PK Null check for PT_DP
print("Null values in CUST_GRP column :",PT_Promo_Final.filter(col("CUST_GRP").isNull()).count())
print("Null values in PROD_CD column :",PT_Promo_Final.filter(col("PROD_CD").isNull()).count())
print("Null values in STRT_DT column :",PT_Promo_Final.filter(col("STRT_DT").isNull()).count())
print("Null values in END_DT column :",PT_Promo_Final.filter(col("END_DT").isNull()).count())
print("Null values in PRMO_DESC column :",PT_Promo_Final.filter(col("PRMO_DESC").isNull()).count())
print("Null values in BU column :",PT_Promo_Final.filter(col("BU").isNull()).count())
print("Null values in CONFIRMED_DT column :",PT_Promo_Final.filter(col("CONFIRMED_DT").isNull()).count())
print("Null values in STTS column :",PT_Promo_Final.filter(col("STTS").isNull()).count())

# COMMAND ----------

# MAGIC %md #Spain Promotions

# COMMAND ----------

#Reading Spain file into Dataframe
SFA_DF = spark.read.format("delta").option("versionAsOf", sfa_latest_version) \
                                 .load(sfa_bronze) \
                                 .withColumn("STRT_DT", to_timestamp(col("Promo_Start"), "dd/MM/yyyy")) \
                                 .withColumn("END_DT",  to_timestamp(col("Promo_Finish"), "dd/MM/yyyy")) \
                                 .withColumn("CONFIRMED_DT", to_timestamp(date_sub(col("STRT_DT"),7)))
                                 
print("Row Count: ", SFA_DF.count())
SFA_DF.display()

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = SFA_DF.agg({"PROCESS_DATE": "max"}).collect()[0][0]


SFA = (SFA_DF.filter(col("PROCESS_DATE")==max_value) 
             .filter(col("Promo_Cadena").isNotNull())
             .filter(col("Promo_Product_Pack_ID").isNotNull())
      )
print("Max Date value in bronze layer: ",max_value)
print("Row Count: ", SFA.count())
display(SFA)

# COMMAND ----------

col_name = ["Promo_Description","Promo_Cadena","Promo_Display_Type"]

for name in col_name:
  check = check_spl_char(SFA,name)
  if check == 0:
    display(SFA.filter(col(name).contains("�")))
    raise Exception("Special character � found in "+name+" column, Please inform Edu to save the source files in CSV UTF-8 format")

# COMMAND ----------

## Defining Dependent Paths for Spain Dataset

dpndntdatapath_list = dpndntdatapath.split(";")
for path in dpndntdatapath_list:
  srcPath = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if 'sfa-customer-mapping' in path:
    customer_mapping = spark.read.format('delta').load(srcPath).withColumnRenamed("Promo_Cadena","Promo_Cadena_Mapping")

print("Record count for customer mapping: ",customer_mapping.count())

# COMMAND ----------

SFA_F = SFA.filter("left(Promo_Product_Pack_ID,1) = 'F'")
SFA_F_Cust = SFA_F.join(customer_mapping.select("Promo_Cadena_Mapping","G2M").withColumnRenamed("G2M","DMDGROUP"), SFA_F.Promo_Cadena==customer_mapping.Promo_Cadena_Mapping, 'left') 

SFA_B = SFA.filter("left(Promo_Product_Pack_ID,1) = 'B'")
SFA_B_Cust = SFA_B.join(customer_mapping.select("Promo_Cadena_Mapping","SAP").withColumnRenamed("SAP","DMDGROUP"), SFA_B.Promo_Cadena==customer_mapping.Promo_Cadena_Mapping, 'left') 

SFA_A = SFA.filter("left(Promo_Product_Pack_ID,1) = 'A'")
SFA_A_Cust = SFA_A.join(customer_mapping.select("Promo_Cadena_Mapping","AS400").withColumnRenamed("AS400","DMDGROUP"), SFA_A.Promo_Cadena==customer_mapping.Promo_Cadena_Mapping, 'left')

SFA_Customer = SFA_F_Cust.union(SFA_B_Cust.union(SFA_A_Cust))
SFA_Customer.display()

print("SFA:", SFA.count())
print("SFA_F_Cust:", SFA_F_Cust.count())
print("SFA_B_Cust:", SFA_B_Cust.count())
print("SFA_A_Cust:", SFA_A_Cust.count())
print("SFA_Customer:", SFA_Customer.count())

print("Total:", SFA_Customer.count())
print("Total distinct customer:", SFA_Customer.select("Promo_Cadena").distinct().count())
print("Total distinct customer mapped:", SFA_Customer.filter(col("DMDGroup").isNotNull()).select("Promo_Cadena").distinct().count())
print("Total distinct customer not mapped:", SFA_Customer.filter(col("DMDGroup").isNull()).select("Promo_Cadena").distinct().count())

# COMMAND ----------

DistinctDMDGROUP = SFA_Customer.select("Promo_Cadena","DMDGROUP").distinct()
DistinctDMDGROUP.filter("DMDGROUP is null").display()

# COMMAND ----------

SFA_Customer_Promo = (SFA_Customer.withColumnRenamed("Promo_Description", "Promo_Desc")
                                   .withColumnRenamed("DMDGroup", "CUST_GRP")
                                   .withColumnRenamed("dmdunit", "PROD_CD")
                                   .withColumnRenamed("Promo_Display_Type", "PRMO_DISP")
                                   .withColumnRenamed("Promo_Status", "STTS")
								   .select("CUST_GRP"
										  ,"PROD_CD"
										  ,"STRT_DT"
										  ,"END_DT"
										  ,"Promo_Desc"
										  ,"PRMO_DISP"
										  ,"STTS"
										  ,"CONFIRMED_DT"
										  ,lit(None).alias("EST_CS")
										  ,lit(None).alias("ACT_CS_W-1")
										  ,lit(None).alias("ACT_CS_W0")
										  ,lit(None).alias("ACT_CS_W+1")
										  ,lit("ES").alias("BU")
										  ,lit("SFA").alias("SRC"))
								   .dropDuplicates() 
                      )

# COMMAND ----------

SFA_Promo = SFA_Customer_Promo.groupby("CUST_GRP"
                                      ,"PROD_CD"
                                      ,"STRT_DT"
                                      ,"END_DT"
                                      ,"Promo_Desc"
                                      ,"STTS"
                                      ,"CONFIRMED_DT"
                                      ,"EST_CS"
                                      ,"ACT_CS_W-1"
                                      ,"ACT_CS_W0"
                                      ,"ACT_CS_W+1"
                                      ,"BU"
                                      ,"SRC").agg(f.concat_ws(" | ",f.collect_list(col("PRMO_DISP"))).alias("PRMO_DISP"))

# COMMAND ----------

promotions = SFA_Promo.select("Promo_Desc").distinct()

for col_name in promotions.columns:
  promotions = (promotions.withColumn(f"{col_name}_cleaned", f.trim(f.col(col_name)))
                          .withColumn(f"{col_name}_cleaned", f.lower(f"{col_name}_cleaned"))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", '(\d+),(\d+)', '$1.$2'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'ª', 'a'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'º', 'a'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", '([0-9])\s+([0-9][0-9]%)', '$1a$2'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'euro(\W)', '€$1'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'eur(\W)', '€$1'))
               )
display(promotions)

# COMMAND ----------


ordered_tactic_patterns = [
  ("loyalty_InStore",              1, 0, "cheque|crece|chq|club|próxima compra"),
  ("loyalty_General",              1, 0, "vuelve|%v"),
  ("freegoods_Premium",            1, 0, "regalo|item"),
  ("tpr_DiscountPrice",            1, 0, r"\btpr\b|\btrp\b|descuento"),
  ("multibuy_BuyXforY",            1, 0, "[2-9]\s*x\s*[0-9\.]+\s*€"),
  ("tpr_DiscountPrice",            2, 0, "^[0-9\.]+\s*€"),
  ("multi_tiered_BuyMoreSaveMore", 1, 0, "[2-9]\s*a*\s*al*\s*[0-9\.]+\s*%*|compra\s*[2-9]\s*dto|dobleahorro|cajas-[0-9]|\d\s*€\s*x\s*compra\s*\d+€"),
  ("multibuy_BuyXforY",            2, 0, "[2-9]\s*por\s*[0-9\.]+\s*€*|latas|lote|[2-9]\s*x\s*[0-9\.]+|[0-9]+\s*\+\s*[0-9]+"),
  ("tpr_DiscountPerc",             1, 0, "[0-9\.]+\s*%"),
  ("tpr_DiscountPrice",            3, 0, "[0-9]\.[0-9][0-9]\s*€?|[0-9]+\s*€|precio|dto"),
  ("display_General",              1, 0, "cooler|lineal|exp|lin|pilada|chimenea|balda|exhibicion|cab|espacio|eve|floore?stand|pall?et|box")
]


promo_mapped = (promotions.withColumn("PRMO_TAC", 
                        when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[0][3]), ordered_tactic_patterns[0][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[1][3]), ordered_tactic_patterns[1][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[2][3]), ordered_tactic_patterns[2][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[3][3]), ordered_tactic_patterns[3][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[4][3]), ordered_tactic_patterns[4][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[5][3]), ordered_tactic_patterns[5][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[6][3]), ordered_tactic_patterns[6][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[7][3]), ordered_tactic_patterns[7][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[8][3]), ordered_tactic_patterns[8][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[9][3]), ordered_tactic_patterns[9][0])
                       .when(f.col("Promo_Desc_cleaned").rlike(ordered_tactic_patterns[10][3]), ordered_tactic_patterns[10][0])
                     )
               )


# COMMAND ----------

promo_mapped.display()

# COMMAND ----------

SFA_Promo = SFA_Promo.withColumnRenamed("Promo_Desc", "PRMO_DESC")
SFA_CustomerProductPromo = SFA_Promo.join(promo_mapped, SFA_Promo.PRMO_DESC == promo_mapped.Promo_Desc, 'left') \
                                 . select("CUST_GRP", 
										  "PROD_CD", 
										  "STRT_DT", 
										  "END_DT", 
										  "PRMO_DESC", 
										  "PRMO_TAC", 
										  "PRMO_DISP", 
										  "STTS", 
										  "CONFIRMED_DT",  
										  "EST_CS", 
										  "ACT_CS_W-1", 
										  "ACT_CS_W0", 
										  "ACT_CS_W+1", 
										  "BU", 
										  "SRC"
                                         )

SFA_CustomerProductPromo.display()

# COMMAND ----------

#PK Duplicate check for SFA
dup_df = SFA_CustomerProductPromo.groupBy("CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_DESC","BU","CONFIRMED_DT","STTS").count().filter("count > 1")
print("There are total of "+str(dup_df.count())+" rows with duplicate values on PK columns")
dup_df.display()

# COMMAND ----------

##PK Null check for SFA
print("Null values in CUST_GRP column :",SFA_CustomerProductPromo.filter(col("CUST_GRP").isNull()).count())
print("Null values in PROD_CD column :",SFA_CustomerProductPromo.filter(col("PROD_CD").isNull()).count())
print("Null values in STRT_DT column :",SFA_CustomerProductPromo.filter(col("STRT_DT").isNull()).count())
print("Null values in END_DT column :",SFA_CustomerProductPromo.filter(col("END_DT").isNull()).count())
print("Null values in PRMO_DESC column :",SFA_CustomerProductPromo.filter(col("PRMO_DESC").isNull()).count())
print("Null values in BU column :",SFA_CustomerProductPromo.filter(col("BU").isNull()).count())
print("Null values in CONFIRMED_DT column :",SFA_CustomerProductPromo.filter(col("CONFIRMED_DT").isNull()).count())
print("Null values in STTS column :",SFA_CustomerProductPromo.filter(col("STTS").isNull()).count())

# COMMAND ----------

# MAGIC %md
# MAGIC # SFA ALDI Promotion

# COMMAND ----------

#Reading Spain file into Dataframe
SFA_ALDI_DF = spark.read.format("delta").option("versionAsOf", sfa_kam_latest_version) \
                                 .load(sfa_kam_bronze) \
								 .withColumn("STRT_DT", to_timestamp(col("Promo_Start_Date"))) \
								 .withColumn("END_DT", to_timestamp(col("Promo_End_Date"))) \
                                 .withColumn('SRC', lit('KAM'))\
                                 .withColumn('BU', lit('ES'))\
                                 .withColumn("CONFIRMED_DT",when(col("Ask_Start_Date").isNull(),to_timestamp(col("Promo_Start_Date"))) \
                                                  .otherwise(to_timestamp(col("Ask_Start_Date")))) \
								 .withColumnRenamed('DMDUNIT', 'PROD_CD') \
                                 .withColumnRenamed('Promo_Mechanic', 'PRMO_DESC')\
								 .withColumnRenamed('DMDGROUP', 'CUST_GRP') \
								 .withColumnRenamed('Customer_Proposal_QTY','EST_CS')
								 
                                 
print("Row Count: ", SFA_ALDI_DF.count())
SFA_ALDI_DF.display()

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = SFA_ALDI_DF.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print("Max Process Date In Bronze Layer: ",max_value)

sfa_aldi_promo = SFA_ALDI_DF.filter(col("PROCESS_DATE")==max_value) 

print("Record count in bronze layer after filter: ",sfa_aldi_promo.count())
display(sfa_aldi_promo)

# COMMAND ----------

ALDI_promo  = sfa_aldi_promo.select("PRMO_DESC").distinct()

for col_name in ALDI_promo.columns:
  ALDI_promo = (ALDI_promo.withColumn(f"{col_name}_cleaned", f.trim(f.col(col_name)))
                          .withColumn(f"{col_name}_cleaned", f.lower(f"{col_name}_cleaned"))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", '(\d+),(\d+)', '$1.$2'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'ª', 'a'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'º', 'a'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", '([0-9])\s+([0-9][0-9]%)', '$1a$2'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'euro(\W)', '€$1'))
                          .withColumn(f"{col_name}_cleaned", f.regexp_replace(f"{col_name}_cleaned", 'eur(\W)', '€$1'))
               )
display(ALDI_promo)

# COMMAND ----------

ordered_ALDI_tactic_patterns = ordered_tactic_patterns + [("tpr_DiscountPrice", 4, 0, "push|hi\s*['-]*cone")]

# COMMAND ----------

ALDI_promo_mapped = ALDI_promo.withColumn("PRMO_TAC", 
                      f.when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[0][3]), ordered_ALDI_tactic_patterns[0][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[1][3]), ordered_ALDI_tactic_patterns[1][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[2][3]), ordered_ALDI_tactic_patterns[2][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[3][3]), ordered_ALDI_tactic_patterns[3][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[4][3]), ordered_ALDI_tactic_patterns[4][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[5][3]), ordered_ALDI_tactic_patterns[5][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[6][3]), ordered_ALDI_tactic_patterns[6][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[7][3]), ordered_ALDI_tactic_patterns[7][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[8][3]), ordered_ALDI_tactic_patterns[8][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[9][3]), ordered_ALDI_tactic_patterns[9][0])
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[10][3]), ordered_ALDI_tactic_patterns[10][0]) 
                       .when(f.col("PRMO_DESC_cleaned").rlike(ordered_ALDI_tactic_patterns[11][3]), ordered_ALDI_tactic_patterns[11][0])                                          
                     )

# COMMAND ----------

ALDI_final = SFA_ALDI_DF.join(ALDI_promo_mapped['PRMO_DESC','PRMO_TAC'], on = ['PRMO_DESC'], how ='left')
ALDI_final.filter("PRMO_TAC is null").count()
ALDI_final.display()

# COMMAND ----------

ALDI_promo_final = ALDI_final.select("CUST_GRP", 
                                     "PROD_CD", 
                                     "STRT_DT",
                                     "END_DT",
                                     "PRMO_DESC",
                                     "PRMO_TAC",
                                     lit(None).alias("PRMO_DISP"),
                                     lit(" ").alias("STTS"),
                                     "CONFIRMED_DT",
                                     "EST_CS",
                                     lit(None).alias("ACT_CS_W-1"),
								     lit(None).alias("ACT_CS_W0"),
								     lit(None).alias("ACT_CS_W+1"),
								     "BU", 
									 "SRC"
                                    ).dropDuplicates()
								     
ALDI_promo_final.display()


# COMMAND ----------

#PK Duplicate check for KAM
dup_df = ALDI_promo_final.groupBy("CUST_GRP","PROD_CD","STRT_DT","END_DT","PRMO_DESC","BU","CONFIRMED_DT","STTS").count().filter("count > 1")
print("There are total of "+str(dup_df.count())+" rows with duplicate values on PK columns")
dup_df.display()

# COMMAND ----------

##PK Null check for KAM
print("Null values in CUST_GRP column :",ALDI_promo_final.filter(col("CUST_GRP").isNull()).count())
print("Null values in PROD_CD column :",ALDI_promo_final.filter(col("PROD_CD").isNull()).count())
print("Null values in STRT_DT column :",ALDI_promo_final.filter(col("STRT_DT").isNull()).count())
print("Null values in END_DT column :",ALDI_promo_final.filter(col("END_DT").isNull()).count())
print("Null values in PRMO_DESC column :",ALDI_promo_final.filter(col("PRMO_DESC").isNull()).count())
print("Null values in BU column :",ALDI_promo_final.filter(col("BU").isNull()).count())
print("Null values in CONFIRMED_DT column :",ALDI_promo_final.filter(col("CONFIRMED_DT").isNull()).count())
print("Null values in STTS column :",ALDI_promo_final.filter(col("STTS").isNull()).count())

# COMMAND ----------

IBP_SL_Promotions = SFA_CustomerProductPromo.union(PT_Promo_Final.union(ALDI_promo_final)).dropDuplicates()

silver = IBP_SL_Promotions.withColumn("EST_CS",col("EST_CS").cast("float")) \
                          .withColumn("ACT_CS_W-1",col("ACT_CS_W-1").cast("float")) \
                          .withColumn("ACT_CS_W0",col("ACT_CS_W0").cast("float")) \
                          .withColumn("ACT_CS_W+1",col("ACT_CS_W+1").cast("float"))

print("SFA ALDI rows:", silver.filter("SRC = 'KAM'").count())
print("SFA rows:", silver.filter("SRC = 'SFA'").count())
print("PT rows:", silver.filter("SRC = 'PT_DP'").count())
print("Total rows:", silver.count())
silver.display()

# COMMAND ----------

silver.select("PRMO_TAC").distinct().where(~f.col("PRMO_TAC").isin(tactics)).display()

# COMMAND ----------

silver = silver.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

#Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.promotions") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.promotions

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver.count()))