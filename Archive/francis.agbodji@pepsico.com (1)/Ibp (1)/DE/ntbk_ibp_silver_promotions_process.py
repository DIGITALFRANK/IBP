# Databricks notebook source
#imports
from pyspark.sql.functions import *
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
srcPath = dbutils.widgets.get("sourcePath")
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

# MAGIC 
# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

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
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList

# COMMAND ----------

## Defining the Source path for Portugal and Spain
srcPath_list = srcPath.split(";")

for path in srcPath_list:
  if 'promotions-sfa' in path:
    sfa_bronze = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if 'portugal-promotions' in path:
    pt_promo_bronze = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path

# COMMAND ----------

# MAGIC %md # Protugal Promotions

# COMMAND ----------

# Reading Portugal file into Dataframe

PT_Promo = (spark.read.format("delta").load(pt_promo_bronze)
                                      .withColumn("STRT_DT", to_date(col("DATA_INICIO")))
                                      .withColumn("END_DT", to_date(col("DATA_FIM")))
           )

PT_Promo.printSchema()
PT_Promo.display()

# COMMAND ----------

PT_Promo_Formatted = PT_Promo.withColumn("DMDGROUP",  when(col("CLIENTE") == "FROIZ", "PT_OT_Others") \
                                             .when(col("CLIENTE") == "LIDL", "PT_OT_LIDL") \
                                             .when(col("CLIENTE") == "ITM", "PT_OT_ITM") \
                                             .when(col("CLIENTE") == "LECLERC", "PT_OT_Others") \
                                             .when(col("CLIENTE") == "ESTEVÃO NEVES", "PT_DTS_Madeira") \
                                             .when(col("CLIENTE") == "LIDOSOL", "PT_DTS_Madeira") \
                                             .when(col("CLIENTE") == "ECI",  "PT_OT_Others") \
                                             .when(col("CLIENTE") == "AUCHAN", "PT_OT_AUCHAN") \
                                             .when(col("CLIENTE") == "ALDI", "PT_OT_Others") \
                                             .when(col("CLIENTE") == "BOLAMA", "PT_OT_Others") \
                                             .when(col("CLIENTE") == "SONAE", "PT_OT_SONAE") \
                                             .when(col("CLIENTE") == "DIA", "PT_OT_DIA") \
                                             .when(col("CLIENTE") == "MAKRO", "PT_OT_Others") \
                                             .when(col("CLIENTE") == "PINGO DOCE", "PT_OT_PINGO_DOCE") \
                                             .otherwise("NULL"))

# COMMAND ----------

PT_Promo_Formatted = PT_Promo_Formatted.select("DMDGroup", "DMDUNIT", "STRT_DT", "END_DT", "MECANICA", "STATUS", lit("PT").alias("BU"), lit("PT_DP").alias("SRC")).dropDuplicates() 

PT_Promo_Formatted.display()

# COMMAND ----------

## Bucket the promotions into types
distinct_promos = from_column_to_list(PT_Promo_Formatted.filter(col("MECANICA").isNotNull()).select("MECANICA").distinct(), "MECANICA")

# Get numerical values without any %/€ to be considered as lowered prices
num_list = []
for value in distinct_promos:
    try:
        try:
            num_list.append(str(int(value)))
        except:
            num_list.append(str(float(value)))
    except ValueError:
        continue
        
promo_sub_groups = {}

promo_sub_groups['tpr_DiscountPerc'] = [p for p in distinct_promos if ('%' in p) and ('SP' in p)]
promo_sub_groups['tpr_DiscountPrice'] = [p for p in distinct_promos if ('€' in p)] + num_list
promo_sub_groups['coupons_InStore'] = [p for p in distinct_promos if ('TALÃO' in p) or ('CARTÃO' in p)]
promo_sub_groups['ad_flyer_OnlineAd'] = [p for p in distinct_promos if 'SEM' in p]
promo_sub_groups['ad_flyer_Solo'] = [p for p in distinct_promos if 'S/' in p]
promo_sub_groups['multi_tiered_BuyMoreSaveMore'] = [p for p in distinct_promos if re.search(r'L\dP\d', p)]
promo_sub_groups['multibuy_BuyXforY'] = [p for p in distinct_promos if 'PACK' in p]
promo_sub_groups['purchase_with_purchase'] = [p for p in distinct_promos if 'MENU' in p]


# COMMAND ----------

PT_Promo_Formatted = PT_Promo_Formatted.withColumn("PRMO_TAC", lit(" "))
for group in promo_sub_groups.keys():
  PT_Promo_Formatted = PT_Promo_Formatted.withColumn("PRMO_TAC", 
                      when(col("MECANICA").isin(promo_sub_groups[group]),lit(group)).otherwise(col("PRMO_TAC")))

# COMMAND ----------

display(PT_Promo_Formatted)

# COMMAND ----------

# MAGIC %md #Spain Promotions

# COMMAND ----------

#Reading Spain file into Dataframe
SFA = spark.read.format("delta").load(sfa_bronze) \
                                 .withColumn("STRT_DT", when(length(col("Promo_Start")) < 10 ,to_date(col("Promo_Start"), "dd/MM/yy")) \
                                    .otherwise(to_date(col("Promo_Start"))))\
                                 .withColumn("END_DT", when(length(col("Promo_Finish")) < 10 ,to_date(col("Promo_Finish"), "dd/MM/yy")) \
                                    .otherwise(to_date(col("Promo_Finish"))))
                                 

SFA.display()

# COMMAND ----------

## Defining Dependent Paths for Spain Dataset

dpndntdatapath_list = dpndntdatapath.split(";")
for path in dpndntdatapath_list:
  srcPath = "abfss://"+source_container+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/sap/' in path:
    customerMappingDF_Beverages = spark.read.format('delta').load(srcPath)
  if '/gtm-oracle/' in path:
    customerMappingDF_Snacks = spark.read.format('delta').load(srcPath)
  if '/as400/' in path:
    customerMappingDF_Alvalle = spark.read.format('delta').load(srcPath)
  if 'product-case-mapping' in path:
    product_snacks = spark.read.format('delta').load(srcPath)
  if 'product-mapping' in path:
    product_bevs = spark.read.format('delta').load(srcPath)
  if 'product-master' in path:
    ProlinkProductMaster = spark.read.format('delta').load(srcPath).filter("sys_id = 564 and right(PLANG_MTRL_GRP_VAL,2) in ('04','05','06')").withColumn("Selling_SKU",expr("substring(PLANG_MTRL_GRP_VAL, 1, length(PLANG_MTRL_GRP_VAL)-3)"))

    
product_alvalle = spark.read.csv("/FileStore/tables/temp/Alvelle_Mapping.csv", header="true", inferSchema="true")

# COMMAND ----------

df1 = (customerMappingDF_Beverages.withColumn("Source",lit("SAP"))
                                  .select(col("Customer_Id").alias("Customer_Code")
                                         ,col("Demand_Group").alias("DMDGroup")
                                         ,col("Customer_Name").alias("Desc")
                                         ,col("Source"))
      )
df1 = df1.withColumn('Customer_Code', trim(df1.Customer_Code))

df2 = (customerMappingDF_Snacks.withColumn("Source",lit("GTM"))
                              .select(col("Customer_Id").alias("Customer_Code")
                                     ,col("Demand_Group").alias("DMDGroup")
                                     ,col("Customer_Name").alias("Desc")
                                     ,col("Source"))
      )
df2 = df2.withColumn('Customer_Code', trim(df2.Customer_Code))


df3 = (customerMappingDF_Alvalle.withColumn("Source",lit("AS400"))
                                .withColumn("Desc",lit("N/A"))
                                .select(col("Customer_ID").alias("Customer_Code")
                                       ,col("Demand_Group").alias("DMDGroup")
                                       ,col("Customer_Name").alias("Desc")
                                       ,col("Source"))
      )
df3 = df3.withColumn('Customer_Code', trim(df3.Customer_Code))

Customer_Groups_Combined_DF = df3.union(df1.union(df2))
print("After Union :", Customer_Groups_Combined_DF.count())
Customer_Groups_Combined_DF = Customer_Groups_Combined_DF.withColumn("MU", expr("left(DMDGroup, 2)")).filter("MU='ES'")
print("After Filter :", Customer_Groups_Combined_DF.count())

# COMMAND ----------

SFA_Customer = SFA.join(Customer_Groups_Combined_DF, SFA.Promo_Cadena==Customer_Groups_Combined_DF.Desc, 'left')
print("Total:", SFA_Customer.count())
print("Total distinct customer:", SFA_Customer.select("Promo_Cadena").distinct().count())
print("Total distinct customer not mapped:", SFA_Customer.filter(col("DMDGroup").isNull()).select("Promo_Cadena").distinct().count())
SFA_Customer.filter(col("DMDGroup").isNull()).display()

# COMMAND ----------

SFA_Customer_F = SFA_Customer.filter("left(Promo_Product_Pack_ID,1) = 'F'")

# # Now join to Snacks, Grains and Juices Mapping file
product_snacks = product_snacks.select("CD_PRODUCT_SELLING","CD_PRODUCT_CASE").distinct()
SFA_snacks_mapped = SFA_Customer_F.join(product_snacks, SFA_Customer_F.Promo_Product_Pack_ID == product_snacks.CD_PRODUCT_SELLING , 'left')
SFA_snacks_mapped = SFA_snacks_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('SNACKS','GRAINS')"), SFA_snacks_mapped.CD_PRODUCT_CASE == ProlinkProductMaster.Selling_SKU , 'left')
# Rename
SFA_snacks_mapped = SFA_snacks_mapped.select(["DMDGroup", "Promo_Cadena","Promo_Product_Pack_ID","PLANG_MTRL_GRP_VAL","STRT_DT","END_DT","Promo_Description","Promo_Status"]) 


print('Total:', SFA_Customer.count())
print('Total distinct:', SFA_Customer.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('Food only:' ,SFA_Customer_F.count())
print('Food only distinct:' ,SFA_Customer_F.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('After Joins:', SFA_snacks_mapped.count())
print('After Joins distinct:', SFA_snacks_mapped.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())

display(SFA_snacks_mapped.filter("Promo_Cadena = 'ALCAMPO' and PLANG_MTRL_GRP_VAL = '11230106_05' and Promo_Start = '3/4/2021'"))


# COMMAND ----------

### Product Join for Bevs.
SFA_Customer_B = SFA_Customer.filter("left(Promo_Product_Pack_ID,1) = 'B'")

product_bevs = product_bevs.withColumnRenamed("CD_PROD_SELLING", "CD_PRODUCT_SELLING").select("CD_PRODUCT_SELLING","CD_PRODUCT").distinct()
# Now join to Snacks, Grains and Juices Mapping file
SFA_bevs_mapped = SFA_Customer_B.join(product_bevs ,SFA_Customer_B.Promo_Product_Pack_ID == product_bevs.CD_PRODUCT_SELLING, 'left')
SFA_bevs_mapped = SFA_bevs_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('BEVERAGES','JUICE')"), SFA_bevs_mapped.CD_PRODUCT == ProlinkProductMaster.Selling_SKU , 'left')
# # Rename
SFA_bevs_mapped = SFA_bevs_mapped.select(["DMDGroup", "Promo_Cadena","Promo_Product_Pack_ID","PLANG_MTRL_GRP_VAL","STRT_DT","END_DT","Promo_Description","Promo_Status"]) 

print('Total:', SFA_Customer.count())
print('Total distinct:', SFA_Customer.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('Beverages only:' ,SFA_Customer_B.count())
print('Beverages only distinct:' ,SFA_Customer_B.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('After Joins:', SFA_bevs_mapped.count())
print('After Joins distinct:', SFA_bevs_mapped.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())

display(SFA_bevs_mapped)

# COMMAND ----------

### Product Join for Juices
# 06 lookup
SFA_Customer_A = SFA_Customer.filter("left(Promo_Product_Pack_ID,1) = 'A'")

product_alvalle = product_alvalle.withColumnRenamed("CD_PROD_SELLING", "CD_PRODUCT_SELLING").withColumnRenamed("06 lookup ", "PLANG_MTRL_GRP_VAL").select("CD_PRODUCT_SELLING","PLANG_MTRL_GRP_VAL").distinct()
# Now join to Snacks, Grains and Juices Mapping file
SFA_alvalle_mapped = SFA_Customer_A.join(product_alvalle ,SFA_Customer_A.Promo_Product_Pack_ID == product_alvalle.CD_PRODUCT_SELLING, 'left')
# pricing_alvalle_mapped = pricing_alvalle_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('JUICES')"), pricing_alvalle_mapped.CD_PRODUCT == ProlinkProductMaster.Selling_SKU , 'left')
# Rename
SFA_alvalle_mapped = SFA_alvalle_mapped.select(["DMDGroup", "Promo_Cadena","Promo_Product_Pack_ID","PLANG_MTRL_GRP_VAL","STRT_DT","END_DT","Promo_Description","Promo_Status"]) 

print('Total:', SFA_Customer.count())
print('Total distinct:', SFA_Customer.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('Alvalle only:' ,SFA_Customer_A.count())
print('Alvalle only distinct:' ,SFA_Customer_A.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())
print('After Joins:', SFA_alvalle_mapped.count())
print('After Joins distinct:', SFA_alvalle_mapped.select("Promo_Cadena","Promo_Product_Pack_ID","STRT_DT","END_DT","Promo_Description").distinct().count())

display(SFA_alvalle_mapped )

# COMMAND ----------

SFA_Customer_Null = SFA_Customer.filter("left(Promo_Product_Pack_ID,1) not in ('F','B','A') or Promo_Product_Pack_ID is null").select(["DMDGroup", "Promo_Cadena","Promo_Product_Pack_ID",lit("null").alias("PLANG_MTRL_GRP_VAL"),"STRT_DT","END_DT","Promo_Description","Promo_Status"]) 
SFA_Customer_Null.display()

# COMMAND ----------

SFA_CustomerProduct = SFA_bevs_mapped.union(SFA_snacks_mapped.union(SFA_alvalle_mapped.union(SFA_Customer_Null))).distinct()
print(SFA_CustomerProduct.count())
SFA_CustomerProduct.display()

# COMMAND ----------

SFA_CustomerProduct = (SFA_CustomerProduct.withColumnRenamed("Promo_Description", "Promo_Desc")
                        .select("DMDGroup", "PLANG_MTRL_GRP_VAL", "STRT_DT", "END_DT", "Promo_Desc", 
                         "Promo_Status", lit("ES").alias("BU"), lit("SFA").alias("SRC")).dropDuplicates() 
                      )

# COMMAND ----------

promotions = SFA_CustomerProduct.select("Promo_Desc").distinct()

for col_name in promotions.columns:
  promotions = promotions.withColumn(f"{col_name}_cleaned", trim(col(col_name))).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", ',', '.')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'ª', 'a')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'º', 'a')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", '([0-9]) ([0-9][0-9]%)', '$1a$2')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", ' ', '')).withColumn(f"{col_name}_cleaned", lower(f"{col_name}_cleaned")).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'euro', '€')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'eur', '€'))
  
display(promotions)

# COMMAND ----------

print("Total promos: ",promotions.count())
promo_sub_groups = {}

#Multi-tiered - Buy More Save More More Units you buy the more you save
pattern_bmsm = r'.*[0-9]a?al?[0-9][0-9].*|.*[0-9]a[0-9]\.?[0-9][0-9]?.*|.*compra[0-9]dto.*|.*dobleahorro.*|.*cajas-[0-9]*'
bmsm_promos = promotions.filter(promotions["Promo_Desc_cleaned"].rlike(pattern_bmsm))
bmsm = from_column_to_list(bmsm_promos,"Promo_Desc_cleaned")

promotions_updated = promotions.where(promotions.Promo_Desc_cleaned.isin(bmsm) == False)
promo_sub_groups["multi_tiered_BuyMoreSaveMore"] = bmsm
print(promotions_updated.count())

#Coupons - In Store Available in store only
pattern_cpstore = r'.*cheque.*|.*chqcrc.*|.*cqcrece.*|.*chqcrece.*|.*club.*'
cpstore_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_cpstore))
cpstore = from_column_to_list(cpstore_promos,"Promo_Desc_cleaned")
promo_sub_groups['coupons_InStore'] = cpstore

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(cpstore) == False)
print(promotions_updated.count())

#Loyalty Points - Generic All Other - Loyalty Points
pattern_lpgeneral = r'.*vuelve.*|.*%v.*'
lpgeneral_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_lpgeneral))
lpgeneral = from_column_to_list(lpgeneral_promos,"Promo_Desc_cleaned")
promo_sub_groups['loyalty_points_Generic'] = lpgeneral

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(lpgeneral) == False)
print(promotions_updated.count())

#Free Goods – Free Premium Buy a certain amount / $ value, you get free premiums e.g. collectable plates/mugs etc
pattern_freecol = r'.*regalo.*'
freecol_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_freecol))
freecol = from_column_to_list(freecol_promos,"Promo_Desc_cleaned")
promo_sub_groups['free_goods_Premium'] = freecol

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(freecol) == False)
print(promotions_updated.count())

#TPR - Discount % TPR discount % communicated at store level only
pattern_tprperc = r'.*[0-9]*\.?[0-9]*\%.*'
tprperc_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_tprperc))
tprperc = from_column_to_list(tprperc_promos,"Promo_Desc_cleaned")
promo_sub_groups['tpr_DiscountPerc'] = tprperc

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(tprperc) == False)
print(promotions_updated.count())

#TPR - Discount Price TPR discount price communicated at store level only
pattern_tprprice = r'.*[0-9]€.*|.*[0-9]\.[0-9][0-9]€?.*'
tprprice_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_tprprice))
tprprice = from_column_to_list(tprprice_promos,"Promo_Desc_cleaned")
promo_sub_groups['tpr_DiscountPrice'] = tprprice

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(tprprice) == False)
print(promotions_updated.count())

#Multibuy Buy x for y Buy X for Y
pattern_buyxy = r'.*[0-9]x[0-9].*|.*[0-9]\+[0-9].*|.*latas.*|.*lote.*'
buyxy_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_buyxy))
buyxy = from_column_to_list(buyxy_promos,"Promo_Desc_cleaned")
promo_sub_groups['multibuy_BuyXforY'] = buyxy

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(buyxy) == False)
print(promotions_updated.count())

#Display - General
pattern_display = r'.*cooler.*|.*lineal.*|.*exp.*|.*lin.*|.*pilada.*|.*chimenea.*|.*balda.*|.*exhibicion.*|.*cab.*|.*espacio.*|.*eve.*|.*floore?stand.*|.*pall?et.*|.*box.*|.*item.*|.*trp.*|.*tpr.*'
display_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_display))
dsp = from_column_to_list(display_promos,"Promo_Desc_cleaned")
promo_sub_groups['display_Generic'] = dsp

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(dsp) == False)
print(promotions_updated.count())

# COMMAND ----------

promotions = promotions.withColumn("PRMO_TAC", lit(" "))
for group in promo_sub_groups.keys():
  promotions = promotions.withColumn("PRMO_TAC", when(col("Promo_Desc_cleaned").isin(promo_sub_groups[group]), lit(group)).otherwise(col('PRMO_TAC')))
  
display(promotions)

# COMMAND ----------

SFA_CustomerProduct = SFA_CustomerProduct.withColumnRenamed("Promo_Desc", "Promo_Description")
SFA_CustomerProductPromo = SFA_CustomerProduct.join(promotions, SFA_CustomerProduct.Promo_Description == promotions.Promo_Desc, 'left')
SFA_CustomerProductPromo = SFA_CustomerProductPromo.select("DMDGroup", "PLANG_MTRL_GRP_VAL", "STRT_DT", "END_DT",  "Promo_Desc", "Promo_Status", "BU", "SRC", "PRMO_TAC")

SFA_CustomerProductPromo.display()

# COMMAND ----------

IBP_SL_Promotions = SFA_CustomerProductPromo.union(PT_Promo_Formatted)
IBP_SL_Promotions = (IBP_SL_Promotions.withColumnRenamed("DMDGroup","CUST_GRP")
                                      .withColumnRenamed("PLANG_MTRL_GRP_VAL","PROD_CD")
                                      .withColumnRenamed("Promo_Desc","PRMO_DESC")
                                      .withColumnRenamed("no_of_promo_days","NMBER_PRM_DAYS")
                                      .withColumnRenamed("no_of_weekend_promo_days","NMBR_WKEND_PRM_DAYS")
                                      .withColumnRenamed("Promo_Status","STTS")
                                      .dropDuplicates()
                    )
print("SFA rows:", SFA_CustomerProductPromo.count())
print("PT rows:", PT_Promo_Formatted.count())
print("Total rows:", IBP_SL_Promotions.count())
IBP_SL_Promotions.display()

# COMMAND ----------

IBP_SL_Promotions.write.format("delta").option("overwriteSchema", "true").mode(loadType).save(tgtPath)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(IBP_SL_Promotions.count()))