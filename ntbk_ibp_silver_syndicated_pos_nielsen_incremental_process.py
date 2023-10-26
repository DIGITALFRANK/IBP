# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql.types import *

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
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
dpndntdatapath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#splitting the sourcePath in different variables
sourcePathval = dbutils.widgets.get("sourcePath")

print(sourcePathval)

SourcePath_list = sourcePathval.split(';')
print(SourcePath_list)

portugalSnacks = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+SourcePath_list[0]
spainSnacks = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+SourcePath_list[1]
spainBeverages = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+SourcePath_list[2]

# COMMAND ----------

print(portugalSnacks)
print(spainSnacks)
print(spainBeverages)

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#Reading the data from the bronze path
portugalSnacks_deltaTable = DeltaTable.forPath(spark, portugalSnacks)
spainSnacks_deltaTable = DeltaTable.forPath(spark, spainSnacks)
spainBeverages_deltaTable = DeltaTable.forPath(spark, spainBeverages)
portugalSnacks_version = portugalSnacks_deltaTable.history().select(max(col('version'))).collect()[0][0]
spainSnacks_version = spainSnacks_deltaTable.history().select(max(col('version'))).collect()[0][0]
spainBeverages_version = spainBeverages_deltaTable.history().select(max(col('version'))).collect()[0][0]

# COMMAND ----------

#Reading the source data from bonze layer
portugalSnacks_df= spark.read.format("delta").option("versionAsOf", portugalSnacks_version).load(portugalSnacks)
spainSnacks_df = spark.read.format("delta").option("versionAsOf", spainSnacks_version).load(spainSnacks)
spainBeverages_df = spark.read.format("delta").option("versionAsOf", spainBeverages_version).load(spainBeverages)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value_portSnac = portugalSnacks_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
max_value_spainSnac = spainSnacks_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
max_value_spainBev = spainBeverages_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]

portugalSnacks_df2 = portugalSnacks_df.filter(col("PROCESS_DATE")==max_value_portSnac)
spainSnacks_df2 = spainSnacks_df.filter(col("PROCESS_DATE")==max_value_spainSnac)
spainBeverages_df2 = spainBeverages_df.filter(col("PROCESS_DATE")==max_value_spainBev)

# COMMAND ----------

#splitting the dpendentPath in different variables
dependentPathval = dbutils.widgets.get("dependentDatasetPath")

print(dependentPathval)

DependentPath_list = dependentPathval.split(';')
print(DependentPath_list)

map_ean_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[0]
udt_mat_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[1]
prod_df_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[2]
ship_df_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[3]
map_ean_bev_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[4]
map_ean_snacks_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+DependentPath_list[5]

# COMMAND ----------

print(map_ean_path)
print(udt_mat_path)
print(prod_df_path)
print(ship_df_path)
print(map_ean_bev_path)
print(map_ean_bev_path)
print(map_ean_snacks_path)

# COMMAND ----------

#Reading the data from the bronze path
map_ean_deltaTable = DeltaTable.forPath(spark, map_ean_path)
udt_mat_deltaTable = DeltaTable.forPath(spark, udt_mat_path)
prod_df_deltaTable = DeltaTable.forPath(spark, prod_df_path)
ship_df_deltaTable = DeltaTable.forPath(spark, ship_df_path)
map_ean_version = map_ean_deltaTable.history().select(max(col('version'))).collect()[0][0]
udt_mat_version = udt_mat_deltaTable.history().select(max(col('version'))).collect()[0][0]
prod_df_version = prod_df_deltaTable.history().select(max(col('version'))).collect()[0][0]
ship_df_version = ship_df_deltaTable.history().select(max(col('version'))).collect()[0][0]
map_bf_version = map_bev_deltaTable.history().select(max(col('version'))).collect()[0][0]
map_snacks_version = map_snacks_deltaTable.history().select(max(col('version'))).collect()[0][0]

# COMMAND ----------

#Reading the dependent data from bonze layer
map_ean_df= spark.read.format("delta").option("versionAsOf", map_ean_version).load(map_ean_path)
udt_mat_df = spark.read.format("delta").option("versionAsOf", udt_mat_version).load(udt_mat_path)
prod_df = spark.read.format("delta").option("versionAsOf", prod_df_version).load(prod_df_path)
ship_df = spark.read.format("delta").option("versionAsOf", ship_df_version).load(ship_df_path)
map_bev_df = spark.read.format("delta").option("versionAsOf", map_bf_version).load(map_ean_bev_path)
map_snacks_df = spark.read.format("delta").option("versionAsOf", map_snacks_version).load(map_ean_snacks_path)
max_value_ship = ship_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
shipment_df2 = ship_df.filter(col("PROCESS_DATE")==max_value_ship)

# COMMAND ----------

#Reading Shipment Data
def get_shipment():
  shipmentDF = (shipment_df2.filter("right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06') and year(HSTRY_TMFRM_STRT_DT) in (2020,2021)")
                            .withColumn('MU', expr('left(PLANG_CUST_GRP_VAL, 2)'))
                            .withColumnRenamed("PLANG_MTRL_GRP_VAL","Ship_DMDUNIT")
                            .groupby('MU',"Ship_DMDUNIT").agg(sum('DMND_HSTRY_QTY').alias('QTY'))
             )
  return shipmentDF

# COMMAND ----------

#Reading Product Master and applying changes as applicable
def get_filtered_prolink(MU, material_country_filtered):
  
  udtMaterialCountryDF = udt_mat_df
  udtMaterialCountryDF.createOrReplaceTempView('UDT_Material_Country')
  
  productDF = prod_df
  ProlinkProductMaster = productDF.filter("SYS_ID = 564 and right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06')")
  ProlinkProductMaster.createOrReplaceTempView("Prolink_Product")

  ProlinkProduct = spark.sql("""
  select PLANG_MTRL_GRP_VAL as Prod_CD
         ,PLANG_MTRL_GRP_NM as Prod_NM
         ,MLTPCK_INNR_CNT as UNITSIZE
         ,HRCHY_LVL_4_NM as SUBBRND
         ,HRCHY_LVL_3_NM as SIZE
         ,HRCHY_LVL_2_NM as FLVR
         ,PLANG_PROD_TYP_NM as LVL
         ,BRND_NM as BRND
         ,PLANG_MTRL_BRND_GRP_NM as BRND_GRP
         ,SRC_CTGY_1_NM as CTGY
         ,PLANG_MTRL_FAT_PCT as FAT
         ,PLANG_PROD_KG_QTY as KG
         ,PLANG_MTRL_EA_VOL_LITR_QTY as LITRES
         ,PLANG_MTRL_PCK_CNTNR_NM as PCK_CNTNR
         ,SRC_CTGY_2_NM as PROD_LN
         ,PLANG_MTRL_STTS_NM as PROD_STTS
         ,PLANG_MTRL_EA_PER_CASE_CNT as CS
         ,PLANG_MTRL_EA_PER_CASE_STD_CNT as CS2
         ,AUTMTC_DMNDFCST_UNIT_CREATN_FLG
         ,PLANG_PROD_8OZ_QTY as 8OZ
         ,PCK_SIZE_SHRT_NM as SIZE_SHRT
         ,PCK_CNTNR_SHRT_NM as PCK_CNTNR_SHRT
         ,FLVR_NM as FLVR_SHRT
         ,PLANG_MTRL_GRP_DSC as LOCL_DSC
         ,CASE_TYP_ID_USE_FLG as CASE_TYP
         ,PLANG_MTRL_PRODTN_TCHNLGY_NM as PRODUCTION_TECHNOLOGY
         ,SUBBRND_SHRT_NM as SUBBRAND_SHORT
   from Prolink_Product
   """)
  ProlinkProduct.createOrReplaceTempView('Product')

  materialCountry = spark.sql("""With CTE as (SELECT  EJDA_MTRL_ID AS ITEM
          ,MTRL_XREF_ID AS MATERIAL
          ,MKT_CDV AS JDA_CODE
          ,APPROVED
          ,MTRL_STTS_NM as STATUS
          ,MTRL_TYP_CDV
          ,CTRY_CDV as SOURCE_SYSTEM_COUNTRY
   FROM UDT_Material_Country
   )
   , CTE2 as (
       Select  ITEM
               ,MATERIAL
               ,JDA_CODE
               ,APPROVED
               ,STATUS
               ,MTRL_TYP_CDV
               ,count(distinct SOURCE_SYSTEM_COUNTRY) as Source_System_Cnt 
               ,ROW_NUMBER() over(Partition by ITEM Order by APPROVED desc, STATUS desc) as ROW_Id
       from CTE
       where JDA_CODE in ('04','05','06')
       group by ITEM, MATERIAL, JDA_CODE, APPROVED, STATUS, MTRL_TYP_CDV
   )
   Select * 
   from CTE2
   where ROW_Id = 1
   """)
  materialCountry.createOrReplaceTempView('FMT_Material_Country')
  materialCountry.count()
  # ####################################################################
  filteredProductDF = spark.sql("""
  with CTE_ITEM as (
       Select left(PROD_CD,CHARINDEX('_',PROD_CD)-1) as Selling_SKU
               ,right(PROD_CD, (length(PROD_CD) - CHARINDEX('_',PROD_CD))) as ERP_ID
               ,*
       from Product
   where LVL = 'SB-S-FL-ITEM' AND right(PROD_CD,(length(PROD_CD) - CHARINDEX('_',PROD_CD))) in ('04','05','06')
   )
   , CTE_ITEM2 as (
       select Selling_SKU, count(1) as Distinct_DMDUNIT from CTE_ITEM
       group by Selling_SKU
   )
   select b.Distinct_DMDUNIT
           ,case when a.Selling_SKU LIKE '%[A-Za-z]%' then 1 else 0 end as Contains_Letters
           ,ROW_NUMBER() over (Partition by a.Selling_SKU order by a.PROD_CD) as DMDUNIT_Row_Id
           ,a.* 
   from CTE_ITEM a
   left join CTE_ITEM2 b
   on a.Selling_SKU = b.Selling_SKU
   --where a.PROD_STTS = 'ACTIVE'
   """)
  filteredProductDF.createOrReplaceTempView('Filtered_Product')
  # ####################################################################
  ProlinkActiveProduct = spark.sql("""
       select a.*
               ,b.APPROVED
               ,b.STATUS
               ,b.MTRL_TYP_CDV
       from Filtered_Product a
       left join FMT_Material_Country b
       on a.PROD_CD = b.ITEM
       where b.Status = 'ACTIVE' and a.PROD_STTS = 'ACTIVE'
   """)
  
  if not material_country_filtered:
    #Load prolink, which has dmdunits and the remaining product info
    prolink = (prod_df.withColumnRenamed("SRC_CTGY_1_NM","CTGY")
                      .filter("SYS_ID = 564")
              )
  else:
    prolink = ProlinkActiveProduct
    prolink = (prolink.withColumnRenamed("Prod_CD","PLANG_MTRL_GRP_VAL")
                      .withColumnRenamed("Prod_NM","PLANG_MTRL_GRP_NM")
              )
    
  shipmentDF = get_shipment()
    
  prolink = prolink.withColumn("DMDUNIT",expr("substring(PLANG_MTRL_GRP_VAL, 1, length(PLANG_MTRL_GRP_VAL)-3)"))
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.contains("CABERTA") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("TR.*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("IT .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("FR .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("UK .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("RU .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("CY .*") == False)
  prolink = prolink.join(shipmentDF, prolink.PLANG_MTRL_GRP_VAL==shipmentDF.Ship_DMDUNIT, 'inner')
  prolink = prolink.filter(col("MU")==MU)
  
  return prolink

# COMMAND ----------

#Loading Required Data Set and applying filter as applicable.
def load_data(nielsen_version, prolink_filtered = False):
  nielsen = None
  nielsen_with_eans = None
  prolink = None
  
  shipmentDF = get_shipment()
  
  if nielsen_version == "ES_SNACKS":
    
    MU = 'ES'
    CAT = 'SNACKS'
    
    #Load Nielsen (no eans yet)
    nielsen = spainSnacks_df2
    nielsen = nielsen.withColumn("Year",expr("substring(WEEK, 11, 2)"))

    #Load the mapping file to add eans to Nielsen
    sergi_mapping = map_snacks_df.select(col("description"),col("ean_code"),col("tag_code")).distinct()
  
    #Add eans to Nielsen
    nielsen_with_eans = nielsen.join(sergi_mapping, nielsen.TAG==sergi_mapping.tag_code, how="inner")

  elif nielsen_version == "PT":
    
    MU = 'PT'
    CAT = None
    
    #pep = ["SANTA.ANA","LAY.S", "MUNCHOS", "RUFFLES", "SUNBITES", "CHEETOS", "FRITOS", "3.D.S", "DORITOS", "MATUTANO", "BOCA.BITS","PALA.PALA","MIXUPS"]
    nielsen = portugalSnacks_df2
    nielsen = nielsen.withColumn("Year",expr("substring(Week, 5, 2)"))
    #nielsen = nielsen.filter(nielsen.PRODUTO != "SENSITIVE")
    #nielsen = nielsen.filter(nielsen.PRODUTO != "ND")
    #nielsen = nielsen.filter(nielsen.MARCA.isin(pep))
    nielsen_with_eans = nielsen.withColumn("ean_code", nielsen["EAN"]).drop("EAN")

  elif nielsen_version == "ES_BEV":
    
    MU = 'ES'
    CAT = 'BEVERAGES'
    
    nielsen = spainBeverages_df2
    #nielsen = nielsen.where((nielsen.COMPANIA == "CIA PEPSICO"))
    nielsen = nielsen.withColumn("Year",expr("substring(WEEK, 11, 2)"))
    beverage_mapping = map_bev_df
    beverage_mapping = beverage_mapping.withColumn('ean_code', regexp_replace('ean_code',r'^[0]*', ''))
    #beverage_mapping = spark.read.csv("/FileStore/tables/temp/Beverage_Mappingfile_1to1.csv", header="true", inferSchema="true")
    nielsen_with_eans = nielsen.join(beverage_mapping, nielsen.LDESC == beverage_mapping.description, how="inner")

  else:
    raise Exception("Wrong nielsen version. Options: ES_SNACKS, PT, ES_BEV")
    
    
  #Load ean to dmd mapping file to add dmdunits
  ean_to_dmd = map_ean_df
  ean_to_dmd = ean_to_dmd.withColumnRenamed('CD_EAN_G2M','EAN')
  ean_to_dmd = ean_to_dmd.withColumn('EAN', regexp_replace('EAN',r'^[0]*', ''))
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.contains("CABERTA") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("TR.*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("IT .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("FR .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("UK .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("RU .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("CY .*") == False)
  
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.CD_SELLING_G2M.rlike("99[0-9][0-9][0-9]") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.CD_SELLING_G2M.rlike("5[0-9][0-9][0-9]") == False)

  prolink = get_filtered_prolink(MU, prolink_filtered)

   #Filter prolink by category to avoid overlapping code between Snacks and Beverages
  if CAT == 'BEVERAGES':
    prolink =  prolink.filter("CTGY <> 'SNACKS'")
  else:
    prolink = prolink.filter("CTGY <> 'BEVERAGES'")

  return nielsen ,nielsen_with_eans, ean_to_dmd, prolink

# COMMAND ----------

#If you want to use the filtered prolink, DMDUNIT is referred to as Selling_SKU; Getting Mapped Data

def get_mapping(nielsen_version, prolink_filtered=False):
  print(nielsen_version)
  _, nielsen, ean2dmd, prolink = load_data(nielsen_version, prolink_filtered)  
  total = nielsen.where(nielsen.ean_code.isNotNull()).select("ean_code").distinct().count()
  print("Total eans",total)
  
  nielsen2dmd = nielsen.join(ean2dmd, ean2dmd.EAN == nielsen.ean_code, how="left").select(nielsen['*'],ean2dmd['EAN'],ean2dmd['PCCD_ITEM_G2MPCCD_CASE_G2M'])
  
  nielsen_eans_with_dmd = nielsen2dmd.where(nielsen2dmd.EAN.isNotNull()).select("ean_code").distinct().count()
  print("Total eans with dmd",nielsen_eans_with_dmd)
  print("Ratio", nielsen_eans_with_dmd/total)
  
  #mapping = nielsen2dmd.join(prolink, prolink.Selling_SKU == nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M, how="left")
  mapping = nielsen2dmd.join(prolink, prolink.DMDUNIT == nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M, how="left").select(nielsen2dmd['*'],prolink['DMDUNIT'],prolink['PLANG_MTRL_GRP_VAL'])
  
  full = mapping.where(mapping.DMDUNIT.isNotNull()).select("ean_code").distinct().count()
  #full = mapping.where(mapping.Selling_SKU.isNotNull()).select("ean_code").distinct().count()
  print("Completely mapped:",full)
  print("Mapping rate:", full/total)
  print()
  return nielsen, mapping

# COMMAND ----------

# Using False for second argument in get_mapping when building pipeline rather than calculating mapping percentage

nielsen_es_snacks , mapping_es_snacks = get_mapping("ES_SNACKS", False)
mapping_es_snacks_full = mapping_es_snacks.where(mapping_es_snacks.DMDUNIT.isNotNull())
# mapping_es_snacks_full = mapping_es_snacks.where(mapping_es_snacks.Selling_SKU.isNotNull())

nielsen_es_bev , mapping_es_bev = get_mapping("ES_BEV", False)
mapping_es_bev_full = mapping_es_bev.where(mapping_es_bev.DMDUNIT.isNotNull())
# mapping_es_bev_full = mapping_es_bev_full.where(mapping_es_bev_full.Selling_SKU.isNotNull())

nielsen_pt , mapping_pt = get_mapping("PT", False)
mapping_pt_full = mapping_pt.where(mapping_pt.DMDUNIT.isNotNull())
# mapping_pt_full = mapping_pt_full.where(mapping_pt_full.Selling_SKU.isNotNull())

###Then Union this

## Creat then map to CUST_GRP

# COMMAND ----------

print('PortugalSnacks')
print(portugalSnacks_df2.count())
print(mapping_pt.count())
print('SpainSnacks')
print(spainSnacks_df2.count())
print(mapping_es_snacks.count())
print('SpainBeverages')
print(spainBeverages_df2.count())
print(mapping_es_bev.count())

# COMMAND ----------

# Gettige Week MaxDate for Further Calculation
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
mapping_pt_snacks_df = mapping_pt\
                          .withColumn("lagyear",year(date_sub(current_date(),365)))\
                          .withColumn("lagweek",weekofyear(date_sub(current_date(),365)))\
                          .withColumn("weekVal",concat(col("lagyear"),col("lagweek")).cast(IntegerType()))\
                          .withColumn("WeekCol",concat(expr("substring(WEEK, 3, 4)"),expr("substring(WEEK, 8, 2)")).cast(IntegerType()))\
                          .withColumn("WeekCol2",concat(expr("substring(WEEK, 3, 4)"),lit("-W"),expr("substring(WEEK, 8, 2)")))\
                          .withColumn("WeekMaxDate", max(to_date(concat("WeekCol2", lit("-1")), "YYYY-'W'ww-u")).over(Window.partitionBy("PRODUCT_DESCRIPTION","MARKET","TAG","PLANG_MTRL_GRP_VAL","ean_code")))\
                          .withColumn("Thresholdweek", add_months(col("WeekMaxDate"),-3))\
                          .withColumn("quartyear",year(col("Thresholdweek")))\
                          .withColumn("quartweek",weekofyear(col("Thresholdweek")))\
                          .withColumn("QuartweekVal",concat(col("quartyear"),col("quartweek")).cast(IntegerType()))\
                          .drop("lagyear","lagweek","WeekCol2","WeekMaxDate","Thresholdweek","quartyear","quartweek")

mapping_sp_snacks_df = mapping_es_snacks\
                          .withColumn("lagyear",year(date_sub(current_date(),365)))\
                          .withColumn("lagweek",weekofyear(date_sub(current_date(),365)))\
                          .withColumn("weekVal",concat(col("lagyear"),col("lagweek")))\
                          .withColumn("WeekCol",concat(lit("20"),expr("substring(WEEK, 11, 2)"),expr("substring(WEEK, 8, 2)")))\
                          .withColumn("WeekCol2",concat(lit("20"),expr("substring(WEEK, 11, 2)"),lit("-W"),expr("substring(WEEK, 8, 2)")))\
                          .withColumn("WeekMaxDate", max(to_date(concat("WeekCol2", lit("-1")), "YYYY-'W'ww-u")).over(Window.partitionBy("PRODUCT_DESCRIPTION","MARKET","TAG","PLANG_MTRL_GRP_VAL","ean_code")))\
                          .withColumn("Thresholdweek", add_months(col("WeekMaxDate"),-3))\
                          .withColumn("quartyear",year(col("Thresholdweek")))\
                          .withColumn("quartweek",format_string("%02d",weekofyear(col("Thresholdweek"))))\
                          .withColumn("QuartweekVal",concat(col("quartyear"),col("quartweek")))\
                          .drop("lagyear","lagweek","WeekCol2","WeekMaxDate","Thresholdweek","quartyear","quartweek")

mapping_sp_beverages_df = mapping_es_bev\
                          .withColumn("lagyear",year(date_sub(current_date(),365)))\
                          .withColumn("lagweek",weekofyear(date_sub(current_date(),365)))\
                          .withColumn("weekVal",concat(col("lagyear"),col("lagweek")).cast(IntegerType()))\
                          .withColumn("WeekCol",concat(lit("20"),expr("substring(WEEK, 11, 2)"),expr("substring(WEEK, 8, 2)")).cast(IntegerType()))\
                          .withColumn("WeekCol2",concat(lit("20"),expr("substring(WEEK, 11, 2)"),lit("-W"),expr("substring(WEEK, 8, 2)")))\
                          .withColumn("WeekMaxDate", max(to_date(concat("WeekCol2", lit("-1")), "YYYY-'W'ww-u")).over(Window.partitionBy("LDESC","SDESC","PLANG_MTRL_GRP_VAL","ean_code")))\
                          .withColumn("Thresholdweek", add_months(col("WeekMaxDate"),-3))\
                          .withColumn("quartyear",year(col("Thresholdweek")))\
                          .withColumn("quartweek",format_string("%02d",weekofyear(col("Thresholdweek"))))\
                          .withColumn("QuartweekVal",concat(col("quartyear"),col("quartweek")).cast(IntegerType()))\
                          .drop("lagyear","lagweek","WeekCol2","WeekMaxDate","Thresholdweek","quartyear","quartweek")

# COMMAND ----------

#Creating COlumnList to union
col_port_snacks = ['"PORTUGAL_INC" as CNTRY_SRC_NM','NULL as AROMA','Numerical_Distribution_S as NUMRC_DSTRBTN','NULL AS NUMRC_DSTRBTN_PROMO','Wtd_distribution_S as WGHTD_DSTRBTN','Wtd_distribution_S_temporary_reduction_of_price as WGHTD_DSTRBTN_PRC_RDCTN','MANUFACTURER AS MANUFACTURER','PRODUCT_DESCRIPTION as LDESC1','BRAND as BRAND','WEIGHT AS WEIGHT','NULL AS PRC_PER_QTY','NULL AS PRC_ON_PROMO','FLAVOUR as FLAVOR','MARKET AS SDESC','SEGMENT AS SEGMENT','NULL AS SEGMENTS_GTC','TAG AS TAG','NULL AS PRODUCT_TYPE_VARIETY','Universe AS UNIVERSE','Units_Sales as UNITS_SLS','Value_Sales as VAL_SLS','Baseline_Value_Sales as BS_SLS_VAL','Value_sales_in_promo as VAL_SLS_PROMO','Wtd_distribution_S_temporary_reduction_of_price as Ventas_en_valor_reduccion_temp_de_prec','Baseline_Volume_Sales as BS_SLS_QTY','WEEK as WEEK','ean_code as EAN_CODE','PLANG_MTRL_GRP_VAL as PROD_CD','weekVal','WeekCol','QuartweekVal','"PT" as MU']

col_spain_snacks = ['"SPAIN_INC" as CNTRY_SRC_NM','NULL as AROMA','Numeric_Distribution_S  as NUMRC_DSTRBTN','Numeric_Distribution_S_in_promo AS NUMRC_DSTRBTN_PROMO','WTD_Distribution_S  as WGHTD_DSTRBTN','WTD_Distribution_S_temporary_reduction_of_price  as WGHTD_DSTRBTN_PRC_RDCTN','MANUFACTURER as MANUFACTURER','PRODUCT_DESCRIPTION  as LDESC1','BRAND  as BRAND','WEIGHT AS WEIGHT','Price_per_vol  as PRC_PER_QTY','NULL  as PRC_ON_PROMO','FLAVOUR  as FLAVOR','MARKET as SDESC','SEGMENT as SEGMENT','NULL AS SEGMENTS_GTC','TAG  as TAG','PRODUCT_TYPE AS PRODUCT_TYPE_VARIETY','Universe AS UNIVERSE','Unit_Sales as UNITS_SLS','Value_Sales as VAL_SLS','Baseline_value_sales as BS_SLS_VAL','Value_sales_in_promo AS VAL_SLS_PROMO','NULL as Ventas_en_valor_reduccion_temp_de_prec','Baseline_Volume_sales as BS_SLS_QTY','WEEK as WEEK','ean_code as EAN_CODE','PLANG_MTRL_GRP_VAL as PROD_CD','weekVal','WeekCol','QuartweekVal','"ES" as MU']

col_spain_bev = ['"BEVERAGES_INC" as CNTRY_SRC_NM','NULL as AROMA','Distribucion_numerica_S as NUMRC_DSTRBTN','Distribucion_numerica_S_en_promocion as NUMRC_DSTRBTN_PROMO','Distribucion_ponderada_S AS WGHTD_DSTRBTN','Distribucion_ponderada_S_en_promocion as WGHTD_DSTRBTN_PRC_RDCTN','NULL as MANUFACTURER','LDESC as LDESC1','MARCA as BRAND','NULL AS WEIGHT','Precio  as PRC_PER_QTY','Precio_en_promocion  as PRC_ON_PROMO','SABORES as FLAVOR','SDESC  as SDESC','SEGMENTOS as SEGMENT','SEGMENTOS_GTC as SEGMENTS_GTC','NULL AS TAG','NULL AS PRODUCT_TYPE_VARIETY','Universo as UNIVERSE','Ventas_en_unidades AS UNITS_SLS','Ventas_en_valor as VAL_SLS','Ventas_en_valor_baseline  as BS_SLS_VAL','Ventas_en_valor_en_promocion AS VAL_SLS_PROMO','NULL as Ventas_en_valor_reduccion_temp_de_prec','NULL as BS_SLS_QTY','WEEK as WEEK','ean_code as EAN_CODE','PLANG_MTRL_GRP_VAL as PROD_CD','weekVal','WeekCol','QuartweekVal','"ES" as MU']

# COMMAND ----------

#Reading the required Column
mapping_pt_snacks_df2 = mapping_pt_snacks_df.selectExpr(col_port_snacks)
mapping_sp_snacks_df2 = mapping_sp_snacks_df.selectExpr(col_spain_snacks)
mapping_sp_beverages_df2 = mapping_sp_beverages_df.selectExpr(col_spain_bev)

# COMMAND ----------

# Performing Union Operation on all the three Dataset
nielsen_union_df = mapping_pt_snacks_df2.union(mapping_sp_snacks_df2).union(mapping_sp_beverages_df2)

# COMMAND ----------

nielsen_union_df2 = nielsen_union_df.fillna('0',subset=['VAL_SLS','VAL_SLS_PROMO','WGHTD_DSTRBTN','NUMRC_DSTRBTN','UNITS_SLS'])

# COMMAND ----------

#Checking for Duplicates
display(nielsen_union_df.where(col("PROD_CD").isNotNull()).groupBy(col("SOURCE"),col("PROD_CD"),col("EAN_CODE"),col("LDESC1"),col("TAG"),col("WEEK"))\
       .agg(count(col('SEGMENT')).alias("DUP_CHECK"))\
       .where(col("DUP_CHECK")>1))

# COMMAND ----------

# Creating Required Calculative Field
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
silverr_df = nielsen_union_df.na.fill('0').withColumn("CUST_GRP",when(col("SDESC")=='TOTAL ALCAMPO','ES_OT_CARREFOUR')\
                                                        .when(col("SDESC")=='TOTAL PINGO DOCE','PT_OT_PINGO_DOCE')\
                                                        .when(col("SDESC")=='TOTAL CARREFOUR HIPER','ES_OT_ALCAMPO').otherwise(lit('NULL')))\
                          .withColumn("Sals_Val_Num",sum(col("VAL_SLS").cast('float')).over(Window.partitionBy("MU","EAN_CODE","PROD_CD","TAG","WEEK","CUST_GRP","SDESC","LDESC1")))\
                          .withColumn("Sals_Val_Den",sum(col("VAL_SLS").cast('float')).over(Window.partitionBy("MU","WEEK","CUST_GRP")))\
                          .withColumn("PRCNTG_SHR",(col("Sals_Val_Num")/col("Sals_Val_Den")*100))\
                          .withColumn("Sals_BY_SEG_Den",sum(col("VAL_SLS").cast('float')).over(Window.partitionBy("MU","WEEK","CUST_GRP","SEGMENT")))\
                          .withColumn("PRCNTG_SHR_PR_SGMT",(col("Sals_Val_Num")/col("Sals_BY_SEG_Den")*100))\
                          .withColumn("Val_Sals_Ann_Den",when(col("WeekCol").cast(IntegerType())>=col("weekVal").cast(IntegerType())\
                                                              ,sum(col("VAL_SLS").cast('float')).over(Window.partitionBy("MU","CUST_GRP","EAN_CODE","PROD_CD","TAG","SDESC","LDESC1"))))\
                          .withColumn("PRCNTG_SHR_YA",(col("Sals_Val_Num")/col("Val_Sals_Ann_Den"))*100)\
                          .withColumn("PRCNTG_SLS_PROMO",((sum(col('VAL_SLS_PROMO').cast('float')).over(Window.partitionBy("MU","EAN_CODE","PROD_CD","TAG","WEEK","CUST_GRP","SDESC","LDESC1")))\
                                      /(sum(col("VAL_SLS").cast('float')).over(Window.partitionBy("MU","EAN_CODE","PROD_CD","TAG","WEEK","CUST_GRP","SDESC","LDESC1"))))*100)\
                          .withColumn("INCR_SLS_WTD_DST",((col("WGHTD_DSTRBTN").cast('float'))\
                                                          /((col("UNITS_SLS").cast('float'))*(100-col("WGHTD_DSTRBTN").cast('float')))))\
                          .withColumn('PRC_PER_UNIT',((col("VAL_SLS").cast('float'))/(col("UNITS_SLS").cast('float'))))\
                          .withColumn("Avg_Prc_Per_Unit",avg(col('PRC_PER_UNIT').cast('float')).over(Window.partitionBy("MU","SEGMENT")))\
                          .withColumn("PRC_PER_VOL_IND_VS_CTGY",(col("PRC_PER_UNIT").cast('float')/col("Avg_Prc_Per_Unit").cast('float')))\
                          .withColumn("ROS_UNITS",((col("UNITS_SLS").cast('float'))*(col("NUMRC_DSTRBTN").cast('float')/col("WGHTD_DSTRBTN").cast('float'))))\
                          .withColumn("WKS_PROMO",when((col("WeekCol").cast(IntegerType())>=col("QuartweekVal").cast(IntegerType())) &(col("PRCNTG_SLS_PROMO")>30),size(collect_set("WEEK").over(Window.partitionBy("MU","CUST_GRP","EAN_CODE","PROD_CD","TAG","SDESC","LDESC1")))))\
.drop("Sals_Val_Num","Sals_Val_Den","Sals_BY_SEG_Den","Val_Sals_Ann_Den","Avg_Prc_Per_Unit","weekVal","WeekCol","QuartweekVal","PROCESS_DATE","SEGMENTS_GTC","Ventas_en_valor_reduccion_temp_de_prec")

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

#Writing data innto delta lake
#if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df.alias("updates"),
    condition = cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df.write\
  .format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.syndicated_pos_nielsen") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.consumer_mobility_index

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

#Reading the DFU source data from bonze layer
# silverdfff = spark.read.format("delta").load(tgtPath)
# print(silverdfff.count())

# COMMAND ----------

