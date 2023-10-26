# Databricks notebook source
tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"
#storage_account = "cdodevextrblob"
storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
#spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization","true")

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import substring, expr, trim

# COMMAND ----------

def from_column_to_list(df, colname, keepNone=True):
  l = []
  list_col = df.select(colname).collect()
  for elem in list_col:
    if not keepNone:
      if elem[colname] is not None:
        l.append(elem[colname])
    else:
      l.append(elem[colname])
      
  return l

# COMMAND ----------

def load_data(nielsen_version):
  nielsen = None
  nielsen_with_eans = None
  
  if nielsen_version == "ES_SNACKS":
    #Load Nielsen (no eans yet)
    nielsen = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/spain/')
    nielsen = nielsen.withColumn("Year",expr("substring(WEEK, 11, 2)"))

    #Load the mapping file to add eans to Nielsen
    sergi_mapping = spark.read.csv("/FileStore/tables/temp/Sergi_Mappingfile_onlypep.csv", header="true", inferSchema="true")

    #Add eans to Nielsen
    nielsen_with_eans = nielsen.join(sergi_mapping, nielsen.TAG==sergi_mapping.tag_code, how="inner")

  elif nielsen_version == "PT":
    pep = ["SANTA.ANA","LAY.S", "MUNCHOS", "RUFFLES", "SUNBITES", "CHEETOS", "FRITOS", "3.D.S", "DORITOS", "MATUTANO", "BOCA.BITS","PALA.PALA","MIXUPS"]
    nielsen = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/portugal/')
    nielsen = nielsen.withColumn("Year",expr("substring(Week, 5, 2)"))
    nielsen = nielsen.filter(nielsen.PRODUTO != "SENSITIVE")
    nielsen = nielsen.filter(nielsen.PRODUTO != "ND")
    nielsen = nielsen.filter(nielsen.MARCA.isin(pep))
    nielsen_with_eans = nielsen.withColumn("ean_code", nielsen["PRODUTO"])

  elif nielsen_version == "ES_BEV":
    nielsen = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/spain-beverages/')
    nielsen = nielsen.where((nielsen.COMPANIA == "CIA PEPSICO"))
    nielsen = nielsen.withColumn("Year",expr("substring(WEEK, 11, 2)"))
    beverage_mapping = spark.read.csv("/FileStore/tables/temp/Beverage_Mappingfile.csv", header="true", inferSchema="true")
    #beverage_mapping = spark.read.csv("/FileStore/tables/temp/Beverage_Mappingfile_1to1.csv", header="true", inferSchema="true")
    nielsen_with_eans = nielsen.join(beverage_mapping, nielsen.LDESC == beverage_mapping.description, how="inner")

  else:
    raise Exception("Wrong nielsen version. Options: ES_SNACKS, PT, ES_BEV")
    
    
  #Load ean to dmd mapping file to add dmdunits
  ean_to_dmd = spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp-poc/map-ext-dataset-ean/") 
  ean_to_dmd = ean_to_dmd.withColumnRenamed('CD_EAN_G2M','EAN')
  ean_to_dmd = ean_to_dmd.withColumn('EAN', F.regexp_replace('EAN',r'^[0]*', ''))
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.contains("CABERTA") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("TR.*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("IT .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("FR .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("UK .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("RU .*") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.DE_FIELD_G2M.rlike("CY .*") == False)
  
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.CD_SELLING_G2M.rlike("99[0-9][0-9][0-9]") == False)
  ean_to_dmd = ean_to_dmd.where(ean_to_dmd.CD_SELLING_G2M.rlike("5[0-9][0-9][0-9]") == False)

  #Load prolink, which has dmdunits and the remaining product info
  prolink = (spark.read
             .option("header","true")
             .option("delimiter",";")
             .csv("abfss://bronze@cdodevadls2.dfs.core.windows.net/IBP/EDW/Prolink/Product Master/ingestion_dt=2021-06-04/"))
  
  prolink = prolink.withColumn("DMDUNIT",expr("substring(PLANG_MTRL_GRP_VAL, 1, length(PLANG_MTRL_GRP_VAL)-3)"))
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.contains("CABERTA") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("TR.*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("IT .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("FR .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("UK .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("RU .*") == False)
  prolink = prolink.where(prolink.PLANG_MTRL_GRP_NM.rlike("CY .*") == False)
  
  #Load in Product Master tables
  
  udtMaterialCountryDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/udt-material-country/"))


  productDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/product-master/"))
  ProlinkProductMaster = productDF.filter("SYS_ID = 564 and right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06')")

  MosaicProductDF = productDF.filter("sys_id = 695").select("DW_PLANG_MTRL_UNIT_ID", "PLANG_MTRL_GRP_VAL", 'STDRPRT_SUB_CTGY_CDV', 'STDRPRT_SUB_CTGY_NM', 'STDRPRT_SGMNT_CDV', 'STDRPRT_SGMNT_NM', 'STDRPRT_SUB_SGMNT_CDV', 'STDRPRT_SUB_SGMNT_NM').distinct()
  

  mosaicDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/alteryx/ibp-poc/product-master"))
  
  udtMaterialCountryDF.createOrReplaceTempView('UDT_Material_Country')
  ProlinkProductMaster.createOrReplaceTempView("Prolink_Product")
  MosaicProductDF.createOrReplaceTempView("Mosaic_Product")
  mosaicDF.createOrReplaceTempView('Mosaic_Mapping')
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
  return nielsen ,nielsen_with_eans, ean_to_dmd, ProlinkActiveProduct

# COMMAND ----------

def get_active_snacks_prolink(ProlinkActiveProduct):
  prolink_active_snacks = ProlinkActiveProduct.filter(ProlinkActiveProduct.CTGY =="SNACKS")
  return prolink_active_snacks

# COMMAND ----------

  udtMaterialCountryDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/udt-material-country/"))


  productDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/product-master/"))
  ProlinkProductMaster = productDF.filter("SYS_ID = 564 and right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06')")

  MosaicProductDF = productDF.filter("sys_id = 695").select("DW_PLANG_MTRL_UNIT_ID", "PLANG_MTRL_GRP_VAL", 'STDRPRT_SUB_CTGY_CDV', 'STDRPRT_SUB_CTGY_NM', 'STDRPRT_SGMNT_CDV', 'STDRPRT_SGMNT_NM', 'STDRPRT_SUB_SGMNT_CDV', 'STDRPRT_SUB_SGMNT_NM').distinct()
  

  mosaicDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/alteryx/ibp-poc/product-master"))


  udtMaterialCountryDF.createOrReplaceTempView('UDT_Material_Country')
  ProlinkProductMaster.createOrReplaceTempView("Prolink_Product")
  MosaicProductDF.createOrReplaceTempView("Mosaic_Product")
  mosaicDF.createOrReplaceTempView('Mosaic_Mapping')
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
#prolink_active_snacks = ProlinkActiveProduct.filter(ProlinkActiveProduct.CTGY =="SNACKS")
#return prolink_active_snacks
display(ProlinkActiveProduct)

# COMMAND ----------

#Nielsen with eans -> ean-dmdunit map -> prolink
#Loose mapping. Return all rows but with nulls whenever something is not available
def build_mapping_n2p_left(nielsen, ean2dmd, ProlinkActiveProduct):
  nielsen2dmd = nielsen.join(ean2dmd, ean2dmd.EAN == nielsen.ean_code, how="left")
  mapping = nielsen2dmd.join(ProlinkActiveProduct, ProlinkActiveProduct.Selling_SKU == nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M, how="left")
  return mapping

# COMMAND ----------

#prolink active snacks -> ean-dmdunit map -> Nielsen with eans
#Loose mapping. Return all rows but with nulls whenever something is not available
def build_mapping_p2n_left(nielsen, ean2dmd, ProlinkActiveProduct):
  prolink_joined = prolink.join(ean2dmd, ean2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M == ProlinkActiveProduct.ean_code, how="left")
  mapping = prolink_joined.join(nielsen, prolink_joined.EAN == nielsen.ean_code, how="left")
  return mapping

# COMMAND ----------

def get_mapping_left(nielsen_version):
  _, nielsen, ean2dmd, ProlinkActiveProduct = load_data(nielsen_version)
  mapping = build_mapping_n2p_left(nielsen, ean2dmd, ProlinkActiveProduct)
  return nielsen, mapping

# COMMAND ----------

#Export the mapping
#nielsen_version options: ES_SNACKS, ES_BEV, PT
nielsen , mapping = get_mapping_left("ES_SNACKS")
display(mapping.drop("process_date").limit(400000))

# COMMAND ----------

es_mapping = mapping.select("EAN","Prod_CD").distinct()
es_mapping.display()

# COMMAND ----------

pt_mapping = mapping.select("EAN","Prod_CD").distinct()
pt_mapping.display()

# COMMAND ----------

from pyspark.sql.functions import *
shipmentDF = (spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp-poc/shipment-actuals")
                                        .filter("right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06') and year(HSTRY_TMFRM_STRT_DT) in (2020,2021)")
                                        .withColumn('MU', expr('left(PLANG_CUST_GRP_VAL, 2)'))
                                        .groupby('MU','PLANG_CUST_GRP_VAL','PLANG_MTRL_GRP_VAL').agg(sum('DMND_HSTRY_QTY').alias('QTY'))
#                                         .groupby('PLANG_CUST_GRP_VAL','PLANG_MTRL_GRP_VAL','PLANG_LOC_GRP_VAL').agg(sum('DMND_HSTRY_QTY').alias('QTY'))
             )
shipmentDF.display()

# COMMAND ----------

shipment_es_mapped = shipmentDF.filter("MU = 'ES'").join(ProlinkActiveProduct.select("Prod_CD","Prod_NM").distinct(), shipmentDF.PLANG_MTRL_GRP_VAL==ProlinkActiveProduct.Prod_CD, 'inner').groupby("Prod_CD").agg(sum("QTY").alias("QTY"))
print(shipment_es_mapped.count())
print(shipment_es_mapped.distinct().count())
print(shipment_es_mapped.select('Prod_CD').distinct().count())
shipment_es_mapped.display()

# COMMAND ----------

DistinctProduct = es_mapping.filter("Prod_CD is not null").withColumnRenamed("Prod_CD","DMDUNIT").select("DMDUNIT").distinct()
ESProductMapping = shipment_es_mapped.join(DistinctProduct, shipment_es_mapped.Prod_CD==DistinctProduct.DMDUNIT, 'left')
print(ESProductMapping.count())
print(ESProductMapping.filter("DMDUNIT is not null").count())

# COMMAND ----------

display(ESProductMapping.agg(sum("QTY")))
display(ESProductMapping.filter("DMDUNIT is not null").agg(sum("QTY")))

# COMMAND ----------

shipment_pt_mapped = shipmentDF.filter("MU = 'PT'").join(ProlinkActiveProduct.select("Prod_CD","Prod_NM").distinct(), shipmentDF.PLANG_MTRL_GRP_VAL==ProlinkActiveProduct.Prod_CD, 'inner').groupby("Prod_CD").agg(sum("QTY").alias("QTY"))
print(shipment_pt_mapped.count())
print(shipment_pt_mapped.distinct().count())
print(shipment_pt_mapped.select('Prod_CD').distinct().count())
shipment_pt_mapped.display()

# COMMAND ----------

DistinctProduct = pt_mapping.filter("Prod_CD is not null").withColumnRenamed("Prod_CD","DMDUNIT").select("DMDUNIT").distinct()
PTProductMapping = shipment_pt_mapped.join(DistinctProduct, shipment_pt_mapped.Prod_CD==DistinctProduct.DMDUNIT, 'left')
print(PTProductMapping.count())
print(PTProductMapping.filter("DMDUNIT is not null").count())

# COMMAND ----------

display(PTProductMapping.agg(sum("QTY")))
display(PTProductMapping.filter("DMDUNIT is not null").agg(sum("QTY")))

# COMMAND ----------

display(mapping.where(mapping.Prod_CD.isNull()))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> Stats </h3>

# COMMAND ----------

def compute_mapping_rate(nielsen, mapping):
  nielsen_ean_size = nielsen.select("ean_code").distinct().count()
  mapping_size = mapping.select("ean_code").distinct().count()
  print("Mapping rate", mapping_size/nielsen_ean_size)
  

# COMMAND ----------

def get_sales_new(df, column="Ventas_en_valor"):
  '''
    Accepted values for column: 
        ES: Ventas_en_valor, Ventas_en_unidades
        PT: Vendas_em_Valor, Vendas_em_Unidades
  '''
  sum_vol = None
  sum_vol_2020 = None
  sum_vol_2021 = None
  
  df = df.dropDuplicates(["ean_code",column])
  recent = df.filter(df.Year >= 20)
  df_2020 = recent.filter(recent.Year == 20)
  df_2021 = recent.filter(recent.Year == 21)
  
  vol = recent.groupby('ean_code').agg(F.sum(column).alias("sales"))
  sum_vol = vol.select(F.sum("sales")).collect()[0][0]

  vol_2020 = df_2020.groupby('ean_code').agg(F.sum(column).alias("sales"))
  sum_vol_2020 = vol_2020.select(F.sum("sales")).collect()[0][0]

  vol_2021 = df_2021.groupby('ean_code').agg(F.sum(column).alias("sales"))
  sum_vol_2021 = vol_2021.select(F.sum("sales")).collect()[0][0]

  return sum_vol, sum_vol_2020, sum_vol_2021

# COMMAND ----------

def print_stats(total_recent, total_2020, total_2021, mapping_recent, mapping_2020, mapping_2021, mode):
  print("Volume Mapping", mode)
  print("-----------------------")
  print("TOTAL 20 and 21 Mapped Volume",mapping_recent)
  print("TOTAL 20 and 21 Volume",total_recent)
  print("RATIO:" ,mapping_recent/total_recent)

  print("TOTAL 20 Mapped Volume",mapping_2020)
  print("TOTAL 20 Volume", total_2020)
  print("RATIO:" ,mapping_2020/total_2020)

  print("TOTAL 21 Mapped Volume",mapping_2021)
  print("TOTAL 21 Volume",total_2021)
  print("RATIO:" ,mapping_2021/total_2021)
  print("-----------------------")

# COMMAND ----------

def compute_volume_coverage(nielsen, mapping, column):  
  total_recent, total_2020, total_2021 = get_sales_new(nielsen, column)
  mapping_recent, mapping_2020, mapping_2021 = get_sales_new(mapping, column)
  print_stats(total_recent, total_2020, total_2021, mapping_recent, mapping_2020, mapping_2021, column)


# COMMAND ----------

def get_shipments(df):
  df = df.dropDuplicates(["QTY","DMDUNIT_ship","WEEK_OF_YEAR"])
  recent = df.filter(df.Year_ship >= 2020)
  shipments_2020 = recent.filter(recent.Year_ship == 2020)
  shipments_2021 = recent.filter(recent.Year_ship == 2021)

  ship_recent = recent.groupby('DMDUNIT_ship').agg(F.sum('QTY').alias('shipped_units'))
  sum_ship = ship_recent.select(F.sum("shipped_units")).collect()[0][0]
  
  ship_2020 = shipments_2020.groupby('DMDUNIT_ship').agg(F.sum('QTY').alias('shipped_units'))
  sum_ship_2020 = ship_2020.select(F.sum("shipped_units")).collect()[0][0]

  ship_2021 = shipments_2021.groupby('DMDUNIT_ship').agg(F.sum('QTY').alias('shipped_units'))
  sum_ship_2021 = ship_2021.select(F.sum("shipped_units")).collect()[0][0]

  return sum_ship, sum_ship_2020, sum_ship_2021

# COMMAND ----------

def compute_shipment_coverage(mapping, country="ES"):
  shipmentDF = spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/shipment-actuals")

  shipmentDF = shipmentDF.withColumn("DMDUNIT_ship",expr("substring(PROD_CD, 1, length(PROD_CD)-3)"))
  shipmentDF = shipmentDF.withColumn("Year_ship",expr("substring(WEEK_OF_YEAR, 1, 4)"))
  
  display(shipmentDF)
  total_ship, total_ship_2020, total_ship_2021 = get_shipments(shipmentDF)
  
  mapped_shipments = shipmentDF.join(mapping, shipmentDF.DMDUNIT_ship == mapping.PCCD_ITEM_G2MPCCD_CASE_G2M, how="inner")
  mapped_ship, mapped_ship_2020, mapped_ship_2021 = get_shipments(mapped_shipments)

  print_stats(total_ship, total_ship_2020, total_ship_2021, mapped_ship, mapped_ship_2020, mapped_ship_2021, "Shipments")

# COMMAND ----------

#prolink active snacks -> ean-dmdunit map -> Nielsen with eans
#Strict mapping. Only return rows that are complete.

def build_mapping_p2n(nielsen, ean2dmd, ProlinkActiveProduct):
  #prolink_active_snacks = get_active_snacks_prolink(prolink)
  prolink_active_snacks_ean = ProlinkActiveProduct.join(ean2dmd, ean2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M == ProlinkActiveProduct.Selling_SKU, how="inner")
  mapping = prolink_active_snacks_ean.join(nielsen, prolink_active_snacks_ean.EAN == nielsen.ean_code, how="inner")
  return mapping

# COMMAND ----------

#Nielsen with eans -> ean-dmdunit map -> prolink active snacks
#Strict mapping. Only return rows that are complete.

def build_mapping_n2p(nielsen, ean2dmd, ProlinkActiveProduct):
  print("Nielsen eans UNIQUE",nielsen.select("ean_code").distinct().count())
  
  nielsen2dmd = nielsen.join(ean2dmd, ean2dmd.EAN == nielsen.ean_code, how="inner")
  print("Nielsen with dmdunit UNIQUE", nielsen2dmd.select("ean_code").distinct().count())

  prolink_active_snacks = get_active_snacks_prolink(ProlinkActiveProduct)
  mapping = nielsen2dmd.join(prolink_active_snacks, prolink_active_snacks.Selling_SKU == nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M, how="inner")
  print("Nielsen mapped with prolink UNIQUE", mapping.select("ean_code").distinct().count())

  return mapping

# COMMAND ----------

def get_mapping_inner(nielsen_version):
  _, nielsen, ean2dmd, ProlinkActiveProduct = load_data(nielsen_version)
  mapping = build_mapping_n2p(nielsen, ean2dmd, ProlinkActiveProduct)
  return nielsen, mapping

# COMMAND ----------

#Just to compute some stats
#OPTIONS: ES_SNACKS, ES_BEV, PT

nielsen , mapping = get_mapping_inner("ES_SNACKS")

#Of the nielsen data in 2020, 2021 and both, how much do the mapped elements represent 
#In terms of value and units
#ES: Ventas_en_valor, Ventas_en_unidades
#PT: Vendas_em_Valor, Vendas_em_Unidades

compute_volume_coverage(nielsen, mapping, "Ventas_en_valor")
compute_volume_coverage(nielsen, mapping, "Ventas_en_unidades")

#Same but with shipment info
country = "ES"
compute_shipment_coverage(mapping, country)


# COMMAND ----------

# MAGIC %md
# MAGIC <h3> Soeren Data Export </h3>

# COMMAND ----------

_, nielsen, ean2dmd, prolink = load_data("PT")
print(prolink.select("DMDUNIT").distinct().count())
item_codes = spark.read.csv("/FileStore/tables/temp/item_exclusion_list.csv", header="true", inferSchema="true")
item_list = from_column_to_list(item_codes,"item_code")


# COMMAND ----------

##########################################
############### NO FILTER ################
##########################################

print("NO FILTER")
total_rows = prolink.select("DMDUNIT").distinct().count()
mapping = build_mapping_p2n_left(nielsen, ean2dmd, prolink)
dmdmapped = mapping.where(mapping.ean_code.isNotNull()).select("DMDUNIT").distinct().count()
ratio = dmdmapped / total_rows
print("Total:", total_rows)
print("Mapped:", dmdmapped)
print("RATIO:", ratio)

# COMMAND ----------

##########################################
# Filter Prolink to keep only Pep Brands #
##########################################
print("ONLY PEP BRANDS")

not_pep_brands = ["ARTIACH", "PANRICO", "TOPPS", "PALA PALHA", "BIMBO", "DOLCI PREZIOSI", "ADAMS", "NESTLE", "PERNIGOTTI", "DULCESOL", "CHUPA CHUPS", "SUCHARD", "LACASA", "LINDT", "DAMEL", "KRAFT", "PRINCIPE", "CHIPITA", "OSP", "OTEP", "ASSORTED BRANDS", "MULTIPACK", "OTHERS", "MULTIBRAND CSD", "LAVAZZA", "SOUTHERN REFRESHERS" ]

#we now remove the items that are not from pepsico brands
prolink_f = prolink.where(prolink.BRND_NM.isin(not_pep_brands) == False)
total_only_pep = prolink_f.select("DMDUNIT").distinct().count()

mapping = build_mapping_p2n_left(nielsen, ean2dmd, prolink_f)
dmdmapped = mapping.where(mapping.ean_code.isNotNull()).select("DMDUNIT").distinct().count()
ratio = dmdmapped / total_only_pep
print("Total:", total_only_pep)
print("Mapped:", dmdmapped)
print("RATIO:", ratio)
print()

##########################################
# only Pep Brands and item exclusion list#
##########################################

print("ONLYPEP + ITEM EXCLUSION LIST")
prolink_f = prolink_f.where(prolink_f.PLANG_MTRL_GRP_VAL.isin(item_list) == False)
total_without_soeren = prolink_f.select("DMDUNIT").distinct().count()

mapping = build_mapping_p2n_left(nielsen, ean2dmd, prolink_f)
dmdmapped = mapping.where(mapping.ean_code.isNotNull()).select("DMDUNIT").distinct().count()
ratio = dmdmapped / total_without_soeren
print("Total:", total_without_soeren)
print("Mapped:", dmdmapped)
print("RATIO:", ratio)
print()

##########################################
# onlyPep item exclusion and only snacks #
##########################################

print("Onlypep + items excluded + only snacks")
prolink_only_snacks = prolink_f.filter(prolink_f.SRC_CTGY_1_NM =="SNACKS")
#prolink_only_snacks = prolink_f.filter(prolink_f.SRC_CTGY_1_NM =="BEVERAGES")
total_snacks = prolink_only_snacks.select("DMDUNIT").distinct().count()

mapping = build_mapping_p2n_left(nielsen, ean2dmd, prolink_only_snacks)
dmdmapped = mapping.where(mapping.ean_code.isNotNull()).select("DMDUNIT").distinct().count()

ratio = dmdmapped / total_snacks

print("Total:", total_snacks)
print("Mapped:", dmdmapped)
print("RATIO:", ratio)
print()

#################################################
# onlyPep item exclusion and only active snacks #
#################################################

print("Onlypep + items excluded + only active snacks")

prolink_active_snacks = prolink_only_snacks.filter(prolink_only_snacks.PLANG_MTRL_STTS_NM =="ACTIVE")
total_active_snacks = prolink_active_snacks.select("DMDUNIT").distinct().count()

mapping = build_mapping_p2n_left(nielsen, ean2dmd, prolink_active_snacks)
dmdmapped = mapping.where(mapping.ean_code.isNotNull()).select("DMDUNIT").distinct().count()

ratio = dmdmapped / total_active_snacks

print("Total:", total_active_snacks)
print("Mapped:", dmdmapped)
print("RATIO:", ratio)
print()
display(mapping.where(mapping.ean_code.isNull()))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <table class="tg">
# MAGIC <thead>
# MAGIC   <tr>
# MAGIC     <th class="tg-amwm">Filter</th>
# MAGIC     <th class="tg-amwm">Prolink DMDUNITS</th>
# MAGIC     <th class="tg-amwm">Mapped to EAN</th>
# MAGIC     <th class="tg-amwm">Ratio</th>
# MAGIC   </tr>
# MAGIC </thead>
# MAGIC <tbody>
# MAGIC   <tr>
# MAGIC     <td class="tg-amwm">No Filter</td>
# MAGIC     <td class="tg-baqh">5773</td>
# MAGIC     <td class="tg-baqh">2428</td>
# MAGIC     <td class="tg-baqh">42,06%</td>
# MAGIC   </tr>
# MAGIC   <tr>
# MAGIC     <td class="tg-amwm">Only Pep</td>
# MAGIC     <td class="tg-baqh">4371</td>
# MAGIC     <td class="tg-baqh">2393</td>
# MAGIC     <td class="tg-baqh">54,75%</td>
# MAGIC   </tr>
# MAGIC   <tr>
# MAGIC     <td class="tg-amwm">Only Pep, excluded items</td>
# MAGIC     <td class="tg-baqh">4369</td>
# MAGIC     <td class="tg-baqh">2393</td>
# MAGIC     <td class="tg-baqh">54,77%</td>
# MAGIC   </tr>
# MAGIC    <tr>
# MAGIC     <td class="tg-amwm">Only Pep, excluded items, only snacks</td>
# MAGIC     <td class="tg-baqh">3306</td>
# MAGIC     <td class="tg-baqh">2393</td>
# MAGIC     <td class="tg-baqh">72,38%</td>
# MAGIC   </tr>
# MAGIC   <tr>
# MAGIC     <td class="tg-amwm">Only Pep, excluded items, only active snacks</td>
# MAGIC     <td class="tg-baqh">2311</td>
# MAGIC     <td class="tg-baqh">1781</td>
# MAGIC     <td class="tg-baqh">77,07%</td>
# MAGIC   </tr>
# MAGIC </tbody>
# MAGIC </table>

# COMMAND ----------

# MAGIC %md
# MAGIC <h3> Beverages </h3>
# MAGIC <ul> 
# MAGIC   <li>Need to get Sergi Mapping file to put EANs in the Nielsen data. Currently beverage descriptions and EANs are not matched. Also the Nielsen beverage data does not contain a TAG column &#9745;</li>
# MAGIC   <li>Need to get EAN to DMDUnit mapping file. The current ean2dmd table contains beverages &#9745;</li>
# MAGIC </ul>

# COMMAND ----------

#Export the mapping
#nielsen_version options: ES_SNACKS, ES_BEV, PT

nielsen , mapping = get_mapping_left("ES_BEV")

print(nielsen.select("ean_code").distinct().count())
print(mapping.where(mapping.DMDUNIT.isNotNull()).select("ean_code").distinct().count())

#Just to download a small sample
display(mapping.drop("process_date").limit(50000))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Portugal</h3>

# COMMAND ----------

#Export the mapping
#nielsen_version options: ES_SNACKS, ES_BEV, PT
nielsen , mapping = get_mapping_left("PT")

#Just to download a small sample
display(mapping.drop("process_date").limit(50000))

# COMMAND ----------

nielsen_pt, nielsen_pt_ean, ean_to_dmd, prolink = load_data("PT")
display(nielsen_pt_ean)
print(nielsen_pt_ean.select("PRODUTO").distinct().count())

# COMMAND ----------

#ES: Ventas_en_valor, Ventas_en_unidades
#PT: Vendas_em_Valor, Vendas_em_Unidades

joined = nielsen_pt_ean.join(ean_to_dmd, nielsen_pt_ean.PRODUTO == ean_to_dmd.EAN, how="inner")
print(joined.where(joined.EAN.isNotNull()).select("PRODUTO").distinct().count())

joined_prolink = joined.join(prolink, joined.PCCD_ITEM_G2MPCCD_CASE_G2M == prolink.DMDUNIT, how="inner")
print(joined_prolink.where(joined_prolink.DMDUNIT.isNotNull()).select("PRODUTO").distinct().count())

#compute_volume_coverage(nielsen_pt_ean, joined_prolink, "Vendas_em_Valor")
#compute_volume_coverage(nielsen_pt_ean, joined_prolink, "Vendas_em_Unidades")

compute_shipment_coverage(joined_prolink)