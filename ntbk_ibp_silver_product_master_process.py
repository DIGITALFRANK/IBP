# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

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
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
dependentPath = dbutils.widgets.get("dependentDatasetPath")

# COMMAND ----------

#splitting the dependentDatasetPath in different variables
print(dependentPath)

dependentPath_list = dependentPath.split(';')

for path in dependentPath_list:
  if '/udt-material-country' in path:
    udt_material = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/alteryx/ibp/product-master' in path:
    alteryx_product = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/prolink-edw/ibp/dfu' in path:
    dfu_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/europe-dl/ibp/product-mapping/' in path:
    product_mapping_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/europe-dl/ibp/product-case-mapping/' in path:
    product_case_mapping_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/europe-dl/ibp/sourcing/' in path:
    sourcing_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/prolink-edw/ibp/item/' in path:
    item_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/edw/ibp/sku-master/' in path:
    sku_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/iberia/europe-dl/ibp/udt-item-prod-mapping/' in path:
    udt_item_prod_mapping_path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  

# COMMAND ----------

print(udt_material)
print(alteryx_product)
print(dfu_path)

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  merge_cond = " and ".join(ls)
else :
  merge_cond = "target."+pkList+" = updates."+pkList
merge_cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#Reading the delta history from the bronze path of EDW Product Master
product_deltaTable = DeltaTable.forPath(spark, srcPath)
product_latest_version = product_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(product_latest_version)
display(product_deltaTable.history())

# COMMAND ----------

# DBTITLE 1,Read product-master
# prolink = spark.read.format("delta").load(srcPath)
#Reading the EDW Product Master source data from bronze layer
prolink = spark.read.format("delta").option("versionAsOf", product_latest_version).load(srcPath)

# COMMAND ----------

MosaicProductDF = prolink.filter("sys_id = 695").select("DW_PLANG_MTRL_UNIT_ID", "PLANG_MTRL_GRP_VAL", 'SCTR_PROD_CTGY_NM', 'STDRPRT_SUB_CTGY_CDV', 'STDRPRT_SUB_CTGY_NM', 'STDRPRT_SGMNT_CDV', 'STDRPRT_SGMNT_NM', 'STDRPRT_SUB_SGMNT_CDV', 'STDRPRT_SUB_SGMNT_NM').distinct()

# COMMAND ----------

MosaicProductDF = MosaicProductDF.withColumnRenamed("PLANG_MTRL_GRP_VAL", "MOSAIC_PRODUCT_PLANG_MTRL_GRP_VAL")

# COMMAND ----------

# DBTITLE 1,helper functions
def check_pk(df, df_name, df_keys, null_condition):
  duplicates = df.groupBy(*df_keys).count().where("count > 1").count()
  records = df.count()
  nulls = df.where(null_condition).count()
  print(f"Dataframe: {df_name}, key: {', '.join(df_keys)}")
  print(f"No of nulls in PK: {nulls}")  
  print(f"No of duplicates: {duplicates}")
  print(f"No of records: {records}")

# COMMAND ----------

# DBTITLE 1,Check for uniqueness and count: prolink
check_pk(prolink, "prolink", ["PLANG_MTRL_GRP_VAL"], prolink.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,Read udt-material-country, mosaic, dfu
udt = (spark.read.format("delta").load(udt_material)
       .filter(col("APPROVED") == 1)
       .withColumn("RN", row_number().over( Window.partitionBy("EJDA_MTRL_ID").orderBy(col("DW_UPDT_DTM").desc())))
       .where("RN == 1"))

mosaic = (spark.read.format("delta").load(alteryx_product)
          .filter(col("SOURCE_SYSTEM_ID") == '01')
          .filter(col("SPOT_ATTRIBUTE_10_NM") == 'ACTIVE')
          .filter(~col("MOSAIC_SKU_NM").startswith('NPD'))
          .select("SELLING_SKU_CDV", "SELLING_SKU_NM", "MOSAIC_SKU_CDV", "MOSAIC_SKU_NM", "PROD_CTGY_NM", "LEVEL_4_CDV", "LEVEL_4_NM")
         )

dfu = (spark.read.format("delta")
       .load(dfu_path)
       .filter("DMNDFCST_UNIT_LVL_VAL= 'SB-S-FL-ITEM_CLIENT_DC'")
       .withColumn("MU", expr("left(PLANG_CUST_GRP_VAL,2)"))
       .select("PLANG_MTRL_GRP_VAL","MU")
       .distinct())

# COMMAND ----------

# DBTITLE 1,Check for uniqueness and count: udt, mosaic, dfu
#udt
check_pk(udt, "udt", ["EJDA_MTRL_ID"], udt.EJDA_MTRL_ID.isNull())
#mosaic
check_pk(mosaic, "mosaic", ["SELLING_SKU_CDV"], mosaic.SELLING_SKU_CDV.isNull())
#dfu
check_pk(dfu, "dfu", ["PLANG_MTRL_GRP_VAL", "MU"], dfu.PLANG_MTRL_GRP_VAL.isNull() | dfu.MU.isNull())

# COMMAND ----------

# DBTITLE 1,Read remaining datasets
product_list = spark.read.format('delta').load(product_mapping_path)
product_case = spark.read.format('delta').load(product_case_mapping_path)
df_sourcing = spark.read.format('delta').load(sourcing_path)
df_item = spark.read.format('delta').load(item_path)
df_sku = spark.read.format('delta').load(sku_path)
df_udt_item_prod_mapping = spark.read.format('delta').load(udt_item_prod_mapping_path)

# COMMAND ----------

# DBTITLE 1,Extract only items for Iberia - ends with _04/_05/_06
prolink = (prolink.filter(prolink.PLANG_MTRL_GRP_VAL.endswith("_04") | 
                         prolink.PLANG_MTRL_GRP_VAL.endswith("_05") | 
                         prolink.PLANG_MTRL_GRP_VAL.endswith("_06")))
prolink.count()

# COMMAND ----------

# DBTITLE 1,Create a new column PLANG_MTRL_GRP_VAL_2 without the "_xx"
prolink = prolink.withColumn("PLANG_MTRL_GRP_VAL_2", split(prolink.PLANG_MTRL_GRP_VAL, "_")[0])

#check the result
display(prolink.select("PLANG_MTRL_GRP_VAL", "PLANG_MTRL_GRP_VAL_2").limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Flags - TR, Confectionery, Costco, CBS

# COMMAND ----------

# DBTITLE 1,Create TR Flag
prolink = (prolink.withColumn("TR_FLG",
                              when(((prolink.PLANG_MTRL_GRP_NM.startswith("TR")) 
                                    & (prolink.SRC_CTGY_1_NM.contains("SNACKS")) 
                                    & (prolink.SRC_CTGY_2_NM != "CONFECTIONARY")), "TRUE")
                              .otherwise("FALSE")))

# COMMAND ----------

# DBTITLE 1,Create Confectionery Flag
# use this one 
prolink = prolink.withColumn("CONFECTIONERY_FLG", when((prolink.SRC_CTGY_2_NM == "CONFECTIONARY"), "TRUE").otherwise("FALSE"))

# COMMAND ----------

# DBTITLE 1,Create Costco Flag
prolink = prolink.withColumn("COSTCO_FLG", when(col("PLANG_MTRL_GRP_NM").startswith("COSTCO"), "TRUE").otherwise("FALSE"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### CBS Flag

# COMMAND ----------

# DBTITLE 1,Create views of datasets for SQL
udt.createOrReplaceTempView("v_udt_material_country_idm");
df_sourcing.createOrReplaceTempView("v_sourcing");
df_item.createOrReplaceTempView("v_item");
df_sku.createOrReplaceTempView("v_sku");
df_udt_item_prod_mapping.createOrReplaceTempView("v_udt_item_prod_mapping");

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW v_cross_border_product_mapping
# MAGIC AS SELECT DISTINCT
# MAGIC     s.ITEM AS CB_CODE,
# MAGIC     v_item.MTRL_NM AS DESCR, -- It seems like v_item.MTRL_NM is same as DESCR but Need to confirm this with data owner.
# MAGIC     s.SOURCE,
# MAGIC     CASE
# MAGIC       WHEN instr(s.SOURCE, '_ES_') > 0 THEN 'ES'
# MAGIC       WHEN instr(s.SOURCE, '_PT_') > 0 THEN 'PT'
# MAGIC       WHEN instr(s.SOURCE, '_TR_') > 0 THEN 'TR'
# MAGIC       WHEN instr(s.SOURCE, '_BY_') > 0 THEN 'BY'
# MAGIC       WHEN instr(s.SOURCE, '_RS_') > 0 THEN 'RS'
# MAGIC       WHEN instr(s.SOURCE, '_GE_') > 0 THEN 'GE'
# MAGIC       WHEN instr(s.SOURCE, '_BE_') > 0 THEN 'BE'
# MAGIC       WHEN instr(s.SOURCE, '_AM_') > 0 THEN 'AM'
# MAGIC       WHEN instr(s.SOURCE, '_FR_') > 0 THEN 'FR'
# MAGIC       WHEN instr(s.SOURCE, '_IT_') > 0 THEN 'IT'
# MAGIC       WHEN instr(s.SOURCE, '_KG_') > 0 THEN 'KG'
# MAGIC       WHEN instr(s.SOURCE, '_GR_') > 0 THEN 'GR'
# MAGIC       WHEN instr(s.SOURCE, '_CY_') > 0 THEN 'CY'
# MAGIC       WHEN instr(s.SOURCE, '_ZA_') > 0 THEN 'ZA'
# MAGIC       WHEN instr(s.SOURCE, '_GB_') > 0 THEN 'GB'
# MAGIC       WHEN instr(s.SOURCE, '_BA_') > 0 THEN 'BA'
# MAGIC       WHEN instr(s.SOURCE, '_RU_') > 0 THEN 'RU'
# MAGIC       WHEN instr(s.SOURCE, '_RO_') > 0 THEN 'RO'
# MAGIC       WHEN instr(s.SOURCE, '_DE_') > 0 THEN 'DE'
# MAGIC       WHEN instr(s.SOURCE, '_PL_') > 0 THEN 'PL'
# MAGIC       WHEN instr(s.SOURCE, '_UA_') > 0 THEN 'UA'
# MAGIC       WHEN instr(s.SOURCE, '_NL_') > 0 THEN 'NL'
# MAGIC       WHEN instr(s.SOURCE, '_KZ_') > 0 THEN 'KZ'
# MAGIC       WHEN instr(s.SOURCE, '_AZ_') > 0 THEN 'AZ'
# MAGIC       ELSE ''
# MAGIC     END AS SOURCE_BU,
# MAGIC     s.DEST,
# MAGIC     CASE
# MAGIC       WHEN instr(s.DEST, '_ES_') > 0 THEN 'ES'
# MAGIC       WHEN instr(s.DEST, '_PT_') > 0 THEN 'PT'
# MAGIC       WHEN instr(s.DEST, '_TR_') > 0 THEN 'TR'
# MAGIC       WHEN instr(s.DEST, '_BY_') > 0 THEN 'BY'
# MAGIC       WHEN instr(s.DEST, '_RS_') > 0 THEN 'RS'
# MAGIC       WHEN instr(s.DEST, '_GE_') > 0 THEN 'GE'
# MAGIC       WHEN instr(s.DEST, '_BE_') > 0 THEN 'BE'
# MAGIC       WHEN instr(s.DEST, '_AM_') > 0 THEN 'AM'
# MAGIC       WHEN instr(s.DEST, '_FR_') > 0 THEN 'FR'
# MAGIC       WHEN instr(s.DEST, '_IT_') > 0 THEN 'IT'
# MAGIC       WHEN instr(s.DEST, '_KG_') > 0 THEN 'KG'
# MAGIC       WHEN instr(s.DEST, '_GR_') > 0 THEN 'GR'
# MAGIC       WHEN instr(s.DEST, '_CY_') > 0 THEN 'CY'
# MAGIC       WHEN instr(s.DEST, '_ZA_') > 0 THEN 'ZA'
# MAGIC       WHEN instr(s.DEST, '_GB_') > 0 THEN 'GB'
# MAGIC       WHEN instr(s.DEST, '_BA_') > 0 THEN 'BA'
# MAGIC       WHEN instr(s.DEST, '_RU_') > 0 THEN 'RU'
# MAGIC       WHEN instr(s.DEST, '_RO_') > 0 THEN 'RO'
# MAGIC       WHEN instr(s.DEST, '_DE_') > 0 THEN 'DE'
# MAGIC       WHEN instr(s.DEST, '_PL_') > 0 THEN 'PL'
# MAGIC       WHEN instr(s.DEST, '_UA_') > 0 THEN 'UA'
# MAGIC       WHEN instr(s.DEST, '_NL_') > 0 THEN 'NL'
# MAGIC       WHEN instr(s.DEST, '_KZ_') > 0 THEN 'KZ'
# MAGIC       WHEN instr(s.DEST, '_AZ_') > 0 THEN 'AZ'
# MAGIC       ELSE ''
# MAGIC     END AS DEST_BU,
# MAGIC     s.FACTOR,
# MAGIC     v_udt_material_country_idm.CRSSELL_MTRL_ID AS ORIGINAL_ITEM_CB_CODE,
# MAGIC     v_udt_item_prod_mapping.PROD_ITEM AS ORIGINAL_ITEM_IPM
# MAGIC   FROM v_sourcing AS s
# MAGIC     INNER JOIN v_sku ON s.ITEM = v_sku.MTRL_ID
# MAGIC     LEFT OUTER JOIN v_udt_material_country_idm ON s.ITEM = v_udt_material_country_idm.EJDA_MTRL_ID
# MAGIC     LEFT OUTER JOIN v_udt_item_prod_mapping ON s.ITEM = v_udt_item_prod_mapping.DEMAND_ITEM
# MAGIC     LEFT OUTER JOIN v_item ON s.ITEM = v_item.MTRL_ID
# MAGIC   WHERE s.FACTOR <> 0
# MAGIC     AND v_sku.LOC_MTRL_STTS_CDV = 'ACTIVE';

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC CREATE OR REPLACE TEMPORARY VIEW v_cross_border_sales_product_mapping_final
# MAGIC AS SELECT DISTINCT CB_CODE,
# MAGIC     DESCR,
# MAGIC     SOURCE,
# MAGIC     SOURCE_BU,
# MAGIC     DEST,
# MAGIC     DEST_BU,
# MAGIC     FACTOR,
# MAGIC     ORIGINAL_ITEM_CB_CODE,
# MAGIC     ORIGINAL_ITEM_IPM
# MAGIC   FROM v_cross_border_product_mapping
# MAGIC   WHERE SOURCE_BU = 'ES' AND DEST_BU NOT IN ('ES', 'PT')
# MAGIC   UNION ALL
# MAGIC   SELECT DISTINCT CB_CODE,
# MAGIC     DESCR,
# MAGIC     SOURCE,
# MAGIC     SOURCE_BU,
# MAGIC     DEST,
# MAGIC     DEST_BU,
# MAGIC     FACTOR,
# MAGIC     ORIGINAL_ITEM_CB_CODE,
# MAGIC     ORIGINAL_ITEM_IPM
# MAGIC   FROM v_cross_border_product_mapping
# MAGIC   WHERE SOURCE_BU = 'PT' AND DEST_BU NOT IN ('ES', 'PT')

# COMMAND ----------

cbs_df = (spark.table("v_cross_border_sales_product_mapping_final")
          .withColumn("CBS_FLG", 
                      when(col("SOURCE_BU") == col("DEST_BU"), lit("FALSE"))
                      .otherwise(lit("TRUE")))
          .where(col("CBS_FLG") == "TRUE")
          .select(col("CB_CODE"), col("CBS_FLG"))
          .distinct())

# COMMAND ----------

# DBTITLE 1,Create CBS Flag
product = (prolink.join(cbs_df, prolink.PLANG_MTRL_GRP_VAL == cbs_df.CB_CODE, how="left")
                  .drop("CB_CODE")
                  .withColumn("CBS_FLG", 
                              when(col("CBS_FLG").isNull(), lit("FALSE"))
                              .otherwise(col("CBS_FLG"))
                             )
          )

# COMMAND ----------

# DBTITLE 1,Check PK holds for product
check_pk(product, "product", ["PLANG_MTRL_GRP_VAL"], product.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,Check for invalid FLG values. All counts should be 0
valid_FLG_values = ["TRUE", "FALSE"]
print(f"TR_FLG: {product.where(~product.TR_FLG.isin(valid_FLG_values)).count()}")
print(f"CONFECTIONERY_FLG: {product.where(~product.CONFECTIONERY_FLG.isin(valid_FLG_values)).count()}")
print(f"COSTCO_FLG: {product.where(~product.COSTCO_FLG.isin(valid_FLG_values)).count()}")
print(f"CBS_FLG: {product.where(~product.CBS_FLG.isin(valid_FLG_values)).count()}")

# COMMAND ----------

# DBTITLE 1,Add PLANG_MTRL_GRP_DESC for COSTCO products

product = (product
           .withColumn("PLANG_MTRL_GRP_DESC", 
                       when(col("PLANG_MTRL_GRP_NM").startswith("COSTCO"),
                            trim(split(product.PLANG_MTRL_GRP_NM, "COSTCO")[1]))
                       .otherwise(product.PLANG_MTRL_GRP_NM))
          )

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC #### Join Product List and Product Case Tables - Product List Case Joined Table with PRODUCT table

# COMMAND ----------

product_list_case_joined = product_list.join(product_case, ["CD_PRODUCT", "CD_PROD_ORIGIN", "CD_PROD_COUNTRY"], how = "left")

# COMMAND ----------

prolink_product = (product.join(product_list_case_joined, 
                               (product.PLANG_MTRL_GRP_VAL_2 == product_list_case_joined.CD_PRODUCT_CASE), 
                               how = "left"))
prolink_product.count()

# COMMAND ----------

prolink_product

# COMMAND ----------

# MAGIC %md
# MAGIC #### Start mapping prolink with mosaic
# MAGIC 
# MAGIC Note: jid (or joinid) is an attribute added to every mapping as a marker of the type of join used to create that mapping.

# COMMAND ----------

# DBTITLE 1,1 - Map confectionery products [SB]
# map confectionery items from prolink to mosaic
mapped_sb = (prolink_product
              .filter(prolink_product.CONFECTIONERY_FLG == "TRUE")
              .join(mosaic.filter(mosaic.SELLING_SKU_CDV.startswith("SB")), 
                    prolink_product.CD_PROD_SUB_BRAND == mosaic.SELLING_SKU_CDV, 
                    how = "inner")
              .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("SB").alias("jid"))
            )

check_pk(mapped_sb, "mapped_sb", ["PLANG_MTRL_GRP_VAL"], mapped_sb.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,Mosaic from this point on should be non-SB
mosaic_NoSB = mosaic.filter(~mosaic.SELLING_SKU_CDV.startswith("SB"))

# COMMAND ----------

# DBTITLE 1,2 - Map costco products
# map costco items from prolink to mosaic
mapped_costco = (prolink_product
              .filter(prolink_product.COSTCO_FLG == "TRUE")
              .join(mosaic_NoSB, 
                    prolink_product.PLANG_MTRL_GRP_DESC == mosaic_NoSB.SELLING_SKU_NM, 
                    how = "inner")
              .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("COSTCO").alias("jid"))
             )

check_pk(mapped_costco, "mapped_costco", ["PLANG_MTRL_GRP_VAL"], mapped_costco.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,Get products where all flags are FALSE
prolink_product_active = (prolink_product.filter(prolink_product.PLANG_MTRL_STTS_NM == 'ACTIVE'))
prolink_product_false = (prolink_product_active
                         .filter((prolink_product_active.COSTCO_FLG == "FALSE") 
                                 & (prolink_product_active.TR_FLG == "FALSE") 
                                 & (prolink_product_active.CBS_FLG == "FALSE") 
                                 & (prolink_product_active.CONFECTIONERY_FLG == "FALSE"))
                        )

check_pk(prolink_product_false, "prolink_product_false", ["PLANG_MTRL_GRP_VAL"], prolink_product_false.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### 3 - Map products where all flags are FALSE

# COMMAND ----------

# DBTITLE 1,Create empty mapping table
mapped_false = (prolink_product_active
                .withColumn("jid", lit(None))
                .withColumn("SELLING_SKU_CDV_MAPPED", lit(None)))

check_pk(mapped_false, "mapped_false", ["PLANG_MTRL_GRP_VAL"], mapped_false.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,3.1. Map PLANG_MTRL_GRP_VAL_2 == SELLING_SKU_CDV
mapped_dmdunit = (prolink_product_false
                   .join(mosaic_NoSB,
                         prolink_product_false.PLANG_MTRL_GRP_VAL_2 == mosaic_NoSB.SELLING_SKU_CDV,
                         how="inner")
                   .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("DMD").alias("J"))
               )

check_pk(mapped_dmdunit, "mapped_dmdunit", ["PLANG_MTRL_GRP_VAL"], mapped_dmdunit.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,3.2. Map BEVERAGES == SELLING_SKU_CDV starts with B
mapped_B = (prolink_product_false
                   .filter(col("SRC_CTGY_1_NM") == "BEVERAGES")
                   .join(mosaic_NoSB.filter(mosaic_NoSB.SELLING_SKU_CDV.startswith("B")),
                         concat(lit("B"), prolink_product_false.PLANG_MTRL_GRP_VAL_2) == mosaic_NoSB.SELLING_SKU_CDV,
                         how="inner")
                   .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("B").alias("J"))
                  )

check_pk(mapped_B, "mapped_B", ["PLANG_MTRL_GRP_VAL"], mapped_B.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,3.3. Map CD_PRODUCT_CASE
mapped_pcase = (prolink_product_false
                   .join(mosaic_NoSB,
                         prolink_product_false.CD_PRODUCT_CASE == mosaic_NoSB.SELLING_SKU_CDV,
                         how="inner")
                   .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("CASE").alias("J"))
                  )

check_pk(mapped_pcase, "mapped_pcase", ["PLANG_MTRL_GRP_VAL"], mapped_pcase.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

# DBTITLE 1,3.4. Map CD_PCASE_ALTERNATE_CODE_1
mapped_pcase_alt = (prolink_product_false
                   .join(mosaic_NoSB,
                         prolink_product_false.CD_PCASE_ALTERNATE_CODE_1 == mosaic_NoSB.SELLING_SKU_CDV,
                         how="inner")
                   .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV", lit("ALT").alias("J"))
                  )

check_pk(mapped_pcase_alt, "mapped_pcase_alt", ["PLANG_MTRL_GRP_VAL"], mapped_pcase_alt.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

def left_join_to_mapping(base, add, overwrite=False):
  """
  if overwrite == True:
    base left join add 
    for the same PLANG_MTRL_GRP_VAL, if add SELLING_SKU_CDV has a value, it overwrites the value in base
  else:
    base left join add 
    for the same PLANG_MTRL_GRP_VAL, apply add SELLING_SKU_CDV only if base SELLING_SKU_CDV is null
  """
  if overwrite:
    df =       (base
                  .join(add, "PLANG_MTRL_GRP_VAL", how="left")
                  .withColumn("SELLING_SKU_CDV_MAPPED", 
                              when(add.SELLING_SKU_CDV.isNotNull(), add.SELLING_SKU_CDV)
                              .otherwise(base.SELLING_SKU_CDV_MAPPED)
                             )
                  .withColumn("jid", 
                              when(add.J.isNotNull(), add.J)
                              .otherwise(base.jid)
                             )
                  .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV_MAPPED", "jid")
                 )
  else:
    df =       (base
                  .join(add, "PLANG_MTRL_GRP_VAL", how="left")
                  .withColumn("SELLING_SKU_CDV_MAPPED", 
                              when(base.SELLING_SKU_CDV_MAPPED.isNotNull(), base.SELLING_SKU_CDV_MAPPED)
                              .otherwise(add.SELLING_SKU_CDV)
                             )
                  .withColumn("jid", 
                              when(base.jid.isNotNull(), base.jid)
                              .otherwise(add.J)
                             )
                  .select("PLANG_MTRL_GRP_VAL", "SELLING_SKU_CDV_MAPPED", "jid")
                 )    

  return df

# COMMAND ----------

# 3.1 - apply mapped_dmdunit - overwrite existing
mapped_false = left_join_to_mapping(mapped_false, mapped_dmdunit, overwrite=True)

# COMMAND ----------

# 3.2 - apply mapped_B - overwrite existing
mapped_false = left_join_to_mapping(mapped_false, mapped_B, overwrite=True)

# COMMAND ----------

# 3.3 - apply mapped_pcase - apply if null/do not overwrite
mapped_false = left_join_to_mapping(mapped_false, mapped_pcase, overwrite=False)

# COMMAND ----------

# 3.4 - apply mapped_pcase_alt -- apply if null/do not overwrite
mapped_false = left_join_to_mapping(mapped_false, mapped_pcase_alt, overwrite=False)

# COMMAND ----------

mapped_false = mapped_false.where(col("SELLING_SKU_CDV_MAPPED").isNotNull())

check_pk(mapped_false, "mapped_false", ["PLANG_MTRL_GRP_VAL"], mapped_false.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

mapped_all = (mapped_false.withColumnRenamed("SELLING_SKU_CDV_MAPPED", "SELLING_SKU_CDV")
              .union(mapped_sb)
              .union(mapped_costco)
             )

check_pk(mapped_all, "mapped_all", ["PLANG_MTRL_GRP_VAL"], mapped_all.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

product_mapped = (prolink_product
                  .join(mapped_all, "PLANG_MTRL_GRP_VAL", how="left")
                  .join(mosaic, "SELLING_SKU_CDV", how="left")
                  .drop("mapped_all.PLANG_MTRL_GRP_VAL", "mapped_all.SELLING_SKU_CDV")
                  .join(udt.select("EJDA_MTRL_ID", "APPROVED", "MTRL_TYP_CDV", "MTRL_STTS_NM"),
                        udt.EJDA_MTRL_ID == prolink_product.PLANG_MTRL_GRP_VAL,
                        how="left"
                       )
                 )

check_pk(product_mapped, "product_mapped", ["PLANG_MTRL_GRP_VAL"], product_mapped.PLANG_MTRL_GRP_VAL.isNull())

# COMMAND ----------

#Creating product_mapped DF
product_mapped_final = product_mapped.select(product_mapped.PLANG_MTRL_GRP_VAL.alias("PROD_CD"),
                                               product_mapped.PLANG_MTRL_GRP_NM.alias("PROD_NM"),
                                               product_mapped.MLTPCK_INNR_CNT.alias("UNITSIZE"),
                                               product_mapped.HRCHY_LVL_4_NM.alias("SUBBRND"),
                                               product_mapped.HRCHY_LVL_3_NM.alias("SIZE"),
                                               product_mapped.HRCHY_LVL_2_NM.alias("FLVR"),
                                               product_mapped.PLANG_PROD_TYP_NM.alias("LVL"),
                                               product_mapped.BRND_NM.alias("BRND"),
                                               product_mapped.PLANG_MTRL_BRND_GRP_NM.alias("BRND_GRP"),
                                               product_mapped.SRC_CTGY_1_NM.alias("CTGY"),
                                               product_mapped.PLANG_MTRL_FAT_PCT.alias("FAT"),
                                               product_mapped.PLANG_PROD_KG_QTY.alias("KG"),
                                               product_mapped.PLANG_MTRL_EA_VOL_LITR_QTY.alias("LITRES"),
                                               product_mapped.PLANG_MTRL_PCK_CNTNR_NM.alias("PCK_CNTNR"),
                                               product_mapped.SRC_CTGY_2_NM.alias("PROD_LN"),
                                               product_mapped.PLANG_MTRL_DEL_DT.alias("DEL_DT"),
                                               product_mapped.PLANG_MTRL_STTS_NM.alias("PROD_STTS"),
                                               product_mapped.PLANG_MTRL_EA_PER_CASE_CNT.alias("CS"),
                                               product_mapped.PLANG_MTRL_EA_PER_CASE_STD_CNT.alias("CS2"),
                                               product_mapped.AUTMTC_DMNDFCST_UNIT_CREATN_FLG.alias("CRTDFU"),
                                               product_mapped.PLANG_PROD_8OZ_QTY.alias("8OZ"),
                                               product_mapped.PCK_SIZE_SHRT_NM.alias("SIZE_SHRT"),
                                               product_mapped.PCK_CNTNR_SHRT_NM.alias("PCK_CNTNR_SHRT"),
                                               product_mapped.FLVR_NM.alias("FLVR_SHRT"),
                                               product_mapped.PLANG_MTRL_GRP_DSC.alias("LOCL_DSC"),
                                               product_mapped.CASE_TYP_ID_USE_FLG.alias("CASE_TYP"),
                                               product_mapped.PLANG_MTRL_PRODTN_TCHNLGY_NM.alias("PRODUCTION_TECHNOLOGY"),
                                               product_mapped.SUBBRND_SHRT_NM.alias("SUBBRAND_SHORT"),
                                               product_mapped.SELLING_SKU_CDV,
                                               product_mapped.SELLING_SKU_NM,
                                               product_mapped.MOSAIC_SKU_CDV,
                                               product_mapped.MOSAIC_SKU_NM,
                                               product_mapped.PROD_CTGY_NM,
                                               product_mapped.LEVEL_4_CDV,
                                               product_mapped.LEVEL_4_NM,
                                               product_mapped.MTRL_TYP_CDV,
                                               product_mapped.MTRL_STTS_NM,
                                               product_mapped.TR_FLG,
                                               product_mapped.CONFECTIONERY_FLG,
                                               product_mapped.COSTCO_FLG,
                                               product_mapped.CBS_FLG,
                                               product_mapped.APPROVED,
                                               product_mapped.jid,
                                            )

check_pk(product_mapped_final, "product_mapped_final", ["PROD_CD"], product_mapped_final.PROD_CD.isNull())

# COMMAND ----------

# DBTITLE 1,Analysis: mapping breakdown by Join ID, ERP, CTGY, etc
(product_mapped_final.where(col("MOSAIC_SKU_CDV").isNotNull())\
.select("jid", "PROD_CD", "PROD_NM", "MOSAIC_SKU_CDV", "MOSAIC_SKU_NM", "CTGY", 
        regexp_replace(col("SELLING_SKU_CDV"), "[0-9_]+$", "").alias("CD"), "PROD_CTGY_NM", 
        split(col("PROD_CD"), "_")[1].alias("ERP"),
        regexp_extract(col("MOSAIC_SKU_NM"), "^NPD", 1).alias("NPD"))
.groupBy("jid", "CTGY", "PROD_CTGY_NM", "CD", "NPD", "ERP")
.count()
.select("jid", col("CTGY").alias("PROLINK-SRC_CTGY_1_NM"), col("PROD_CTGY_NM").alias("MOSAIC-PROD_CTGY_NM"), "CD", "NPD", "ERP", "count")
.orderBy("jid", "ERP")
.display())

# COMMAND ----------

#creating the final dataframe after joining with Mosiac_Product
silver_df = (product_mapped_final
                  .join(dfu, product_mapped_final.PROD_CD == dfu.PLANG_MTRL_GRP_VAL, "left")
                  .join(MosaicProductDF, 
                        product_mapped_final.MOSAIC_SKU_CDV == MosaicProductDF.MOSAIC_PRODUCT_PLANG_MTRL_GRP_VAL, 
                        "left")
                  .select( product_mapped_final.PROD_CD,
                           product_mapped_final.PROD_NM,
                           product_mapped_final.UNITSIZE,
                           product_mapped_final.SUBBRND,
                           product_mapped_final.SIZE,
                           product_mapped_final.FLVR,
                           product_mapped_final.LVL,
                           product_mapped_final.BRND,
                           product_mapped_final.BRND_GRP,
                           product_mapped_final.CTGY,
                           product_mapped_final.FAT.cast("float"),
                           product_mapped_final.KG.cast('float'),
                           product_mapped_final.LITRES.cast("float"),
                           product_mapped_final.PCK_CNTNR,
                           product_mapped_final.PROD_LN,
                           when(((product_mapped_final.PROD_STTS == 'ACTIVE') & 
                                 (year(product_mapped_final.DEL_DT)==1970)), lit('9999-01-01T00:00:00.000+0000'))
                           .otherwise(product_mapped_final.DEL_DT).cast('timestamp').alias("DEL_DT"),
                           product_mapped_final.PROD_STTS,
                           product_mapped_final.CS.cast('float'),
                           product_mapped_final.CS2.cast("float"),
                           product_mapped_final.CRTDFU,
                           product_mapped_final["8OZ"].cast('float'),
                           product_mapped_final.SIZE_SHRT,
                           product_mapped_final.PCK_CNTNR_SHRT,
                           product_mapped_final.FLVR_SHRT,
                           product_mapped_final.LOCL_DSC,
                           product_mapped_final.CASE_TYP,
                           product_mapped_final.PRODUCTION_TECHNOLOGY,
                           product_mapped_final.SUBBRAND_SHORT,
                           product_mapped_final.MTRL_TYP_CDV,
                           product_mapped_final.MTRL_STTS_NM,
                           product_mapped_final.MOSAIC_SKU_CDV,
                           product_mapped_final.MOSAIC_SKU_NM,
                           product_mapped_final.LEVEL_4_CDV,
                           product_mapped_final.LEVEL_4_NM,
                           MosaicProductDF.STDRPRT_SUB_CTGY_CDV.cast("int").alias("STDRPRT_SUB_CTGY_CDV"),
                           MosaicProductDF.STDRPRT_SUB_CTGY_NM.alias("STDRPRT_SUB_CTGY_NM"),
                           MosaicProductDF.STDRPRT_SGMNT_CDV.cast("int").alias("STDRPRT_SGMNT_CDV"),
                           MosaicProductDF.STDRPRT_SGMNT_NM.alias("STDRPRT_SGMNT_NM"),
                           MosaicProductDF.STDRPRT_SUB_SGMNT_CDV.cast("int").alias("STDRPRT_SUB_SGMNT_CDV"),
                           MosaicProductDF.STDRPRT_SUB_SGMNT_NM.alias("STDRPRT_SUB_SGMNT_NM"),
                           dfu.MU.alias("MU")
                         )
                 )

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

check_pk(silver_df, "silver_df", ["PROD_CD"], silver_df.PROD_CD.isNull())

# COMMAND ----------

mapped = silver_df.where(silver_df.MOSAIC_SKU_CDV.isNotNull()).count()
total = silver_df.count()
print(f"Rows mapped: {mapped}")
print(f"Rows total: {total}")
print(f"% Rows mapped: {mapped/total}")

# COMMAND ----------

#Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.product_master") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.product_master

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

