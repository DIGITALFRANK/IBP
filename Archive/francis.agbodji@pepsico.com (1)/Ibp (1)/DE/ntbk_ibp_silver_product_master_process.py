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

# COMMAND ----------

print(udt_material)
print(alteryx_product)

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

#Reading the EDW Product Master source data from bronze layer
productDF = spark.read.format("delta").option("versionAsOf", product_latest_version).load(srcPath)
display(productDF)

# COMMAND ----------

###Reading the dependent datasets into dataframes
udtMaterialCountryDF = spark.read.format('delta').load(udt_material)
mosaicDF = spark.read.format('delta').load(alteryx_product)

# COMMAND ----------

#Creating additional filtered dataframes
ProlinkProductMaster = productDF.filter("SYS_ID = 564 and right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06')")
MosaicProductDF = productDF.filter("sys_id = 695").select("DW_PLANG_MTRL_UNIT_ID", "PLANG_MTRL_GRP_VAL", 'STDRPRT_SUB_CTGY_CDV', 'STDRPRT_SUB_CTGY_NM', 'STDRPRT_SGMNT_CDV', 'STDRPRT_SGMNT_NM', 'STDRPRT_SUB_SGMNT_CDV', 'STDRPRT_SUB_SGMNT_NM').distinct()

# COMMAND ----------

print(ProlinkProductMaster.count())
print(MosaicProductDF.count())

# COMMAND ----------

#Creating Prolink_Product DF
ProlinkProductDF = ProlinkProductMaster.select(ProlinkProductMaster.PLANG_MTRL_GRP_VAL.alias("PROD_CD"),
                                               ProlinkProductMaster.PLANG_MTRL_GRP_NM.alias("PROD_NM"),
                                               ProlinkProductMaster.MLTPCK_INNR_CNT.alias("UNITSIZE"),
                                               ProlinkProductMaster.HRCHY_LVL_4_NM.alias("SUBBRND"),
                                               ProlinkProductMaster.HRCHY_LVL_3_NM.alias("SIZE"),
                                               ProlinkProductMaster.HRCHY_LVL_2_NM.alias("FLVR"),
                                               ProlinkProductMaster.PLANG_PROD_TYP_NM.alias("LVL"),
                                               ProlinkProductMaster.BRND_NM.alias("BRND"),
                                               ProlinkProductMaster.PLANG_MTRL_BRND_GRP_NM.alias("BRND_GRP"),
                                               ProlinkProductMaster.SRC_CTGY_1_NM.alias("CTGY"),
                                               ProlinkProductMaster.PLANG_MTRL_FAT_PCT.alias("FAT"),
                                               ProlinkProductMaster.PLANG_PROD_KG_QTY.alias("KG"),
                                               ProlinkProductMaster.PLANG_MTRL_EA_VOL_LITR_QTY.alias("LITRES"),
                                               ProlinkProductMaster.PLANG_MTRL_PCK_CNTNR_NM.alias("PCK_CNTNR"),
                                               ProlinkProductMaster.SRC_CTGY_2_NM.alias("PROD_LN"),
                                               ProlinkProductMaster.PLANG_MTRL_DEL_DT.alias("DEL_DT"),
                                               ProlinkProductMaster.PLANG_MTRL_STTS_NM.alias("PROD_STTS"),
                                               ProlinkProductMaster.PLANG_MTRL_EA_PER_CASE_CNT.alias("CS"),
                                               ProlinkProductMaster.PLANG_MTRL_EA_PER_CASE_STD_CNT.alias("CS2"),
                                               ProlinkProductMaster.AUTMTC_DMNDFCST_UNIT_CREATN_FLG.alias("CRTDFU"),
                                               ProlinkProductMaster.PLANG_PROD_8OZ_QTY.alias("8OZ"),
                                               ProlinkProductMaster.PCK_SIZE_SHRT_NM.alias("SIZE_SHRT"),
                                               ProlinkProductMaster.PCK_CNTNR_SHRT_NM.alias("PCK_CNTNR_SHRT"),
                                               ProlinkProductMaster.FLVR_NM.alias("FLVR_SHRT"),
                                               ProlinkProductMaster.PLANG_MTRL_GRP_DSC.alias("LOCL_DSC"),
                                               ProlinkProductMaster.CASE_TYP_ID_USE_FLG.alias("CASE_TYP"),
                                               ProlinkProductMaster.PLANG_MTRL_PRODTN_TCHNLGY_NM.alias("PRODUCTION_TECHNOLOGY"),
                                               ProlinkProductMaster.SUBBRND_SHRT_NM.alias("SUBBRAND_SHORT"))

# COMMAND ----------

display(udtMaterialCountryDF)

# COMMAND ----------

#Operations on UDT_Material
CTE_df = udtMaterialCountryDF.select(udtMaterialCountryDF.EJDA_MTRL_ID.alias('ITEM'), udtMaterialCountryDF.MTRL_XREF_ID.alias('MATERIAL'), udtMaterialCountryDF.MKT_CDV.alias('JDA_CODE'), udtMaterialCountryDF.APPROVED, udtMaterialCountryDF.MTRL_STTS_NM.alias('STATUS'), udtMaterialCountryDF.CTRY_CDV.alias('SOURCE_SYSTEM_COUNTRY'), udtMaterialCountryDF.MTRL_TYP_CDV, udtMaterialCountryDF.MTRL_STTS_NM)

display(CTE_df)   

# COMMAND ----------

CTE2_df_temp = CTE_df.filter(CTE_df.JDA_CODE.isin('04','05','06')).groupBy("ITEM", "MATERIAL", "JDA_CODE", "APPROVED", "STATUS", "MTRL_TYP_CDV", "MTRL_STTS_NM").agg(countDistinct("SOURCE_SYSTEM_COUNTRY").alias("Source_System_Cnt"))

CTE2_df = CTE2_df_temp.select(CTE2_df_temp.ITEM, CTE2_df_temp.MATERIAL, CTE2_df_temp.JDA_CODE, CTE2_df_temp.APPROVED, CTE2_df_temp.STATUS, CTE2_df_temp.MTRL_TYP_CDV, CTE2_df_temp.MTRL_STTS_NM, CTE2_df_temp.Source_System_Cnt, row_number().over( Window.partitionBy("ITEM").orderBy(col("APPROVED").desc(), col("STATUS").desc())).alias("ROW_Id"))

display(CTE2_df)

# COMMAND ----------

#Creating MaterialCountry DF
materialCountry = CTE2_df.filter(CTE2_df.ROW_Id == 1).select(CTE2_df.ITEM, CTE2_df.MATERIAL, CTE2_df.JDA_CODE, CTE2_df.APPROVED, CTE2_df.STATUS, CTE2_df.MTRL_TYP_CDV, CTE2_df.MTRL_STTS_NM, CTE2_df.Source_System_Cnt, CTE2_df.ROW_Id)

print(materialCountry.count())

display(materialCountry)

# COMMAND ----------

#Creating Filtered Product Dataframe
CTE_ITEM_df = ProlinkProductDF.filter(ProlinkProductDF.LVL == 'SB-S-FL-ITEM').filter(substring_index(ProlinkProductDF.PROD_CD, '_', -1).isin('04','05','06')).withColumn('Selling_SKU', substring_index(ProlinkProductDF.PROD_CD, '_', 1)).withColumn('ERP_ID', substring_index(ProlinkProductDF.PROD_CD, '_', -1))

print(CTE_ITEM_df.count())

CTE_ITEM2_df = CTE_ITEM_df.groupBy("Selling_SKU").agg(count("Selling_SKU").alias("Distinct_DMDUNIT"))
CTE_ITEM2_df = CTE_ITEM2_df.select(CTE_ITEM2_df.Selling_SKU, CTE_ITEM2_df.Distinct_DMDUNIT)

display(CTE_ITEM2_df)

fil_df_join_cond = [CTE_ITEM_df.Selling_SKU == CTE_ITEM2_df.Selling_SKU]

#filteredProductDF = CTE_ITEM_df.join(CTE_ITEM2_df, fil_df_join_cond, "left").filter(CTE_ITEM_df.PROD_STTS == 'ACTIVE').withColumn("Contains_Letters", when(CTE_ITEM_df.Selling_SKU.like('%[A-Za-z]%'),1).otherwise(0)).drop(CTE_ITEM2_df.Selling_SKU)

filteredProductDF = CTE_ITEM_df.join(CTE_ITEM2_df, fil_df_join_cond, "left").withColumn("Contains_Letters", when(CTE_ITEM_df.Selling_SKU.like('%[A-Za-z]%'),1).otherwise(0)).drop(CTE_ITEM2_df.Selling_SKU)

# COMMAND ----------

filteredProductDF = filteredProductDF.withColumn("DMDUNIT_Row_Id", row_number().over( Window.partitionBy("Selling_SKU").orderBy(col("PROD_CD"))))

display(filteredProductDF)

# COMMAND ----------

print(filteredProductDF.count())
print(len(filteredProductDF.columns))

# COMMAND ----------

#Filtering mosiacDF and selecting required columns
mosiacDF_filtered = mosaicDF.filter(mosaicDF.SOURCE_SYSTEM_ID == '01').filter(mosaicDF.SPOT_ATTRIBUTE_10_CDV == '1').select(mosaicDF.SELLING_SKU_CDV, mosaicDF.SELLING_SKU_NM, mosaicDF.MOSAIC_SKU_CDV, mosaicDF.MOSAIC_SKU_NM, mosaicDF.PROD_CTGY_NM, mosaicDF.LEVEL_4_CDV, mosaicDF.LEVEL_4_NM)

# COMMAND ----------

prolink_join_cond  = [filteredProductDF.PROD_CD == materialCountry.ITEM]

#prolink_df = filteredProductDF.join(materialCountry, prolink_join_cond, "left").filter((materialCountry.STATUS == 'ACTIVE') & ((filteredProductDF.Distinct_DMDUNIT == 1) | ((filteredProductDF.Distinct_DMDUNIT != 1) & (filteredProductDF.ERP_ID != '04')))).drop(materialCountry.ITEM).drop(materialCountry.MATERIAL).drop(materialCountry.JDA_CODE).drop(materialCountry.Source_System_Cnt).drop(materialCountry.ROW_Id)

prolink_df = filteredProductDF.join(materialCountry, prolink_join_cond, "left").filter(((filteredProductDF.Distinct_DMDUNIT == 1) | ((filteredProductDF.Distinct_DMDUNIT != 1) & (filteredProductDF.ERP_ID != '04')))).drop(materialCountry.ITEM).drop(materialCountry.MATERIAL).drop(materialCountry.JDA_CODE).drop(materialCountry.Source_System_Cnt).drop(materialCountry.ROW_Id)

# COMMAND ----------

#prolink_df_2 = filteredProductDF.join(materialCountry, prolink_join_cond, "left").filter((materialCountry.STATUS == 'ACTIVE') & (filteredProductDF.Distinct_DMDUNIT != 1) & (filteredProductDF.ERP_ID == '04')).drop(materialCountry.ITEM).drop(materialCountry.MATERIAL).drop(materialCountry.JDA_CODE).drop(materialCountry.Source_System_Cnt).drop(materialCountry.ROW_Id)

prolink_df_2 = filteredProductDF.join(materialCountry, prolink_join_cond, "left").filter((filteredProductDF.Distinct_DMDUNIT != 1) & (filteredProductDF.ERP_ID == '04')).drop(materialCountry.ITEM).drop(materialCountry.MATERIAL).drop(materialCountry.JDA_CODE).drop(materialCountry.Source_System_Cnt).drop(materialCountry.ROW_Id)

# COMMAND ----------

#creating the Prolink_Mosaic_Mapping dataframe
fianl_join_1 = [prolink_df.Selling_SKU == mosiacDF_filtered.SELLING_SKU_CDV]
mappingProlinkMosaic_1 = prolink_df.join(mosiacDF_filtered, fianl_join_1, "left")

fianl_join_2 = [concat(lit("B"), prolink_df_2.Selling_SKU) == mosiacDF_filtered.SELLING_SKU_CDV]
mappingProlinkMosaic_2 = prolink_df_2.join(mosiacDF_filtered, fianl_join_2, "left")

mappingProlinkMosaic = mappingProlinkMosaic_1.unionAll(mappingProlinkMosaic_2)

# COMMAND ----------

print(mappingProlinkMosaic.count())
print(len(mappingProlinkMosaic.columns))
display(mappingProlinkMosaic)

# COMMAND ----------

display(MosaicProductDF)

# COMMAND ----------

#creating the final dataframe after joining with Mosiac_Product
silver_df = mappingProlinkMosaic.join(MosaicProductDF, mappingProlinkMosaic.MOSAIC_SKU_CDV == MosaicProductDF.PLANG_MTRL_GRP_VAL, "left").select(mappingProlinkMosaic.PROD_CD,
  mappingProlinkMosaic.PROD_NM,
                                       mappingProlinkMosaic.UNITSIZE,
                                       mappingProlinkMosaic.SUBBRND,
                                       mappingProlinkMosaic.SIZE,
                                       mappingProlinkMosaic.FLVR,
                                       mappingProlinkMosaic.LVL,
                                       mappingProlinkMosaic.BRND,
                                       mappingProlinkMosaic.BRND_GRP,
                                       mappingProlinkMosaic.CTGY,
                                       mappingProlinkMosaic.FAT,
                                       mappingProlinkMosaic.KG.cast('float'),
                                       mappingProlinkMosaic.LITRES,
                                       mappingProlinkMosaic.PCK_CNTNR,
                                       mappingProlinkMosaic.PROD_LN,
                                       when(((mappingProlinkMosaic.PROD_STTS == 'ACTIVE') & (year(mappingProlinkMosaic.DEL_DT)==1970)), lit('9999-01-01T00:00:00.000+0000')).otherwise(mappingProlinkMosaic.DEL_DT).cast('timestamp').alias("DEL_DT"),
                                       mappingProlinkMosaic.PROD_STTS,
                                       mappingProlinkMosaic.CS.cast('float'),
                                       mappingProlinkMosaic.CS2,
                                       mappingProlinkMosaic.CRTDFU,
                                       mappingProlinkMosaic["8OZ"].cast('float'),
                                       mappingProlinkMosaic.SIZE_SHRT,
                                       mappingProlinkMosaic.PCK_CNTNR_SHRT,
                                       mappingProlinkMosaic.FLVR_SHRT,
                                       mappingProlinkMosaic.LOCL_DSC,
                                       mappingProlinkMosaic.CASE_TYP,
                                       mappingProlinkMosaic.PRODUCTION_TECHNOLOGY,
                                       mappingProlinkMosaic.SUBBRAND_SHORT,
                                       mappingProlinkMosaic.MOSAIC_SKU_CDV,
                                       mappingProlinkMosaic.MOSAIC_SKU_NM,                                                                                                         
									   mappingProlinkMosaic.LEVEL_4_CDV,
									   mappingProlinkMosaic.LEVEL_4_NM,
									   MosaicProductDF.STDRPRT_SUB_CTGY_CDV.alias("STDRPRT_SUB_CTGY_CDV"),
                                       MosaicProductDF.STDRPRT_SUB_CTGY_NM.alias("STDRPRT_SUB_CTGY_NM"),
                                       MosaicProductDF.STDRPRT_SGMNT_CDV.alias("STDRPRT_SGMNT_CDV"),
                                       MosaicProductDF.STDRPRT_SGMNT_NM.alias("STDRPRT_SGMNT_NM"),
                                       MosaicProductDF.STDRPRT_SUB_SGMNT_CDV.alias("STDRPRT_SUB_SGMNT_CDV"),
                                       MosaicProductDF.STDRPRT_SUB_SGMNT_NM.alias("STDRPRT_SUB_SGMNT_NM"),
                                       mappingProlinkMosaic.MTRL_TYP_CDV,
                                       mappingProlinkMosaic.MTRL_STTS_NM)

# COMMAND ----------

silver_df.printSchema()

# COMMAND ----------

#silver_df = silver_df \
#  .withColumn("8OZ" ,
#              silver_df["Course_Fees"]
#              .cast(FloatType()))   \
#  .withColumn("KG",
#              silver_df["Payment_Done"]
#              .cast(StringType()))    \
#  .withColumn("CS"  ,
#              silver_df["Start_Date"]
#              .cast(DateType())) \

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

display(silver_df.select(col('MTRL_TYP_CDV')).distinct())

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

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

