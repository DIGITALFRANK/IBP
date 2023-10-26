# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"
#storage_account = "cdodevextrblob"

# COMMAND ----------

storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
#spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization","true")

# COMMAND ----------

pep = ["SANTA.ANA","LAY.S", "MUNCHOS", "RUFFLES", "SUNBITES", "CHEETOS", "FRITOS", "3.D.S", "DORITOS", "MATUTANO", "BOCA.BITS","PALA.PALA","MIXUPS"]
nielsen = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/portugal/')
nielsen = nielsen.withColumn("Year",expr("substring(Week, 5, 2)"))
nielsen = nielsen.filter(nielsen.PRODUTO != "SENSITIVE")
nielsen = nielsen.filter(nielsen.PRODUTO != "ND")
nielsen = nielsen.filter(nielsen.MARCA.isin(pep))

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

#Load Silver layer Product Master, which has dmdunits and the remaining product info
prod = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/product-master")
prod = prod.withColumnRenamed("PROD_CD", "DMDUNIT")
prod = prod.withColumn("DMDUNIT2", when((prod.DMDUNIT.contains("_")), expr("substring(DMDUNIT, 1, length(DMDUNIT)-3)")).otherwise(col("DMDUNIT")))
prod2 = prod.select("DMDUNIT", "DMDUNIT2", "CTGY").distinct()


# COMMAND ----------

rename_pt_dict = {"Vendas_em_Valor":"Value_Sales",
"Vendas_em_Unidades":"Unit_Sales",
"Vendas_em_Quantidade":"Volume_Sales",
"Nmero_de_Lojas_do_Universo":"Total_Points_Of_Sales_Dealing",
"Vendas_em_Unidades_com_promoo":"Unit_Promo_Sales",
"Vendas_em_Valor_com_promoo":"Value_Promo_Sales",
"Vendas_em_Valor_exposio_especial_e_folleto":"Value_Sales_SEL",
"Vendas_em_Valor_exposio_especial_sem__folleto":"Value_Sales_SE",
"Vendas_em_Valor_folheto_sem_exposio_especial":"Value_Sales_L",
"Vendas_em_Valor_reduo_de_preo":"Value_Sales_TPR",
"Distribuio_Numrica_S":"Numeric_Distribution",
"Distribuio_Ponderada_S":"Wtd_Distribution",
"Distribuio_Ponderada_S_com_promoo":"Wtd_Distribution_Promo",
"Dist_Ponderada_S_exposio_especial_e_folleto":"Wtd_Distribution_SEL",
"Dist_Ponderada_S_exposio_especial_sem_folleto":"Wtd_Distribution_SE",
"Dist_Ponderada_S_folheto_sem_exposiao_especial":"Wtd_Distribution_L",
"Distribuio_Ponderada_S_reduo_de_preo":"Wtd_Distribution_TPR",
"Preo_por_Quantidade":"Price_Per_Qty",
"Promo_Price_pr_LKG":"Promo_Price_Per_Volume",
"Vendas_Base_em_Unidades":"Base_Sales_Unit",
"Vendas_Base_em_Quantidade":"Base_Sales_Qty",
"Vendas_Base_em_Valor":"Base_Sales_Value",
"Vendas_em_Quantidade_com_promoo":"Qty_Sales_Promo",
"Vendas_em_Quantidade_exposio_especial_e_folleto":"Qty_Sales_SEL",
"Vendas_em_Quantidade_exposio_especial_sem_folleto":"Qty_Sales_SE",
"Vendas_em_Quantidade_folheto_sem_exposiao_especial":"Qty_Sales_L",
"Vendas_em_Quantidade_reduo_de_preo":"Qty_Sales_TPR",
}

for old_name, new_name in rename_pt_dict.items():
    nielsen = nielsen.withColumnRenamed(old_name, new_name)
display(nielsen)

# COMMAND ----------

# MAGIC %md #### Get EAN code

# COMMAND ----------

nielsen_with_eans = nielsen.withColumn("ean_code", nielsen["PRODUTO"])

# COMMAND ----------

# MAGIC %md ####EAN code mapping

# COMMAND ----------

nielsen2dmd = nielsen_with_eans.join(ean_to_dmd.drop("PROCESS_DATE"), ean_to_dmd.EAN == nielsen_with_eans.ean_code, how="left")

# COMMAND ----------

# MAGIC %md #### DMDUNIT mapping

# COMMAND ----------

print(nielsen2dmd.count())
nielsen2dmd_pt_snacks = nielsen2dmd.join(prod2, nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M==prod2.DMDUNIT2, how="left").drop("DMDUNIT2")
nielsen2dmd_pt_snacks = nielsen2dmd_pt_snacks.filter(col("DMDUNIT").isNotNull())
print(nielsen2dmd_pt_snacks.count())
display(nielsen2dmd_pt_snacks.groupby("CTGY").agg(count("*")))

# COMMAND ----------

cols_keywords = ["Qty", "Base", "Promo", "Price", "Wtd", "Value", "Unit", "Volume", "Total", "Numeric"]
cols_to_agg = []
for i in cols_keywords:
  cols_to_agg = cols_to_agg + [x for x in nielsen2dmd_pt_snacks.columns if i in x]
cols_to_agg = list(set(cols_to_agg))
print(cols_to_agg)

# COMMAND ----------

nielsen2dmd_pt_snacks = nielsen2dmd_pt_snacks.withColumn('WEEK', regexp_replace('WEEK', 'S', ''))\
.withColumn('WEEK', regexp_replace('WEEK', ' ', '')).withColumnRenamed("WEEK", "Week_Of_Year")

nielsen2dmd_pt_snacks = nielsen2dmd_pt_snacks.withColumn("COUNTRY", lit("PT"))

group_cols = ["Week_Of_Year", "DMDUNIT", "COUNTRY"] 
nielsen2dmd_pt_snacks = nielsen2dmd_pt_snacks.groupBy(group_cols).agg(*[sum(c).alias(c) for c in cols_to_agg])
display(nielsen2dmd_pt_snacks)

# COMMAND ----------

print(nielsen2dmd_pt_snacks.count())
print(nielsen2dmd_pt_snacks.select("Week_Of_Year", "DMDUNIT").distinct().count())

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##SPAIN

# COMMAND ----------

rename_es_dict = {'Ventas_en_unidades':'Unit_Sales',
 'Ventas_en_unidades_equivalentes':  'Sales_In_Equivalent_Units',
 'Ventas_en_unidades_equivalentes_en_prom':  'Ventas_en_unidades_equivalentes_en_prom',
 'Ventas_en_valor':  'Value_Sales',
 'Ventas_en_valor_en_promocion':  'Sales_in_value_in_promotion',
 'Universo':  'Universo',
 'Ventas_en_unidades_display__folleto':  'Sales_In_Units_Display_and_Brochure',
 'Ventas_en_unidades_display':  'Sales_In_Units_Display',
 'Ventas_en_unidades_folleto':  'Sales_In_Units_Brochure',
 'Ventas_en_unidades_reduccion_temp_de_p':  'Value_Sales_TPR',
 'Ventas_en_unidades_resto_promociones':  'Sales_In_Equivalent_Units_Rest_Promo',
 'Ventas_en_unidades_multicompra':  'Sales_in_multi_purchase_units',
 'Ventas_en_unidades_equivalentes_display__folleto':  'Sales_In_Equivalent_Units_Display_Brochure',
 'Ventas_en_unidades_equivalentes_display':  'Sales_In_Equivalent_Units_Display',
 'Ventas_en_unidades_equivalentes_folleto':  'Sales_In_Equivalent_Units_Brochure',
 'Ventas_en_unidades_equivalentes_reduccion_temp_De_precio':  'Sales_In_Equivalent_Units_Price_Temp_Reduction',
 'Ventas_en_unidades_equivalentes_resto_promo':  'Ventas_en_unidades_equivalentes_resto_promo',
 'Ventas_en_unidades_equivalentes_multicompra':  'Sales_in_multi_buy_equivalent_units',
 'Ventas_en_valor_display__folleto':  'Value_Sales_SEL',
 'Ventas_en_valor_display':  'Value_Sales_SE',
 'Ventas_en_valor_folleto':  'Value_Sales_L',
 'Ventas_en_valor_reduccion_temp_de_prec':  'Sales_in_value_reduction_temp_from_prec',
 'Ventas_en_valor_resto_promociones':  'Value_Promo_Sales',
 'Ventas_en_valor_multicompra':  'Sales_in_multibuy_value',
 'Distribucion_numerica_S':  'Numeric_Distribution',
 'Distribucion_numerica_S_en_promocion':  'Distribucion_numerica_S_en_promocion',
 'Distribucion_ponderada_S':  'Wtd_Distribution',
 'Distribucion_ponderada_S_en_promocion':  'Wtd_Distribution_Promo',
 'Distribucion_ponderada_S_display__folleto':  'Wtd_Distribution_SEL',
 'Distribucion_ponderada_S_display':  'Wtd_Distribution_SE',
 'Distribucion_ponderada_S_folleto':  'Weighted_Distribution_S_Brochure',
 'Distribucion_ponderada_S_reduccion_temp_de_prec':  'Wtd_Distribution_TPR',
 'Distribucion_ponderada_S_resto_promocion':  'Weighted_Distribution_S_Rest_Promotion',
 'Distribucion_ponderada_S_promocion_regalo':  'Weighted_Distribution_S_Present_Promotion',
 'Distribucion_ponderada_S_multicompra':  'Weighted_Spread_S_multi-purchase',
 'Precio':  'Price_Per_Qty',
 'Precio_en_promocion':  'Promo_Price_Per_Volume',
 'Ventas_unidades_baseline':  'Base_Sales_Unit',
 'Ventas_unidades_equivalentes_baseline':  'Base_Sales_Qty',
 'Ventas_en_valor_baseline':  'Base_Sales_Value',
 'Distribucion_numerica_4S':  'Distribucion_numerica_4S',
 'Distribucion_ponderada_4S':  'Distribucion_ponderada_4S'
  }

# COMMAND ----------

# Load Nielsen (no eans yet)
neilsen_spain_snacks_df = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/spain/')
# neilsen_spain_df = neilsen_spain_df.withColumn("Year",expr("substring(WEEK, 11, 2)"))

#Load the mapping file to add eans to Nielsen
sergi_mapping = spark.read.csv("/FileStore/tables/temp/Sergi_Mappingfile_onlypep.csv", header="true", inferSchema="true")

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

#Load Silver layer Product Master, which has dmdunits and the remaining product info
prod = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/product-master")
prod = prod.withColumnRenamed("PROD_CD", "DMDUNIT")
prod = prod.withColumn("DMDUNIT2", when((prod.DMDUNIT.contains("_")), expr("substring(DMDUNIT, 1, length(DMDUNIT)-3)")).otherwise(col("DMDUNIT")))
prod2 = prod.select("DMDUNIT", "DMDUNIT2", "CTGY").distinct()


# COMMAND ----------

neilsen_spain_snacks_df =  neilsen_spain_snacks_df.filter(neilsen_spain_snacks_df.Ventas_en_unidades > 0)
neilsen_spain_snacks_df.select('LDESC').distinct().count()

# COMMAND ----------

# Rename Columns
for old_name, new_name in rename_es_dict.items():
    neilsen_spain_snacks_df = neilsen_spain_snacks_df.withColumnRenamed(old_name, new_name)
display(neilsen_spain_snacks_df)

# COMMAND ----------

# MAGIC %md ###Sergi file join 

# COMMAND ----------

# Add eans to Nielsen
neilsen_spain_snacks_with_eans = neilsen_spain_snacks_df.join(sergi_mapping, neilsen_spain_snacks_df.TAG==sergi_mapping.tag_code, how="inner")

# COMMAND ----------

# MAGIC %md ###EAN code join 

# COMMAND ----------

nielsen2dmd = neilsen_spain_snacks_with_eans.join(ean_to_dmd, ean_to_dmd.EAN == neilsen_spain_snacks_with_eans.ean_code, how="left")

# COMMAND ----------

# MAGIC %md ###DMDUNIT join 

# COMMAND ----------

print(nielsen2dmd.count())
nielsen2dmd_es_snacks = nielsen2dmd.join(prod2, nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M==prod2.DMDUNIT2, how="left").drop("DMDUNIT2")
nielsen2dmd_es_snacks = nielsen2dmd_es_snacks.filter(col("DMDUNIT").isNotNull())
print(nielsen2dmd_es_snacks.count())
display(nielsen2dmd_es_snacks.filter(col("DMDUNIT").isNotNull()))

# COMMAND ----------

cols_keywords = ["Qty", "Base", "Promo", "Price", "Wtd", "Value", "Unit", "Volume", "Total", "Numeric", "Sales", "Ventas", "Distribucion", "Weighted", "Universo"]
cols_to_agg = []
for i in cols_keywords:
  cols_to_agg = cols_to_agg + [x for x in nielsen2dmd_es_snacks.columns if i in x]
cols_to_agg = list(set(cols_to_agg))
print(cols_to_agg)

# COMMAND ----------

nielsen2dmd_es_snacks = nielsen2dmd_es_snacks.withColumn("WEEK",expr("substring(WEEK, length(WEEK)-5, length(WEEK))"))\
.withColumn("WEEK_no",expr("substring(WEEK, 2, length(WEEK)-4)"))\
.withColumn("Year",expr("substring(WEEK, length(WEEK)-1, length(WEEK))"))\
.withColumn("Week_Of_Year", concat_ws("", lit("20"), "Year",'WEEK_no'))

nielsen2dmd_es_snacks = nielsen2dmd_es_snacks.withColumn("COUNTRY", lit("ES"))

group_cols = ["Week_Of_Year", "DMDUNIT", "COUNTRY"] 
nielsen2dmd_es_snacks = nielsen2dmd_es_snacks.groupBy(group_cols).agg(*[sum(c).alias(c) for c in cols_to_agg])
display(nielsen2dmd_es_snacks)

# COMMAND ----------

print(nielsen2dmd_es_snacks.count())
print(nielsen2dmd_es_snacks.select("Week_Of_Year", "DMDUNIT").distinct().count())

# COMMAND ----------

# MAGIC %md #### Beverages Spain

# COMMAND ----------


nielsen = spark.read.format("delta").load('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/access-db/ibp-poc/syndicated-pos-historical/spain-beverages')
nielsen = nielsen.where((nielsen.COMPANIA == "CIA PEPSICO"))
beverage_mapping = spark.read.csv("/FileStore/tables/temp/Beverage_Mappingfile.csv", header="true", inferSchema="true")

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


# COMMAND ----------

nielsen =  nielsen.filter(nielsen.Ventas_en_unidades > 0)
nielsen.select('LDESC').distinct().count()

# COMMAND ----------

# Rename Columns
for old_name, new_name in rename_es_dict.items():
    nielsen = nielsen.withColumnRenamed(old_name, new_name)
display(nielsen)

# COMMAND ----------

# MAGIC %md ####Get EAN code

# COMMAND ----------

nielsen_with_eans = nielsen.join(beverage_mapping, nielsen.LDESC == beverage_mapping.description, how="left")

# COMMAND ----------

# MAGIC %md #### EAN code join 

# COMMAND ----------

nielsen2dmd = nielsen_with_eans.join(ean_to_dmd, ean_to_dmd.EAN == nielsen_with_eans.ean_code, how="left")

# COMMAND ----------

# MAGIC %md ###DMDUNIT join 

# COMMAND ----------

print(nielsen2dmd.count())
nielsen2dmd_es_bev = nielsen2dmd.join(prod2, nielsen2dmd.PCCD_ITEM_G2MPCCD_CASE_G2M==prod2.DMDUNIT2, how="left").drop("DMDUNIT2")
nielsen2dmd_es_bev = nielsen2dmd_es_bev.filter(col("DMDUNIT").isNotNull())
print(nielsen2dmd_es_bev.count())

# COMMAND ----------

cols_keywords = ["Qty", "Base", "Promo", "Price", "Wtd", "Value", "Unit", "Volume", "Total", "Numeric", "Sales", "Ventas", "Distribucion", "Weighted", "Universo", "Promedio", "TDP"]
cols_to_agg = []
for i in cols_keywords:
  cols_to_agg = cols_to_agg + [x for x in nielsen2dmd_es_bev.columns if i in x]
cols_to_agg = list(set(cols_to_agg))
print(cols_to_agg)

# COMMAND ----------


nielsen2dmd_es_beverages = nielsen2dmd_es_bev.withColumn("WEEK",expr("substring(WEEK, length(WEEK)-5, length(WEEK))"))\
.withColumn("WEEK_no",expr("substring(WEEK, 2, length(WEEK)-4)"))\
.withColumn("Year",expr("substring(WEEK, length(WEEK)-1, length(WEEK))"))\
.withColumn("Week_Of_Year", concat_ws("", lit("20"), "Year",'WEEK_no'))

nielsen2dmd_es_beverages = nielsen2dmd_es_beverages.withColumn("COUNTRY", lit("ES"))

group_cols = ["Week_Of_Year", "DMDUNIT", "COUNTRY"]
nielsen2dmd_es_beverages = nielsen2dmd_es_beverages.groupBy(group_cols).agg(*[sum(c).alias(c) for c in cols_to_agg])
display(nielsen2dmd_es_beverages)

# COMMAND ----------

print(nielsen2dmd_es_beverages.count())
print(nielsen2dmd_es_beverages.select(["Week_Of_Year", "DMDUNIT", "COUNTRY"]).distinct().count())

# COMMAND ----------

# MAGIC %md ## Joining Spain & Portugal 

# COMMAND ----------

print("nielsen2dmd_pt_snacks count:", nielsen2dmd_pt_snacks.count())
print("nielsen2dmd_es_snacks count:", nielsen2dmd_es_snacks.count())
print("nielsen2dmd_es_beverages count:", nielsen2dmd_es_beverages.count())
neilsen_df = nielsen2dmd_pt_snacks.join(nielsen2dmd_es_snacks, on=intersect_two_lists(nielsen2dmd_pt_snacks.columns, nielsen2dmd_es_snacks.columns), how="outer")
neilsen_df = neilsen_df.join(nielsen2dmd_es_beverages, on=intersect_two_lists(neilsen_df.columns, nielsen2dmd_es_beverages.columns), how="outer")
neilsen_df = neilsen_df.withColumnRenamed('COUNTRY', 'HRCHY_LVL_3_NM')
print("Joined dataset count:", neilsen_df.count())
print(neilsen_df.select(["Week_Of_Year", "DMDUNIT", "HRCHY_LVL_3_NM"]).distinct().count() == neilsen_df.count())
display(neilsen_df)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(neilsen_df.columns, ['HRCHY_LVL_3_NM', 'DMDUNIT', 'Week_Of_Year'])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("neil_")==False:
    neilsen_df = neilsen_df.withColumnRenamed(i, "neil_"+i)
display(neilsen_df)

# COMMAND ----------

## Neilsen data columns to be retained 
# Retain following columns for modelling: Total_Points_Of_Sales_Dealing, 'Numeric_Distribution','Wtd_Distribution', 'Wtd_Distribution_Promo', 'Wtd_Distribution_SEL', 'Wtd_Distribution_SE', 'Wtd_Distribution_L', 'Wtd_Distribution_TPR', 'Price_Per_Qty', 'Promo_Price_Per_Volume',
# We need to drop following columns after cleansing script: 'Base_Sales_Unit', 'Unit_Sales',
# percent_baseline_unit = Base_Sales_Unit/Unit_Sales
# percent_baseline_volume = Base_Sales_Qty/Volume_Sales
# percent_baseline_value = Base_Sales_Value/Value_Sales
# and all other columns 

## Write as delta table to dbfs
save_df_as_delta(neilsen_df, DBI_NIELSEN, enforce_schema=False)
delta_info = load_delta_info(DBI_NIELSEN)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

temp = load_delta(DBI_NIELSEN)
print(temp.count())
print(temp.select(["Week_Of_Year", "DMDUNIT", "HRCHY_LVL_3_NM"]).distinct().count() == temp.count())
display(temp)

# COMMAND ----------

