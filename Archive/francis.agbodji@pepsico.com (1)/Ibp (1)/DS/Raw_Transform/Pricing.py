# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

from pyspark.sql.functions import substring, length, col, expr
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import desc, lit, col

# COMMAND ----------

tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"
storage_account_uri = f"{storage_account}.dfs.core.windows.net"

# COMMAND ----------

spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)


# COMMAND ----------

## Function to read in files
def read_file(file_location, delim = ","):
  return (spark.read.option("header","true").option("delimiter", delim).csv(file_location))

# COMMAND ----------

## Direct Query
productDF = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-bronze/IBP/EDW/Prolink/Product Master/ingestion_dt=2021-06-04/", delim = ";")

customerMappingDF_Beverages = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Customers_Demand Group (Active_Inactive Cust)/Customers_Demand Groups_Beverages/Customers_Demand Groups_Beverages.csv", delim = ",")
customerMappingDF_Snacks = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Customers_Demand Group (Active_Inactive Cust)/Customers_Demand Groups_Snacks/dmdgroup-ib-snacks.csv", delim = ";")                        
customerMappingDF_Alvalle = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Customers_Demand Group (Active_Inactive Cust)/Customers_Demand groups Alvalle/Customers_Demand groups Alvalle.csv", delim = ";")

pricingDF_2021 = spark.read.parquet("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Pricing/Sales/YEAR=2021/ES-Int-Finance-LocalReporting-Sales.parquet")
pricingDF_2020 = spark.read.parquet("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Pricing/Sales/YEAR=2020/ES-Int-Finance-LocalReporting-Sales.parquet")
pricingDF_2019 = spark.read.parquet("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Pricing/Sales/YEAR=2019/ES-Int-Finance-LocalReporting-Sales.parquet")
pricingDF_2018 = spark.read.parquet("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/ibp/Pricing/Sales/YEAR=2018/ES-Int-Finance-LocalReporting-Sales.parquet")
pricingDF_PT = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-bronze/IBP/EDW/PINGO_DOCE/Pricing.csv", delim = ",")

product_case = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/edw/ibp/product-master/CASE.csv", delim = ",")
product_PRODUCT = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/edw/ibp/product-master/PRODUCT.csv", delim = ",")
product_sweet = read_file("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-landing/edw/ibp/product-master/Sweet Products.csv", delim = ",")

# COMMAND ----------

# checking distinct lengths of customer codes in PT pricing
display(pricingDF_PT.withColumn("len_cd_cust", length(col("CD_CUSTOMER_SOURCE").cast(IntegerType()).cast(StringType()))).select("len_cd_cust").distinct())

# COMMAND ----------

## Clean column names and combine tables

## Rename and select fields
df1 = customerMappingDF_Beverages.withColumn("Source_Cat",lit("Beverage")).select(col("Cliente")
          .alias("Customer_Code"),
                 col("Demand Group3")
          .alias("DMDGroup"),
                 col("Nombre 1")
          .alias("Desc"),"Source_Cat")

df1 = df1.withColumn('Customer_Code', trim(df1.Customer_Code))

df2 = customerMappingDF_Snacks.withColumn("Source_Cat",lit("Snacks")).select(col("Customer Id")
          .alias("Customer_Code"),
                 col("Demand Group")
          .alias("DMDGroup"),
                 col("Customer Name")
          .alias("Desc"),"Source_Cat")

df2 = df2.withColumn('Customer_Code', trim(df2.Customer_Code))


df3 = customerMappingDF_Alvalle.withColumn("Source_Cat",lit("Alvalle")).withColumn("Desc",lit("N/A")).select(col("Customer_ID")
          .alias("Customer_Code"),col("Demand Group")
          .alias("DMDGroup"),
          "Desc",
          "Source_Cat")

df3 = df3.withColumn('Customer_Code', trim(df3.Customer_Code))

## This is the three customer tables appended onto one another
Customer_Groups_Combined_DF = df3.union(df1.union(df2))

## Join merge to pricing data
p1 = pricingDF_2021.withColumn("file_year",lit("2021"))
p2 = pricingDF_2020.withColumn("file_year",lit("2020"))
p3 = pricingDF_2019.withColumn("file_year",lit("2019"))
p4 = pricingDF_2018.withColumn("file_year",lit("2018"))

## Now do the same with pricing
pricing_full_es = p4.union(p3.union(p2.union(p1)))
pricing_full_es = pricing_full_es.withColumn('CD_CUSTOMER_SOURCE', trim(pricing_full_es.CD_CUSTOMER_SOURCE)).drop("file_year")
pricing_full_pt = pricingDF_PT.withColumn('CD_CUSTOMER_SOURCE', col("CD_CUSTOMER_SOURCE") - lit(90000000))
pricing_full_pt = pricing_full_pt.withColumn('CD_CUSTOMER_SOURCE', col("CD_CUSTOMER_SOURCE").cast(IntegerType()))
pricing_full_pt = pricing_full_pt.withColumn('CD_CUSTOMER_SOURCE', trim(pricing_full_pt.CD_CUSTOMER_SOURCE)).drop("CD_CUSTOMER", "ME_ON_INVOICE_ACT")
print(pricing_full_es.columns == pricing_full_pt.columns)

display(pricing_full_es)
display(pricing_full_pt)

# COMMAND ----------

display(pricing_full_es.select("CD_COUNTRY").distinct())
display(pricing_full_pt.select("CD_COUNTRY").distinct())

# COMMAND ----------

## Join Pricing to Customer
pricing_full = pricing_full_es.union(pricing_full_pt)
print(pricing_full.count() == pricing_full_es.count() + pricing_full_pt.count())
pricing_join_erp = pricing_full.join(Customer_Groups_Combined_DF,  pricing_full.CD_CUSTOMER_SOURCE == Customer_Groups_Combined_DF.Customer_Code, 'left')

## Creating Week_Of_Year column to facilitate merge with mrd
pricing_join_erp = pricing_join_erp.withColumn("Week_Of_Year", pricing_join_erp["ID_TIME_WEEK"].cast(StringType()))
pricing_join_erp = pricing_join_erp.withColumn("Week_Of_Year", concat(substring('Week_Of_Year', 1, 4), substring('Week_Of_Year', 7, 2)))
pricing_join_erp = pricing_join_erp.withColumn("Week_Of_Year", pricing_join_erp["Week_Of_Year"].cast(IntegerType()))

pricing_join_erp = pricing_join_erp.withColumn("Month_Of_Year", pricing_join_erp["ID_TIME_MONTH"].cast(IntegerType()))
display(pricing_join_erp)

# COMMAND ----------

display(pricing_full.describe())

# COMMAND ----------

display(pricing_join_erp.filter(col("DMDGroup").contains("PT_")).select("DMDGroup").distinct())

# COMMAND ----------

display(pricing_join_erp.filter(col("DMDGroup") == "PT_OT_PINGO_DOCE").select("CD_CUSTOMER_SOURCE").distinct())

# COMMAND ----------

# MAGIC %md #### Product Join for Categories

# COMMAND ----------

## Format the product master table
productDF = productDF.withColumn('DmdUnit', trim(productDF.PLANG_MTRL_GRP_VAL))
productDF = productDF.withColumn("DmdUnit",expr("substring(DmdUnit, 1, length(DmdUnit)-3)"))

# COMMAND ----------

def category_mapping(time_var, category="BEVERAGES"):  
  
  ## Calculate Prices
  pricing_dmd_group = pricing_join_erp.groupby('DMDGroup','CD_PROD_SELLING', time_var).agg({'ME_GROSS_SALES_ACT': 'sum','ME_TOTAL_DISCOUNTS_ACT': \
                                                                                            'sum', 'ME_SALES_UNITS_BAG_ACT': 'sum'})
  
  pricing_dmd_group = pricing_dmd_group.withColumn("LIST_PRICE_BAG",col("sum(ME_GROSS_SALES_ACT)")/col("sum(ME_SALES_UNITS_BAG_ACT)"))
  pricing_dmd_group = pricing_dmd_group.withColumn("NET_PRICE_BAG",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col("sum(ME_SALES_UNITS_BAG_ACT)"))
  
   ## Join to Category Mapping file, then to Prolink
  if category == "BEVERAGES":
    product_join = pricing_dmd_group.join(product_PRODUCT, on = "CD_PROD_SELLING", how = 'left')
    product_ctgy_master = product_join.join(productDF, product_join.CD_PRODUCT == productDF.DmdUnit , 'left')
  else:
    product_join = pricing_dmd_group.join(product_case, pricing_dmd_group.CD_PROD_SELLING == product_case.CD_PRODUCT_SELLING , 'left')
    product_ctgy_master = product_join.join(productDF, product_join.CD_PRODUCT_CASE == productDF.DmdUnit , 'left')
    
  ## For retaining necessary columns
  price_cols = [c for c in product_ctgy_master.columns if "PRICE" in c]
  cols_to_drop = list(set(product_ctgy_master.columns) - set(["DMDGroup", "PLANG_MTRL_GRP_VAL", time_var, "SRC_CTGY_1_NM"] + price_cols))  
  product_ctgy_master = product_ctgy_master.filter(product_ctgy_master.SRC_CTGY_1_NM == category).drop(*cols_to_drop)  

  ## Keeping rows with non null DMDGroup AND non null DMDUnit
  product_ctgy_master = product_ctgy_master.filter(col("DMDGroup").isNotNull() & col("PLANG_MTRL_GRP_VAL").isNotNull())

  return product_ctgy_master

# COMMAND ----------

# def category_mapping(volume_col, time_var, category="BEVERAGES"):  
  
#   ## Calculate Prices
#   pricing_dmd_group = pricing_join_erp.groupby('DMDGroup','CD_PROD_SELLING', time_var).agg({'ME_GROSS_SALES_ACT': 'sum','ME_TOTAL_DISCOUNTS_ACT': \
#                                                                                             'sum', volume_col: 'sum', "ME_SALES_UNITS_P_CASES_ACT": 'sum',\
#                                                                                             'ME_SALES_UNITS_BAG_ACT': 'sum'})
#   pricing_dmd_group = pricing_dmd_group.withColumn("LIST_PRICE_VOLUME",col("sum(ME_GROSS_SALES_ACT)")/col(f"sum({volume_col})"))
#   pricing_dmd_group = pricing_dmd_group.withColumn("NET_PRICE_VOLUME",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col(f"sum({volume_col})"))
  
#   pricing_dmd_group = pricing_dmd_group.withColumn("LIST_PRICE_BAG",col("sum(ME_GROSS_SALES_ACT)")/col("sum(ME_SALES_UNITS_BAG_ACT)"))
#   pricing_dmd_group = pricing_dmd_group.withColumn("NET_PRICE_BAG",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col("sum(ME_SALES_UNITS_BAG_ACT)"))
  
#   pricing_dmd_group = pricing_dmd_group.withColumn("LIST_PRICE_CASES",col("sum(ME_GROSS_SALES_ACT)")/col("sum(ME_SALES_UNITS_P_CASES_ACT)"))
#   pricing_dmd_group = pricing_dmd_group.withColumn("NET_PRICE_CASES",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col("sum(ME_SALES_UNITS_P_CASES_ACT)"))

#   ## Join to Category Mapping file, then to Prolink
#   if category == "BEVERAGES":
#     product_join = pricing_dmd_group.join(product_PRODUCT, on = "CD_PROD_SELLING", how = 'left')
#     product_ctgy_master = product_join.join(productDF, product_join.CD_PRODUCT == productDF.DmdUnit , 'left')
#   else:
#     product_join = pricing_dmd_group.join(product_case, pricing_dmd_group.CD_PROD_SELLING == product_case.CD_PRODUCT_SELLING , 'left')
#     product_ctgy_master = product_join.join(productDF, product_join.CD_PRODUCT_CASE == productDF.DmdUnit , 'left')
    
#   ## For retaining necessary columns
#   price_cols = [c for c in product_ctgy_master.columns if "PRICE" in c]
#   cols_to_drop = list(set(product_ctgy_master.columns) - set(["DMDGroup", "PLANG_MTRL_GRP_VAL", time_var, "SRC_CTGY_1_NM"] + price_cols))  
#   product_ctgy_master = product_ctgy_master.filter(product_ctgy_master.SRC_CTGY_1_NM == category).drop(*cols_to_drop)  

#   ## Keeping rows with non null DMDGroup AND non null DMDUnit
#   product_ctgy_master = product_ctgy_master.filter(col("DMDGroup").isNotNull() & col("PLANG_MTRL_GRP_VAL").isNotNull())

#   return product_ctgy_master

# COMMAND ----------

# MAGIC %md ### Weekly Aggregation

# COMMAND ----------

pricing_snacks_weekly = category_mapping(time_var = "Week_Of_Year", category = "SNACKS").distinct()
print(pricing_snacks_weekly.count(), len(pricing_snacks_weekly.columns))   

pricing_bev_weekly = category_mapping(time_var = "Week_Of_Year", category = "BEVERAGES").distinct()
print(pricing_bev_weekly.count(), len(pricing_bev_weekly.columns))

pricing_juice_weekly = category_mapping(time_var = "Week_Of_Year", category = "JUICE").distinct()
print(pricing_juice_weekly.count(), len(pricing_juice_weekly.columns))

pricing_grains_weekly = category_mapping(time_var = "Week_Of_Year", category = "GRAINS").distinct()
print(pricing_grains_weekly.count(), len(pricing_grains_weekly.columns))

# COMMAND ----------

pricing_weekly = pricing_snacks_weekly.union(pricing_bev_weekly.union(pricing_juice_weekly.union(pricing_grains_weekly)))

# COMMAND ----------

mrd_merge_cols = [c for c in pricing_weekly.columns if "PRICE" not in c]

if pricing_weekly.count() != pricing_weekly.select(mrd_merge_cols).distinct().count():
  price_cols = [c for c in pricing_weekly.columns if "PRICE" in c]
  agg_dict = {x:"mean" for x in price_cols}
  pricing_weekly = pricing_weekly.groupBy(mrd_merge_cols).agg(agg_dict)
  for price_col in price_cols:
    pricing_weekly = pricing_weekly.withColumnRenamed("avg(" + price_col + ")", price_col) 
    
print(pricing_weekly.count() == pricing_weekly.select(mrd_merge_cols).distinct().count())
print(pricing_weekly.count())

# COMMAND ----------

display(pricing_weekly)

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(pricing_weekly, DBI_PRICING_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PRICING_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# MAGIC %md ### Monthly Aggregation

# COMMAND ----------

pricing_snacks_monthly = category_mapping(time_var = "Month_Of_Year", category = "SNACKS").distinct()
print(pricing_snacks_monthly.count(), len(pricing_snacks_monthly.columns))   

pricing_bev_monthly = category_mapping(time_var = "Month_Of_Year", category = "BEVERAGES").distinct()
print(pricing_bev_monthly.count(), len(pricing_bev_monthly.columns))

pricing_juice_monthly = category_mapping(time_var = "Month_Of_Year", category = "JUICE").distinct()
print(pricing_juice_monthly.count(), len(pricing_juice_monthly.columns))

pricing_grains_monthly = category_mapping(time_var = "Month_Of_Year", category = "GRAINS").distinct()
print(pricing_grains_monthly.count(), len(pricing_grains_monthly.columns))

# COMMAND ----------

pricing_monthly = pricing_snacks_monthly.union(pricing_bev_monthly.union(pricing_juice_monthly.union(pricing_grains_monthly)))

# COMMAND ----------

mrd_merge_cols = [c for c in pricing_monthly.columns if "PRICE" not in c]

if pricing_monthly.count() != pricing_monthly.select(mrd_merge_cols).distinct().count():
  price_cols = [c for c in pricing_monthly.columns if "PRICE" in c]
  agg_dict = {x:"mean" for x in price_cols}
  pricing_monthly = pricing_monthly.groupBy(mrd_merge_cols).agg(agg_dict)
  for price_col in price_cols:
    pricing_monthly = pricing_monthly.withColumnRenamed("avg(" + price_col + ")", price_col) 
    
print(pricing_monthly.count() == pricing_monthly.select(mrd_merge_cols).distinct().count())
print(pricing_monthly.count())

# COMMAND ----------

display(pricing_monthly)

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(pricing_monthly, DBI_PRICING_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PRICING_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# MAGIC %md #### Below table has logic for price 

# COMMAND ----------

dfuviewDF = read_file("abfss://bronze@cdodevadls2.dfs.core.windows.net/IBP/EDW/Prolink/DFU View/ingestion_dt=2021-06-04/", delim = ";")
dfuviewDF.select("DMNDPLN_FCST_UNIT_UOM_CDV").distinct().show() 

# COMMAND ----------

dfuviewDF.count()

# COMMAND ----------

dfuviewDF.select("DMND_FMLY_NM", "DMNDPLN_FCST_UNIT_UOM_CDV").distinct().count()

# COMMAND ----------

display(dfuviewDF.select("DMND_FMLY_NM", "DMNDPLN_FCST_UNIT_UOM_CDV").distinct())

# COMMAND ----------

display(dfuviewDF.select("SPLY_FMLY_NM", "DMNDPLN_FCST_UNIT_UOM_CDV").distinct())

# COMMAND ----------

display(pricing_full)

# COMMAND ----------

display(dfuviewDF)

# COMMAND ----------



# COMMAND ----------

