# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

# Check paths
print(DBI_PROMO_ES_WEEKLY) 
print(DBI_PROMO_ES_MONTHLY)

# COMMAND ----------

# daily calendar creation
from pyspark.sql import Row

df = spark.sparkContext.parallelize([Row(start_date='2016-01-01', end_date='2025-12-31')]).toDF()
df = df \
  .withColumn('start_date', F.col('start_date').cast('date')) \
  .withColumn('end_date', F.col('end_date').cast('date'))\
  .withColumn('cal_date', F.explode(F.expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
  .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
  .withColumn("Week_end_date",date_add("Week_start_date",6))\
  .withColumn('week_year',F.when((year(col('Week_start_date'))==year(col('cal_date'))) &          (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
              .otherwise(year('Week_end_date')))\
  .withColumn('month_year',year(col('cal_date')))\
  .withColumn('week',F.when((year(col('Week_start_date'))==year(col('Week_end_date'))),F.weekofyear(col("Week_end_date")))\
                     .otherwise(F.weekofyear(col("Week_end_date"))))\
  .withColumn('month',F.month("cal_date"))

calendar_df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+F.month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .select('cal_date','Week_start_date','Week_end_date','Week_Of_Year','Month_Of_Year').distinct()

# COMMAND ----------

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"

storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
spark.catalog.clearCache()
spark.conf.set("spark.worker.cleanup.enabled", "true")

# COMMAND ----------

## Function to read in files
def read_file(file_location, delim = ","):
  return (spark.read.option("header","true").option("delimiter", delim).csv(file_location))

# COMMAND ----------

SFA_DF = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp-poc/promotions-sfa")
product_SNACK = load_delta('abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp-poc/product-case-mapping')
product_BEV = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp-poc/product-mapping")
productDF = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp-poc/product-master") 
product_AV = spark.read.csv("/FileStore/tables/temp/Alvelle_Mapping.csv", header="true", inferSchema="true")

# COMMAND ----------

SFA_DF_filter = SFA_DF.select("dmdgroups", "Promo_Start", "Promo_Finish", "Promo_Product_Pack_ID", "Promo_Description")
print(SFA_DF_filter.count())
SFA_DF_filter = SFA_DF_filter.withColumnRenamed('dmdgroups','DMDGROUP')\
                             .withColumnRenamed('Promo_Start','start_date')\
                             .withColumnRenamed('Promo_Finish','end_date')\
                             .withColumnRenamed('Promo_Product_Pack_ID','Prod_Selling_SKU')\
                             .withColumnRenamed('Promo_Description','Promo_Desc').distinct()
print(SFA_DF_filter.count())
display(SFA_DF_filter)

# COMMAND ----------

## Mapping the DTS DemandGroups to Servicio
cust = load_delta(DBI_CUSTOMER)
cust_es_dts = cust.filter(col("HRCHY_LVL_2_ID") == "ES_DTS").withColumn("DMDGROUP", lit("SERVICIO DC")).select("DMDGROUP", "PLANG_CUST_GRP_VAL").distinct()

prev_count = SFA_DF_filter.filter(col("DMDGROUP") == "SERVICIO DC").count()
SFA_DF_filter = SFA_DF_filter.join(cust_es_dts, on = "DMDGROUP", how = "left")
SFA_DF_filter = SFA_DF_filter.withColumn("DMDGROUP", when(col("DMDGROUP") == "SERVICIO DC", col("PLANG_CUST_GRP_VAL")).otherwise(col("DMDGROUP"))).drop("PLANG_CUST_GRP_VAL")

print(prev_count*2 == SFA_DF_filter.filter(col("DMDGROUP") == "ES_DTS_DISTRIBUTORS").count() + SFA_DF_filter.filter(col("DMDGROUP") == "ES_DTS_OTHERS").count())

# COMMAND ----------

def get_date_format(df, col_name):  
  
  split_col = pyspark.sql.functions.split(df[col_name], '/')
 
  df = df.withColumn("month", split_col.getItem(0)).withColumn("day", split_col.getItem(1)).withColumn("year", split_col.getItem(2))
  df = df.withColumn("month", F.when(length(F.col("month")) == 1, concat(lit('0'), F.col("month"))).otherwise(F.col("month")))
  df = df.withColumn("day", F.when(length(F.col("day")) == 1, concat(lit('0'), F.col("day"))).otherwise(F.col("day")))
  df = df.withColumn(col_name, concat(F.col("year"), lit('-'), F.col("month"), lit('-'), F.col("day")))
  df = df.withColumn(col_name, to_date(col(col_name),"yyyy-MM-dd")).drop("month", "day", "year")
 
  return df

# COMMAND ----------

es_promo = get_date_format(get_date_format(SFA_DF_filter, "start_date"), "end_date")
print (es_promo.count())

# Join with calendar to get TIME_VAR, WEEK start and end dates
es_promo = es_promo.crossJoin(calendar_df).filter((col('cal_date')>=col('start_date')) & (col('cal_date')<=col('end_date')))              
print (es_promo.count())
es_promo = es_promo.withColumn("Day_Of_Week", datediff(col("cal_date"), col("Week_start_date")) + 1)
es_promo = es_promo.withColumn("weekend_flag", F.when(F.col("Day_Of_Week") >= 6, lit(1)).otherwise(0)).drop("Day_Of_Week", "Week_start_date", "Week_end_date")
es_promo = es_promo.withColumn("promo_day", lit(1)).distinct()

# COMMAND ----------

dmdunit_mapping_SNACK = product_SNACK.select("CD_PRODUCT_SELLING","CD_PRODUCT_CASE")
dmdunit_mapping_BEV = product_BEV.select("CD_PROD_SELLING","CD_PRODUCT")
dmdunit_mapping_AV = product_AV.select("CD_PROD_SELLING","DMDUnit")

#Split the data by category
es_promo = es_promo.withColumn("CT",expr("substring(Prod_Selling_SKU, 1, 1)"))
SFA_SNACK_DISTINCT = es_promo.filter(es_promo.CT == "F").select('Prod_Selling_SKU').distinct()
SFA_ALVELLE_DISTINCT = es_promo.filter(es_promo.CT == "A").select('Prod_Selling_SKU').distinct()
SFA_BEV_DISTINCT = es_promo.filter(es_promo.CT == "B").select('Prod_Selling_SKU').distinct()

SFA_SNACK_DISTINCT = SFA_SNACK_DISTINCT.join(dmdunit_mapping_SNACK, SFA_SNACK_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_SNACK.CD_PRODUCT_SELLING , how='left')
SFA_BEV_DISTINCT = SFA_BEV_DISTINCT.join(dmdunit_mapping_BEV, SFA_BEV_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_BEV.CD_PROD_SELLING , how='left')
SFA_ALVELLE_DISTINCT = SFA_ALVELLE_DISTINCT.join(dmdunit_mapping_AV, SFA_ALVELLE_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_AV.CD_PROD_SELLING , how='left')

SFA_ALL_DISTINCT = SFA_SNACK_DISTINCT.union(SFA_BEV_DISTINCT.union(SFA_ALVELLE_DISTINCT))

print(SFA_SNACK_DISTINCT.count())
print(SFA_BEV_DISTINCT.count())
print(SFA_ALVELLE_DISTINCT.count())
print(SFA_ALL_DISTINCT.count() == SFA_SNACK_DISTINCT.count() + SFA_BEV_DISTINCT.count() + SFA_ALVELLE_DISTINCT.count())

# COMMAND ----------

# Get PLANG_MTRL_GRP_VAL from product master
productDF = productDF.withColumn("DMDunit_Product_master",expr("substring(PLANG_MTRL_GRP_VAL, 1, length(PLANG_MTRL_GRP_VAL)-3)")) 
SFA_ALL_DISTINCT = SFA_ALL_DISTINCT.join(productDF, SFA_ALL_DISTINCT.CD_PRODUCT_CASE == productDF.DMDunit_Product_master , how='left')\
                                    .select("Prod_Selling_SKU", "PLANG_MTRL_GRP_VAL").distinct()

print(es_promo.count())
es_promo = es_promo.join(SFA_ALL_DISTINCT, on = "Prod_Selling_SKU", how = "left").withColumnRenamed("PLANG_MTRL_GRP_VAL", "DMDUNIT").drop("CT")

# COMMAND ----------

# Promotion column - General cleaning
promotions = es_promo.select("Promo_Desc").distinct()

for col_name in promotions.columns:
  promotions = promotions.withColumn(f"{col_name}_cleaned", trim(col(col_name))).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", ',', '.')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'ª', 'a')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'º', 'a')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", '([0-9]) ([0-9][0-9]%)', '$1a$2')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", ' ', '')).withColumn(f"{col_name}_cleaned", lower(f"{col_name}_cleaned")).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'euro', '€')).withColumn(f"{col_name}_cleaned", regexp_replace(f"{col_name}_cleaned", 'eur', '€'))
  
display(promotions)
promotions.count()

# COMMAND ----------

## Categorising the promotions based on patterns
print("Total promos: ",promotions.count())
promo_sub_groups = {}

#Multi-tiered - Buy More Save More More Units you buy the more you save
pattern_bmsm = r'.*[0-9]a?al?[0-9][0-9].*|.*[0-9]a[0-9]\.?[0-9][0-9]?.*|.*compra[0-9]dto.*|.*dobleahorro.*|.*cajas-[0-9]*'
bmsm_promos = promotions.filter(promotions["Promo_Desc_cleaned"].rlike(pattern_bmsm))
bmsm = convertDFColumnsToList(bmsm_promos,"Promo_Desc_cleaned")

promotions_updated = promotions.where(promotions.Promo_Desc_cleaned.isin(bmsm) == False)
promo_sub_groups["multi_tiered_BuyMoreSaveMore"] = bmsm
print(promotions_updated.count())

#Coupons - In Store Available in store only
pattern_cpstore = r'.*cheque.*|.*chqcrc.*|.*cqcrece.*|.*chqcrece.*|.*club.*'
cpstore_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_cpstore))
cpstore = convertDFColumnsToList(cpstore_promos,"Promo_Desc_cleaned")
promo_sub_groups['coupons_InStore'] = cpstore

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(cpstore) == False)
print(promotions_updated.count())

#Loyalty Points - Generic All Other - Loyalty Points
pattern_lpgeneral = r'.*vuelve.*|.*%v.*'
lpgeneral_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_lpgeneral))
lpgeneral = convertDFColumnsToList(lpgeneral_promos,"Promo_Desc_cleaned")
promo_sub_groups['loyalty_points_Generic'] = lpgeneral

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(lpgeneral) == False)
print(promotions_updated.count())

#Free Goods – Free Premium Buy a certain amount / $ value, you get free premiums e.g. collectable plates/mugs etc
pattern_freecol = r'.*regalo.*'
freecol_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_freecol))
freecol = convertDFColumnsToList(freecol_promos,"Promo_Desc_cleaned")
promo_sub_groups['free_goods_Premium'] = freecol

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(freecol) == False)
print(promotions_updated.count())

#TPR - Discount % TPR discount % communicated at store level only
pattern_tprperc = r'.*[0-9]*\.?[0-9]*\%.*'
tprperc_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_tprperc))
tprperc = convertDFColumnsToList(tprperc_promos,"Promo_Desc_cleaned")
promo_sub_groups['tpr_DiscountPerc'] = tprperc

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(tprperc) == False)
print(promotions_updated.count())

#TPR - Discount Price TPR discount price communicated at store level only
pattern_tprprice = r'.*[0-9]€.*|.*[0-9]\.[0-9][0-9]€?.*'
tprprice_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_tprprice))
tprprice = convertDFColumnsToList(tprprice_promos,"Promo_Desc_cleaned")
promo_sub_groups['tpr_DiscountPrice'] = tprprice

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(tprprice) == False)
print(promotions_updated.count())

#Multibuy Buy x for y Buy X for Y
pattern_buyxy = r'.*[0-9]x[0-9].*|.*[0-9]\+[0-9].*|.*latas.*|.*lote.*'
buyxy_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_buyxy))
buyxy = convertDFColumnsToList(buyxy_promos,"Promo_Desc_cleaned")
promo_sub_groups['multibuy_BuyXforY'] = buyxy

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(buyxy) == False)
print(promotions_updated.count())

#Display - General
pattern_display = r'.*cooler.*|.*lineal.*|.*exp.*|.*lin.*|.*pilada.*|.*chimenea.*|.*balda.*|.*exhibicion.*|.*cab.*|.*espacio.*|.*eve.*|.*floore?stand.*|.*pall?et.*|.*box.*|.*item.*|.*trp.*|.*tpr.*'
display_promos = promotions_updated.filter(promotions_updated["Promo_Desc_cleaned"].rlike(pattern_display))
dsp = convertDFColumnsToList(display_promos,"Promo_Desc_cleaned")
promo_sub_groups['display_Generic'] = dsp

promotions_updated = promotions_updated.where(promotions_updated.Promo_Desc_cleaned.isin(dsp) == False)
print(promotions_updated.count())

# COMMAND ----------

#Get promo types with all percentage values
perc_promo_types = []
for group in promo_sub_groups:
  if len([p for p in promo_sub_groups[group] if "%" in p]) == len(promo_sub_groups[group]):
    perc_promo_types.append(group)

#Currency extraction from tpr_DiscountPrice 

tpr_disc = promo_sub_groups['tpr_DiscountPrice']
disc_price_map = {}

pattern1 = r'.*([0-9]\.[0-9][0-9]?€).*'
pattern2 = r'.*([0-9]€).*'
pattern3 = r'.*([0-9]\.[0-9][0-9]).*'

p1 = re.compile(pattern1)
p2 = re.compile(pattern2)
p3 = re.compile(pattern3)

for promo in tpr_disc:
  extracted = None
  search_results = p1.search(promo)
  try:
    currency_part = search_results.group(1)
    extracted = (promo , currency_part)
  except:
    search_results = p2.search(promo)
    try:
      currency_part = search_results.group(1)
      extracted = (promo , currency_part)
    except:
      search_results = p3.search(promo)
      currency_part = search_results.group(1)
      extracted = (promo , currency_part)
    
  disc_price_map[promo] = currency_part
     
print(disc_price_map)

# COMMAND ----------

# Map promo types to descriptions
promotions_mapped = promotions.withColumn("promo_type", lit(" "))
for group in promo_sub_groups.keys():
  promotions_mapped = promotions_mapped.withColumn("promo_type", when(F.col("Promo_Desc_cleaned").isin(promo_sub_groups[group]), lit(group)).otherwise(F.col('promo_type')))
  
# Create column with numerical extracts of numeric promo types
price_mapping_expr = create_map([lit(x) for x in chain(*disc_price_map.items())])
promotions_mapped = promotions_mapped.withColumn("numeric_val", price_mapping_expr.getItem(col("Promo_Desc_cleaned")))

promotions_mapped = promotions_mapped.withColumn("numeric_val", when(col("promo_type").isin(perc_promo_types), regexp_extract(col('Promo_Desc_cleaned'), '[0-9]*\.?[0-9]*\%', 0)).otherwise(col("numeric_val")))
promotions_mapped = promotions_mapped.withColumn("numeric_val", regexp_replace('numeric_val', '%', '')) \
                                     .withColumn("numeric_val", regexp_replace('numeric_val', '€', ''))

promotions_mapped.count()

# COMMAND ----------

# Get promo types as features
num_promo_type = convertDFColumnsToList(promotions_mapped.filter(col("numeric_val").isNotNull()).select("promo_type").distinct(), "promo_type")
non_num_promo_type = subtract_two_lists(promo_sub_groups.keys(), num_promo_type)
for group in promo_sub_groups.keys():
  if group in num_promo_type:
    promotions_mapped = promotions_mapped.withColumn(group, when(F.col("promo_type") == group, F.col("numeric_val")).otherwise(lit(0)))
  else:
    promotions_mapped = promotions_mapped.withColumn(group, when(F.col("promo_type") == group, lit(1)).otherwise(lit(0)))

# COMMAND ----------

# Map promo categories back to ES promotions
es_promo = es_promo.join(promotions_mapped.drop("Promo_Desc_cleaned", "numeric_val"), on = "Promo_Desc", how = 'left')
print(es_promo.count())
es_promo = es_promo.filter(col("promo_type") != " ").filter(col("DMDGROUP") != "#N/A").filter(col("DMDUNIT").isNotNull())

es_promo = es_promo.drop("Promo_Desc", "Prod_Selling_SKU", "start_date", "end_date", "promo_type").drop_duplicates()
print(es_promo.count())

# COMMAND ----------

## Features specific to scenario modeling
# Create avg_discount feature
es_promo = es_promo.withColumn("promo_avg_DiscountPerc", col("tpr_DiscountPerc"))
es_promo = es_promo.withColumn("promo_avg_DiscountPerc", when(col("promo_avg_DiscountPerc") == 0, None).otherwise(col("promo_avg_DiscountPerc")))

# Create count of promo feature
for promo in promo_sub_groups.keys():
  es_promo = es_promo.withColumn(f"flag_{promo}", when(col(promo)>0, lit(1)).otherwise(0))
  
all_flags = [f for f in es_promo.columns if "flag_" in f]
expression = '+'.join(all_flags)
es_promo = es_promo.withColumn('promo_count', expr(expression)).drop(*all_flags)

# Create on promo flag
es_promo = es_promo.withColumn("promo_on", lit(1))

print(es_promo.count())

# COMMAND ----------

# Ensure 1-1 mapping of cal_date and time variables
es_promo.select("cal_date", "Week_Of_Year", "Month_Of_Year").distinct().count() == es_promo.select("cal_date").distinct().count()

# COMMAND ----------

def rename_cols(df):
  
  agg_types = ["sum", "avg", "max", "min"]
  for c in df.columns:
    for a in agg_types:
      if a in c:
        df = df.withColumnRenamed(c, c.replace(f"{a}(", "").replace(")", ""))
  
  return df                                

# COMMAND ----------

## Aggregating upto daily level for each DMDUNIT, DMDGROUP
day_cols = ["weekend_flag", "promo_day"]
max_cols = list(promo_sub_groups.keys()) + day_cols + ["Week_Of_Year", "Month_Of_Year"] + ['promo_on']
avg_cols = ["promo_avg_DiscountPerc"]
sum_cols = ['promo_count']

max_dict = {x: "max" for x in max_cols}
avg_dict = {x: "avg" for x in avg_cols}
sum_dict = {x: "sum" for x in sum_cols}
all_dict = {**max_dict, **avg_dict, **sum_dict}

es_promo_daily = es_promo.groupBy("DMDGROUP", "DMDUNIT", "cal_date").agg(all_dict)
es_promo_daily = rename_cols(es_promo_daily)
es_promo_daily = es_promo_daily.withColumnRenamed("promo_day", "promo_no_of_days") \
                               .withColumnRenamed("weekend_flag", "promo_no_of_weekend_days")

# COMMAND ----------

print(es_promo_daily.count() == es_promo_daily.distinct().count() == es_promo_daily.select("DMDGROUP", "DMDUNIT", "cal_date").distinct().count())
print(es_promo_daily.count())

# COMMAND ----------

def agg_func(df, time_var):
  
  no_of_days_cols = [c for c in df.columns if "no_of" in c]
  max_cols = ["tpr_DiscountPerc", "loyalty_points_Generic"] + ["promo_on"]
  min_cols = ["tpr_DiscountPrice"]
  sum_cols = non_num_promo_type + no_of_days_cols + ["promo_count"]
  avg_cols = ["promo_avg_DiscountPerc"]

  max_dict = {x: "max" for x in max_cols}
  min_dict = {x: "min" for x in min_cols}
  sum_dict = {x: "sum" for x in sum_cols}
  avg_dict = {x: "avg" for x in avg_cols}
  all_dict = {**max_dict, **min_dict, **sum_dict, **avg_dict}

  df_agg = df.groupBy("DMDGROUP", "DMDUNIT", time_var).agg(all_dict)
  df_agg = rename_cols(df_agg)
  df_agg = df_agg.na.fill(value=0,subset=["promo_avg_DiscountPerc"])
  
  return df_agg

# COMMAND ----------

## Aggregating upto Weekly, Monthly levels
es_promo_agg_weekly = agg_func(es_promo_daily, "Week_Of_Year")
es_promo_agg_monthly = agg_func(es_promo_daily, "Month_Of_Year")

# COMMAND ----------

# Prefixing promo type columns with "promo"
for group in promo_sub_groups.keys():
  es_promo_agg_weekly = es_promo_agg_weekly.withColumnRenamed(group, f"promo_{group}")
  es_promo_agg_monthly = es_promo_agg_monthly.withColumnRenamed(group, f"promo_{group}")

# COMMAND ----------

display(es_promo_agg_weekly.describe())
display(es_promo_agg_monthly.describe())

# COMMAND ----------

print(es_promo_agg_weekly.count() == es_promo_agg_weekly.distinct().count() == es_promo_agg_weekly.select("DMDGROUP", "DMDUNIT", "Week_Of_Year").distinct().count())
print(es_promo_agg_weekly.count())

print(es_promo_agg_monthly.count() == es_promo_agg_monthly.distinct().count() == es_promo_agg_monthly.select("DMDGROUP", "DMDUNIT", "Month_Of_Year").distinct().count())
print(es_promo_agg_monthly.count())

# COMMAND ----------

# MAGIC %md ### Write out the datasets

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(es_promo_agg_weekly, DBI_PROMO_ES_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_ES_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(es_promo_agg_monthly, DBI_PROMO_ES_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_ES_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

