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
print(DBI_PROMO_PT_WEEKLY)
print(DBI_PROMO_PT_MONTHLY)

# COMMAND ----------

tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"
storage_account_uri = f"{storage_account}.dfs.core.windows.net"
spark.conf.set("fs.azure.sas.supplychain-ibp.cdodevextrblob.blob.core.windows.net","h_ttps://cdodevextrblob.blob.core.windows.net/?sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacuptfx&se=2022-06-08T21:16:16Z&st=2021-06-09T13:16:16Z&spr=https&sig=ciFKbQI6uFpYKTh%2FOdnd2vw8qAxWUOmSX0LtPvfkt3Y%3D") 
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
spark.conf.set("fs.azure.sas.ibp-manual.cdodevextrblob.blob.core.windows.net","h_ttps://cdodevextrblob.blob.core.windows.net/?sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacuptfx&se=2022-06-08T21:16:16Z&st=2021-06-09T13:16:16Z&spr=https&sig=ciFKbQI6uFpYKTh%2FOdnd2vw8qAxWUOmSX0LtPvfkt3Y%3D") 

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

raw_pt_promo = load_delta("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/excel/ibp-poc/portugal-promotions")
display(raw_pt_promo)
raw_pt_promo.count()

# COMMAND ----------

#Column translations
translated_pt_promo = raw_pt_promo.withColumnRenamed('CLIENTE','customer')\
.withColumnRenamed('DATA_INICIO','start_date')\
.withColumnRenamed('DATA_FIM','end_date')\
.withColumnRenamed('MARCA','brand')\
.withColumnRenamed('SUBMARCA','subbrand')\
.withColumnRenamed('SABOR','flavour')\
.withColumnRenamed('FORMATO','size_format')\
.withColumnRenamed('PARENT','parent')\
.withColumnRenamed('CODIGO_LOGISTICO','logistic_code')\
.withColumnRenamed('DESCRIO','description')\
.withColumnRenamed('CODIGO CLIENTE','customer_code')\
.withColumnRenamed('MECANICA','mechanical')\
.withColumnRenamed('DATA_DE_CONFIRMAO','Confirmation_Date')

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

translated_pt_promo = get_date_format(get_date_format(get_date_format(translated_pt_promo, "start_date"), "end_date"), "Confirmation_Date")
translated_pt_promo = translated_pt_promo.withColumn("DMDGROUP", when(col("customer") == "SONAE", "PT_OT_SONAE")\
                                            .when(col("customer") == "PINGO DOCE", "PT_OT_PINGO_DOCE")\
                                             .when(col("customer") == "ITM", "PT_OT_ITM")\
                                             .when(col("customer") == "AUCHAN", "PT_OT_AUCHAN")\
                                             .when(col("customer") == "DIA", "PT_OT_DIA")\
                                             .when(col("customer") == "LIDL", "PT_OT_LIDL")\
                                             .when((col("customer") == "ESTEVÃO NEVES") | \
                                                   (col("customer") == "LIDOSOL"), "PT_DTS_MADEIRA").otherwise("PT_OT_OTHERS")).drop("customer") 

# COMMAND ----------

print(translated_pt_promo.count())
pt_promo = translated_pt_promo.filter(col("STATUS") == "CONFIRMADO").filter(col("DMDUNIT") != "#N/A").drop('STATUS')
pt_promo = pt_promo.select("start_date", "end_date", "DMDUNIT", "DMDGROUP", "mechanical").distinct()
print (pt_promo.count())

pt_promo = pt_promo.crossJoin(calendar_df).filter((col('cal_date')>=col('start_date')) & (col('cal_date')<=col('end_date')))
pt_promo = pt_promo.withColumn("Day_Of_Week", datediff(col("cal_date"), col("Week_start_date")) + 1)
pt_promo = pt_promo.withColumn("weekend_flag", F.when(F.col("Day_Of_Week") >= 6, lit(1)).otherwise(0)).drop("Day_Of_Week", "Week_start_date", "Week_end_date")
pt_promo = pt_promo.withColumn("promo_day", lit(1)).distinct()
print (pt_promo.count())

# COMMAND ----------

## Bucket the promotions into types
distinct_promos = convertDFColumnsToList(pt_promo.filter(col("mechanical").isNotNull()).select("mechanical").distinct(), "mechanical")

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

non_num_promo_type = ['ad_flyer_OnlineAd', 'ad_flyer_Solo', 'multi_tiered_BuyMoreSaveMore', 'multibuy_BuyXforY', 'purchase_with_purchase']
num_promo_type = ["tpr_DiscountPerc", "tpr_DiscountPrice", "coupons_InStore"]

# COMMAND ----------

subtract_two_lists(distinct_promos, promo_sub_groups['tpr_DiscountPerc'] + promo_sub_groups['tpr_DiscountPrice'] + promo_sub_groups['coupons_InStore'] + \
                   promo_sub_groups['ad_flyer_OnlineAd'] + promo_sub_groups['ad_flyer_Solo'] + promo_sub_groups['multi_tiered_BuyMoreSaveMore'] + \
                   promo_sub_groups['multibuy_BuyXforY'] + promo_sub_groups['purchase_with_purchase'])

# COMMAND ----------

print(pt_promo.count())

pt_promo = pt_promo.withColumn("promo_type", lit(" "))
for group in promo_sub_groups.keys():
  pt_promo = pt_promo.withColumn("promo_type", when(F.col("mechanical").isin(promo_sub_groups[group]), lit(group)).otherwise(F.col('promo_type')))
pt_promo = pt_promo.filter(col("promo_type") != " ")

pt_promo = pt_promo.withColumn('mecanica', regexp_replace('mechanical', '%', ''))\
                    .withColumn('mecanica', regexp_replace('mecanica', 'SP', ''))\
                    .withColumn('mecanica', regexp_replace('mecanica', '€', ''))\
                    .withColumn('mecanica', regexp_replace('mecanica', ',', '.'))\
                    .withColumn('mecanica', regexp_replace('mecanica', 'TALÃO', ''))\
                    .withColumn('mecanica', regexp_replace('mecanica', 'CARTÃO', ''))\
                    .drop("mechanical")

print(pt_promo.count())

# COMMAND ----------

for group in promo_sub_groups.keys():
  if group in num_promo_type:
    pt_promo = pt_promo.withColumn(group, when(F.col("promo_type") == group, F.col("mecanica")).otherwise(lit(0)))
  else:
    pt_promo = pt_promo.withColumn(group, when(F.col("promo_type") == group, lit(1)).otherwise(lit(0)))

# COMMAND ----------

pt_promo_final = pt_promo.drop("start_date", "end_date", "mecanica")
pt_promo_final = pt_promo_final.withColumnRenamed("coupons_InStore", "coupons_InStore_value")
pt_promo_final = pt_promo_final.withColumn("coupons_InStore", when(col("coupons_InStore_value") > 0, lit(1)).otherwise(0))

# COMMAND ----------

## Features specific to scenario modeling
# Create avg_discount feature
pt_promo_final = pt_promo_final.withColumn("promo_avg_DiscountPerc", col("tpr_DiscountPerc"))
pt_promo_final = pt_promo_final.withColumn("promo_avg_DiscountPerc", when(col("promo_avg_DiscountPerc") == 0, None).otherwise(col("promo_avg_DiscountPerc")))

# Create count of promo feature
for promo in promo_sub_groups.keys():
  pt_promo_final = pt_promo_final.withColumn(f"flag_{promo}", when(col(promo)>0, lit(1)).otherwise(0))
  
all_flags = [f for f in pt_promo_final.columns if "flag_" in f]
expression = '+'.join(all_flags)
pt_promo_final = pt_promo_final.withColumn('promo_count', expr(expression)).drop(*all_flags)

# Create on promo flag
pt_promo_final = pt_promo_final.withColumn("promo_on", lit(1))

print(pt_promo_final.count())

# COMMAND ----------

# Ensure 1-1 mapping of cal_date and time variables
pt_promo_final.select("cal_date", "Week_Of_Year", "Month_Of_Year").distinct().count() == pt_promo_final.select("cal_date").distinct().count()

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
max_cols = list(promo_sub_groups.keys()) + day_cols +["coupons_InStore_value", "Week_Of_Year", "Month_Of_Year"] + ['promo_on']
avg_cols = ["promo_avg_DiscountPerc"]
sum_cols = ['promo_count']
max_dict = {x: "max" for x in max_cols}

max_dict = {x: "max" for x in max_cols}
avg_dict = {x: "avg" for x in avg_cols}
sum_dict = {x: "sum" for x in sum_cols}
all_dict = {**max_dict, **avg_dict, **sum_dict}

pt_promo_daily = pt_promo_final.groupBy("DMDGROUP", "DMDUNIT", "cal_date").agg(all_dict)
pt_promo_daily = rename_cols(pt_promo_daily)
pt_promo_daily = pt_promo_daily.withColumnRenamed("promo_day", "promo_no_of_days") \
                               .withColumnRenamed("weekend_flag", "promo_no_of_weekend_days")

# COMMAND ----------

print(pt_promo_daily.count() == pt_promo_daily.distinct().count() == pt_promo_daily.select("DMDGROUP", "DMDUNIT", "cal_date").distinct().count())
print(pt_promo_daily.count())

# COMMAND ----------

def agg_func(df, time_var):
  
  no_of_days_cols = [c for c in df.columns if "no_of" in c]
  max_cols = ["tpr_DiscountPerc", "coupons_InStore_value"] + ["promo_on"] 
  min_cols = ["tpr_DiscountPrice"]
  sum_cols = non_num_promo_type + no_of_days_cols + ["coupons_InStore"] + ["promo_count"]
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
pt_promo_agg_weekly = agg_func(pt_promo_daily, "Week_Of_Year")
pt_promo_agg_monthly = agg_func(pt_promo_daily, "Month_Of_Year")

# COMMAND ----------

# Prefixing promo type columns with "promo"
for group in list(promo_sub_groups.keys()) + ["coupons_InStore_value"]:
  pt_promo_agg_weekly = pt_promo_agg_weekly.withColumnRenamed(group, f"promo_{group}")
  pt_promo_agg_monthly = pt_promo_agg_monthly.withColumnRenamed(group, f"promo_{group}")

# COMMAND ----------

display(pt_promo_agg_weekly.describe())
display(pt_promo_agg_monthly.describe())

# COMMAND ----------

print(pt_promo_agg_weekly.count() == pt_promo_agg_weekly.distinct().count() == pt_promo_agg_weekly.select("DMDGROUP", "DMDUNIT", "Week_Of_Year").distinct().count())
print(pt_promo_agg_weekly.count())

print(pt_promo_agg_monthly.count() == pt_promo_agg_monthly.distinct().count() == pt_promo_agg_monthly.select("DMDGROUP", "DMDUNIT", "Month_Of_Year").distinct().count())
print(pt_promo_agg_monthly.count())

# COMMAND ----------

# MAGIC %md ### Write out the datasets

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(pt_promo_agg_weekly, DBI_PROMO_PT_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_PT_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(pt_promo_agg_monthly, DBI_PROMO_PT_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_PT_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

