# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql import Row
# clean column names and combine tables
from pyspark.sql.functions import desc
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import split

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

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

#Reading the data from the bronze path of Portugal table
promtion_deltaTable = DeltaTable.forPath(spark, srcPath)
promtion_deltaTable_version = promtion_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(promtion_deltaTable)
display(promtion_deltaTable.history())

# COMMAND ----------

#Reading the Protugal source data from bonze layer
PORT_DF = spark.read.format("delta").option("versionAsOf", promtion_deltaTable_version).load(srcPath)

# COMMAND ----------

# DBTITLE 1,testing command
PORT_DF.count()


# COMMAND ----------

display(PORT_DF)

# COMMAND ----------

PORT_DF.filter(col("CLIENTE") == "PINGO DOCE").count()

# COMMAND ----------

#Column translations
translated_pt_promo = PORT_DF.withColumnRenamed('ANO','year')\
.withColumnRenamed('CLIENTE','customer')\
.withColumnRenamed('SEMANA_SERVIO','week_service')\
.withColumnRenamed('SEMANA_IN-MARKET','in_market_week')\
.withColumnRenamed('DATA_INICIO','start_date')\
.withColumnRenamed('DATA_FIM','end_date')\
.withColumnRenamed('DURAO','duration')\
.withColumnRenamed('MARCA','brand')\
.withColumnRenamed('SUBMARCA','subbrand')\
.withColumnRenamed('SABOR','flavour')\
.withColumnRenamed('FORMATO','format')\
.withColumnRenamed('PARENT','parent')\
.withColumnRenamed('CODIGO_LOGISTICO','logistic_code')\
.withColumnRenamed('DESCRIO','description')\
.withColumnRenamed('BOLSASCAIXA','bag/box')\
.withColumnRenamed('CODIGO_CLIENTE','customer_code')\
.withColumnRenamed('DESCRIO_CLIENTE','customer_description')\
.withColumnRenamed('ACO','action')\
.withColumnRenamed('MECANICA','mechanical')\
.withColumnRenamed('DATA_DE_CONFIRMAO','Confirmation_Date')\
.withColumnRenamed('ESTIMATIVA_KAM_UNIDADES','estimate_kam(units)')\
.withColumnRenamed('ESTIMATIVA_CLIENTE_UNIDADES','customer_estimate(units)')\
.withColumnRenamed('ESTIMATIVA_SUPPLY_UNIDADES','supply_estimate(units)')\
.withColumnRenamed('ESTIMATIVA_FINAL_UNIDADES','final_estimate(units)')\
.withColumnRenamed('ESTIMATIVA_FINAL_CAIXAS','FinalEstimateBox')\
.withColumnRenamed('PEDIDO_S-1','Order_Prev')\
.withColumnRenamed('PEDIDO_S0','Order_Curr')\
.withColumnRenamed('PEDIDO_S+1','Order_Next')\
.withColumnRenamed('SELL-OUT','sell_out')\
.withColumnRenamed('FOLHETO','leaflet')\
.withColumnRenamed('COMENTRIOS','comments')

# COMMAND ----------

display(translated_pt_promo)

# COMMAND ----------

def get_date_format(df, col_name):  
  split_col = split(df[col_name], '/') 
  df = df.withColumn("month", split_col.getItem(0)).withColumn("day", split_col.getItem(1)).withColumn("year", split_col.getItem(2))
  df = df.withColumn("month", when(length(col("month")) == 1, concat(lit('0'), col("month"))).otherwise(col("month")))
  df = df.withColumn("day", when(length(col("day")) == 1, concat(lit('0'), col("day"))).otherwise(col("day")))
  df = df.withColumn(col_name, concat(col("year"), lit('-'), col("month"), lit('-'), col("day")))
  df = df.withColumn(col_name, to_date(col(col_name),"yyyy-MM-dd")).drop("month", "day", "year")
 
  return df

# COMMAND ----------

translated_pt_promo = get_date_format(get_date_format(get_date_format(translated_pt_promo, "start_date"), "end_date"), "Confirmation_Date")


# COMMAND ----------

display(translated_pt_promo)

# COMMAND ----------

# daily calendar creation
from pyspark.sql import Row

df = spark.sparkContext.parallelize([Row(start_date='2016-01-01', end_date='2025-12-31')]).toDF()
df = df \
  .withColumn('start_date', col('start_date').cast('date')) \
  .withColumn('end_date', col('end_date').cast('date'))\
  .withColumn('cal_date', explode(expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
  .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
  .withColumn("Week_end_date",date_add("Week_start_date",6))\
  .withColumn('week_year',when((year(col('Week_start_date'))==year(col('cal_date'))) &          (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
              .otherwise(year('Week_end_date')))\
  .withColumn('month_year',year(col('cal_date')))\
  .withColumn('week',when((year(col('Week_start_date'))==year(col('Week_end_date'))),weekofyear(col("Week_end_date")))\
                     .otherwise(weekofyear(col("Week_end_date"))))\
  .withColumn('month',month("cal_date"))

calendar_df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .select('cal_date','Week_start_date','Week_end_date','Week_Of_Year','Month_Of_Year_WSD')

# COMMAND ----------

display(calendar_df)

# COMMAND ----------

display(translated_pt_promo.select(col("customer")).distinct())

# COMMAND ----------

translated_pt_promo = translated_pt_promo.withColumn("DMDGROUP", when(col("customer") == "SONAE", "PT_OT_SONAE")\
                                            .when(col("customer") == "PINGO DOCE", "PT_OT_PINGO_DOCE")\
                                             .when(col("customer") == "ITM", "PT_OT_ITM")\
                                             .when(col("customer") == "AUCHAN", "PT_OT_AUCHAN")\
                                             .when(col("customer") == "DIA", "PT_OT_DIA")\
                                             .when(col("customer") == "LIDL", "PT_OT_LIDL")\
                                             .when((col("customer") == "ESTEVÃO NEVES") | \
                                                   (col("customer") == "LIDOSOL"), "PT_DTS_MADEIRA").otherwise("PT_OT_OTHERS")).drop("customer") 

# COMMAND ----------

# DBTITLE 1,Filtering on Confirmed status
print(translated_pt_promo.count())
pt_promo = translated_pt_promo.filter(col("STATUS") == "CONFIRMADO").filter(col("DMDUNIT") != "#N/A").drop('STATUS')
pt_promo = pt_promo.select("start_date", "end_date", "Confirmation_Date", "DMDUNIT", "DMDGROUP", "mechanical", "sell_out", "Order_Curr").distinct()
print (pt_promo.count())

# Join with calendar to get TIME_VAR, WEEK start and end dates
pt_promo = pt_promo.crossJoin(calendar_df).filter((col('cal_date')>=col('start_date')) & (col('cal_date')<=col('end_date')))\
                    .withColumn('week_diff',floor((datediff(col('start_date'),col('Confirmation_Date')))/7))\
                    .withColumn('month_diff',floor((datediff(col('start_date'),col('Confirmation_Date')))/30))
pt_promo = pt_promo.withColumn("Day_Of_Week", datediff(col("cal_date"), col("Week_start_date")) + 1)
pt_promo = pt_promo.withColumn("weekend_flag", when(col("Day_Of_Week") >= 6, lit(1)).otherwise(0))
print (pt_promo.count())


# COMMAND ----------

## Impute confirmation dates that are null or later to start date 
week_diff_mode = pt_promo.groupby("week_diff").count().sort(desc("count")).select("week_diff").collect()[0][0]
month_diff_mode = pt_promo.groupby("month_diff").count().sort(desc("count")).select("month_diff").collect()[0][0]

pt_promo = pt_promo.withColumn("week_diff", when(((col("Confirmation_Date").isNull()) | (col("Confirmation_Date")>col("start_date"))), \
                                                 lit(week_diff_mode)).otherwise(col("week_diff")))
pt_promo = pt_promo.withColumn("month_diff", when(((col("Confirmation_Date").isNull()) | (col("Confirmation_Date")>col("start_date"))), \
                                                  lit(month_diff_mode)).otherwise(col("month_diff")))
pt_promo = pt_promo.withColumn("Confirmation_Date", when(((col("Confirmation_Date").isNull()) | (col("Confirmation_Date")>col("start_date"))), \
                                                 date_sub(pt_promo['start_date'], week_diff_mode * 7)).otherwise(col("Confirmation_Date")))

## Get month and week of year for Confirmation Date
conf_calendar_df = calendar_df.select("cal_date", "Week_of_Year", "Month_Of_Year_WSD").distinct().withColumnRenamed("cal_date", "Confirmation_Date")\
                                                                                              .withColumnRenamed("Week_of_Year", "Promo_Planned_Week_Of_Year")\
                                                                                              .withColumnRenamed("Month_Of_Year_WSD", "Promo_Planned_Month_Of_Year")
pt_promo = pt_promo.join(conf_calendar_df, on = "Confirmation_Date", how = "left").drop("week_diff", "month_diff")
pt_promo.count()
# temp = temp.withColumn("Promo_Planned_Week_Of_Year", col("Week_Of_Year") - col("week_diff"))\
#            .withColumn("Promo_Planned_Month_Of_Year", col("Month_Of_Year") - col("month_diff"))

# COMMAND ----------

display(pt_promo)

# COMMAND ----------

display(pt_promo.select(col("mechanical")).distinct())

# COMMAND ----------

# Create additional features
promo_day_count_level = list(set(pt_promo.columns) - set(["cal_date", "Month_Of_Year_WSD",\
                                                  "week_diff", "month_diff", "Day_Of_Week", "weekend_flag"]))
print (pt_promo.count()) 
pt_promo  = pt_promo.withColumn('no_of_promo_days', max('Day_Of_Week').over(Window.partitionBy(promo_day_count_level)) - \
                                min('Day_Of_Week').over(Window.partitionBy(promo_day_count_level)) + lit(1))
pt_promo  = pt_promo.withColumn('no_of_weekend_promo_days', sum('weekend_flag').over(Window.partitionBy(promo_day_count_level)))\
                     .drop("cal_date", "Day_Of_Week", "weekend_flag").withColumnRenamed("Month_Of_Year_WSD", "Month_Of_Year")

pt_promo = pt_promo.dropDuplicates()     

print (pt_promo.count()) 

# COMMAND ----------

def convertDFColumnsToList(df, col):
  "Converts Dataframe column to list and returns"
  return(df.select(col).distinct().rdd.map(lambda r: r[0]).collect()) 

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
# promo_sub_groups['direct_discount'] = [p for p in distinct_promos if ('% SP' in p) or ('%SP' in p)]
# promo_sub_groups['lowered_price'] = [p for p in distinct_promos if ('€ SP' in p) or ('€' in p)] + num_list
# promo_sub_groups['coupon'] = [p for p in distinct_promos if 'TALÃO' in p]
# promo_sub_groups['rebate'] = [p for p in distinct_promos if 'CARTÃO' in p]
# promo_sub_groups['without_discount'] = [p for p in distinct_promos if 'PROMO' in p]
# promo_sub_groups['buy_more_pay_less'] = [p for p in distinct_promos if re.search(r'L\dP\d', p)]
# promo_sub_groups['pack'] = [p for p in distinct_promos if 'PACK' in p]
# promo_sub_groups['cross_sell'] = [p for p in distinct_promos if 'MENU' in p]

promo_sub_groups['tpr_DiscountPerc'] = [p for p in distinct_promos if ('%' in p) and ('SP' in p)]
promo_sub_groups['tpr_DiscountPrice'] = [p for p in distinct_promos if ('€' in p)] + num_list
promo_sub_groups['coupons_InStore'] = [p for p in distinct_promos if ('TALÃO' in p) or ('CARTÃO' in p)]
promo_sub_groups['ad_flyer_OnlineAd'] = [p for p in distinct_promos if 'SEM' in p]
promo_sub_groups['ad_flyer_Solo'] = [p for p in distinct_promos if 'S/' in p]
promo_sub_groups['multi_tiered_BuyMoreSaveMore'] = [p for p in distinct_promos if 'L\dP\d' in p]
promo_sub_groups['multibuy_BuyXforY'] = [p for p in distinct_promos if 'PACK' in p]
promo_sub_groups['purchase_with_purchase'] = [p for p in distinct_promos if 'MENU' in p]

# COMMAND ----------

# tpr = promo_sub_groups['direct_discount'] + promo_sub_groups['lowered_price'] + promo_sub_groups['coupon']
# non_num_promo_type = ['without_discount', 'buy_more_pay_less', 'pack', 'cross_sell']
# num_promo_type = ["tpr", "direct_discount", "lowered_price", "coupon", "rebate"]

tpr = promo_sub_groups['tpr_DiscountPerc'] + promo_sub_groups['tpr_DiscountPrice'] 
non_num_promo_type = ['ad_flyer_OnlineAd', 'ad_flyer_Solo', 'multi_tiered_BuyMoreSaveMore', 'multibuy_BuyXforY', 'purchase_with_purchase']
num_promo_type = ["tpr_DiscountPerc", "tpr_DiscountPrice", "coupons_InStore"]


# COMMAND ----------

# DBTITLE 1,only for check
# subtract_two_lists(distinct_promos, promo_sub_groups['direct_discount'] +\
#                    promo_sub_groups['lowered_price'] + promo_sub_groups['coupon'] + promo_sub_groups['rebate'] + \
#                    promo_sub_groups['without_discount'] + promo_sub_groups['buy_more_pay_less'] + promo_sub_groups['pack'] + promo_sub_groups['cross_sell'])
subtract_two_lists(distinct_promos, promo_sub_groups['tpr_DiscountPerc'] + promo_sub_groups['tpr_DiscountPrice'] + promo_sub_groups['coupons_InStore'] + \
                   promo_sub_groups['ad_flyer_OnlineAd'] + promo_sub_groups['ad_flyer_Solo'] + promo_sub_groups['multi_tiered_BuyMoreSaveMore'] + \
                   promo_sub_groups['multibuy_BuyXforY'] + promo_sub_groups['purchase_with_purchase'])

# COMMAND ----------

# DBTITLE 1,do we have to include promo_type in Promo_id?
# pt_promo_final = pt_promo
# for promo in non_num_promo_type:
#   pt_promo_final = get_dummies(pt_promo_final, [promo]).drop(promo, f"{promo}_0")
  
# dummy_cols = subtract_two_lists(pt_promo_final.columns, pt_promo.columns)
# pt_promo_final = pt_promo_final.withColumn("promo_id", concat(F.col("DMDGROUP"), F.col("DMDUNIT"), F.col("start_date"), F.col("end_date"),\
#                                             F.col("Confirmation_Date"), F.col("promo_type"), F.col("mecanica"))).drop('mecanica', "start_date", "end_date", "Confirmation_Date")

pt_promo_final = pt_promo.withColumn("promo_id", concat(col("DMDGROUP"), col("DMDUNIT"), col("start_date"), col("end_date"),\
                                            col("Confirmation_Date"), col("mechanical"))).drop('mecanica', "start_date", "end_date", "Confirmation_Date")

# COMMAND ----------

print(pt_promo.count())
pt_promo = pt_promo.filter(col("mechanical").isin(subtract_two_lists(distinct_promos,['N/A'])))
print(pt_promo.count())

pt_promo = pt_promo.withColumn("promo_type", lit(" "))
for group in promo_sub_groups.keys():
  pt_promo = pt_promo.withColumn("promo_type", when(F.col("mechanical").isin(promo_sub_groups[group]), lit(group)).otherwise(F.col('promo_type')))
  
pt_promo = pt_promo.withColumn("tpr", when(F.col("mechanical").isin(tpr), lit(1)).otherwise(lit(0)))

# COMMAND ----------

pt_promo = pt_promo.withColumn('mechanical', regexp_replace('mechanical', '%', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', 'SP', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', '€', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', ',', '.'))\
                    .withColumn('mechanical', regexp_replace('mechanical', 'TALÃO', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', 'CARTÃO', ''))\
                    .withColumn('sell_out', regexp_replace('sell_out', ',', ''))\
                    .withColumn('Order_Curr', regexp_replace('Order_Curr', ',', '')).drop("mechanical")

# COMMAND ----------

pt_promo = pt_promo.filter(pt_promo.mechanical.like("%SP%"))
pt_promo = pt_promo.withColumn('mechanical', regexp_replace('mechanical', '%', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', ' ', ''))\
                    .withColumn('mechanical', regexp_replace('mechanical', 'SP', ''))\
                    .withColumnRenamed("mechanical", "perc_discount")

# Get action types as dummies
# pt_promo = get_dummies(pt_promo, ["action"]).drop("action")  

# Clean up estimate kamunits column
pt_promo = pt_promo.fillna(value=0, subset=["STATUS"])

for estimate_col in ["estimate_kam_units", "final_estimate_units"]:
  pt_promo = pt_promo.withColumn(estimate_col, regexp_replace(estimate_col, ',', '')).withColumn(estimate_col, regexp_replace(estimate_col, ' ', ''))
  pt_promo = pt_promo.withColumn(estimate_col, when(col(estimate_col).contains("�"), None).otherwise(col(estimate_col))) 
  pt_promo = pt_promo.withColumn(estimate_col, col(estimate_col).cast("integer"))
  pt_promo = pt_promo.fillna(value=0, subset=[estimate_col])

# pt_promo = pt_promo.withColumn("promo_estimate_ratio", col("final_estimate_units")/col("estimate_kam_units")).fillna(value=1, subset=["promo_estimate_ratio"])
pt_promo = pt_promo.withColumn("promo_id", concat(col("perc_discount"), col("DMDGROUP"), col("DMDUNIT"), col("start_date"), col("end_date"), col("Confirmation_Date")))

print (pt_promo.count()) 

# COMMAND ----------

display(pt_promo)

# COMMAND ----------

# for group in promo_sub_groups.keys():
#   pt_promo = pt_promo.withColumn(group, when(F.col("promo_type") == group, F.col("mecanica")).otherwise(lit('0')))

for group in promo_sub_groups.keys():
  if group in num_promo_type:
    pt_promo = pt_promo.withColumn(group, when(col("promo_type") == group, col("mecanica")).otherwise(lit(0)))
  else:
    pt_promo = pt_promo.withColumn(group, when(col("promo_type") == group, lit(1)).otherwise(lit(0)))
    

# COMMAND ----------

# pt_promo_final = pt_promo
# for promo in non_num_promo_type:
#   pt_promo_final = get_dummies(pt_promo_final, [promo]).drop(promo, f"{promo}_0")
  
# dummy_cols = subtract_two_lists(pt_promo_final.columns, pt_promo.columns)
# pt_promo_final = pt_promo_final.withColumn("promo_id", concat(F.col("DMDGROUP"), F.col("DMDUNIT"), F.col("start_date"), F.col("end_date"),\
#                                             F.col("Confirmation_Date"), F.col("promo_type"), F.col("mecanica"))).drop('mecanica', "start_date", "end_date", "Confirmation_Date")

pt_promo_final = pt_promo.withColumn("promo_id", concat(F.col("DMDGROUP"), F.col("DMDUNIT"), F.col("start_date"), F.col("end_date"),\
                                            F.col("Confirmation_Date"), F.col("promo_type"), F.col("mecanica"))).drop('mecanica', "start_date", "end_date", "Confirmation_Date")

# COMMAND ----------

def rename_cols(df):
  
  agg_types = ["sum", "avg", "max", "min"]
  for c in df.columns:
    for a in agg_types:
      if a in c:
        df = df.withColumnRenamed(c, c.replace(f"{a}(", "").replace(")", ""))
  
  return df     

# COMMAND ----------

def agg_promo(time_var):
  
  if time_var == "Week_Of_Year":
    df = pt_promo_final.drop("Promo_Planned_Month_Of_Year").drop_duplicates()
  elif time_var == "Month_Of_Year":
    df = pt_promo_final.drop("Week_Of_Year", "Promo_Planned_Week_Of_Year").drop_duplicates()
    
  group_cols = ["DMDGROUP", "DMDUNIT", time_var, "promo_id", f"Promo_Planned_{time_var}"]
  sum_cols = ["Order_Curr", "sell_out"] + [c for c in df.columns if "no_of" in c]
#   avg_cols = dummy_cols + num_promo_type
  avg_cols = list(promo_sub_groups.keys())
  
  df = df.groupBy(group_cols).agg({**{x: "sum" for x in sum_cols}, **{x: "mean" for x in avg_cols}}).drop("promo_id")
  
  df = rename_cols(df)
  df1 = df
  max_cols = [c for c in df.columns if "no_of" in c] + avg_cols
      
#   df = df.withColumn(diff_col[0], round(diff_col[0]))
  df = df.groupBy(["DMDGROUP", "DMDUNIT", time_var]).agg({**{x: "max" for x in max_cols},\
                                                          **{x: "sum" for x in ["Order_Curr", "sell_out"]}, **{f"Promo_Planned_{time_var}": "min"}})
  df = rename_cols(df)
  
  df = df.withColumn("order_sellout_ratio", col("Order_Curr")/col("sell_out")).withColumnRenamed("without_discount_S/PROMO", "without_discount_S_PROMO")
  df = df.na.fill(value = 1,subset=["order_sellout_ratio"])
    
  return df

# COMMAND ----------

pt_promo_agg_weekly = agg_promo("Week_Of_Year")
pt_promo_agg_monthly = agg_promo("Month_Of_Year")

# COMMAND ----------

display(pt_promo_agg_weekly)

# COMMAND ----------

display(pt_promo_agg_monthly)

# COMMAND ----------

print(pt_promo_agg_weekly.count() == pt_promo_agg_weekly.distinct().count() == pt_promo_agg_weekly.select("DMDGROUP", "DMDUNIT", "Week_Of_Year").distinct().count())
print(pt_promo_agg_weekly.count())
print(pt_promo_agg_monthly.count() == pt_promo_agg_monthly.distinct().count() == pt_promo_agg_monthly.select("DMDGROUP", "DMDUNIT", "Month_Of_Year").distinct().count())
print(pt_promo_agg_monthly.count())

# COMMAND ----------

display(pt_promo_agg_weekly.describe())
display(pt_promo_agg_monthly.describe())

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