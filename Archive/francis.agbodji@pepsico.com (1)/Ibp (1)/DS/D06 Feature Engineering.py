# Databricks notebook source
# MAGIC %md 
# MAGIC ##06- Feature Engineering
# MAGIC 
# MAGIC This script develops additional features for modeling.
# MAGIC 
# MAGIC Key source code (src) libraries to reference: feature_engineering, utilities

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

#initiate widget if needed, otherwise does nothing        
check_or_init_dropdown_widget("TIME_VAR_LOCAL","Week_Of_Year",["Week_Of_Year","Month_Of_Year"]) 

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

## Check that configurations exist for this script
required_configs = [
  DBA_MRD_EXPLORATORY, DBA_MRD_CLEAN, DBA_MRD,\
  TARGET_VAR,\
  DIST_PROXY_LEVEL_UNIV, DIST_PROXY_SELL_LEVELS_LIST
]


print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

# DBTITLE 1,Load Data
## Load cleansed MRD
try:
  
#   mrd_df = load_delta(DBA_DRIVER_FORECASTS, 11) #monthly
#   mrd_df = load_delta(DBA_DRIVER_FORECASTS, 7) #weekly    
  mrd_df = load_delta(DBA_DRIVER_FORECASTS)
  print("mrd size", mrd_df.count())
  
  delta_info = load_delta_info(DBA_DRIVER_FORECASTS)
  set_delta_retention(delta_info, '90 days')
  display(delta_info.history())
  
except:
  dbutils.notebook.exit("Exploratory mrd load failed")
  
try:
  calendar = load_delta(DBI_CALENDAR)
except:
  print("Calendar not available - please map to calendar using alternative method")
  
try:
  model_ids = load_delta(DBA_MODELIDS)
except:
  print("Model ID data load failed")
  
try:
  # This harcoding needs to be dropped once silver layer integration code is integrated in the pipeline
  silver_product_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/product-master"
  prod = load_delta(silver_product_path)
except:
  print("Model ID data load failed")  


# COMMAND ----------

# Loading driver data to drop columns                     
if TIME_VAR == 'Month_Of_Year':
  pricing = load_delta(DBI_PRICING_MONTHLY)
  covid_macro_vars = load_delta(DBI_EXTERNAL_VARIABLES_MONTHLY)
  weather = load_delta(DBI_WEATHER_MONTHLY)
  holidays_clean = load_delta(DBI_HOLIDAYS_MONTHLY)
  promo_df_es = load_delta(DBI_PROMO_ES_MONTHLY)
  promo_df_pt = load_delta(DBI_PROMO_PT_MONTHLY)
else: 
  pricing = load_delta(DBI_PRICING_WEEKLY)
  covid_macro_vars = load_delta(DBI_EXTERNAL_VARIABLES_WEEKLY)
  weather = load_delta(DBI_WEATHER_WEEKLY)
  holidays_clean = load_delta(DBI_HOLIDAYS_WEEKLY)
  promo_df_es = load_delta(DBI_PROMO_ES_WEEKLY)
  promo_df_pt = load_delta(DBI_PROMO_PT_WEEKLY)
  
media_clean = load_delta(DBI_MEDIA)

covid_macro_cols = list(set(covid_macro_vars.columns) - set([TIME_VAR] + ["country_name"]))
weather_cols = list(set(weather.columns) - set([TIME_VAR] + ["country_name"]))

pricing_cols = subtract_two_lists(pricing.columns, ['DMDGroup', 'PLANG_MTRL_GRP_VAL', 'SRC_CTGY_1_NM', TIME_VAR])\
                                            + ['ownSame_1_NET_PRICE_BAG', 'ownSame_2_NET_PRICE_BAG', 'ownSame_3_NET_PRICE_BAG',\
                                               'ownSame_4_NET_PRICE_BAG', 'ownSame_5_NET_PRICE_BAG', 'Dollar_Discount', 'Discount_Depth']

media_clean_cols = subtract_two_lists(media_clean.columns, ['Week_Of_Year', 'HRCHY_LVL_3_NM', 'SRC_CTGY_1_NM', 'BRND_NM'])
inventory_cols = [c for c in mrd_df.columns if "inventory" in c]

# COMMAND ----------

# DBTITLE 1,Include features related to maturity
win = Window.partitionBy("FCST_START_DATE", "MODEL_ID").orderBy(TIME_VAR)
win2 = Window.partitionBy("FCST_START_DATE", "MODEL_ID")

# COMMAND ----------

# historical age of the product
# first non-zero shipments 
mrd_df = mrd_df.withColumn("rnum", row_number().over(win)) \
    .withColumn("delta", first(when(col(TARGET_VAR) == 0, None).otherwise(col("rnum")), ignorenulls=True).over(win))\
    .withColumn("age_rolling", when(col("delta").isNull(), None).otherwise(col("rnum")-col("delta")+1)).drop("rnum", "delta")
# fill 0 for age after 
mrd_df = mrd_df.withColumn("age_aux", when(col(TIME_VAR)>=col('FCST_START_DATE'), 0).otherwise(col('age_rolling')))
# drop rolling age for simplicity
mrd_df = mrd_df.withColumn("age", max(col('age_aux')).over(win2)).drop("age_rolling", "age_aux")

# COMMAND ----------

# total number of non-zero shipments
mrd_df = mrd_df.withColumn("TARGET_VAR_copy", when(col(TIME_VAR)>=col('FCST_START_DATE'), 0).otherwise(col(TARGET_VAR)))
mrd_df = mrd_df.withColumn("non_zero_shipments_rolling", F.sum((F.col("TARGET_VAR_copy") != 0).cast("int")).over(win))
# drop rolling non-zero shipments for simplicity
mrd_df = mrd_df.withColumn("non_zero_shipments", max(col('non_zero_shipments_rolling')).over(win2)).drop("non_zero_shipments_rolling", "TARGET_VAR_copy")

# for sparsity
mrd_df = mrd_df.withColumn("shipments_sparsity", mrd_df.non_zero_shipments / mrd_df.age)
# display(mrd_df)

# COMMAND ----------

# DBTITLE 1,Generating Time Features
## Pulling time-oriented features
existing_cols = mrd_df.columns
mrd_df = get_time_vars(mrd_df, SHIPMENT_TIME_REF, funcs=TIME_VARS_TO_GENERATE).drop(SHIPMENT_TIME_REF)

## Review and checks
TIME_FEAT = subtract_two_lists(mrd_df.columns,existing_cols)
print('New time features:', TIME_FEAT)

mrd_df.cache()
print("mrd shape", mrd_df.count(), len(mrd_df.columns) ) 

# COMMAND ----------

# DBTITLE 1,Generating Ratios, Velocity, Counts
mrd_df = get_dummies(mrd_df, ["MonthIndex"])
mrd_df = calc_unique_count_pyspark(mrd_df, CATEGORY_FIELD, PROD_MERGE_FIELD)
mrd_df = calc_unique_count_pyspark(mrd_df, [CATEGORY_FIELD, BRAND_FIELD], PROD_MERGE_FIELD)
mrd_df = calc_unique_count_pyspark(mrd_df, [MARKET_FIELD], CUST_MERGE_FIELD)
mrd_df = calc_unique_count_pyspark(mrd_df, [CUST_MERGE_FIELD], [BRAND_FIELD, PROD_MERGE_FIELD])

mrd_df.cache()
mrd_df.count()

# COMMAND ----------

# DBTITLE 1,Lag Features
# logging inventory columns before lagging        
mrd_df = log_columns_ps(mrd_df, inventory_cols)
mrd_df.cache()
mrd_df.count()

#TO-DO: This will lag Syndicated Distribution as well which we don't want (that would be a driver forecast)
COLS_TO_LAG = [col_name for col_name in mrd_df.columns \
               if 'Weighted_Distribution' in col_name or (TARGET_VAR in col_name and 'orig' not in col_name)] + inventory_cols

COLS_TO_LAG = intersect_two_lists(COLS_TO_LAG, mrd_df.columns)
print('Columns to lag:', sorted(COLS_TO_LAG))

mrd_df = do_lags_N(mrd_df, order=TIME_VAR, lagvars = COLS_TO_LAG, 
                      n=LAGGED_PERIODS_1, partition=MODEL_ID_HIER+['FCST_START_DATE'],drop_lagvars=False)
print("mrd shape", mrd_df.count(), len(mrd_df.columns)) 

# COMMAND ----------

## Checking if 'original' columns for which we are lagging have been dropped
## Note - must leave TARGET_VAR in for now - this is handled in our modeling functions
cols_to_drop = list(set(COLS_TO_LAG) - set([TARGET_VAR]))  ## leakage cols to be dropped

existing_cols = mrd_df.columns
if len(intersect_two_lists(cols_to_drop, mrd_df.columns)) > 0:
  print (f"{intersect_two_lists(cols_to_drop, mrd_df.columns)} have not been dropped")
  mrd_df = mrd_df.drop(*cols_to_drop) #dropping now
  print(len(existing_cols) - len(cols_to_drop) == len(mrd_df.columns))   # should be true
else:
  print (f"All columns in {cols_to_drop} have been dropped")
print('TARGET_VAR retention check:', TARGET_VAR in mrd_df.columns)

## Imputing NA values using the average of each lagged column
cols = [ str(col) + '_lag' for col in COLS_TO_LAG]
cols_to_impute = [col + str(lag) for col in cols for lag in LAGGED_PERIODS_1]

w = Window().partitionBy('MODEL_ID', 'FCST_START_DATE')
for lag in cols_to_impute:
  mrd_df = mrd_df.withColumn(lag, when(col(lag).isNull(), avg(col(lag)).over(w)).otherwise(col(lag)))

mrd_df.cache()
print("mrd shape", mrd_df.count(), len(mrd_df.columns)) 

# COMMAND ----------

## log required variables
COLS_TO_LOG = media_clean_cols + pricing_cols + neilsen_cols_to_frcst
COLS_TO_LOG = intersect_two_lists(COLS_TO_LOG, mrd_df.columns)

try:
  print("Logging columns : " , COLS_TO_LOG)
  mrd_df = log_columns_ps(mrd_df, COLS_TO_LOG)
  mrd_df.cache()
  mrd_df.count()
except:
  print('No columns were logged!')


# COMMAND ----------

# index columns
COLS_TO_INDEX = COLS_TO_INDEX + ['SRC_CTGY_2_NM', 'PCK_SIZE_SHRT_NM'] + ['ABC', 'XYZ']
COLS_TO_INDEX = set(COLS_TO_INDEX)

COLS_TO_INDEX = intersect_two_lists(COLS_TO_INDEX, mrd_df.columns)


try:
  print("Indexing columns : ", COLS_TO_INDEX)
  print(mrd_df.count())
  mrd_df = index_cols_ps(mrd_df, COLS_TO_INDEX, "_index")
  mrd_df.cache()
  print(mrd_df.count())
except:
  print('No columns indexed!')

# COMMAND ----------

# DBTITLE 1,STD Treatment / Transformations
# ## Note - this occurs after cleansing as we need to develop variables on the treated target and need to cleanse on all variables
# ## Note - this is set for after lagging ... but log transformation has occurred already in pipeline
# ## TODO - confirm that we want to set in this way for now

driver_cols = list(set(covid_macro_cols + weather_cols ))
driver_cols = list(set(driver_cols))
COLS_TO_STD = sorted(set([x for x in mrd_df.columns for y in driver_cols if y in x]))
COLS_TO_STD = COLS_TO_STD + [col_name for col_name in mrd_df.columns if ('Weighted_Distribution' in col_name) or ('distinct' in col_name)]

cols_std = intersect_two_lists(COLS_TO_STD, mrd_df.columns)
print("len of columns: ", len(cols_std))
print("Standardizing level : ", STD_LEVEL)

try:
  mrd_df = standardize_columns_pyspark(mrd_df, ['FCST_START_DATE'], cols_std)
  print("Standardizing columns : ", sorted(cols_std))
except:
  print("No columns were standardized!")

mrd_df.cache()
mrd_df.count()

# COMMAND ----------

# DBTITLE 1,Promo Feature Engg
# Function to flag top top_num dmdunits with high value of given promo var, or dmdunits with positive avg disc percentages
def get_top_by_promo_var(group_df, merge_df, top_num, group_field, promo_var, agg_type = 'sum'):
 
  if agg_type == 'avg':
    field_promo_var_df = group_df.groupBy(group_field).agg(avg(promo_var).alias(promo_var))
    field_promo_var_df = field_promo_var_df.filter(col(promo_var) > 0).withColumn(promo_var.replace("promo", "promo_pos"), lit(1)).drop(promo_var)
  else:
    field_promo_var_df = group_df.groupBy(group_field).agg(sum(promo_var).alias(promo_var))\
                               .orderBy(col(promo_var).desc()).limit(top_num)
    field_promo_var_df = field_promo_var_df.withColumn(promo_var.replace("promo", "promo_high"), lit(1)).drop(promo_var)
  
  flag_col = subtract_two_lists(field_promo_var_df.columns, [group_field])
  if group_df.select(TIME_VAR).distinct().count() < merge_df.select(TIME_VAR).distinct().count():
    field_promo_var_df = field_promo_var_df.withColumnRenamed(flag_col[0], flag_col[0] + '_last_yr')
    flag_col = subtract_two_lists(field_promo_var_df.columns, [group_field])

  df1 = merge_df.join(field_promo_var_df, on = group_field, how = 'left')
  df1 = df1.na.fill(0, subset = flag_col)
  
  return df1

# COMMAND ----------

# Function to capture continuous promotional activities  
if TIME_VAR == "Month_Of_Year":
  time_thresh = 3
  promo_count_thresh = 10
  promo_no_of_days_thresh = 10
elif TIME_VAR == "Week_Of_Year":
  time_thresh = 12
  promo_count_thresh = 5
  promo_no_of_days_thresh = 5
  
def get_consecutive_flags(df, promo_var, flag_col, target_thres):
  
  df1 = get_velocity_flag_pyspark(df, [PROD_MERGE_FIELD, CUST_MERGE_FIELD], promo_var, TIME_VAR, velocity_type="high", target_threshold = target_thres, time_threshold = time_thresh)
  df1 = df1.withColumnRenamed('High_Velocity_Flag', flag_col)
  
  return df1

# COMMAND ----------

# Get cutoff for past 1 year of history
max_time_var = mrd_df.filter(col(TARGET_VAR).isNotNull()).select(max(TIME_VAR)).collect()[0][0]

# COMMAND ----------

# Get flags for full historicals
mrd_df = get_top_by_promo_var(mrd_df, mrd_df, 100, PROD_MERGE_FIELD, "promo_no_of_days", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df, mrd_df, 100, PROD_MERGE_FIELD, "promo_on", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df, mrd_df, 100, PROD_MERGE_FIELD, "promo_count", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df, mrd_df, 100, PROD_MERGE_FIELD, "promo_avg_DiscountPerc", agg_type = 'avg')

# Get flags for previous 1 year of history
mrd_df = get_top_by_promo_var(mrd_df.filter((max_time_var - 100 < col(TIME_VAR)) & (col(TIME_VAR) <= max_time_var)), mrd_df, 100, PROD_MERGE_FIELD, "promo_no_of_days", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df.filter((max_time_var - 100 < col(TIME_VAR)) & (col(TIME_VAR) <= max_time_var)), mrd_df, 100, PROD_MERGE_FIELD, "promo_on", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df.filter((max_time_var - 100 < col(TIME_VAR)) & (col(TIME_VAR) <= max_time_var)), mrd_df, 100, PROD_MERGE_FIELD, "promo_count", agg_type = 'sum')
mrd_df = get_top_by_promo_var(mrd_df.filter((max_time_var - 100 < col(TIME_VAR)) & (col(TIME_VAR) <= max_time_var)), mrd_df, 100, PROD_MERGE_FIELD, "promo_avg_DiscountPerc", agg_type = 'avg')

# Get flags for consecutive promo vars
mrd_df = get_consecutive_flags(mrd_df, 'promo_on', 'promo_consec_binary_flag', 1)
mrd_df = get_consecutive_flags(mrd_df, 'promo_count', 'promo_consec_count', promo_count_thresh)
mrd_df = get_consecutive_flags(mrd_df, 'promo_no_of_days', 'promo_consec_days_flag', promo_no_of_days_thresh)

# Momentum and Ratio feature calculation for promo count and promo dsys
mrd_df = calc_ratio_vs_average_pyspark(mrd_df, PROD_MERGE_FIELD, 'promo_count')
mrd_df = calc_ratio_vs_average_pyspark(mrd_df, PROD_MERGE_FIELD, 'promo_no_of_days')

mrd_df = calc_ratio_vs_prior_period_pyspark(mrd_df, PROD_MERGE_FIELD, 'promo_count', TIME_VAR)
mrd_df = calc_ratio_vs_prior_period_pyspark(mrd_df, PROD_MERGE_FIELD, 'promo_no_of_days', TIME_VAR)

# COMMAND ----------

# Impute nulls in momentum and ratio features with zero
promo_momentum_cols = [c for c in mrd_df.columns if ("promo_" in c) & ("momentum" in c)]
mrd_df = mrd_df.na.fill(0, subset = promo_momentum_cols)

# COMMAND ----------

# DBTITLE 1,Drop Leakage Columns
LAGGED_FEAT = [col_name for col_name in mrd_df.columns if "_lag" in col_name] 
INDEX_FEAT = [s for s in mrd_df.columns if "_index" in s]


# Note: dropping holiday features. need to revist the construction 
HOLIDAY_FEAT = subtract_two_lists(holidays_clean.columns, [TIME_VAR, "hol_spain_hol_flag", "hol_portugal_hol_flag"])
all_hol_cols = []
for hol in HOLIDAY_FEAT:
  for i in range(1, 5):
    hol_cols = [hol, f"{hol}_lag{i}", f"{hol}_lead{i}"] 
    all_hol_cols = all_hol_cols + hol_cols

original_cols_lag = list(set([col_string[:col_string.index("_lag")] for col_string in LAGGED_FEAT]))   ## list comp to find "orig" cols - lag
original_cols_idx = list(set([col_string[:col_string.index("_index")] for col_string in INDEX_FEAT]))  ## list comp to find "orig" cols - index
original_cols = original_cols_lag + original_cols_idx

# drop_cols = list(set(original_cols + [TARGET_VAR + '_orig'] + HOLIDAY_FEAT) - set(INDEX_FEAT))    ## ensure no crossover
drop_cols = list(set(original_cols + [TARGET_VAR + '_orig']) - set(INDEX_FEAT))    ## ensure no crossover

# Following step is to get MonthIndex columns with less than 1 index   
MONTH_INDEX_FEAT = [c for c in mrd_df.columns if "MonthIndex" in c]
pos_month_index = [m for m in MONTH_INDEX_FEAT if any(char.isdigit() for char in m) and int(m[11:]) > 0]
drop_month_index = subtract_two_lists(MONTH_INDEX_FEAT, pos_month_index)

drop_cols = drop_cols + ['QTY', 'QuarterYear', 'CV', 'country_name', 
                         'Quarter_Of_Year', 'HSTRY_TMFRM_STRT_DT', 'PLANG_MTRL_STTS_NM', 'PLANG_MTRL_DEL_DT'] \
                        + drop_month_index     #Adding stuff that got missed above
hol_cols_in_drop_cols = intersect_two_lists(drop_cols, all_hol_cols)     # Making sure no holiday column is getting dropped
drop_cols = subtract_two_lists(drop_cols, hol_cols_in_drop_cols)

if TARGET_VAR in drop_cols:
  print('Removing {} from drop columns'.format(TARGET_VAR))
  drop_cols.remove(TARGET_VAR)

if TIME_VAR in drop_cols:
  print('Removing {} from drop columns'.format(TIME_VAR))
  drop_cols.remove(TIME_VAR)
  
mrd_df = mrd_df.drop(*drop_cols)
print('Dropped columns:', sorted(drop_cols))

mrd_df.cache()
mrd_df.count()

# COMMAND ----------

# DBTITLE 1,Clean Columns Output
## Cleanse column names if necessary - occasionally causes an issue (ASCI, UTF8, etc.)
new_names = mrd_df.columns
new_names = [re.sub('[^A-Za-z0-9_]+', '', s) for s in new_names]
new_names = [re.sub(' ', '', s) for s in new_names]
new_names = [re.sub(',', '', s) for s in new_names]
new_names = [re.sub('=', '', s) for s in new_names]

mrd_df = mrd_df.toDF(*new_names)
print('TARGET_VAR check:', TARGET_VAR in mrd_df.columns)  ## should read TRUE
print('Cols in final_df', len(mrd_df.columns))
print('All columns:', sorted(mrd_df.columns))

# COMMAND ----------

## Note - not printing out NA counts here given the runtime associated with doing so
## If there are any NaNs, they will be filled with 0
print('NaNs in mrd_df, if present, to be filled with 0')
display(mrd_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in mrd_df.columns]))
mrd_df = mrd_df.na.fill(-99999, subset=neilsen_cols_to_frcst)
mrd_df = mrd_df.fillna(0)

# COMMAND ----------

# MAGIC %md #### Removing Phased out records

# COMMAND ----------

# model_ids = load_delta('dbfs:/mnt/adls/Tables/Month_Of_Year/DBA_MODELIDS')
model_ids = model_ids.select("MODEL_ID", "DMDUNIT").distinct()
mrd_df = mrd_df.join(model_ids, on="MODEL_ID", how="left")

prod = prod.withColumnRenamed("PROD_CD", "DMDUNIT")
prod = prod.select("DMDUNIT", "PROD_STTS", "MTRL_STTS_NM", "DEL_DT")
calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct()

if PROD_MERGE_FIELD=="DMDUNIT":
  mrd_df = mrd_df.join(prod, on=PROD_MERGE_FIELD, how='left')
  mrd_df = mrd_df.join(calendar_df, on=TIME_VAR, how='left')
  mrd_del = mrd_df.filter((col("PROD_STTS")=="DELETE") & (col("MTRL_STTS_NM")=="DELETE") )
  mrd_del = mrd_del.filter(col(CALENDAR_DATEVAR) < col("DEL_DT"))
  mrd_active = mrd_df.filter((col("PROD_STTS")=="ACTIVE") | (col("MTRL_STTS_NM")=="ACTIVE"))
  mrd_df = mrd_active.union(mrd_del.select(mrd_active.columns))
  mrd_df = mrd_df.drop(*["DMDUNIT", "PROD_STTS", "MTRL_STTS_NM", "DEL_DT", CALENDAR_DATEVAR])

# COMMAND ----------

# DBTITLE 1,Save Output
## Save to DBA_MRD - write as delta table to dbfs
save_df_as_delta(mrd_df, DBA_MRD, enforce_schema=False)

## Load and review delta table information
delta_info = load_delta_info(DBA_MRD)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Create subset of MRD for experiments to be run on
## Get list of MODEL_ID's to be filtered for
if TIME_VAR == "Month_Of_Year":
  subset_mi_version = 4
else:
  subset_mi_version = 9
  
sample_mi = convertDFColumnsToList(load_delta("dbfs:/mnt/adls/Tables/chosen_mi", subset_mi_version), "value")
mrd_sub = mrd_df.filter(col("MODEL_ID").isin(sample_mi))

# COMMAND ----------

## Write out the MRD Subset to another table
save_df_as_delta(mrd_sub, DBA_MRD_SUB, enforce_schema=False)

## Load and review delta table information
delta_info = load_delta_info(DBA_MRD_SUB)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# MAGIC %md #### Creating Driver Categorization table

# COMMAND ----------

featureSchema = StructType([       
    StructField('variable', StringType(), True)
])

mrd_features = spark.createDataFrame(data=[[x] for x in mrd_df.columns], schema = featureSchema)

mrd_features = mrd_features.withColumn("driver_group", when(col("variable").startswith("media_"), lit("Media"))\
                                       .otherwise(when(col("variable").startswith("cc_"), lit("Covid"))\
                                       .otherwise(when(col("variable").startswith("cg_"), lit("Covid"))\
                                       .otherwise(when(col("variable").startswith("weather_"), lit("Weather"))\
                                       .otherwise(when(col("variable").startswith("ownSame_"), lit("Cannibalization"))\
                                       .otherwise(when(col("variable").startswith("promo_"), lit("Promo"))\
                                       .otherwise(when(col("variable").startswith("gm_"), lit("Google_mobility"))\
                                       .otherwise(when(col("variable").startswith("pg_"), lit("Demographics"))\
                                       .otherwise(when(col("variable").startswith("ph_"), lit("Demographics"))\
                                       .otherwise(when(col("variable").startswith("pl_"), lit("Demographics"))\
                                       .otherwise(when(col("variable").contains("_inventory"), lit("Inventory"))\
                                       .otherwise(when(col("variable").startswith("wb_"), lit("Macroeconomic"))\
                                       .otherwise(when(col("variable").startswith("hol_"), lit("Holidays"))\
                                       .otherwise(when(col("variable").startswith("neil_"), lit("Neilsen"))\
                                       .otherwise(when(col("variable").contains("PRICE_"), lit("Pricing"))\
                                       .otherwise(when(col("variable").contains("Discount"), lit("Pricing"))))))))))))))))))
mrd_features = mrd_features.na.fill("Base", subset=["driver_group"])
mrd_features = mrd_features.withColumn("base_vs_incremental", when(col("driver_group")=="Pricing", lit("Incremental"))\
                                       .otherwise(when(col("driver_group")=="Promo", lit("Incremental"))\
                                       .otherwise(when(col("driver_group")=="Media", lit("Incremental")))))
mrd_features = mrd_features.na.fill("Base", subset=["base_vs_incremental"])


# COMMAND ----------

features_list = convertDFColumnsToList(mrd_features, "variable")
secondary_ml_features = [x for x in features_list if "_lag" in x] + ["MODEL_ID", "FCST_START_DATE", TARGET_VAR]
mrd_features = mrd_features.withColumn("include_in_secondary_ml", when(col("variable").isin(secondary_ml_features), lit(0)).otherwise(lit(1)) )

# COMMAND ----------

# DBTITLE 1,Save Driver Categories
# DBP_DECOMP
save_df_as_delta(mrd_features, DBP_DECOMP, enforce_schema=False)

## Load and review delta table information
delta_info = load_delta_info(DBP_DECOMP)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

