# Databricks notebook source
# MAGIC %md 
# MAGIC ## Driver Forecasting
# MAGIC 
# MAGIC This script predicts a list of features using traditional time series (arima, holt, etc.), lagging last year values or historical averaging. 
# MAGIC * It can be used as a benchmark for the target variable to compare against the ML models
# MAGIC * It can also be used to predict future variables in time (e.g., weather, distribution, etc.)
# MAGIC * It runs the predictions using a rolling holdout method and outputs the prediction table in a similar format to Stage 1 & 2 models
# MAGIC 
# MAGIC Note(s):
# MAGIC 
# MAGIC * Requires pmdamira be installed on cluster

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

#Check configurations exist for this script
required_configs = [DBA_MRD_CLEAN, DBA_MODELIDS]
print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# override config
DIST_PROXY_LEVEL_UNIV = [TIME_VAR, 'HRCHY_LVL_1_NM']

# COMMAND ----------

# Used only in case of Development Runs
if TIME_VAR =='Week_Of_Year':
  DATA_VERSION = 37 #version of the delta table
elif TIME_VAR =='Month_Of_Year':
  DATA_VERSION = 38 #version of the delta table

# COMMAND ----------

# DBTITLE 1,Load data
# mrd_df = load_delta(DBA_MRD_CLEAN, DATA_VERSION)
mrd_df = load_delta(DBA_MRD_CLEAN)

model_ids = mrd_df.select('MODEL_ID').distinct()

# loading calendar data
calendar = load_delta(DBI_CALENDAR)
calendar = calendar.select([TIME_VAR, CALENDAR_DATEVAR]).dropDuplicates()

# loading inventory data
inventory_df = load_delta(DBI_INVENTORY)

# Neilsen data
nielson = load_delta(DBI_NIELSEN)

# load holiday and promo data
if TIME_VAR == 'Month_Of_Year':
  holidays_clean = load_delta(DBI_HOLIDAYS_MONTHLY)
  promo_df_es = load_delta(DBI_PROMO_ES_MONTHLY)
  promo_df_pt = load_delta(DBI_PROMO_PT_MONTHLY)
else:
  holidays_clean = load_delta(DBI_HOLIDAYS_WEEKLY)
  promo_df_es = load_delta(DBI_PROMO_ES_WEEKLY)
  promo_df_pt = load_delta(DBI_PROMO_PT_WEEKLY)

# COMMAND ----------

# DBTITLE 1,creating a subset of mrd for development(Remove in Prod/cleanup)
# mrd_stat = mrd_df.groupby("MODEL_ID").agg(min(TIME_VAR).alias('min'), 
#                                          max(TIME_VAR).alias('max'), 
#                                          countDistinct(TIME_VAR).alias('count'))

# mrd_dev = mrd_stat.withColumn('tiebreak', monotonically_increasing_id())\
#                   .withColumn("row",rank().over(Window.partitionBy("min").orderBy('tiebreak')))\
#                   .filter(col('row') == 1).drop('row', 'tiebreak')

# dev_ids = mrd_dev.select('MODEL_ID').rdd.flatMap(lambda x: x).collect()
# mrd_df = mrd_df.filter(col('MODEL_ID').isin(dev_ids))

# COMMAND ----------

#Palaash TODO: This needs to be modularised
# Filling the rolling FORECAST_START_DATES list  based on FORECAST_START_DATE, PERIODS and INTERVAL
for i in range(PERIODS):
  next_idx = i*INTERVAL
  if (next_idx<PERIODS):
    date_var=next_time_var(TIME_VAR,FORECAST_START_DATE,next_idx,calendar,CALENDAR_DATEVAR)
    FORECAST_START_DATES.append(date_var)
    
# Appended next week/next month from max_mrd_date to make sure number of forward forecast happing from the max_mrd_date
max_mrd_date = mrd_df.select(max(col(TIME_VAR))).collect()[0][0]
max_forecast_start_date = next_time_var(TIME_VAR,max_mrd_date,1,calendar,CALENDAR_DATEVAR) # Finding the next week/month 
FORECAST_START_DATES.append(max_forecast_start_date)

# Checking max allowed FORECAST_START_DATES. It will make sure the driver data is present. 
# Driver data is present until max mrd date
FORECAST_START_DATES = [x for x in FORECAST_START_DATES if x <= max_forecast_start_date]

FORECAST_START_DATES = list(set(FORECAST_START_DATES))
print(FORECAST_START_DATES)

# COMMAND ----------

#TO-DO: Should add country_name to DBA_MODELIDS table so we can read all ID columns directly from there rather than entire dataset
#TO-DO: If we have separate output tables for things such as this it needs to be consistent for all tables
if TIME_VAR =='Week_Of_Year':
  external_vars = load_delta(DBI_EXTERNAL_VARIABLES_WEEKLY)
  weather = load_delta(DBI_WEATHER_WEEKLY_HISTORICALS, 1)
elif TIME_VAR =='Month_Of_Year':
  external_vars = load_delta(DBI_EXTERNAL_VARIABLES_MONTHLY)
  weather = load_delta(DBI_WEATHER_MONTHLY_HISTORICALS, 1)

# filter external variables from master_df
external_columns = subtract_two_lists(set(external_vars.columns).intersection(mrd_df.columns),  [TIME_VAR,"country_name"])

# COMMAND ----------

## construct the variables to forward forcast
# reading this file since govt policies are mostly catagorical and needs to be fwd filled if in mrd
# filtering the non catagorical values
covid_govt_policies = read_file(covid_govt_policies_path).columns
covid_govt_policies = [x for x in covid_govt_policies if ('_flag' not in x) and ('confirmed_cases' not in x) and ('deaths' not in x) ] 

# COMMAND ----------

# get the top holidaya columns from mrd_clean
holiday_columns = sorted(set([x for x in holidays_clean.columns for y in mrd_df.columns if x in y]))
holidays_clean = holidays_clean.select(holiday_columns)

# get lag and lead of holiday
holiday_to_lag = subtract_two_lists(holiday_columns, [TIME_VAR, 'hol_spain_hol_flag', 'hol_portugal_hol_flag'])
holiday_lag = do_lags_N(holidays_clean, TIME_VAR, holiday_to_lag, 4, partition = None, drop_lagvars=False)

# drop holiday columns
columns_drop = sorted(set([y for x in holiday_to_lag for y in mrd_df.columns if x in y]))
mrd_df = mrd_df.drop(*columns_drop + ['hol_spain_hol_flag', 'hol_portugal_hol_flag'])

# Keeping the columns that are to be replaced in mrd
holiday_lag = holiday_lag.select([TIME_VAR, 'hol_spain_hol_flag', 'hol_portugal_hol_flag'] + columns_drop)
holiday_lag = holiday_lag.select(*[col(x).alias(x.replace('_lag', '_LAG')) for x in holiday_lag.columns])

# COMMAND ----------

# Dropping Promo columns for POC
promo_es_cols = subtract_two_lists(promo_df_es.columns, ['DMDGROUP', 'DMDUNIT', TIME_VAR])
promo_pt_cols = subtract_two_lists(promo_df_pt.columns, ['DMDGROUP', 'DMDUNIT', TIME_VAR])
promo_cols = intersect_two_lists(promo_es_cols + promo_pt_cols, mrd_df.columns)
existing_cols = mrd_df.columns
mrd_df = mrd_df.drop(*promo_cols)
print(len(mrd_df.columns) == len(existing_cols) - len(promo_es_cols + promo_pt_cols))

# COMMAND ----------

# DBTITLE 1,Feature Engineering: Distribution Proxy Creation
## Generating sum of TARGET by customer ('HRCHY_LVL_1_NM') and by TIME (week or month)

store_sales_universe = mrd_df.groupby(DIST_PROXY_LEVEL_UNIV)\
                             .agg(sum(TARGET_VAR)\
                             .alias('univ_sales'))\
                             .na.fill(0,'univ_sales')

LISTINDEX =  list(map(str,list(range(0, len(DIST_PROXY_SELL_LEVELS_LIST)))))       ## creating numeric suffixes
sell_sales_list = ['sell_sales' + '_L' + i for i in LISTINDEX]                     ## creating list for loop below
Wtd_Distribution_list = ['Weighted_Distribution_Proxy' + '_L' + i for i in LISTINDEX]   ## creating list for loop below

print('Pre-Merge Shape:', (mrd_df.count(), len(mrd_df.columns)))

for EACH_LEVEL in DIST_PROXY_SELL_LEVELS_LIST:
  
  ## Set up the looping designations - by different product levels (L0 - highest level)
  full_sales_level = DIST_PROXY_LEVEL_UNIV + [EACH_LEVEL]  
  sell_sales = sell_sales_list[DIST_PROXY_SELL_LEVELS_LIST.index(EACH_LEVEL)]
  Wtd_Distribution = Wtd_Distribution_list[DIST_PROXY_SELL_LEVELS_LIST.index(EACH_LEVEL)]
  select_cols = DIST_PROXY_LEVEL_UNIV + [EACH_LEVEL] + [Wtd_Distribution]
  print(select_cols)
  
  ## Generating the sum of TARGET by customer, time, AND NOW PRODUCT (based on product level)
  store_sales_selling_item = mrd_df.groupby(full_sales_level)\
                                   .agg(sum(TARGET_VAR)\
                                   .alias(sell_sales))\
                                   .na.fill(0, sell_sales)
  
  ## The ratio of these gives us a sense of how shipments for that customer break down across products
  dist_proxy = store_sales_selling_item.join(store_sales_universe, on=DIST_PROXY_LEVEL_UNIV, how='left')\
                                       .withColumn(Wtd_Distribution, col(sell_sales)/col("univ_sales"))\
                                       .na.fill(0, Wtd_Distribution)\
                                       .select(select_cols)
  
  ## Visualize/validate the above results in histogram
  plt.hist(np.array(dist_proxy.select(Wtd_Distribution).distinct().collect()),50)
  plt.title('Histogram: Weighted_Distribution_' + sell_sales + ' --- ' + EACH_LEVEL)
  plt.show()
  
  ## Iteratively merge each on core dataframe
  mrd_df = mrd_df.join(dist_proxy, on=full_sales_level, how='left')

## Checking shape after loop 
print('Post-Merge Shape:', (mrd_df.count(),len(mrd_df.columns)))

# COMMAND ----------

# DBTITLE 1,Generating Top / Bottom "Performers"
existing_cols = mrd_df.columns

## For 'top' performers - in 80th+ percentile when looking at TARGET_VAR
## Note: removing [stddev] from function since this requires time-related fills 
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, PROD_MERGE_FIELD, TARGET_VAR, [sum, max], "top")
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, CUST_MERGE_FIELD, TARGET_VAR, [sum, max], "top")
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, LOC_MERGE_FIELD,  TARGET_VAR, [sum, max], "top")

## For 'bottom' performers - 20th percentile when looking at TARGET_VAR
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, PROD_MERGE_FIELD, TARGET_VAR, [sum, max], "bottom")
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, CUST_MERGE_FIELD, TARGET_VAR, [sum, max], "bottom")
mrd_df = get_performance_flag(mrd_df, MARKET_FIELD, LOC_MERGE_FIELD, TARGET_VAR,  [sum, max], "bottom")

## Review and checks
AUTO_DEMAND_FEAT = subtract_two_lists(mrd_df.columns, existing_cols)
print('New engineered features:', sorted(AUTO_DEMAND_FEAT))

mrd_df.cache()
print("mrd shape", mrd_df.count(), len(mrd_df.columns) )   ## Note: wont be same size if we used above time cutoff used

# COMMAND ----------

# DBTITLE 1,Velocity, ratio
existing_cols = mrd_df.columns

mrd_df = calc_ratio_vs_prior_period_pyspark(mrd_df, MODEL_ID_HIER, [TARGET_VAR], TIME_VAR)
mrd_df = calc_ratio_vs_average_pyspark(mrd_df, MODEL_ID_HIER, [TARGET_VAR])

mrd_df = get_velocity_flag_pyspark(df              = mrd_df, 
                                    group_cols       = MODEL_ID_HIER, 
                                    sales_var        = TARGET_VAR + "_orig", 
                                    time_var         = TIME_VAR, 
                                    velocity_type    = "high", 
                                    target_threshold = HIGH_VELOCTIY_TARGET_THRESH, 
                                    time_threshold   = HIGH_VELOCTIY_TIME_THRESH)

mrd_df = get_velocity_flag_pyspark(df              = mrd_df, 
                                    group_cols       = MODEL_ID_HIER, 
                                    sales_var        = TARGET_VAR + "_orig", 
                                    time_var         = TIME_VAR, 
                                    velocity_type    = "low", 
                                    target_threshold = LOW_VELOCITY_TARGET_THRESH, 
                                    time_threshold   = LOW_VELOCTIY_TIME_THRESH)

## Review and checks
AUTO_DEMAND_FEAT = subtract_two_lists(mrd_df.columns, existing_cols)
print('New engineered features:', sorted(AUTO_DEMAND_FEAT))

mrd_df.cache()
mrd_df.count()

# COMMAND ----------

# create imputation list to forward fill
impute_ffill_list = [x[0] for x in mrd_df.dtypes if x[1]=='string'] \
                    + [x for x in mrd_df.columns if x.endswith('_index')]\
                    + [x for x in mrd_df.columns if ('_performer_' in x) or ('_Velocity_' in x)]\
                    + ['STAT_CLUSTER', 'PLANG_PROD_8OZ_QTY', 'PLANG_PROD_KG_QTY', 'PLANG_MTRL_EA_PER_CASE_CNT', 'CV']\
                    + list(set(mrd_df.columns).intersection(covid_govt_policies))

# FORECAST_LAG_VARS = sorted(set([y for x in holidays_clean.columns for y in mrd_df.columns if x in y]))
# FORECAST_LAG_VARS.remove(TIME_VAR)
FORECAST_LAG_VARS = None 

# COMMAND ----------

# DBTITLE 1,Future Shell
# create future shell
if TIME_VAR=="Week_Of_Year":
  calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct()
  temp = mrd_df.join(calendar_df, on=[TIME_VAR], how="left")
  max_mrd_date = temp.select(max (TIME_VAR)).collect()[0][0]
  temp = temp.withColumn('time_var_temp', date_add(col(CALENDAR_DATEVAR), NUM_FWD_FRCST*7)).select(to_timestamp(col("time_var_temp")).alias("time_var_temp"))
  temp = temp.join(calendar_df, temp.time_var_temp==calendar_df.Week_start_date, how="left")
  max_date = temp.select(max(TIME_VAR)).collect()[0][0]
  temp = temp.drop('time_var_temp', TIME_VAR, CALENDAR_DATEVAR)
  
elif TIME_VAR=="Month_Of_Year":
  calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct()
  temp = mrd_df.join(calendar_df, on=[TIME_VAR], how="left")
  max_mrd_date = temp.select(max (TIME_VAR)).collect()[0][0]
  temp = temp.withColumn('time_var_temp', F.add_months(temp[CALENDAR_DATEVAR], NUM_FWD_FRCST)).select(col("time_var_temp"), date_format(col("time_var_temp"), "yyyyMM").alias("time_var_temp2"))
  max_date = temp.select(max("time_var_temp2")).collect()[0][0]
  temp = temp.drop('time_var_temp') # CALENDAR_DATEVAR

# max_date = max_mrd_date + NUM_FWD_FRCST
min_date = mrd_df.select(min (TIME_VAR)).collect()[0][0]
print("max date: {}, min_date: {}".format(max_date,min_date))

#Note: This is leading to dates > 52 weeks when rolling forecast spans over december / january
#Create cartesian product of MODEL_ID/DATE
master_df = spark.sql("SELECT sequence(" + str(min_date)  +" , " + str(max_date) + ") as " + TIME_VAR  +"    ")
master_df = master_df.withColumn(TIME_VAR, explode(col(TIME_VAR)))
master_df = master_df.withColumn(TIME_VAR,master_df[TIME_VAR].cast(IntegerType()))
master_df = master_df.crossJoin(model_ids)
master_df = master_df.join(calendar, on=[TIME_VAR], how="inner") #Join to calendar to remove > 52 weeks rows and get true date field

# filtering additional null columns added by above cross join
min_filter_df = mrd_df.groupby("MODEL_ID").agg(min(TIME_VAR).alias('min_time'))
master_df = master_df.join(mrd_df, on=['MODEL_ID', TIME_VAR], how='left')
master_df = master_df.join(min_filter_df, on='MODEL_ID', how='left')\
                     .filter(col(TIME_VAR)>=col('min_time')).drop('min_time')
print(master_df.count())

# COMMAND ----------

# DBTITLE 1,Forward Fill
# This forward fill is done to replicate the catagorical values into future shell
#Forward fill required null variables
try:
  master_df = impute_cols_ts(master_df, list(set(impute_ffill_list)), order_col=["MODEL_ID", TIME_VAR], fill_type = "ffill")
except:
  print("Forward fill imputation not required")

# COMMAND ----------

# DBTITLE 1,add holiday features
# adding holiday columns
master_df = master_df.join(holiday_lag, on=[TIME_VAR], how='left')

# COMMAND ----------

# DBTITLE 1,Lag Forecast
# MAGIC %md 
# MAGIC Here we use last year values as the assumption for a particular driver going forward.  We perform this using the lag calculation as it is less computationally expensive than running naive time series.

# COMMAND ----------

# add one year lag
print(FORECAST_LAG_VARS)

#TO-DO: This code needs to be tested given configuration updates, especially for Monthly data
if FORECAST_LAG_VARS != None:
  #Join historical data to needed forecasting dates
  historical_data = master_df.select(["MODEL_ID",TIME_VAR]+FORECAST_LAG_VARS)

  #Lag data for the same period last year and two years ago.  Two years is used as a default for when we have missing data last year 
  historical_dates2 = do_lags_N(historical_data, TIME_VAR, FORECAST_LAG_VARS, [ONE_YEAR_LAG], "MODEL_ID")
  historical_dates3 = do_lags_N(historical_data, TIME_VAR, FORECAST_LAG_VARS, [TWO_YEAR_LAG], "MODEL_ID")
  historical_data = historical_data.join(historical_dates2, on=["MODEL_ID",TIME_VAR], how="left")
  historical_data = historical_data.join(historical_dates3, on=["MODEL_ID",TIME_VAR], how="left")

  #Set future data to none (not actualized as of yet)
  for i in  FORECAST_LAG_VARS:
    historical_data = historical_data.withColumn(i, lit(None))

  #Impute NULL variables (i.e., future values which are not actualized as of yet)
  #We will first impute to lag52 and if it is still NULL, we will impute to lag104
  impute_val_cols = [x + "_lag" + str(ONE_YEAR_LAG) for x in  FORECAST_LAG_VARS] 
  historical_data = impute_to_col(historical_data, FORECAST_LAG_VARS ,impute_val_cols)
  impute_val_cols = [x + "_lag" + str(TWO_YEAR_LAG) for x in  FORECAST_LAG_VARS] 
  historical_data = impute_to_col(historical_data, FORECAST_LAG_VARS ,impute_val_cols)

  historical_data = historical_data.select(["MODEL_ID",TIME_VAR] + FORECAST_LAG_VARS)
  historical_data = historical_data.filter(col(TIME_VAR)>=max_mrd_date)
  historical_data = master_df.select(["MODEL_ID",TIME_VAR]+FORECAST_LAG_VARS).filter(col(TIME_VAR)<max_mrd_date)\
                              .union(historical_data) #joining history with future
  
  master_df = master_df.drop(*FORECAST_LAG_VARS)
  master_df = master_df.join(historical_data, on = ["MODEL_ID",TIME_VAR], how="left")
  print(master_df.count())

# COMMAND ----------

# DBTITLE 1,Load External Data
#TO-DO: Should add country_name to DBA_MODELIDS table so we can read all ID columns directly from there rather than entire dataset
#TO-DO: If we have separate output tables for things such as this it needs to be consistent for all tables
if TIME_VAR =='Week_Of_Year':
  external_vars = load_delta(DBI_EXTERNAL_VARIABLES_WEEKLY)
  weather = load_delta(DBI_WEATHER_WEEKLY_HISTORICALS, 1)
elif TIME_VAR =='Month_Of_Year':
  external_vars = load_delta(DBI_EXTERNAL_VARIABLES_MONTHLY)
  weather = load_delta(DBI_WEATHER_MONTHLY_HISTORICALS, 1)

# filter external variables from master_df
external_columns = subtract_two_lists(set(external_vars.columns).intersection(master_df.columns),  [TIME_VAR,"country_name"])

# COMMAND ----------

# variable to univariate forcast
external_columns = subtract_two_lists(set(external_vars.columns).intersection(master_df.columns),  [TIME_VAR,"country_name"])
external_columns = [x for x in external_columns if x not in covid_govt_policies]

# subset master for external forcast
# external variables are at country level, execution will be faster if we subset and run the externals seperately
external_vars = master_df.select([TIME_VAR,"country_name",CALENDAR_DATEVAR]+list(external_columns)).distinct()

# COMMAND ----------

# DBTITLE 1,Time Series Forecast
# MAGIC %md 
# MAGIC Here we use time series to predict future driver assumptions.

# COMMAND ----------

nielson = load_delta(DBI_NIELSEN)
neilsen_cols = [x for x in nielson.columns if "neil_" in x] + ['neil_percent_baseline_unit', 'neil_percent_baseline_volume', 'neil_percent_baseline_value']
COLS_TO_DROP = subtract_two_lists(neilsen_cols, neilsen_cols_to_frcst)

master_df = master_df.drop(*COLS_TO_DROP)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,defining the forcast parameters
# Time Series Configs
# These configs must be placed after dat is read in since the dataframe location is referenced in dictionary
# TO-DO: Update config file so it reads dataframe in as string and evaluates location keys post dataframe read in
predict_dict = {master_df     : {
                  'PARTITION_COLS' : ["MODEL_ID"],
                  'ALGO_TO_USE' : Holt,
                  'COLS_TO_FORECAST' : neilsen_cols_to_frcst + ['LIST_PRICE_BAG', 'NET_PRICE_BAG','ownSame_1_NET_PRICE_BAG','ownSame_2_NET_PRICE_BAG',
                                       'ownSame_3_NET_PRICE_BAG','ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG']
                },
                external_vars : {
                  'PARTITION_COLS' : ["country_name"],
                  'ALGO_TO_USE' :  Holt,
                  'COLS_TO_FORECAST' : external_columns
                }
               }

# COMMAND ----------

# # workaround to speedup computation
# save_df_as_delta(master_df, 'dbfs:/mnt/adls/Tables/test2', enforce_schema=False)
# master_df = load_delta('dbfs:/mnt/adls/Tables/test2')

# COMMAND ----------

df = spark.createDataFrame(FORECAST_START_DATES, "int").toDF("FCST_START_DATE")
all_dates = master_df.select("MODEL_ID", "country_name", TIME_VAR).distinct()
#print(all_dates.count())
all_dates = all_dates.crossJoin(df)
#print(all_dates.count())
    
# all_dates.count()

all_periods = all_dates.select(TIME_VAR).distinct().rdd.map(lambda x: x[0]).collect() #get distinct months in dataset

# get the max forcast that should be available each forcast start date. 
max_forcast = pd.DataFrame(list(zip(FORECAST_START_DATES, [get_maxforcast(x, NUM_FWD_FRCST, all_periods) for x in FORECAST_START_DATES])),
                    columns=['FCST_START_DATE', 'max_forcast'])
max_forcast = spark.createDataFrame(max_forcast)

# filter model_id that started after FCST_START_DATE
all_dates = all_dates.withColumn('min', min(TIME_VAR).over(Window.partitionBy('MODEL_ID','FCST_START_DATE')))\
                 .filter(col('min')<col('FCST_START_DATE')).drop('min')

# filter dataframe to have only the holdout period
all_dates = all_dates.join(max_forcast, on=['FCST_START_DATE'], how='left')\
                     .filter((col(TIME_VAR)<=col('max_forcast'))&(col(TIME_VAR)>=col('FCST_START_DATE')))\
                     .drop('max_forcast')

# COMMAND ----------

for ts_dat in predict_dict.keys():
  #print(i)
  this_dict = predict_dict.get(ts_dat)
  partition_cols = this_dict.get('PARTITION_COLS')
  algo_to_use = this_dict.get('ALGO_TO_USE')
  cols_to_predict = this_dict.get('COLS_TO_FORECAST')
  cols_to_predict = intersect_two_lists(ts_dat.columns, cols_to_predict)
  print(this_dict)
  print(cols_to_predict)
  print(len(cols_to_predict))
  
  #If predict columns are found in dataset, commence prediction
  if len(cols_to_predict) > 0:
    #Stack data for rolling runs
    #Append rolling holdout samples
    for start_period in FORECAST_START_DATES:
      this_roll = ts_dat
      this_roll = this_roll.withColumn("TRAIN_IND", when(col(TIME_VAR)<start_period,1).otherwise(0))
      this_roll = this_roll.withColumn("FCST_START_DATE", lit(start_period))

      #Filter out model_id's without enough observations
      this_roll = this_roll.filter(col("TRAIN_IND")==1)
      low_sales = aggregate_data(this_roll, partition_cols, [cols_to_predict[0]], [count])
      this_roll = this_roll.join(low_sales, on=partition_cols, how="left")
      this_roll = this_roll.filter(col("count_"+cols_to_predict[0]) >= MIN_ROWS_THRESHOLD)

      if start_period == FORECAST_START_DATES[0]:
        rolling_df = this_roll
      else:
        rolling_df = rolling_df.union(this_roll)

    rolling_df.cache()
    print(rolling_df.count())
    # Filtering the training set
    rolling_df = rolling_df.filter(col('TRAIN_IND')==1)
    print(rolling_df.count())

    #Melt dataframe so each variable is a row
    train_data = melt_df_pyspark(rolling_df, id_vars = partition_cols + [CALENDAR_DATEVAR] +['FCST_START_DATE'], value_vars=cols_to_predict)

    # Run Time Series
    #Setup modeling dictionary
    model_info_dict = dict(
          algo_func         = algo_to_use,
          time_field        = CALENDAR_DATEVAR, 
          fcst_start_field  = 'FCST_START_DATE', 
          target_field      = 'value',
          freq              = MODEL_FREQ,
          n_ahead           = NUM_FWD_FRCST ,
          model_id          = partition_cols + ["variable"]
      )
    model_info_cls = TimeSeriesModelInfo(**model_info_dict)

    auto_schema = train_data.limit(1)
    auto_schema = auto_schema.withColumn('pred', lit(10.00))
    auto_schema = auto_schema.withColumn('FCST_START_DATE', lit(10))
    auto_schema = auto_schema.select(model_info_cls.model_id + [model_info_cls.fcst_start_field] + [model_info_cls.time_field] + ["pred"])
    auto_schema = change_schema_in_pyspark(auto_schema, string_cols = [model_info_cls.time_field])
    schema = auto_schema.schema

    #Note: auto arima calls a different function than generic statsmodels time series since it uses a different library
    if algo_to_use == auto_arima:
      @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
      def predict_time_series_udf(data):
          return run_auto_arima(data, model_info_cls)
    else:
      @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
      def predict_time_series_udf(data):
          return run_time_series(data, model_info_cls)

    preds_ts = train_data.groupBy(partition_cols + ['variable' , 'FCST_START_DATE']).apply(predict_time_series_udf)
    preds_ts.cache()
    preds_ts = preds_ts.groupBy(*partition_cols, CALENDAR_DATEVAR,"FCST_START_DATE").pivot("variable").max("pred")


    calendar=calendar.withColumn(CALENDAR_DATEVAR,to_date(col(CALENDAR_DATEVAR)).cast("string"))
    preds_ts = preds_ts.join(calendar, on=[CALENDAR_DATEVAR], how='left').drop(CALENDAR_DATEVAR)
    all_dates = all_dates.join(preds_ts, on = partition_cols + [TIME_VAR, "FCST_START_DATE"], how="left") 

all_dates = all_dates.filter(col(TIME_VAR)>=col('FCST_START_DATE'))

# COMMAND ----------

# # hack to speedup computation
# # all_dates
# save_df_as_delta(all_dates, 'dbfs:/mnt/adls/Tables/test', enforce_schema=False)
# all_dates = load_delta('dbfs:/mnt/adls/Tables/test')

# COMMAND ----------

# TODO: get this from "predict_dict"
forcast_columns = ['LIST_PRICE_BAG', 'NET_PRICE_BAG','ownSame_1_NET_PRICE_BAG','ownSame_2_NET_PRICE_BAG', 
 'ownSame_3_NET_PRICE_BAG',  'ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG'] \
+ subtract_two_lists(set(external_vars.columns).intersection(master_df.columns), [TIME_VAR,"country_name",CALENDAR_DATEVAR]) + neilsen_cols_to_frcst

# COMMAND ----------

df = spark.createDataFrame(FORECAST_START_DATES, "int").toDF("FCST_START_DATE")
#print(master_df.count())
stacked_data = master_df.crossJoin(df)
#print(stacked_data.count())

# append the driver forcast table to this
driver_actuals = stacked_data.select(['MODEL_ID', TIME_VAR, 'country_name', 'FCST_START_DATE'] + forcast_columns )
driver_actuals = driver_actuals.filter(col(TIME_VAR)<col('FCST_START_DATE'))
stacked_data = stacked_data.drop(*forcast_columns)

#joining the forcast
driver_full = driver_actuals.union(all_dates.select(driver_actuals.columns)).cache()
driver_full.count()

# COMMAND ----------

# impute nulls with average for model_id that were not forcasted for lack of History
window = Window().partitionBy('MODEL_ID', 'FCST_START_DATE')
average_impute_cols = ['LIST_PRICE_BAG', 'NET_PRICE_BAG','ownSame_1_NET_PRICE_BAG','ownSame_2_NET_PRICE_BAG', 
               'ownSame_3_NET_PRICE_BAG',  'ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG'] 

for column in average_impute_cols:
  driver_full = driver_full.withColumn(column, 
                              when(col(column).isNull(),avg(col(column)).over(window)).otherwise(col(column)))

# COMMAND ----------

# # impute nulls with average for Neisen model_id & country that were not forcasted for lack of History
# window = Window().partitionBy('MODEL_ID', 'country_name', 'FCST_START_DATE')

# for column in neilsen_cols_to_frcst:
#   driver_full = driver_full.withColumn(column, 
#                               when(col(column).isNull(),avg(col(column)).over(window)).otherwise(col(column)))

# COMMAND ----------

# merge forcasted drivers with main df 
stacked_data = stacked_data.join(driver_full, on=['MODEL_ID', TIME_VAR, 'country_name', 'FCST_START_DATE'], how='right')

# COMMAND ----------

# print(master_df.count())
display(stacked_data.groupby('FCST_START_DATE').agg(count('MODEL_ID'), countDistinct('MODEL_ID'), max(TIME_VAR)))

# COMMAND ----------

# DBTITLE 1,Historical Averaging
# MAGIC %md 
# MAGIC Here we summarize historical averages by a certain group for forward looking assumptions.
# MAGIC 
# MAGIC 1. We summarize variables by different groups depending on how we want to do the imputation (e.g., Display ACV's should be summarized by Display Type, W_Distribution should be summarized by baseline/promo weeks). In order to do this imputation, we take in a dictionary of the basic assumptions.
# MAGIC 
# MAGIC 2. Variables that were already forecasted above may also be included here if want to override "bad" forecast values with historical averages (e.g., W_Distribution < 0).
# MAGIC 
# MAGIC 3. Since these averages have a different level of aggregation (e.g., not by DATE but by Promotion_Flag), it cannot be appended into time series predictions. Thus, we output these forecasts as a separate table.

# COMMAND ----------

stacked_data = stacked_data.drop(*subtract_two_lists(weather.columns, [WEATHER_VAR,'country_name']))

# COMMAND ----------

# Temporary merge of historical average variables until auto averaging is setup
# Merging with the weather data
stacked_data = stacked_data.withColumn(TIME_VAR,stacked_data[TIME_VAR].cast('string'))
stacked_data=stacked_data.withColumn(WEATHER_VAR,substring(col(TIME_VAR),5,7))
stacked_data = stacked_data.join(weather,on=[WEATHER_VAR,'country_name'],how='left')
stacked_data = stacked_data.drop(WEATHER_VAR)
stacked_data = stacked_data.withColumn(TIME_VAR,stacked_data[TIME_VAR].cast('int'))

# COMMAND ----------

# DBTITLE 1,Pricing variable computation
# Custom Fields
ratio_dict = {'Discount_Depth':{'top_var':"NET_PRICE_BAG",'bottom_var':"LIST_PRICE_BAG"}}
stacked_data = calculate_ratio(stacked_data,ratio_dict) 
stacked_data = stacked_data.withColumn("Discount_Depth", (1 - col("Discount_Depth"))*100)
stacked_data = stacked_data.withColumn("Discount_Depth", when(col("Discount_Depth") < 0, 0).otherwise(col("Discount_Depth")))
stacked_data = stacked_data.withColumn("Discount_Depth", when(col("Discount_Depth").isNull(), 0).otherwise(col("Discount_Depth")))

#Dollar Discount
stacked_data = stacked_data.withColumn("Dollar_Discount", col("LIST_PRICE_BAG")-col("NET_PRICE_BAG"))

# COMMAND ----------

# DBTITLE 1,MEDIA
# media time
MEDIA_COLS = [
                'media_DIGITAL', 'media_OOHDigital', 'media_OOHSpecial', 'media_OOHTraditional', 'media_OtherCinema', 'media_OtherPrint', 
                'media_OtherProduction', 'media_OtherRadio', 'media_OtherSponsorship', 'media_TVTraditional'
              ]

DBI_MEDIA = 'dbfs:/mnt/adls/Tables/DBI_MEDIA'
MEDIA_MERGE_FIELD_M = ["HRCHY_LVL_3_NM", "SRC_CTGY_1_NM", "BRND_NM", TIME_VAR]

MEDIA_MERGE_FIELD_W = ["HRCHY_LVL_3_NM", "SRC_CTGY_1_NM", "BRND_NM", "Week_Of_Year"]
# dropping existing media columns
stacked_data = stacked_data.drop(*MEDIA_COLS)

## MEDIA Spend mapping
try:
  media_clean = load_delta(DBI_MEDIA)
except:
  print("Media not available for this market, they will not be included as features")

# Temp code to remove duplicates from Media data. This needs to be fixed in Raw_transform script later
media_clean = media_clean.groupBy(MEDIA_MERGE_FIELD_W).agg(*[sum(c).alias(c) for c in media_clean.columns if c not in MEDIA_MERGE_FIELD_W]) 

try:
  if TIME_VAR=='Week_Of_Year':
      stacked_data2 = stacked_data.join(media_clean, on=MEDIA_MERGE_FIELD_M, how='left')
  else :
      calendar = load_delta(DBI_CALENDAR)
      calendar_df=calendar.select("Month_Of_Year", "Week_Of_Year", "month_ratio").distinct()
      media_clean = media_clean.join(calendar_df, on="Week_Of_Year", how='left')\
                         .withColumnRenamed('month_ratio', 'ratio').drop("Week_Of_Year")

      for c in subtract_two_lists(media_clean.columns, MEDIA_MERGE_FIELD_M):
          media_clean = media_clean.withColumn(c, col("ratio")*col(c))

      #Removing data for incomplete month
      media_clean2 = media_clean.drop("ratio","Week_Of_Year")
      media_agg = media_clean2.groupBy(MEDIA_MERGE_FIELD_M).agg(*[sum(c).alias(c) for c in media_clean2.columns if c not in MEDIA_MERGE_FIELD_M]) 

      stacked_data2 = stacked_data.join(media_agg, on=MEDIA_MERGE_FIELD_M, how="left")
  
except:
  print('Media Spend not used in this market')

# COMMAND ----------

if TIME_VAR == "Month_Of_Year":
  if 'Week_Of_Year' in stacked_data2.columns:
    stacked_data2 = stacked_data2.drop('Week_Of_Year')
else:
  if 'Month_Of_Year' in stacked_data2.columns:
    stacked_data2 = stacked_data2.drop('Month_Of_Year')

# COMMAND ----------

# DBTITLE 1,Inventory Forecast and Integration
# For easier computation, retaining only DMDUNIT and LOC combos present in mrd
dmdunit_loc_to_keep = master_df.select(INVENTORY_MERGE_FIELD).distinct()

inventory_df = inventory_df.withColumnRenamed("MTRL_ID", PROD_MERGE_FIELD).withColumnRenamed("LOC_ID", LOC_MERGE_FIELD)
inventory_df = inventory_df.join(dmdunit_loc_to_keep, on = INVENTORY_MERGE_FIELD, how = "inner")

# COMMAND ----------

# Replicate inventory data for last two years
inventory_past_one = inventory_df.withColumn("Month_Of_Year", col('Month_Of_Year') - lit(100)).withColumn("Week_Of_Year", col('Week_Of_Year') - lit(100))
inventory_past_two = inventory_df.withColumn("Month_Of_Year", col('Month_Of_Year') - lit(200)).withColumn("Week_Of_Year", col('Week_Of_Year') - lit(200))
inventory_df = inventory_df.union(inventory_past_one.select(inventory_df.columns)).union(inventory_past_two.select(inventory_df.columns))

# Get inventory data at TIME_VAR level
if TIME_VAR == "Month_Of_Year":
  inventory_df = inventory_df.withColumn('max_week_of_month', F.max('Week_Of_Year').over(Window.partitionBy(INVENTORY_MERGE_FIELD + ["Month_Of_Year"]))) \
                                          .filter(col('max_week_of_month') == col('Week_Of_Year')).drop("Week_Of_Year", "max_week_of_month")  
else:
  inventory_df = inventory_df.drop("Month_Of_Year")

# Get time variables  
calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct()
inventory_df = inventory_df.join(calendar_df, on = TIME_VAR, how = 'left')
display(inventory_df)

# COMMAND ----------

# Making the inventory data continuous time series data
print(inventory_df.count())
min_filter_df = inventory_df.groupby(INVENTORY_MERGE_FIELD).agg(min(CALENDAR_DATEVAR).alias('min_time'))
inventory = fill_missing_timeseries_pyspark(inventory_df, INVENTORY_MERGE_FIELD, time_variable=CALENDAR_DATEVAR) \
                                        .join(inventory_df, on=[CALENDAR_DATEVAR] + INVENTORY_MERGE_FIELD, how='left').na.fill(0)
# making sure zero fill happens for min date of inventory level onwards to current date
inventory = inventory.join(min_filter_df, on = INVENTORY_MERGE_FIELD, how='left')\
                     .filter(col(CALENDAR_DATEVAR)>=col('min_time')).drop('min_time')
# Dropping TIME_VAR and again joining with Calendar 
inventory = inventory.drop(TIME_VAR)
inventory = inventory.join(calendar_df,on=[CALENDAR_DATEVAR],how='left')

# Create partition field
inventory = inventory.withColumn("DMDUNIT_LOC_ID", concat(col(PROD_MERGE_FIELD), col(LOC_MERGE_FIELD)))
print(inventory.count())

# COMMAND ----------

# Create the prediction dictionary
inv_predict_dict = {inventory    : {
                  'PARTITION_COLS' : ["DMDUNIT_LOC_ID"],
                  'ALGO_TO_USE' : Holt,
                  'ACTUAL_COL' : "actual_inventory",
                  'PROJECTED_COL' : "projected_inventory"
               }
                   }

inv_dict = inv_predict_dict.get(inventory)
partition_cols = inv_dict.get('PARTITION_COLS')
algo_to_use = inv_dict.get('ALGO_TO_USE')
actual_col = inv_dict.get('ACTUAL_COL')
projected_col = inv_dict.get('PROJECTED_COL')
cols_to_predict = intersect_two_lists(inventory_df.columns, [actual_col, projected_col])
print(inv_dict)
print(cols_to_predict)
print(len(cols_to_predict))

# COMMAND ----------

# Create training dataset, separate logic for actual and projected
for inv_col in cols_to_predict:
  
  if inv_col == actual_col:
  
    for start_period in FORECAST_START_DATES:
      this_roll = inventory.withColumn("FCST_START_DATE", lit(start_period))
      this_roll = this_roll.withColumn("TRAIN_IND", when(col(TIME_VAR)<start_period,1).otherwise(0))

      #Filter out model_id's without enough observations  
      this_roll = this_roll.filter(col("TRAIN_IND")==1)
      # low_sales = aggregate_data(this_roll, partition_cols, [cols_to_predict[0]], [count])
      # this_roll = this_roll.join(low_sales, on=partition_cols, how="left")
      # this_roll = this_roll.filter(col("count_" + cols_to_predict[0]) >= MIN_ROWS_THRESHOLD)

      if start_period == FORECAST_START_DATES[0]:
        rolling_df = this_roll
      else:
        rolling_df = rolling_df.union(this_roll.select(rolling_df.columns))
  
  else:
    
    for start_period in FORECAST_START_DATES:
      this_roll = inventory.withColumn("FCST_START_DATE", lit(start_period))
      
      calendar_fcst_start = calendar_df.withColumnRenamed(CALENDAR_DATEVAR, 'fcst_start_date_full').withColumnRenamed(TIME_VAR, "FCST_START_DATE")
      this_roll = this_roll.join(calendar_fcst_start, on = "FCST_START_DATE", how = 'left')
      this_roll = this_roll.withColumn("fcst_start_date_full", to_date(col("fcst_start_date_full")))
      if TIME_VAR == "Month_Of_Year":
        this_roll = this_roll.withColumn("NUM_FWD_day", F.add_months('fcst_start_date_full', NUM_FWD_FRCST-1)).drop("fcst_start_date_full")
      else:
        this_roll = this_roll.withColumn("NUM_FWD_day", F.date_add('fcst_start_date_full', NUM_FWD_FRCST*7)).drop("fcst_start_date_full")
      calendar_num_fwd = calendar_df.withColumnRenamed(CALENDAR_DATEVAR, 'NUM_FWD_day').withColumnRenamed(TIME_VAR, "NUM_FWD_DATE")
      this_roll = this_roll.join(calendar_num_fwd, on = "NUM_FWD_day", how = 'left')
      this_roll = this_roll.withColumn("TRAIN_IND", when(col(TIME_VAR)<col("NUM_FWD_DATE"),1).otherwise(0)).drop("NUM_FWD_day", "NUM_FWD_DATE")

      #Filter out model_id's without enough observations  
      this_roll = this_roll.filter(col("TRAIN_IND")==1)
      # low_sales = aggregate_data(this_roll, partition_cols, [cols_to_predict[0]], [count])
      # this_roll = this_roll.join(low_sales, on=partition_cols, how="left")
      # this_roll = this_roll.filter(col("count_"+cols_to_predict[0]) >= MIN_ROWS_THRESHOLD)

      if start_period == FORECAST_START_DATES[0]:
        rolling_df = this_roll
      else:
        rolling_df = rolling_df.union(this_roll.select(rolling_df.columns))
  
  rolling_df = rolling_df.withColumn("inv_type", lit(inv_col))
  if inv_col == cols_to_predict[0]:
    rolling_df_inv = rolling_df
  else:
    rolling_df_inv = rolling_df_inv.union(rolling_df.select(rolling_df_inv.columns))

# COMMAND ----------

rolling_df_inv.cache()
print(rolling_df_inv.count())

# COMMAND ----------

# Melt dataframe so each variable is a row
train_actual = melt_df_pyspark(rolling_df_inv.filter(col("inv_type") == actual_col), id_vars = partition_cols + [CALENDAR_DATEVAR] +['FCST_START_DATE'], value_vars = [actual_col])
train_projected = melt_df_pyspark(rolling_df_inv.filter(col("inv_type") == projected_col), id_vars = partition_cols + [CALENDAR_DATEVAR] +['FCST_START_DATE'], value_vars = [projected_col])
train_data_inv = train_actual.union(train_projected)

# Run Time Series
#Setup modeling dictionary
model_info_dict = dict(
      algo_func         = algo_to_use,
      time_field        = CALENDAR_DATEVAR, 
      fcst_start_field  = 'FCST_START_DATE', 
      target_field      = 'value',
      freq              = MODEL_FREQ,
      n_ahead           = NUM_FWD_FRCST ,
      model_id          = partition_cols + ["variable"]
  )
model_info_cls = TimeSeriesModelInfo(**model_info_dict)

auto_schema = train_data_inv.limit(1)
auto_schema = auto_schema.withColumn('pred', lit(10.00))
auto_schema = auto_schema.withColumn('FCST_START_DATE', lit(10))
auto_schema = auto_schema.select(model_info_cls.model_id + [model_info_cls.fcst_start_field] + [model_info_cls.time_field] + ["pred"])
auto_schema = change_schema_in_pyspark(auto_schema, string_cols = [model_info_cls.time_field])
schema = auto_schema.schema

# COMMAND ----------

#Note: auto arima calls a different function than generic statsmodels time series since it uses a different library
if algo_to_use == auto_arima:
  @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
  def predict_time_series_udf(data):
      return run_auto_arima(data, model_info_cls)
else:
  @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
  def predict_time_series_udf(data):
      return run_time_series(data, model_info_cls)

preds_ts_inv = train_data_inv.groupBy(partition_cols + ['variable' , 'FCST_START_DATE']).apply(predict_time_series_udf)
preds_ts_inv.cache()

pred_actual = preds_ts_inv.filter(col("variable") == actual_col).withColumnRenamed("pred", actual_col).drop("variable")
pred_projected = preds_ts_inv.filter(col("variable") == projected_col).withColumnRenamed("pred", projected_col).drop("variable")

# COMMAND ----------

calendar_df = calendar_df.withColumn(CALENDAR_DATEVAR,to_date(col(CALENDAR_DATEVAR)).cast("string"))
pred_actual = pred_actual.join(calendar_df, on=[CALENDAR_DATEVAR], how='left').drop(CALENDAR_DATEVAR)
pred_projected = pred_projected.join(calendar_df, on=[CALENDAR_DATEVAR], how='left').drop(CALENDAR_DATEVAR)

# COMMAND ----------

# To validate forecast time frame for actual and projected, projected: training set taken till max available date; actual: training set taken till period before FCST_START_DATE
display(pred_actual.groupBy("FCST_START_DATE").agg(min(TIME_VAR), max(TIME_VAR)))
display(pred_projected.groupBy("FCST_START_DATE").agg(min(TIME_VAR), max(TIME_VAR)))

# COMMAND ----------

# Merge train and forecasted inventories
inv_full_actual = rolling_df_inv.filter(col("inv_type") == actual_col).select(pred_actual.columns).union(pred_actual)
inv_full_projected = rolling_df_inv.filter(col("inv_type") == projected_col).select(pred_projected.columns).union(pred_projected)

# COMMAND ----------

## Join actual and projected inventories
common_cols = intersect_two_lists(inv_full_actual.columns, inv_full_projected.columns)
all_inv_rows = inv_full_actual.select(common_cols).union(inv_full_projected.select(common_cols)).distinct()

inventory_full = all_inv_rows.join(inv_full_actual, on = common_cols, how = "left").join(inv_full_projected, on = common_cols, how = "left")
inventory_full = inventory_full.withColumn("actual_inventory",when(col("actual_inventory") < 0, lit(0)).otherwise(col("actual_inventory"))) \
                                 .withColumn("projected_inventory",when(col("projected_inventory") < 0, lit(0)).otherwise(col("projected_inventory")))
inventory_full = inventory_full.na.fill(value=0, subset=["actual_inventory", "projected_inventory"])

# Bring back dmdunit and loc
dmdunit_loc = inventory.select(["DMDUNIT_LOC_ID"] + INVENTORY_MERGE_FIELD).distinct()
inventory_full = inventory_full.join(dmdunit_loc, on = "DMDUNIT_LOC_ID", how = "left").drop("DMDUNIT_LOC_ID")

display(inventory_full)
print(inventory_full.select(INVENTORY_MERGE_FIELD + [TIME_VAR, "FCST_START_DATE"]).distinct().count() == inventory_full.count() == inventory_full.distinct().count())

# COMMAND ----------

INVENTORY_MERGE_FIELD = INVENTORY_MERGE_FIELD + [TIME_VAR]

# COMMAND ----------

# Merge actual and forecasted inventory with main df 
stacked_data2 = stacked_data2.join(inventory_full, on = INVENTORY_MERGE_FIELD + ["FCST_START_DATE"], how='left')
stacked_data2 = stacked_data2.na.fill(0, subset = ["actual_inventory", "projected_inventory"])
stacked_data2.count()

# COMMAND ----------

# DBTITLE 1,Promotions Replication and Integration
# Rename required fields
promo_df_es = promo_df_es.withColumnRenamed("DMDGROUP", CUST_MERGE_FIELD).withColumnRenamed("DMDUNIT", PROD_MERGE_FIELD)
promo_df_pt = promo_df_pt.withColumnRenamed("DMDGROUP", CUST_MERGE_FIELD).withColumnRenamed("DMDUNIT", PROD_MERGE_FIELD)


# Replicate promotions of latest year for next 2 years
max_time_var_es = promo_df_es.select(max(TIME_VAR)).collect()[0][0]
max_time_var_pt = promo_df_pt.select(max(TIME_VAR)).collect()[0][0]
promo_es_latest_year = promo_df_es.filter(col(TIME_VAR) > max_time_var_es - 100)
promo_pt_latest_year = promo_df_pt.filter(col(TIME_VAR) > max_time_var_pt - 100)

promo_next_one_es = promo_es_latest_year.withColumn(TIME_VAR, col(TIME_VAR) + lit(100))
promo_next_two_es = promo_es_latest_year.withColumn(TIME_VAR, col(TIME_VAR) + lit(200))
promo_next_one_pt = promo_pt_latest_year.withColumn(TIME_VAR, col(TIME_VAR) + lit(100))
promo_next_two_pt = promo_pt_latest_year.withColumn(TIME_VAR, col(TIME_VAR) + lit(200))

promotions_es = promo_df_es.union(promo_next_one_es.select(promo_df_es.columns)).union(promo_next_two_es.select(promo_df_es.columns)).distinct()
promotions_pt = promo_df_pt.union(promo_next_one_pt.select(promo_df_pt.columns)).union(promo_next_two_pt.select(promo_df_pt.columns)).distinct()


# Combine promotions for both countries
promotions = promotions_es.join(promotions_pt, on=intersect_two_lists(promotions_es.columns, promotions_pt.columns), how="outer")

promo_cols = [c for c in promotions.columns if ("promo_" in c)]
for c in promo_cols:
  promotions = promotions.withColumn(c, promotions[c].cast(IntegerType()))
  
promotions = promotions.na.fill(value=0, subset = promo_cols)


PROMO_MERGE_FIELD = PROMO_MERGE_FIELD + [TIME_VAR]


# Merge promotions with main df 
stacked_data2 = stacked_data2.join(promotions, on = PROMO_MERGE_FIELD, how = 'left')
stacked_data2 = stacked_data2.na.fill(0, subset = promo_cols)

# COMMAND ----------

# DBTITLE 1,Output
#Output table
save_df_as_delta(stacked_data2, DBA_DRIVER_FORECASTS, enforce_schema=False)
delta_info = load_delta_info(DBA_DRIVER_FORECASTS)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Historical averaging (Code added for future ref)
# #This dictionary lists variables we want to summarize (value) by the group over which we want to summarize (key)
# historical_vars = {"Sales_Display_Flag"      : ["Any_Disp_ACV","ACVWeightedDistDisplayOnly"],
#                    "Sales_Feature_Flag"      : ["Any_Feat_ACV","ACVWeightedDistFeatureOnly"],
#                    "Promo_Layer_1_PromoType" : ["Any_Promo_ACV"],
#                    "Sales_Disp_Feat_Flag"    : ["ACVWeightedDistFeatureandDisplay"],
#                    "Promo_Layer_3_Flag"      : ["Promo_Layer_3_Duration","Promo_Layer_3_QtyRedeemed"]
#                   }

# #Summarize mrd historical_vars by the provided group levels/summary vars
# i = 0
# for c in historical_vars.keys():
#   #Summarize mrd
#   summary_group = ["MODEL_ID"] + [c]
#   summary_vars = historical_vars.get(c)
#   print(summary_vars)
#   historical_averages = aggregate_data(mrd, summary_group, summary_vars, [avg])
  
#   #Shape into output that be appended
#   historical_averages = historical_averages.withColumn("group_var",lit(c))
#   new_names = ["MODEL_ID","group_val"] + summary_vars + ["group_var"]
#   historical_averages = historical_averages.toDF(*new_names)
#   historical_averages = historical_averages.select(["MODEL_ID","group_var","group_val"]+summary_vars)
 
#   historical_averages = melt(df = historical_averages, 
#                              id_vars = ['MODEL_ID',"group_var","group_val"],
#                              value_vars = summary_vars)
#   if i==0:
#     historical_output = historical_averages
#   else:
#     historical_output = historical_output.union(historical_averages)
    
#   i = i + 1
