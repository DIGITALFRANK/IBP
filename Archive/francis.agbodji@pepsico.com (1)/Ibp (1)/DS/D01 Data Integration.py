# Databricks notebook source
# MAGIC %md 
# MAGIC ##01 - Data Integration
# MAGIC 
# MAGIC This script maps relevant data sources into a "base" modeling dataset.  Downstream processes will perform an EDA, feature engineering and cleanse the data.  If demand drivers are present for a market, they are mapped to the data in this script.  If they are not present for his market, the will not be included in the final output.

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

print(TIME_COLS_TO_DROP)

#Check configurations exist for this script
try:
  required_configs = [
    TARGET_VAR,
    TIME_VAR,
    MODEL_ID_HIER,
    NUM_FWD_FRCST,
    CALENDAR_DATEVAR,  
    PROD_MERGE_FIELD,
    CUST_MERGE_FIELD,
    LOC_MERGE_FIELD,
    NAN_THRESH_PERC,
    TOP_HOL_THRESH,
    DBI_SHIPMENTS,
    DBI_PRODUCTS,
    DBI_CUSTOMER,
    DBI_LOC,
    DBI_CALENDAR,
    DBI_HOLIDAYS_MONTHLY,
    DBI_HOLIDAYS_WEEKLY,
    DBA_MRD_EXPLORATORY 
    ]
  print(json.dumps(required_configs, indent=4))
except:
  dbutils.notebook.exit("Missing required configs")

# Optional Configurations for this script listed below
# DBI_ORDERS
# PROMO_MERGE_FIELD
# MEDIA_MERGE_FIELD
# DBI_MEDIA
# DBI_SYNDICATED
# DBI_HOLIDAYS_MONTHLY
# DBI_HOLIDAYS_WEEKLY
# DBI_EXTERNAL_VARIABLES_MONTHLY
# DBI_EXTERNAL_VARIABLES_WEEKLY
# DBI_PROMO_ES_MONTHLY
# DBI_PROMO_PT_MONTHLY
# DBI_PROMO_ES_WEEKLY
# DBI_PROMO_PT_WEEKLY
# DBI_PRICING_MONTHLY
# DBI_PRICING_WEEKLY
# DBI_WEATHER_MONTHLY
# DBI_WEATHER_WEEKLY

# COMMAND ----------

# DBTITLE 1,Mlflow
#Track details about this data source (e.g., aggregation, included drivers, etc.)
try:
  output_path_mlflow_data_details = f'{OUTPUT_PATH_MLFLOW_EXPERIMENTS}/PEP Data Details'
  mlflow.set_experiment(output_path_mlflow_data_details)
  mlflow.start_run()
  experiment = mlflow.get_experiment_by_name(output_path_mlflow_data_details)
  print("Experiment_id: {}".format(experiment.experiment_id))
  print("Artifact Location: {}".format(experiment.artifact_location))
  print("Tags: {}".format(experiment.tags))
  print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
except:
  print("Mlflow run already active")
  
mlflow.log_param('Target Variable',TARGET_VAR)
mlflow.log_param('Forecast Aggregation Level',MODEL_ID_HIER)

# COMMAND ----------

# DBTITLE 1,Load Curated Datasets
## Orders data extract
try:
  shipments = load_delta(DBI_ORDERS)
except:
  mlflow.log_param('ORDERS_PRESENT', False)
  print('Orders are not present for this market, shipments will be used')
    
## Shipments data extract
try:
  shipments = load_delta(DBI_SHIPMENTS)
except:
  #TO-DO: Check if logic of this if/else statement is still valid
  if TARGET_VAR == 'QTY':
    dbutils.notebook.exit("Shipments load failed")
  else:
    mlflow.log_param('SHIPMENTS_PRESENT',False)
    print('Orders are used to forecast this market')
    
## Hierarchies across our 3 dimensions - PRODUCT
try:
  prod = load_delta(DBI_PRODUCTS)
except:
  dbutils.notebook.exit("Product master load failed")

## Hierarchies across our 3 dimensions - CUSTOMER
try:
  cust = load_delta(DBI_CUSTOMER)
except:
  dbutils.notebook.exit("Customer master load failed")

## Hierarchies across our 3 dimensions - LOCATION
try:
  loc = load_delta(DBI_LOC)
except:
  dbutils.notebook.exit("Location master load failed")

## Calendar
try:
  calendar = load_delta(DBI_CALENDAR)
except:
  dbutils.notebook.exit("Calendar master load failed")

#### DRIVER LOADING ####
try:
  media_clean = load_delta(DBI_MEDIA)
except:
  print("Media not available for this market, they will not be included as features")
  
try:
  # syndicated = load_delta(DBI_SYNDICATED)
  nielson = load_delta(DBI_NIELSEN)
except:
  print("Syndicated data not available for this market, they will not be included as features")
  #TO-DO: DBI_NIELSEN should be DBI_SYNDICATED for markets where IRI is used
  # nielson = load_delta(DBI_NIELSEN)
  # NIELSON_MERGE_FIELD = ["DMDUNIT", "HRCHY_LVL_3_NM", TIME_VAR]

#TO-DO: Weekly / Monthly call outs should be one config, different folders
try:
  if TIME_VAR == 'Month_Of_Year':
    holidays_clean = load_delta(DBI_HOLIDAYS_MONTHLY)
  else:
    holidays_clean = load_delta(DBI_HOLIDAYS_WEEKLY)
except:
  print("Holidays not available for this market, they will not be included as features")

try:
  if TIME_VAR == 'Month_Of_Year':
    covid_macro_vars = load_delta(DBI_EXTERNAL_VARIABLES_MONTHLY)
  else:
    covid_macro_vars = load_delta(DBI_EXTERNAL_VARIABLES_WEEKLY)
except:
  print("External variables not available for this market, they will not be included as features")
  
try:
  if TIME_VAR == 'Month_Of_Year':
    pricing = load_delta(DBI_PRICING_MONTHLY)
  else:
    pricing = load_delta(DBI_PRICING_WEEKLY)
except:
  print("Pricing not available for this market, they will not be included as features")
  
try:
  if TIME_VAR == 'Month_Of_Year':
    weather = load_delta(DBI_WEATHER_MONTHLY)
  else:
    weather = load_delta(DBI_WEATHER_WEEKLY)
except:
  print("Weather not available for this market, they will not be included as features")
  
shipments.cache()
shipments.count()
print("shipments shape:", shipments.count(), len(shipments.columns))

# COMMAND ----------

## Review the minimum value present for shipments
## If lower than zero, will exit the notebook (for now)

min_shipments_value = shipments.agg({"QTY": "min"}).collect()[0][0]   ## alternative: shipments.agg(F.min(TARGET_VAR).collect()[0][0]

if min_shipments_value < 0:
  dbutils.notebook.exit("Target variable values include negative numbers - please confirm next steps with the business")

## Below option will give ability to correct negative values to 0 and retain the originals
## Commenting out the below for now, since we do not need for the current use case
## shipments = shipments.withColumnRenamed(TARGET_VAR, TARGET_VAR + "_orig_val")  ## retain
## shipments = shipments.withColumn(TARGET_VAR, F.when(F.col(TARGET_VAR) > 0, F.col(TARGET_VAR)).otherwise(0))  ## correct

# COMMAND ----------

## Aggregating shipments data to right time horizon (weekly or pro-rated monthly)
## Have retained STARTDATE in both agg's (its used in time_vars feature engineering)
## from pyspark.sql.functions import col

if TIME_VAR=='Week_Of_Year':
    ratio=["week_ratio"]
    calendar_df=calendar.select("Week_Of_Year", "Week_start_date", "week_ratio").distinct()
    shipments=shipments.join(calendar_df, shipments["HSTRY_TMFRM_STRT_DT"] == calendar_df["Week_start_date"], how='left')\
                  .withColumnRenamed('week_ratio', 'ratio')\
                  .withColumn("QTY",col("ratio")*col("QTY"))\
                  .drop("ratio")
    # Retain month_of_year
    shipments=shipments.withColumn('HSTRY_TMFRM_STRT_DT', to_timestamp('HSTRY_TMFRM_STRT_DT', "yyyy-MM-dd HH:mm:ss"))
    shipments=shipments.withColumn('year',year(shipments.HSTRY_TMFRM_STRT_DT))
    shipments=shipments.withColumn('month',month(shipments.HSTRY_TMFRM_STRT_DT))
    shipments=shipments.withColumn('Month_Of_Year',col("year")*lit(100)+col("month"))
    shipments = shipments.drop("year","month")
    shipments = shipments.withColumn('HSTRY_TMFRM_STRT_DT', col('HSTRY_TMFRM_STRT_DT').cast(StringType()))
else : 
    ratio=["month_ratio"]
    calendar_df=calendar.select("Month_Of_Year","Week_Of_Year","Week_start_date","month_ratio", "Month_start_date").distinct()
    shipments=shipments.join(calendar_df, shipments["HSTRY_TMFRM_STRT_DT"] == calendar_df["Week_start_date"], how='left')\
                         .drop("Week_start_date")\
                         .withColumnRenamed('month_ratio', 'ratio')\
                         .withColumn("QTY",col("ratio")*col("QTY"))
    
    # Removing data for incomplete month
    # Pomil - 20210727 - Modified logic for incomplete month 
    dates_df = shipments.select("Month_Of_Year", "Week_Of_Year", "Month_start_date").drop_duplicates()
    dates_df = dates_df.withColumn("Month_end_date", last_day("Month_start_date"))\
                        .withColumn("Month_start_day", dayofmonth(col("Month_start_date")).cast('int'))\
                        .withColumn("Month_end_day", dayofmonth(col("Month_end_date")).cast('int'))
    dates_df = dates_df.join(calendar.select("Week_Of_Year", "Month_Of_Year", "Day_count").distinct(), on=["Week_Of_Year", "Month_Of_Year"], how="left")
    dates_agg = dates_df.groupBy("Month_Of_Year").agg(max("Month_end_day").alias("Month_end_day"), sum("Day_count").alias("Day_count"))
    dates_agg = dates_agg.filter(col("Month_end_day")==col("Day_count")).select("Month_Of_Year")
    
    shipments = shipments.join(dates_agg, on=TIME_VAR, how="inner")
    
    shipments = shipments.drop("ratio","Week_Of_Year")
   
  # Aggregating the data for monthly level
    cols_to_agg = ["QTY"]
    sum_n = [sum] * len(cols_to_agg)
    shipments = aggregate_data(shipments,  MODEL_ID_HIER + [TIME_VAR, "Month_start_date"], cols_to_agg, sum_n)
    shipments = shipments.withColumnRenamed("sum_QTY" ,"QTY" )
    
#agg_count = shipments.select(MODEL_ID_HIER +[TIME_VAR]+['Month_Of_Year']).dropDuplicates().count()

agg_count = shipments.select(MODEL_ID_HIER +[TIME_VAR]).dropDuplicates().count()
raw_count = shipments.count()
print(agg_count)
print(raw_count)
if agg_count != raw_count:
  print('going to get aggregated')
  cols_to_agg = ["QTY"]
  sum_n = [sum] * len(cols_to_agg)

  if TIME_VAR=='Week_Of_Year':
    shipments = aggregate_data(shipments,  
                               MODEL_ID_HIER + [TIME_VAR, "Month_start_date"] + ['HSTRY_TMFRM_STRT_DT'], 
                               cols_to_agg, 
                               sum_n)
  
  shipments = shipments.withColumnRenamed("sum_QTY" ,"QTY" )
shipments.cache()
shipments.count()

# COMMAND ----------

# DBTITLE 1,Make Data Continuous
# Making the shipment data as continuous time series data
min_filter_df = shipments.groupby(MODEL_ID_HIER).agg(min(CALENDAR_DATEVAR).alias('min_time'))
shipments = fill_missing_timeseries_pyspark(shipments, MODEL_ID_HIER, time_variable=CALENDAR_DATEVAR) \
                                        .join(shipments, on=[CALENDAR_DATEVAR]+MODEL_ID_HIER, how='left').na.fill(0)
# making sure zero fill happens for min date of model_id onwards to current date
shipments = shipments.join(min_filter_df, on=MODEL_ID_HIER, how='left')\
                     .filter(col(CALENDAR_DATEVAR)>=col('min_time')).drop('min_time')
# Droping TIME_VAR and again joing with Calendar 
calendar_sub =calendar_df.select(CALENDAR_DATEVAR,TIME_VAR).distinct()
shipments = shipments.drop(TIME_VAR)
print(shipments.count())
shipments =shipments.join(calendar_sub,on=[CALENDAR_DATEVAR],how='left')
print(shipments.count())


# COMMAND ----------

# DBTITLE 1,Removing Model IDs inactive for 1 Year
end_period = shipments.select(max(TIME_VAR)).collect()[0][0]
print("end_period:", end_period)

# start_period = (np.around(int(max_period)/100)-1)*100 + (int(max_period)%100)
# end_period = np.around(int(max_period)/100)*100 + (int(max_period)%100)

# Pomil - 20210727 - Modified logic to get Model IDs inactive for 52 weeks (Handled 53 weeks in Leap year properly)  
mrd_dates = shipments
if CALENDAR_DATEVAR not in mrd_dates.columns:
  calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct() 
  mrd_dates = mrd_dates.join(calendar_df, on=TIME_VAR, how="left").select(TIME_VAR, CALENDAR_DATEVAR).distinct()

if 'week' in TIME_VAR.lower():
  mrd_dates = mrd_dates.withColumn("year_adjusted_date", F.date_sub(mrd_dates[CALENDAR_DATEVAR], 52*7))\
                        .withColumn("year_adjusted_date", to_timestamp(col("year_adjusted_date")))
elif 'month' in TIME_VAR.lower():
  mrd_dates = mrd_dates.withColumn("year_adjusted_date", F.add_months(mrd_dates[CALENDAR_DATEVAR], -12))\
                        .withColumn("year_adjusted_date", to_timestamp(col("year_adjusted_date")))
max_year_adjusted_date = mrd_dates.select(max("year_adjusted_date")).collect()[0][0] 
start_period = calendar_df.filter(col(CALENDAR_DATEVAR)==max_year_adjusted_date).select(TIME_VAR).collect()[0][0]
print("start_period:", start_period)


shipments_subset = shipments.filter((col(TIME_VAR)>start_period)&(col(TIME_VAR)<=end_period))  

shipments_subset = shipments_subset.groupby(MODEL_ID_HIER).agg(sum("QTY").alias('sum_target'))\
                             .filter(col('sum_target')>0)\
                             .drop('sum_target')
shipments = shipments.join(shipments_subset, on=MODEL_ID_HIER, how='inner')
shipments.cache()
print("shipments shape:", shipments.count(), len(shipments.columns) ) 
display(shipments)

# COMMAND ----------

# DBTITLE 1,Create (Combine) & Clean MRDS
total_sales_qc_QTY = shipments.agg(F.sum('QTY')).collect()[0][0]
master_df = shipments.join(prod, on=PROD_MERGE_FIELD, how='left')\
                     .join(cust, on=CUST_MERGE_FIELD, how='left')\
                     .join(loc, on=LOC_MERGE_FIELD, how='left')

## To drop duplicative columns by REGEX columns terms if needed
## Note - the hierarchies seem to have the same column names - thus DO NOT drop 

## Ensure no inadvertent row or column dropping
print(master_df.count() == shipments.count())
print(master_df.count())
print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])

## get the sum of all nan values across the dataframe
##Count the number of nan values
master_df = master_df.drop(CALENDAR_DATEVAR)
nanDict = master_df.select([count(when(isnan(c), c)).alias(c) for c in master_df.columns])\
                      .withColumn('nanDict', F.to_json(F.struct(master_df.columns)))\
                      .select('nanDict')\
                      .collect()[0].nanDict

nanCount= np.array(list(eval(nanDict).values())).sum()

# COMMAND ----------

# Pomil - 20210728 - Dropping DMDUNIT which do not have mapping in Product master table
master_df = master_df.filter((col("PLANG_MTRL_GRP_NM").isNotNull()) | (col("SRC_CTGY_1_NM").isNotNull()) | (col("BRND_NM").isNotNull()) | (col("SUBBRND_SHRT_NM").isNotNull()) | (col("FLVR_NM").isNotNull()) | (col("PCK_CNTNR_SHRT_NM").isNotNull())  | (col("PLANG_PROD_KG_QTY").isNotNull()) | (col("PLANG_PROD_8OZ_QTY").isNotNull())  |(col("PLANG_MTRL_EA_PER_CASE_CNT").isNotNull()))
print('Trimmed master DF shape:', (master_df.count(), len(master_df.columns)))
print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print("%age drop in QTY:", 100*(total_sales_qc_QTY - master_df.agg(F.sum('QTY')).collect()[0][0])/total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Target Variable
#Create target variable if needed

if TARGET_VAR == "CASES":
  master_df = master_df.withColumn("CASES",col("QTY")/col("PLANG_MTRL_EA_PER_CASE_CNT"))
  
total_sales_qc_CASES = master_df.agg(F.sum(TARGET_VAR)).collect()[0][0]

print('NaN % in DF:',nanCount/ (master_df.count() * len(master_df.columns)))

# ## Check NaN and 0 values across TARGET_VAR column
print('TARGET_VAR NaN Value Count:', master_df.agg(*[count(when(isnan(TARGET_VAR), TARGET_VAR)).alias(TARGET_VAR) ]).first())
print('DF Shape when TARGET_VAR = 0:',(master_df[master_df[TARGET_VAR] == 0].count(),len(master_df.columns)))
print('% Rows When TARGET_VAR = 0:', master_df[master_df[TARGET_VAR] == 0].count() / master_df.count())
print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])

# COMMAND ----------

## Quick review of our target variable data
df = master_df.select(TARGET_VAR).filter(col(TARGET_VAR)>0).distinct().collect()
plt.hist(np.array(df),100)  ## plotting unique to avoid overly-viz at 0
plt.show()

# COMMAND ----------

## Quick review of our target variable data when logged
df = master_df.select(TARGET_VAR).filter(col(TARGET_VAR)>0).collect()
plt.hist(np.log1p(df), 100)
plt.show()

# COMMAND ----------

# DBTITLE 1,Dynamic Aggregation
#Check if data is aggregated at desired forecat level
#agg_count = master_df.select(MODEL_ID_HIER + [TIME_VAR]+['Month_Of_Year']).dropDuplicates().count()
# added by as
agg_count = master_df.select(MODEL_ID_HIER + [TIME_VAR]).dropDuplicates().count()
raw_count = master_df.count()
print(agg_count)
print(raw_count)
if agg_count != raw_count:
  cols_to_agg = [TARGET_VAR]
  sum_n = [sum] * len(cols_to_agg)
  master_df = aggregate_data(master_df,  
                             MODEL_ID_HIER + ["HSTRY_TMFRM_STRT_DT"], 
                             cols_to_agg, 
                             sum_n)
  master_df = master_df.withColumnRenamed("sum_" + TARGET_VAR ,TARGET_VAR)
  mlflow.log_param('Data dynamically aggregated',True)
  
#Create concatenation of data hierarchy.  This field is used throughout the pipeline to mege hierarchy information
master_df = get_model_id(master_df,'MODEL_ID',MODEL_ID_HIER)

#QC checks
print(master_df.count())
print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Noise Drop
## Dropping cols with high %% count of nulls
## Note - might be able to build this in as auto-check or auto-review of dataset
## Note - This cell is dropping rows which do not have product mapping
to_convert = set(master_df.columns) # Some set of columns
  
reduce(lambda df, x: df.withColumn(x, blank_as_null(x)), to_convert, master_df)
thresh=int(len(master_df.columns) * NAN_THRESH_PERC)

master_df = master_df.dropna(thresh=thresh, how='all')

# Get the total of nans in the df
nanDict = master_df.select([count(when(isnan(c), c)).alias(c) for c in master_df.columns])\
                      .withColumn('nanDict', F.to_json(F.struct(master_df.columns)))\
                      .select('nanDict')\
                      .collect()[0].nanDict

nanCount= np.array(list(eval(nanDict).values())).sum()

## Review results after column drops
print('Trimmed master DF shape:', (master_df.count(), len(master_df.columns)))
print('% elements as NaN:', nanCount / (master_df.count() * len(master_df.columns)))
print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
# print("%age drop in QTY:", 100*(total_sales_qc_QTY - master_df.agg(F.sum('QTY')).collect()[0][0])/total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])

# COMMAND ----------

# MAGIC %md ###Driver Integration

# COMMAND ----------

# MAGIC %md #### Holidays

# COMMAND ----------

country_flags = [c for c in holidays_clean.columns if "_hol_flag" in c]
lag_cols = list(set(holidays_clean.columns) - set([TIME_VAR] + country_flags))
holidays_clean = do_lags_N(holidays_clean, TIME_VAR, lag_cols, 4, partition = None, drop_lagvars=False)

print(master_df.count())
master_df = master_df.join(holidays_clean, on=[TIME_VAR], how="left")
print(master_df.count())

print(master_df.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print(master_df.agg(F.sum('QTY')).collect()[0][0])
print('Master DF shape:', (master_df.count(),len(master_df.columns)))

# COMMAND ----------

# drop columns conveying no information, where mean = 0                         
all_hol_cols = list(set(holidays_clean.columns) - set([TIME_VAR] + country_flags))

meanDict = master_df.select([mean(c).alias(c) for c in all_hol_cols])\
.withColumn('means', F.to_json(F.struct(all_hol_cols)))\
.select('means').collect()[0].means
meanDict = eval(meanDict)

print(len(all_hol_cols) == len(meanDict))

hol_cols_to_drop = [k for (k,v) in meanDict.items() if v == 0]
old_col_count = len(master_df.columns)
master_df = master_df.drop(*hol_cols_to_drop)
print(len(master_df.columns) + len(hol_cols_to_drop) == old_col_count)

# COMMAND ----------

## Adjusting our 'cols_to_check' accordingly
cols_to_check = list(set(all_hol_cols) - set(hol_cols_to_drop))

## Calculating correlations
holcorrDict = master_df.select([abs(corr(TARGET_VAR,c)).alias(c) for c in cols_to_check])\
.withColumn('correlations', F.to_json(F.struct(cols_to_check)))\
.select('correlations').collect()[0].correlations

hol_corr_dict = eval(holcorrDict)

# COMMAND ----------

# Retaining the top 20 correlated holidays,lags,leads
final_hols_to_drop = top_n_correlation(hol_corr_dict, TOP_HOL_THRESH)[1]
master_df = master_df.drop(*final_hols_to_drop)
master_df = master_df.na.fill(0)

# COMMAND ----------

# MAGIC %md #### Covid & Macro

# COMMAND ----------

# Retaining records of external variables till max(TIME_VAR) period of master_df
covid_macro_vars = covid_macro_vars.filter(col(TIME_VAR) <= master_df.select(max(TIME_VAR)).collect()[0][0])

# COMMAND ----------

covid_macro_cols = list(set(covid_macro_vars.columns) - set([TIME_VAR] + ["country_name"]))
covid_macro_cols = [x for x in covid_macro_cols if ('_flag' not in x) and ('confirmed_cases' not in x)]  #dropping duplicate columns
covid_macro_vars = covid_macro_vars[[TIME_VAR] + ["country_name"] + covid_macro_cols]
#Forward fill required null variables
try:
  covid_macro_vars = impute_cols_ts(covid_macro_vars, covid_macro_cols, 
                                        order_col = [TIME_VAR], fill_type = "ffill", 
                                        partition_cols = 'country_name')
except:
  print("Forward fill imputation failed")
  
# fill nulls at the top of the df with -1
covid_macro_vars = covid_macro_vars.na.fill(-1)   # Encoding nulls as -1

# COMMAND ----------

# Creating column with country name
master_df = master_df.withColumn("country_name", F.when(F.col("PLANG_CUST_GRP_VAL").contains("ES_"), "Spain").otherwise("Portugal"))
master_df = master_df.join(covid_macro_vars, on = [TIME_VAR] + ["country_name"], how = "left")

## Calculating correlations of target variable with covid and macro variables
corr_df_temp = master_df.filter(col(TIME_VAR)>202000) #compute correlation from 2020 onwards
cmcorrDict = corr_df_temp.select([abs(corr(TARGET_VAR,c)).alias(c) for c in covid_macro_cols])\
                      .withColumn('correlations', F.to_json(F.struct(covid_macro_cols)))\
                      .select('correlations').collect()[0].correlations

cm_corr_dict = eval(cmcorrDict)

# some columns produce null correlation coef due to lack of variance in its records, need to account for those to be dropped
missed_cols = subtract_two_lists(covid_macro_cols, list(cm_corr_dict.keys()))

# Retaining the top 50 correlated holidays,lags,leads
cm_cols_to_drop = top_n_correlation(cm_corr_dict, 10)[1] + missed_cols
master_df = master_df.drop(*cm_cols_to_drop)
master_df = master_df.na.fill(0)

print(master_df.count(), len(master_df.columns))

# COMMAND ----------

# MAGIC %md ####Pricing

# COMMAND ----------

pricing = pricing.withColumnRenamed("DMDGroup", "PLANG_CUST_GRP_VAL").withColumnRenamed("PLANG_MTRL_GRP_VAL", "DMDUNIT")
merge_cols = [c for c in pricing.columns if "PRICE" not in c]

print(pricing.count())
display(pricing)

# COMMAND ----------

print(master_df.count())
master_df = master_df.join(pricing, on = merge_cols, how = "left")
print(master_df.count(), len(master_df.columns))

# COMMAND ----------

# MAGIC %md ####Weather

# COMMAND ----------

print(master_df.count())
weather = weather.withColumnRenamed("country", "country_name")
master_df = master_df.join(weather, on = ['country_name', TIME_VAR], how = "left")
print(master_df.count(), len(master_df.columns))

# COMMAND ----------

master_df.cache()
master_df.count()

# COMMAND ----------

# MAGIC %md ####Media

# COMMAND ----------

media_clean = load_delta(DBI_MEDIA)
MEDIA_MERGE_FIELD = MEDIA_MERGE_FIELD + [TIME_VAR]

# COMMAND ----------

# try:
if TIME_VAR=='Week_Of_Year':
    print(master_df.count())
    print(master_df.agg(F.sum('QTY')).collect()[0][0])
    master_df2 = master_df.join(media_clean, on=MEDIA_MERGE_FIELD, how='left')
    print(master_df2.agg(F.sum('QTY')).collect()[0][0])
    print(master_df2.count())

else :
    calendar_df=calendar.select("Month_Of_Year", "Week_Of_Year", "month_ratio").distinct()
    media_clean = media_clean.join(calendar_df, on="Week_Of_Year", how='left')\
                       .withColumnRenamed('month_ratio', 'ratio')

    # MEDIA_MERGE_FIELD2 = list(set(MEDIA_MERGE_FIELD) - set(["Week_Of_Year"])) + [TIME_VAR]
    for c in subtract_two_lists(media_clean.columns, MEDIA_MERGE_FIELD):
        media_clean = media_clean.withColumn(c, col("ratio")*col(c))

    #Removing data for incomplete month
    media_clean2 = media_clean.drop("ratio","Week_Of_Year")
    media_agg = media_clean2.groupBy(MEDIA_MERGE_FIELD).agg(*[sum(c).alias(c) for c in media_clean2.columns if c not in MEDIA_MERGE_FIELD]) 

    print("Month loop:", master_df.count())
    print(master_df.agg(F.sum('QTY')).collect()[0][0])
    master_df2 = master_df.join(media_agg, on=MEDIA_MERGE_FIELD, how="left")
    print(master_df2.agg(F.sum('QTY')).collect()[0][0])
    print(master_df2.count())
  
# except:
#   print('Media Spend not used in this market')


print(master_df2.agg(F.sum('QTY')).collect()[0][0] == total_sales_qc_QTY)
print(master_df2.agg(F.sum('QTY')).collect()[0][0])
print('Master DF shape:', (master_df2.count(),len(master_df2.columns)))

# COMMAND ----------

# master_df2 = master_df2.na.fill(0)
master_df2.cache()
print(master_df2.count(), len(master_df2.columns))

# COMMAND ----------

# MAGIC %md ####Neilsen

# COMMAND ----------

## Neilsen data columns to be retained 
# Retain following columns for modelling: Total_Points_Of_Sales_Dealing, 'Numeric_Distribution','Wtd_Distribution', 'Wtd_Distribution_Promo', 'Wtd_Distribution_SEL', 'Wtd_Distribution_SE', 'Wtd_Distribution_L', 'Wtd_Distribution_TPR', 'Price_Per_Qty', 'Promo_Price_Per_Volume',
# We need to drop following columns after cleansing script: 'Base_Sales_Unit', 'Unit_Sales',
# percent_baseline_unit = Base_Sales_Unit/Unit_Sales
# percent_baseline_volume = Base_Sales_Qty/Volume_Sales
# percent_baseline_value = Base_Sales_Value/Value_Sales
# and all other columns 


# COMMAND ----------

if TIME_VAR == "Month_Of_Year":
  print(nielson.count())
  calendar_df=calendar.select("Month_Of_Year", "Week_Of_Year", "month_ratio").distinct()
  nielson = nielson.join(calendar_df, on="Week_Of_Year", how='left')\
                     .withColumnRenamed('month_ratio', 'ratio')
  
  for c in subtract_two_lists(nielson.columns, NIELSON_MERGE_FIELD):
      nielson = nielson.withColumn(c, col("ratio")*col(c))
  
  nielson = nielson.groupby(NIELSON_MERGE_FIELD).agg(*[sum(c).alias(c) for c in nielson.columns if c not in NIELSON_MERGE_FIELD+["Week_Of_Year"]]).drop("ratio")
  print(nielson.count())
  print(nielson.select(NIELSON_MERGE_FIELD).distinct().count())

# COMMAND ----------

master_df2 = master_df2.join(nielson, on=NIELSON_MERGE_FIELD, how='left')
print(master_df2.count())

# COMMAND ----------

# This will speed up the run
# save_df_as_delta(master_df2, 'dbfs:/mnt/adls/Tables/master_df', enforce_schema=False)
# master_df2 = load_delta('dbfs:/mnt/adls/Tables/master_df')

# COMMAND ----------

 
cols_keywords = ["Qty", "Base", "Promo", "Price", "Wtd", "Value", "Unit", "Volume", "Total", "Numeric", "Sales", "Ventas", "Distribucion", "Weighted", "Universo", "Promedio", "TDP"]
impute_list = []
for i in cols_keywords:
  impute_list = impute_list + [x for x in nielson.columns if i in x]
impute_list = list(set(impute_list))
print(impute_list)

# COMMAND ----------

# Imputation 
auto_schema = master_df2.schema
@pandas_udf(auto_schema, PandasUDFType.GROUPED_MAP)
def interpolate_features_udf(data):
    return interpolate_col(data, impute_list, impute_method="slinear")

if master_df2.filter(col("neil_Numeric_Distribution").isNull()).count()>0:
  master_df2 = master_df2.groupBy(level_at_which_to_impute).apply(interpolate_features_udf)
  master_df2.cache()

master_df2 = impute_cols_ts(master_df2, impute_list, order_col=level_at_which_to_impute, fill_type = "ffill")
master_df2 = impute_cols_ts(master_df2, impute_list, order_col=level_at_which_to_impute, fill_type = "bfill") 
print("Numeric_Distribution Null %age after imputation:", 100*(master_df2.filter(col("neil_Numeric_Distribution").isNull()).count()/master_df2.count() ) )

# COMMAND ----------

# # This will speed up the run 
# save_df_as_delta(master_df_ES, 'dbfs:/mnt/adls/Tables/master_df_ES', enforce_schema=False) 
# master_df_ES = load_delta('dbfs:/mnt/adls/Tables/master_df_ES') 

# COMMAND ----------

master_df2 = master_df2.withColumn("neil_percent_baseline_unit", col("neil_Base_Sales_Unit")/col("neil_Unit_Sales"))\
.withColumn("neil_percent_baseline_volume", col("neil_Base_Sales_Qty")/col("neil_Volume_Sales"))\
.withColumn("neil_percent_baseline_value", col("neil_Base_Sales_Value")/col("neil_Value_Sales"))

display(master_df2.filter(col("neil_percent_baseline_unit").isNotNull()))

# COMMAND ----------

display(master_df2)

# COMMAND ----------

# DBTITLE 1,Dropping Extra Time Columns
master_df2 = master_df2.drop(*TIME_COLS_TO_DROP)

# COMMAND ----------

# DBTITLE 1,Output
## Write as delta table to dbfs
save_df_as_delta(master_df2, DBA_MRD_EXPLORATORY, enforce_schema=False)
delta_info = load_delta_info(DBA_MRD_EXPLORATORY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

#Log data version to mlflow
data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBA_MRD_EXPLORATORY +"`)").collect()
data_version = data_version[0][0]
mlflow.log_param('Delta Version',data_version)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

