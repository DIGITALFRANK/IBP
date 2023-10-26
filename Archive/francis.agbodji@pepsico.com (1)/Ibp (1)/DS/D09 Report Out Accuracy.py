# Databricks notebook source
# MAGIC %md 
# MAGIC # Report Out Accuracy
# MAGIC 
# MAGIC This script outputs the DemandBrain accuracy compared to a reference point (e.g., PEP, adjusted forecast, final DP submission, etc.).
# MAGIC * It calculates DemandBrain accuracy for all present lags and compares to whatever lags are in the PEP forecast
# MAGIC * It reports accuracy at the global accuracy_report_level config

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
check_or_init_dropdown_widget("load_prev_merge", "False", [str(x) for x in [True, False]]) 
check_or_init_dropdown_widget("prev_merge_version", "None", ["None"] + [str(x) for x in list(range(0, 1000))]) 
check_or_init_dropdown_widget("save_curr_merge", "True", [str(x) for x in [True, False]]) 
check_or_init_dropdown_widget("backtest_fcst_version", "None", ["None"] + [str(x) for x in list(range(0, 1000))]) 

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

# MAGIC %md ### Load configurations

# COMMAND ----------

required_configs = [ACCURACY_REPORT_LEVEL, SAMPLING_VARS, PEP_COMPARISON_PRED,
                   FCST_LAG_VAR, ERROR_METRICS_TO_REPORT, VIZ_REPORT_LEVEL,
                   #DBI_PEP_ACC_DATA, 
                    DBO_ACCURACY_COMPARISON, TARGET_VAR_ACCURACY, NUM_FWD_FRCST]
print(json.dumps(required_configs, indent=4))
if required_configs.count(None) >0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

# AD-HOC override of configurations
# TO DO: Correct configurations to make this not necessary

# backtest_fcst_version = 50    
freq = 52

if TIME_VAR == "Month_Of_Year":
  # backtest_fcst_version = 34   
  freq = 12

# Making sure the time var is well defined
ACCURACY_REPORT_LEVEL[0] = TIME_VAR

# Naming target var as CASES_ORIG
TARGET_VAR = TARGET_VAR_ACCURACY

# COMMAND ----------

# DBTITLE 1,Initialize Configuration to blob
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

# COMMAND ----------

# DBTITLE 1,Load Modeling Output Data
## Load using delta so we can load up specific versions
#TO-DO: Accuracies should be stored in Delta Lake as DBI_PEP_ACC_DATA
if load_prev_merge == False:
  pep_mf = spark.read.option("header","true").option("delimiter",";").csv("abfss://bronze@cdodevadls2.dfs.core.windows.net/IBP/EDW/Prolink/DFU to SKU Forecast/")
  pep_mf = pep_mf.drop("ingestion_dt")
  pep_mf = pep_mf.distinct()


# Load demandbrain rolling forecast (will default to versions 50/34 depending on periodicity if not indicated otherwise)
demandbrain_mf = load_delta(DBO_FORECAST_TRAIN_TEST_SPLIT, backtest_fcst_version)
demandbrain_mf = demandbrain_mf.distinct()
demandbrain_mf = demandbrain_mf.withColumn("max_forcast", col("FCST_START_DATE") + col("lag_period") - 1)

# Load calendar and model IDs
calendar = load_delta(DBI_CALENDAR)
model_ids = load_delta(DBA_MODELIDS)

if (TIME_VAR in model_ids.columns):
  model_ids = model_ids.drop(TIME_VAR).distinct()

# Load previous merge if indicated (to measure accuracy)
if load_prev_merge == True:
  all_predictions = load_delta(DBO_ACC_MERGED, prev_merge_version)
  ### The following can be uncommented and run to get latest relevant all_predictions (versions to be updated accordingly ###
  # if TIME_VAR == "Week_Of_Year":
  #   all_predictions = load_delta(DBO_ACC_MERGED, 31)  # weekly
  # else:
  #   all_predictions = load_delta(DBO_ACC_MERGED, 29)   # monthly 
  if (TIME_VAR not in all_predictions.columns):
    raise Exception("TIME VAR has to be the opposite for the selected merged table version")
#   all_predictions = load_delta(DBO_ACC_MERGED)

  # Make sure no duplicates or negatives are present
  all_predictions = all_predictions.dropDuplicates(['MODEL_ID', TIME_VAR, 'FCST_LAG'])
  all_predictions = all_predictions.withColumn("DMNDFCST_QTY", when(all_predictions["DMNDFCST_QTY"] >= 0, all_predictions["DMNDFCST_QTY"]).otherwise(lit(0)))

# COMMAND ----------

# DBTITLE 1,Clean the data available
# Select only out of sample predictions
if "sample" in demandbrain_mf.columns:
  demandbrain_mf = demandbrain_mf.filter((col("sample") == "OOS") & (col("FCST_START_DATE") <= col(TIME_VAR)))
else:
  demandbrain_mf = demandbrain_mf.filter(col("FCST_START_DATE") <= col(TIME_VAR))
  
# Merge Model IDs data frame to complete demandbrain with extra columns
dup_cols = list(set(intersect_two_lists((model_ids.columns), (demandbrain_mf.columns))) - set(['MODEL_ID']))
demandbrain_mf = demandbrain_mf.join(model_ids.drop(*dup_cols), on=["MODEL_ID"], how="left")

#Check necessary columns are in datasets
try:
  len(subtract_two_lists(ACCURACY_REPORT_LEVEL + SAMPLING_VARS + [TARGET_VAR] , demandbrain_mf.columns)) == 0
except:
  dbutils.notebook.exit("Missing required variables in demandbrain forecast")

if load_prev_merge != True:
  try:
    len(subtract_two_lists(ACCURACY_REPORT_LEVEL + FCST_LAG_VAR + PEP_COMPARISON_PRED, pep_mf.columns)) == 0
  except:
    dbutils.notebook.exit("Missing required variables in PepsiCo forecast")


#TO-DO: This should be transitioned to Data Engineering in the data model
# Cleanse case qty and kg qty
try:
  demandbrain_mf = demandbrain_mf.withColumn("PLANG_PROD_KG_QTY",
                                   round(col("PLANG_PROD_KG_QTY")*lit(1000)))
  demandbrain_mf = demandbrain_mf.withColumn("PLANG_MTRL_EA_PER_CASE_CNT",
                                   round(col("PLANG_MTRL_EA_PER_CASE_CNT")))
  demandbrain_mf = demandbrain_mf.withColumn("PLANG_PROD_KG_QTY", col("PLANG_PROD_KG_QTY").cast('int'))
except Exception as e:
  print (e)
  print("Case qty and Kg qty not present in data")

# COMMAND ----------

# DBTITLE 1,Making sure DB lags are well built
if load_prev_merge == False:
#   demandbrain_mf = demandbrain_mf.withColumn("FCST_LAG", round((col(TIME_VAR) - col("FCST_START_DATE")) / lit(100)) * freq + \
#                                              (col(TIME_VAR) % lit(100) - col("FCST_START_DATE") % lit(100)))
  demandbrain_mf = demandbrain_mf.withColumn("FCST_LAG", col("fcst_periods_fwd"))

  demandbrain_mf = demandbrain_mf.filter(col("FCST_LAG") >= 0)
  demandbrain_mf = demandbrain_mf.withColumn("FCST_LAG", (col("FCST_LAG")).cast("int")).drop("fcst_periods_fwd")   

# COMMAND ----------

# DBTITLE 1,Merge DB and PEP forecasts
# Change column names and add relevant columns

if load_prev_merge == False:
  #TO-DO: This should be transitioned to data engineering so incoming forecast is as expected
  pep_mf = pep_mf.withColumn('DMNDFCST_WK_STRT_DT',to_timestamp(pep_mf.DMNDFCST_WK_STRT_DT, "yyyy-MM-dd HH:mm:ss.SSSSSSS"))\
                  .withColumn('DMNDFCST_GNRTN_DTM',to_timestamp(pep_mf.DMNDFCST_GNRTN_DTM, "yyyy-MM-dd HH:mm:ss.SSSSSSS"))

  calendar_df = calendar.select(TIME_VAR,"Week_start_date").distinct()
  pep_mf = pep_mf.join(calendar_df, pep_mf["DMNDFCST_WK_STRT_DT"] == calendar_df["Week_start_date"], how='left').drop("Week_start_date")

  pep_mf = pep_mf.withColumnRenamed("DMND_GRP_UNIQ_ID_VAL", 'PLANG_CUST_GRP_VAL')\
                  .withColumnRenamed("MTRL_UNIQ_ID_VAL", 'DMDUNIT')\
                  .withColumnRenamed("DMNDFCST_LOC_GRP_UNIQ_ID_VAL", 'LOC')\
                  .withColumnRenamed("LAG_WK_QTY", "FCST_LAG")
  
  pep_mf = pep_mf.withColumn("FCST_LAG", col("FCST_LAG").cast("int"))
  
  pep_mf = get_model_id(pep_mf, "MODEL_ID", ["DMDUNIT", "PLANG_CUST_GRP_VAL", "LOC"])
  
  # Make sure no duplicates or negatives are present
  pep_mf = pep_mf.dropDuplicates(['MODEL_ID', TIME_VAR, 'FCST_LAG'])
  pep_mf = pep_mf.withColumn("DMNDFCST_QTY", when(pep_mf["DMNDFCST_QTY"] >= 0, pep_mf["DMNDFCST_QTY"]).otherwise(lit(0)))

# COMMAND ----------

# DBTITLE 1,Aggregate if TIME_VAR is monthly (we should consider the first week of the month as the snapshot date)
if load_prev_merge == False:
  # Create relevant agg levels
  MODEL_ID_LEVEL = ["PLANG_CUST_GRP_VAL", "DMDUNIT", "LOC", "MODEL_ID"]
  MERGE_LEVEL = ["PLANG_CUST_GRP_VAL", "DMDUNIT", "LOC", "MODEL_ID", TIME_VAR, "FCST_LAG"]
  
  if TIME_VAR == "Month_Of_Year":
    calendar = load_delta(DBI_CALENDAR)
    
    ## Setting up with time-based features using calendar reference
    agg_col = PEP_COMPARISON_PRED[0]
    ratio = ["month_ratio"]
    calendar_df = calendar.select("Week_Of_Year", "Week_start_date", "month_ratio").distinct()
    pep_mf = pep_mf.join(calendar_df, pep_mf["DMNDFCST_WK_STRT_DT"] == calendar_df["Week_start_date"], how='left')\
                           .drop("Week_start_date")\
                           .withColumnRenamed('month_ratio', 'ratio')\
                           .withColumn(agg_col, col("ratio")*col(agg_col))\
                           .drop("ratio", "Week_Of_Year")

    ## Create snapshot date column to recalculate the FCST_LAG based on monthly
    calendar_df_2 = calendar.select(TIME_VAR,"Week_start_date").withColumnRenamed(TIME_VAR, "snapshot_month").sort(col("Week_start_date").asc()).drop_duplicates(["Week_start_date"])
    pep_mf = pep_mf.join(calendar_df_2, pep_mf["DMNDFCST_GNRTN_DTM"] == calendar_df_2["Week_start_date"], how='left').drop("Week_start_date")
    pep_mf = pep_mf.withColumn("FCST_LAG_NEW", round((col(TIME_VAR) - col("snapshot_month")) / lit(100)) * lit(12) + (col(TIME_VAR) % lit(100) - col("snapshot_month") % lit(100)))

    ## Grouping by forecast generation date, sorting ascending and dropping duplicates at the MODEL_ID_LEVEL + [TIME_VAR, "FCST_LAG_new"]
    pep_mf = pep_mf.groupBy(MODEL_ID_LEVEL + ["snapshot_month", TIME_VAR, "FCST_LAG_new", "DMNDFCST_GNRTN_DTM"]).agg(sum(agg_col).alias(agg_col))
    pep_mf = pep_mf.orderBy(col('MODEL_ID').desc(), col("snapshot_month").asc(), col('Month_Of_Year').asc(), col("FCST_LAG_new").asc(), col("DMNDFCST_GNRTN_DTM").asc())
    pep_mf = pep_mf.drop_duplicates(["MODEL_ID", "snapshot_month", 'Month_Of_Year', "FCST_LAG_new"]).drop("DMNDFCST_GNRTN_DTM", "snapshot_month").withColumnRenamed("FCST_LAG_new", "FCST_LAG").orderBy(col('MODEL_ID').desc(), col('Month_Of_Year').asc(), col("FCST_LAG").asc())

# COMMAND ----------

# Perform merges
if load_prev_merge == False:
  
#   print(pep_mf.count())
  # Filter pep_mf only for the model_ids for which there is a prediction in both (replacing dataframes to avoid memory issues)
  pep_mf = pep_mf.join(demandbrain_mf.select(MODEL_ID_LEVEL).distinct(), on=MODEL_ID_LEVEL, how="inner").select(MERGE_LEVEL + PEP_COMPARISON_PRED)
#   print(demandbrain_mf.count())
  demandbrain_mf = demandbrain_mf.join(pep_mf.select(MODEL_ID_LEVEL).distinct(), on=MODEL_ID_LEVEL, how="inner")
  
  # Filter only the predicted dates and lags available in demandbrain so that the outer join later does not generate dates we've not backtested (relies on having at least one record of each date in demandbrain, will not be needed when demandbrain_mf contains all needed dates for each model_id)
  pep_mf = pep_mf.join(demandbrain_mf.select([TIME_VAR, "FCST_LAG"]).distinct(), on=[TIME_VAR, "FCST_LAG"], how="inner")
#   print(pep_mf.count())

  # Select only the lag with the smallest holdout (only discarding +1 periods)
  if only_closest_lag:
    # Search for the minimum lag model (indicated by lag_period) that is able to provide a forecast to that FCST_LAG and filter demandbrain_mf via an inner merge
    temp = demandbrain_mf.groupBy(["MODEL_ID", "FCST_START_DATE", TIME_VAR, "FCST_LAG"]).agg(min(col("lag_period")).alias("lag_period"))
    demandbrain_mf = demandbrain_mf.join(temp, on=["MODEL_ID", "FCST_START_DATE", TIME_VAR, "FCST_LAG", "lag_period"], how="inner")
#   print(demandbrain_mf.count())

# COMMAND ----------

if load_prev_merge == False:
 
  # Separate categoricals to avoid creating nulls at undesired columns
  model_cols = [col_name for col_name in demandbrain_mf.columns if "stage" in col_name and "calc" not in col_name]
  
  cols_num = MERGE_LEVEL + [TARGET_VAR] + model_cols + ['FCST_START_DATE', 'lag_period', 'max_forcast'] 
  cols_ctgy = MODEL_ID_LEVEL + [col_name for col_name in demandbrain_mf.columns if col_name not in cols_num]
  
  if "best_model" in demandbrain_mf.columns:
    model_cols = model_cols + ['final_prediction_value']
    cols_num = cols_num + ['best_model', 'final_prediction_value']
    cols_ctgy.remove('final_prediction_value')
    cols_ctgy.remove('best_model')
  
  # Test to verify we're doing it right
  check_1 = MODEL_ID_LEVEL.copy()
  check_1.sort()
  check_2 = list(set(cols_num) & set(cols_ctgy))
  check_2.sort()
  if (check_1 != check_2):
    print("Issue separating dataframes")

  demandbrain_cat = demandbrain_mf.select(cols_ctgy).distinct()
  demandbrain_num = demandbrain_mf.select(cols_num)

  # Make a merge at the necessary level
  all_predictions = demandbrain_num.join(pep_mf, on=MERGE_LEVEL, how="outer")

  # Fill NAs with zeros for missing records
  all_predictions = all_predictions.fillna(0) # For numerics
  all_predictions = all_predictions.fillna('0') # For numerics being stored as characters

  all_predictions = all_predictions.join(demandbrain_cat, on=MODEL_ID_LEVEL, how="inner")

  # Cache for faster processing
  all_predictions.cache() 
  
  if (all_predictions.count() != all_predictions.select(MERGE_LEVEL).distinct().count()):
    raise Exception("Duplicates being generated! Review categorical and numericals")

# COMMAND ----------

# DBTITLE 1,Save merged table
# Save merged table
if load_prev_merge == False:
  if save_curr_merge == True:
    save_df_as_delta(all_predictions, DBO_ACC_MERGED, enforce_schema=False)

# COMMAND ----------

delta_info = load_delta_info(DBO_ACC_MERGED)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Make aggregation
#Aggregate all predictions
model_cols = [col_name for col_name in all_predictions.columns if "stage" in col_name and "calc" not in col_name]

if "final_prediction_value" in all_predictions.columns:
    model_cols = model_cols + ['final_prediction_value']

cols_to_agg = [TARGET_VAR] + model_cols + PEP_COMPARISON_PRED
sum_n = [sum] * len(cols_to_agg)
agg_dict = dict(zip(cols_to_agg, sum_n))

all_predictions_agg = aggregate_data(all_predictions, 
                                   list(set(SAMPLING_VARS + 
                                   ACCURACY_REPORT_LEVEL +
                                   FCST_LAG_VAR)), cols_to_agg, sum_n)
  
#Remove sum_ from summed variable names
for i in cols_to_agg:
  all_predictions_agg =  all_predictions_agg.withColumnRenamed("sum_" + i,i)
  
all_predictions_agg = get_model_id(all_predictions_agg,"REPORT_ID", FCST_LAG_VAR + SAMPLING_VARS + ACCURACY_REPORT_LEVEL)

# COMMAND ----------

# DBTITLE 1,Calculate Line Item Errors
#Create nested dictionary of metrics we want to calculate
gof_nested_dict = {}

models_to_evaluate = [col_name for col_name in demandbrain_mf.columns if "stage" in col_name and "calc" not in col_name]

if "final_prediction_value" in all_predictions.columns:
    models_to_evaluate = models_to_evaluate + ['final_prediction_value']

models_to_evaluate = models_to_evaluate + PEP_COMPARISON_PRED
models_to_evaluate = intersect_two_lists(models_to_evaluate, all_predictions_agg.columns)

#TO-DO: Make metric calls dynamic to feed off METRICS_TO_REPORT
for i in models_to_evaluate:
  #Append weighted accuracies to dictionary
  this_acc_dict = {'error_func' : calculate_line_item_abs_error,
                   'parameters': [all_predictions_agg, TARGET_VAR, i, ["REPORT_ID"]]
                  }
  gof_nested_dict[i + "_APA"] = this_acc_dict
  
  #Append bias to dictionary
  this_bias_dict = {'error_func' : calculate_line_item_error,
                   'parameters': [all_predictions_agg, TARGET_VAR, i, ["REPORT_ID"]]
                   }
  gof_nested_dict[i + "_Bias"] = this_bias_dict

print(models_to_evaluate)
print (json.dumps(gof_nested_dict, indent=2, default=str))

#Generate gof report
gof_report = calculate_gof(all_predictions_agg, "REPORT_ID", gof_nested_dict)

#Generate accuracy report
metrics_to_calculate = [y + "_" + x for y in models_to_evaluate for x in ERROR_METRICS_TO_REPORT]
acc_report = all_predictions_agg.join(gof_report, on = ["REPORT_ID"], how="left" )

# COMMAND ----------

# DBTITLE 1,Apply filters and calculate results at the desired level for later visualization
## Accuracy Reporting level functions (assuming viz_level is passed as a list containing the column/level we want accuracy at)

# Select relevant columns
accuracy_metrics = [y + "_" + x for y in models_to_evaluate for x in ["APA"]]
bias_metrics = [y + "_" + x for y in models_to_evaluate for x in ["Bias"]]

# Function to generate report at different visualization levels
def generate_report(viz_level, val_list = []):    
  
  df = acc_report
  
  if len(val_list) > 0:
    df = df.filter((col(viz_level[0]).isin(val_list)))
      
  df = df.select(viz_level + [TARGET_VAR] + metrics_to_calculate)  
  df = df.toPandas()
  df = df.fillna(0)
    
  return df

# Function to get top n distinct values of a column by sum of target, (to use only if top n required)
def get_top_n(n, viz_level):
  distinct_vals = convertDFColumnsToList(acc_report.groupBy(viz_level[0]).agg(sum(TARGET_VAR).alias(TARGET_VAR)).sort(col(TARGET_VAR).desc()), viz_level[0])
  
  return distinct_vals[:n]

# COMMAND ----------

# DBTITLE 1,Generate report at different VIZ LEVELS
# Generate report at FCST_LAG level 
# can replace df with generate_report(VIZ_REPORT_LEVEL)
VIZ_REPORT_LEVEL = ["FCST_LAG"]
df = generate_report(VIZ_REPORT_LEVEL)
accuracy_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: 1 - np.sum(x[accuracy_metrics]) / np.sum(x[TARGET_VAR]))  
accuracy_review_pd = pd.DataFrame(dict(zip(accuracy_review_series.index, accuracy_review_series.values))).T
accuracy_review_pd.columns = accuracy_metrics

bias_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: np.sum(x[bias_metrics]) / np.sum(x[TARGET_VAR]))   
bias_review_pd = pd.DataFrame(dict(zip(bias_review_series.index, bias_review_series.values))).T
bias_review_pd.columns = bias_metrics

#Shape data for viz
accuracy_review_pd = accuracy_review_pd.stack().reset_index()
bias_review_pd = bias_review_pd.stack().reset_index()

accuracy_review_pd = pd.concat([accuracy_review_pd, bias_review_pd], axis=0)

accuracy_review_pd.columns = VIZ_REPORT_LEVEL + ['model','value']
accuracy_review_pd = extract_data_to_col(accuracy_review_pd, "model", "metric", ERROR_METRICS_TO_REPORT)
accuracy_review_pd = remove_data_from_col(accuracy_review_pd, ERROR_METRICS_TO_REPORT,"model")
accuracy_review_pd = accuracy_review_pd.pivot(index=subtract_two_lists(accuracy_review_pd.columns,["metric","value"]), 
                                              columns="metric", values="value")

accuracy_review_pd = accuracy_review_pd.unstack(level=VIZ_REPORT_LEVEL)
accuracy_review_pd = accuracy_review_pd.sort_values(accuracy_review_pd.columns[0], ascending=False)

accuracy_review_pd.style.bar(subset=['APA',], color='lightgreen')\
                        .bar(subset=['Bias',], color='lightblue')\
                        .highlight_max(color='goldenrod')\
                        .highlight_min(color='indianred')

# COMMAND ----------

# Generate report at country, fcst_lag, TIME_VAR level
VIZ_REPORT_LEVEL = ["HRCHY_LVL_3_NM", "FCST_LAG", TIME_VAR]
df = generate_report(VIZ_REPORT_LEVEL)
accuracy_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: 1 - np.sum(x[accuracy_metrics]) / np.sum(x[TARGET_VAR]))  
accuracy_review_pd = pd.DataFrame(dict(zip(accuracy_review_series.index, accuracy_review_series.values))).T
accuracy_review_pd.columns = accuracy_metrics

bias_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: np.sum(x[bias_metrics]) / np.sum(x[TARGET_VAR]))   
bias_review_pd = pd.DataFrame(dict(zip(bias_review_series.index, bias_review_series.values))).T
bias_review_pd.columns = bias_metrics

#Shape data for viz
accuracy_review_pd = accuracy_review_pd.stack().reset_index()
bias_review_pd = bias_review_pd.stack().reset_index()

accuracy_review_pd = pd.concat([accuracy_review_pd, bias_review_pd], axis=0)

accuracy_review_pd.columns = VIZ_REPORT_LEVEL + ['model','value']
accuracy_review_pd = extract_data_to_col(accuracy_review_pd, "model", "metric", ERROR_METRICS_TO_REPORT)
accuracy_review_pd = remove_data_from_col(accuracy_review_pd, ERROR_METRICS_TO_REPORT,"model")
accuracy_review_pd = accuracy_review_pd.pivot(index=subtract_two_lists(accuracy_review_pd.columns,["metric","value"]), 
                                              columns="metric", values="value")

accuracy_review_pd = accuracy_review_pd.unstack(level=VIZ_REPORT_LEVEL)
accuracy_review_pd = accuracy_review_pd.sort_values(accuracy_review_pd.columns[0], ascending=False)

accuracy_review_pd.style.bar(subset=['APA',], color='lightgreen')\
                        .bar(subset=['Bias',], color='lightblue')\
                        .highlight_max(color='goldenrod')\
                        .highlight_min(color='indianred')

# COMMAND ----------

# Generate report at country, category, TIME_VAR level
VIZ_REPORT_LEVEL = ["HRCHY_LVL_3_NM", "SRC_CTGY_1_NM"]
df = generate_report(VIZ_REPORT_LEVEL)
accuracy_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: 1 - np.sum(x[accuracy_metrics]) / np.sum(x[TARGET_VAR]))  
accuracy_review_pd = pd.DataFrame(dict(zip(accuracy_review_series.index, accuracy_review_series.values))).T
accuracy_review_pd.columns = accuracy_metrics

bias_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: np.sum(x[bias_metrics]) / np.sum(x[TARGET_VAR]))   
bias_review_pd = pd.DataFrame(dict(zip(bias_review_series.index, bias_review_series.values))).T
bias_review_pd.columns = bias_metrics

#Shape data for viz
accuracy_review_pd = accuracy_review_pd.stack().reset_index()
bias_review_pd = bias_review_pd.stack().reset_index()

accuracy_review_pd = pd.concat([accuracy_review_pd, bias_review_pd], axis=0)

accuracy_review_pd.columns = VIZ_REPORT_LEVEL + ['model','value']
accuracy_review_pd = extract_data_to_col(accuracy_review_pd, "model", "metric", ERROR_METRICS_TO_REPORT)
accuracy_review_pd = remove_data_from_col(accuracy_review_pd, ERROR_METRICS_TO_REPORT,"model")
accuracy_review_pd = accuracy_review_pd.pivot(index=subtract_two_lists(accuracy_review_pd.columns,["metric","value"]), 
                                              columns="metric", values="value")

accuracy_review_pd = accuracy_review_pd.unstack(level=VIZ_REPORT_LEVEL)
accuracy_review_pd = accuracy_review_pd.sort_values(accuracy_review_pd.columns[0], ascending=False)

accuracy_review_pd.style.bar(subset=['APA',], color='lightgreen')\
                        .bar(subset=['Bias',], color='lightblue')\
                        .highlight_max(color='goldenrod')\
                        .highlight_min(color='indianred')

# COMMAND ----------

# Generate report at country, category, TIME_VAR level
VIZ_REPORT_LEVEL = ["HRCHY_LVL_3_NM", "BRND_NM"]
df = generate_report(VIZ_REPORT_LEVEL)
accuracy_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: 1 - np.sum(x[accuracy_metrics]) / np.sum(x[TARGET_VAR]))  
accuracy_review_pd = pd.DataFrame(dict(zip(accuracy_review_series.index, accuracy_review_series.values))).T
accuracy_review_pd.columns = accuracy_metrics

bias_review_series = df.groupby(VIZ_REPORT_LEVEL).apply(lambda x: np.sum(x[bias_metrics]) / np.sum(x[TARGET_VAR]))   
bias_review_pd = pd.DataFrame(dict(zip(bias_review_series.index, bias_review_series.values))).T
bias_review_pd.columns = bias_metrics

#Shape data for viz
accuracy_review_pd = accuracy_review_pd.stack().reset_index()
bias_review_pd = bias_review_pd.stack().reset_index()

accuracy_review_pd = pd.concat([accuracy_review_pd, bias_review_pd], axis=0)

accuracy_review_pd.columns = VIZ_REPORT_LEVEL + ['model','value']
accuracy_review_pd = extract_data_to_col(accuracy_review_pd, "model", "metric", ERROR_METRICS_TO_REPORT)
accuracy_review_pd = remove_data_from_col(accuracy_review_pd, ERROR_METRICS_TO_REPORT,"model")
accuracy_review_pd = accuracy_review_pd.pivot(index=subtract_two_lists(accuracy_review_pd.columns,["metric","value"]), 
                                              columns="metric", values="value")

accuracy_review_pd = accuracy_review_pd.unstack(level=VIZ_REPORT_LEVEL)
accuracy_review_pd = accuracy_review_pd.sort_values(accuracy_review_pd.columns[0], ascending=False)

accuracy_review_pd.style.bar(subset=['APA',], color='lightgreen')\
                        .bar(subset=['Bias',], color='lightblue')\
                        .highlight_max(color='goldenrod')\
                        .highlight_min(color='indianred')

# COMMAND ----------

# If we want to export into excel
print(all_predictions_agg.count())
display(all_predictions_agg)

# COMMAND ----------

# DBTITLE 1,Generate tables to feed PowerBI report (will need to be refactored)
# Define target var of interest
target_level = "cases"
source_level = "cases"

# Define relevant accuracy levels
SKU_level = ['SRC_CTGY_1_NM', 'BRND_NM', 'SUBBRND_SHRT_NM', 'PCK_CNTNR_SHRT_NM', 'FLVR_NM']
acc_level_weekly = ['HRCHY_LVL_3_NM'] + SKU_level + ['PCK_SIZE_SHRT_NM', 'PLANG_MTRL_EA_PER_CASE_CNT', 'LOC']
acc_level_monthly_SC = ['HRCHY_LVL_3_NM', 'SRC_CTGY_1_NM', 'BRND_NM', 'SUBBRND_SHRT_NM', 'PCK_CNTNR_SHRT_NM', 'PCK_SIZE_SHRT_NM']
acc_level_monthly_commercial_channel = acc_level_monthly_SC + ['UDC_CHANNEL']
acc_level_monthly_commercial_cust = acc_level_monthly_commercial_channel +  ['PLANG_CUST_GRP_VAL']

# COMMAND ----------

# Define different aggrupations for later groupbys
hier_level = acc_level_weekly
hist_level = hier_level + ["PERIOD"]
acc_level = hist_level + ["SNAPSHOT", "FCST_LAG"]

# COMMAND ----------

# load data
table_merged = all_predictions
# historical = load_delta(DBO_FORECAST_TRAIN_TEST_SPLIT, backtest_fcst_version)
historical = load_delta(DBA_MRD, 59)
historical = historical.withColumnRenamed("CASES", "CASES_ORIG")

customer_master = load_delta(DBI_CUSTOMER)
customer_master = customer_master.select("PLANG_CUST_GRP_VAL", "UDC_CHANNEL").distinct()
product_master = load_delta(DBI_PRODUCTS, 13)
product_master = product_master.select(["DMDUNIT", "PCK_SIZE_SHRT_NM"]).distinct()
location_master = load_delta(DBI_LOC)
location_master = location_master.select("LOC", "PLANG_LOC_UNIT_TYP_SHRT_NM").distinct()

model_info = load_delta(DBA_MODELIDS, 48)
if ("Month_Of_Year" in model_info.columns):
  model_info = model_info.drop("Month_Of_Year").distinct()
model_info = model_info.join(customer_master, on = "PLANG_CUST_GRP_VAL", how="left").join(product_master, on = 'DMDUNIT', how = 'left').join(location_master, on = "LOC", how = "left")

calendar_df = load_delta(DBI_CALENDAR)

# COMMAND ----------

# Convert to dates and select best_model

if "Week_Of_Year" in table_merged.columns:
  TIME_VAR = "Week_Of_Year"
  
  table_merged = table_merged.join(calendar_df.select([TIME_VAR, "Week_start_date"]).distinct(), on = TIME_VAR, how = "left")
  table_merged = table_merged.withColumnRenamed("Week_start_date", "PERIOD").drop(TIME_VAR)
  table_merged = table_merged.withColumn("PERIOD", col("PERIOD").cast("string").substr(1, 10))
  
  # Put snapshot in date format
  table_merged = table_merged.withColumn("days", (col("FCST_LAG")*lit(7)).cast("int")).withColumn("SNAPSHOT", expr("date_sub(to_date(PERIOD), days)")).drop("days")
  
  # Build historical
  min_date = historical.agg(max(col(TIME_VAR))).collect()[0][0] - 200
  historical = historical.select(["MODEL_ID", TIME_VAR, "CASES_ORIG"]).filter(col(TIME_VAR) >= min_date).distinct()
  historical = historical.join(calendar_df.select([TIME_VAR, "Week_start_date"]).distinct(), on = TIME_VAR, how = "left")
  historical = historical.withColumnRenamed("Week_start_date", "PERIOD").drop(TIME_VAR)
  historical = historical.withColumn("PERIOD", col("PERIOD").cast("string").substr(1, 10))
  
  if "best_model" not in table_merged.columns:
    best_models = load_delta('dbfs:/mnt/adls/Tables/DBO_FORECAST_FUTURE_PERIOD', 37)
    best_models = best_models.select(["MODEL_ID", "best_model"]).filter(col("best_model").isin(["catboost_model_stage1", "lightGBM_model_stage1", "lightGBM_model_stage2"])).distinct()
    table_merged = table_merged.join(best_models, on="MODEL_ID", how="left")
else:
  TIME_VAR = "Month_Of_Year"
  
  table_merged = table_merged.join(calendar_df.select([TIME_VAR, "Month_start_date"]).distinct(), on = TIME_VAR, how = "left")
  table_merged = table_merged.withColumnRenamed("Month_start_date", "PERIOD").drop(TIME_VAR)
  table_merged = table_merged.withColumn("PERIOD", col("PERIOD").cast("string").substr(1, 10))
  
  # Put snapshot in date format
  table_merged = table_merged.withColumn("months", (col("FCST_LAG")*lit(-1)).cast("int")).withColumn("SNAPSHOT", expr("add_months(to_date(PERIOD), months)")).drop("months")
  
  # Build historical
  min_date = historical.agg(max(col(TIME_VAR))).collect()[0][0] - 200
  historical = historical.select(["MODEL_ID", TIME_VAR, "CASES_ORIG"]).filter(col(TIME_VAR) >= min_date).distinct().withColumnRenamed(TIME_VAR, "PERIOD")
  historical = historical.join(calendar_df.select([TIME_VAR, "Month_start_date"]).distinct(), on = TIME_VAR, how = "left")
  historical = historical.withColumnRenamed("Month_start_date", "PERIOD").drop(TIME_VAR)
  historical = historical.withColumn("PERIOD", col("PERIOD").cast("string").substr(1, 10))
  
  if "best_model" not in table_merged.columns:
    best_models = load_delta('dbfs:/mnt/adls/Tables/DBO_FORECAST_FUTURE_PERIOD', 36)
    best_models = best_models.select(["MODEL_ID", "best_model"]).filter(col("best_model").isin(["catboost_model_stage1", "lightGBM_model_stage1", "lightGBM_model_stage2"])).distinct()
    table_merged = table_merged.join(best_models, on="MODEL_ID", how="left")

# Select only lags greater than 0
table_merged = table_merged.filter(col("FCST_LAG") > 0)

# Fill for missing models
table_merged = table_merged.fillna("lightGBM_model_stage2", subset = "best_model")

# Fill final prediction column
if "final_prediction_value" not in table_merged.columns:
  table_merged = table_merged.rdd.map(lambda row: row + (row[row.best_model], )).toDF(table_merged.columns + ["FINAL_PRED"])
else:
  table_merged = table_merged.withColumn("FINAL_PRED", col("final_prediction_value"))

# Select only relevant columns
table_merged = table_merged.select(["MODEL_ID", "SNAPSHOT", "PERIOD", "FCST_LAG", "CASES_ORIG", "FINAL_PRED", "DMNDFCST_QTY"]).withColumnRenamed("DMNDFCST_QTY", "PEP_PRED")

# Merge with master
table_merged = table_merged.join(model_info, on="MODEL_ID", how="left")
historical = historical.join(model_info, on="MODEL_ID", how="left")

# COMMAND ----------

# If switching cases/qty viceversa have it here
if target_level != source_level:
  if source_level == "cases":
    table_merged = table_merged.withColumn("CASES_ORIG",col("CASES_ORIG")*col("PLANG_MTRL_EA_PER_CASE_CNT"))
    table_merged = table_merged.withColumn("FINAL_PRED",col("FINAL_PRED")*col("PLANG_MTRL_EA_PER_CASE_CNT"))
    table_merged = table_merged.withColumn("PEP_PRED",col("PEP_PRED")*col("PLANG_MTRL_EA_PER_CASE_CNT"))
    historical = historical.withColumn("CASES_ORIG",col("CASES_ORIG")*col("PLANG_MTRL_EA_PER_CASE_CNT"))
  if source_level == "qty":
    table_merged = table_merged.withColumn("CASES_ORIG",col("CASES_ORIG")/col("PLANG_MTRL_EA_PER_CASE_CNT"))
    table_merged = table_merged.withColumn("FINAL_PRED",col("FINAL_PRED")/col("PLANG_MTRL_EA_PER_CASE_CNT"))
    table_merged = table_merged.withColumn("PEP_PRED",col("PEP_PRED")/col("PLANG_MTRL_EA_PER_CASE_CNT"))
    historical = historical.withColumn("CASES_ORIG",col("CASES_ORIG")/col("PLANG_MTRL_EA_PER_CASE_CNT"))

# COMMAND ----------

# Aggregate to the relevant levels
table_merged = table_merged.groupBy(acc_level).agg(sum(col("CASES_ORIG")).alias("CASES_ORIG"), sum(col("FINAL_PRED")).alias("FINAL_PRED"), sum(col("PEP_PRED")).alias("PEP_PRED"))
historical = historical.groupBy(hist_level).agg(sum(col("CASES_ORIG")).alias("CASES_ORIG"))

# Compute differences
table_merged = table_merged.withColumn("diff_ibp", col("FINAL_PRED") - col("CASES_ORIG"))
table_merged = table_merged.withColumn("abs_diff_ibp", abs(col("diff_ibp")))

table_merged = table_merged.withColumn("diff_pep", col("PEP_PRED") - col("CASES_ORIG"))
table_merged = table_merged.withColumn("abs_diff_pep", abs(col("diff_pep")))

# COMMAND ----------

# Re-calculate ABC/XYZ at the desired level

abc = historical.groupBy(hier_level).agg(sum("CASES_ORIG").alias("total")).orderBy(col("total").desc())
abc = get_cumsum_simple(abc, hier_level, 'total')
abc = abc.withColumn("ABC", when(abc["cum_pct"] < 0.8, lit("A")).when(abc["cum_pct"] < 0.95, lit("B")).otherwise(lit("C")))

xyz = calculate_cv_segmentation(historical, "CASES_ORIG", hier_level)
quant25 = xyz.approxQuantile(["CV"], [0.25], 0.05)[0][0]
quant50 = xyz.approxQuantile(["CV"], [0.50], 0.05)[0][0]
xyz = xyz.withColumn("XYZ", when(xyz["CV"] < quant25, lit("X")).when(xyz["CV"] > quant50, lit("Z")).otherwise(lit("Y")))

table_merged = table_merged.join(abc.select(hier_level + ["ABC"]), on = hier_level, how = "left").join(xyz.select(hier_level + ["XYZ"]), on = hier_level, how = "left")

# COMMAND ----------

# Calculate segmentation per accuracy

accs = table_merged.groupBy(hier_level).agg((1 - F.sum("abs_diff_pep") / F.sum("CASES_ORIG")).alias("acc_item_pep"), (1 - F.sum("abs_diff_ibp") / F.sum("CASES_ORIG")).alias("acc_item_ibp"))
accs = accs.fillna(0, subset = ["acc_item_pep", "acc_item_ibp"])
accs = accs.withColumn("better", accs["acc_item_ibp"] >= accs["acc_item_pep"])
accs = accs.withColumn("segment", when((accs["acc_item_ibp"] < 0.5) & (accs["acc_item_pep"] < 0.5), lit("LowPerf")).when(accs["acc_item_ibp"] >= (accs["acc_item_pep"] + 0.05), lit("HighConf")).when(accs["acc_item_ibp"] < (accs["acc_item_pep"] - 0.05), lit("Manual")).otherwise(lit("Similar")))
 

total = historical.groupBy(hier_level).agg(sum("CASES_ORIG").alias("total"))
accs = accs.join(total, on=hier_level, how="left")

table_merged = table_merged.join(accs, on = hier_level, how = "left")

# COMMAND ----------

# Calculate "best of both accuracy"

table_merged = table_merged.withColumn("diff_bob", when(col("better") == True, col("diff_ibp")).otherwise(col("diff_pep")))
table_merged = table_merged.withColumn("abs_diff_bob", abs(col("diff_bob")))

# COMMAND ----------

# Include unique ID

table_merged = table_merged.withColumn("ID", concat(*hier_level))

# COMMAND ----------

# Remove nulls

print("REMOVING: " + str(table_merged.filter(col(hier_level[0]).isNull()).count() / table_merged.count()) + " % of null rows")
table_merged = table_merged.dropna()

# COMMAND ----------

# DBTITLE 1,Extractions for PBI
  # Full table extraction
toExtract = table_merged.drop(*hier_level)#.filter((col("FCST_LAG") >= 12) & (col("FCST_LAG") < 17))

print(toExtract.count())
toExtract.display()


# COMMAND ----------

  # Master data extraction
toExtract_md = table_merged.select(["ID"] + hier_level).distinct()

print(toExtract_md.count())
toExtract_md.display()

# COMMAND ----------

#   # Historical + forecast extraction
# hist_fcst = historical.join(table_merged.select(acc_level + ["FINAL_PRED", "PEP_PRED", "ID"]), on = hist_level, how="outer")
# toExtract_h = hist_fcst.drop(*hier_level)#.filter((col("FCST_LAG") >= 2) & (col("FCST_LAG") < 3))

# print(toExtract_h.count())
# toExtract_h.display()

# COMMAND ----------

