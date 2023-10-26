# Databricks notebook source
# MAGIC %md #08 - Modeling
# MAGIC 
# MAGIC This script develops the forecast on the mrd (model ready dataset).
# MAGIC This runs through a train/test period, uses the results for model selection/feature importance, and then runs on forward-looking period.

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

## Check configurations exist for this script
required_configs = [DBA_MRD, DBA_MODELIDS, DBO_FORECAST, DBO_FORECAST_ROLLING, RUN_TYPE, TIME_VAR]
print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

test_load = load_delta_info(DBO_HYPERPARAMATER)
set_delta_retention(test_load, '90 days')
display(test_load.history())

# COMMAND ----------

# DBTITLE 1,Will remove these over-rides after testing/iterating
## Find way for us to set this more easily (alternative is to simply have as ./configs)
## Currently running for Pingo Doce - fine for testing purposes

# ## Versions for Pingo Doce dataset - using for development purposes
# ## PALAASH/ANAND - I kept this for now, to just ensure clean runs of the code
# DBA_MRD_version =  36
# DBA_MODELIDS_version =  15
# DBA_MRD_CLEAN_version =  17
# DBA_MRD_EXPLORATORY_data_version = 28
# DBO_OUTLIERS_version =  9
# DBO_SEGMENTS_version = 21
# DBO_HYPERPARAMATER_version = 15 # weekly with drivers

# ## TODO - delete from production version - no over-rides
# ## PALAASH/ANAND - weekly meta
if TIME_VAR == "Week_Of_Year":
  DBA_MRD_version =  45
  DBA_MODELIDS_version =  33
  DBA_MRD_CLEAN_version =  37
  DBA_MRD_EXPLORATORY_data_version = 32
  DBO_OUTLIERS_version =  16
  DBO_SEGMENTS_version = 28
  DBO_HYPERPARAMATER_version = 9   ## 9 is latest run with full weekly // 15 is Pingo Doce weekly with drivers

## PALAASH/ANAND - monthly meta
if TIME_VAR == "Month_Of_Year":
  DBA_MRD_version =  44
  DBA_MODELIDS_version = 34 
  DBA_MRD_CLEAN_version =  38 
  DBA_MRD_EXPLORATORY_data_version = 34
  DBO_OUTLIERS_version = 17
  DBO_SEGMENTS_version = 27
  DBO_HYPERPARAMATER_version = 14 # monthly with drivers

## TODO - delete from production version - no over-rides
## Override configs
TARGET_VAR_ORIG = "CASES"
TARGET_VAR = "CASES"
DBO_HYPERPARAMATER = 'dbfs:/mnt/adls/Tables/DBO_HYPERPARAMS'

## TODO - add to ./config file instead of over-ride
## TODO - reflect this in Feature Engineering, right?
if TIME_VAR == "Month_Of_Year":
  DYNAMIC_LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 18, 21, 24] 

elif TIME_VAR == "Week_Of_Year":
  DYNAMIC_LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 52, 104]

else: print('User Error - please check the TIME_VAR being used for modeling!!')

# COMMAND ----------

## Spark properties
spark.catalog.clearCache()
spark.conf.set("spark.worker.cleanup.enabled", "true")

# COMMAND ----------

## To end any existing run (which might be needed as we test in development)
## Keep in production version - helpful for preventing MLflow issues

mlflow.end_run()

# COMMAND ----------

# These flags define what will be be run in this pipeline
## Train/Test split vs. Rolling Review vs. Future Prediction 
RUN_ROLLING_XVAL = False
RUN_FUTURE_PREDICTION = True

## User to dictate whether models are re-trained following Train/Test split
## Keep as TRUE for now - issue with feature consistency when not retraining 
RETRAIN_FUTURE_MODELS = True

## User to dictate if we roll-up different lagged models to a single value
## Experiments indicated this led to a benefit in accuracy when tested OOS
AGGREGATE_LAGS_TO_PREDICT = True

## Defines the length of our OOS holdout for the Train-Test split part of this notebook
## Default should be to define this as 8-12 weeks from latest date
HOLDOUT_PERIOD_LEN = 8

## Alpha levels (ie, loss function inputs) to be potentially used for GBM Quantile models
## Including as new global features to avoid hard-coding values we might want to change
LOWER_ALPHA_LEVEL = 0.10
UPPER_ALPHA_LEVEL = 0.95
MID_ALPHA_LEVEL = 0.70
QUANTILE_ALPHA_2 = 0.65
QUANTILE_ALPHA_3 = 0.85

## Allows user to control 'volume' of retained lag values - cut down width and noise of FEATURES that are lagged
## Eg, if this value = 4 then Lag6 model will retain features lagged 7, 8, 9, 10 (ie, the 4 closest periods we can use)
LAGS_TO_KEEP = 4

## Output tables for saving these discrete pickle files
DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT = 'dbfs:/mnt/adls/Tables/DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT'
DBO_PICKLE_STAGE1_ROLLING_BACKTEST = 'dbfs:/mnt/adls/Tables/DBO_PICKLE_STAGE1_ROLLING_BACKTEST'
DBO_PICKLE_STAGE1_FUTURE_PERIOD = 'dbfs:/mnt/adls/Tables/DBO_PICKLE_STAGE1_FUTURE_PERIOD'

## Level in the hierarchy/hierarchies at which we want to set model selection
## A 'best' model is then selected using OOS accuracies
BEST_MODEL_SELECTION = ['HRCHY_LVL_3_NM', 'SUBBRND_SHRT_NM']     ## ['PLANG_CUST_GRP_VAL', 'BRND_NM'] 

## Dictates which tree-based model should be used for feature importance review following Train/Test split
FEAT_IMP_MODEL = 'lightGBM_model'

if TIME_VAR == 'Month_Of_Year':
  # defines which partition to use for backtesting/forcasting
  # adding this since our dataset has two partition in the mrd
  BACKTEST_PARTITION = 202009
  FORECAST_PARTITION = 202105 
  
  ## Note - these will only be used if RUN_ROLLING_XVAL = True 
  ROLLING_START_DATE = 202009 
  PERIODS = 2
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 18  
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [1,6, 12, 18] 
  
else:
  # defines which partition to use for backtesting/forcasting
  # adding this since our dataset has two partition in the mrd
  BACKTEST_PARTITION = 202114
  FORECAST_PARTITION = 202122
  
  ## Note - these will only be used if RUN_ROLLING_XVAL = True 
  ROLLING_START_DATE = 202114 
  PERIODS = 2
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 16  
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [1, 2, 4, 8, 12, 16]

# COMMAND ----------

# DBTITLE 1,Mlflow SetUp
## For experiment tracking
mlflow.set_experiment("/Shared/PEP_Master_Pipeline/Experiments/PEP Experiments")
mlflow.start_run()
experiment = mlflow.get_experiment_by_name("/Shared/PEP_Master_Pipeline/Experiments/PEP Experiments")

print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

# DBTITLE 1,Load Data
try:
  mrd_df = load_delta(DBA_MRD, DBA_MRD_version)
  delta_info = load_delta_info(DBA_MRD)
  display(delta_info.history())
except:
  dbutils.notebook.exit("DBA_MRD load failed, Exiting notebook")
  
## DELETE BELOW ONCE WE CONFIRM NEW VERSION
# try:
#   model_info = load_delta(DBA_MODELIDS, DBA_MODELIDS_version)
# except:
#   dbutils.notebook.exit("Model hierarchy information load failed, Exiting notebook")

try:
  model_info = load_delta(DBA_MODELIDS, DBA_MODELIDS_version)
  if (TIME_VAR in model_info.columns):
    model_info = model_info.drop(TIME_VAR).distinct()
except:
  dbutils.notebook.exit("Model hierarchy information load failed, Exiting notebook")

try:
  model_params = load_delta(DBO_HYPERPARAMATER, DBO_HYPERPARAMATER_version)
  model_params = model_params.withColumnRenamed('Model', 'train_func').toPandas()
  if 'one_model_dummy_seg' in model_params.columns:
    model_params.drop(columns=['one_model_dummy_seg'], inplace=True)
except:
  dbutils.notebook.exit("Hyperparameter load failed, Exiting notebook")

# COMMAND ----------

## Capturing pared-down MRD version for downstream joining/merging 
## MODEL ID will be the unique identifier for downstream merging

mrd_clean_df = mrd_df.select("MODEL_ID", TIME_VAR, TARGET_VAR)
mrd_clean_df = mrd_clean_df.withColumnRenamed(TARGET_VAR, TARGET_VAR + "_ORIG")
mrd_clean_df = mrd_clean_df.withColumn(TARGET_VAR + "_ORIG", exp(col(TARGET_VAR + "_ORIG"))-lit(1))

print(len(mrd_df.columns), mrd_df.count())
print(len(mrd_clean_df.columns), mrd_clean_df.count())  ## only 3 cols as specified above

# COMMAND ----------

display(mrd_df.groupby('FCST_START_DATE').agg(min(TIME_VAR), max(TIME_VAR)))

# COMMAND ----------

# DBTITLE 1,Selecting Backtesting Partition - as discussed re: driver predictions - to remove in full production version
mrd_df = mrd_df.filter(col('FCST_START_DATE')==BACKTEST_PARTITION) # use this for backtesting

# COMMAND ----------

## Adjusting to ensure all lags are caught downstream
## NOTE - intentionally commenting this so our holiday 'LAG' items are not dropped (leaving code for posterity)
## Since these effectively represent a holiday 'plan' that we know ahead of time

# lag_correction_list = [col_name for col_name in mrd_df.columns if '_LAG' in col_name or '_Lag' in col_name]
# for each_col in lag_correction_list:
#   mrd_df = mrd_df.withColumnRenamed(each_col, each_col.lower())

# COMMAND ----------

# DBTITLE 1,Set Train/Test Holdout Period Using Historical Dates
## Pulling dates as references for downstream
## Our full dataset now contains historicals and future shell details
historicals = load_delta(DBA_MRD_CLEAN, DBA_MRD_CLEAN_version)
max_historical_date = historicals.select(max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

full_data = load_delta(DBA_MRD, DBA_MRD_version)
max_future_date = full_data.select(max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

print('Max historical date = {}'.format(max_historical_date))
print('Max full dataset date = {}'.format(max_future_date))

## Pulling our calendar - to handle edge cases for when crossing into another year
calendar_df = load_delta(DBI_CALENDAR)
calendar_pd = calendar_df.toPandas()

## To use as a reference (for edge cases) in cell below
calendar_sorted_periods = sorted([i[TIME_VAR] for i in calendar_df.select(TIME_VAR).distinct().collect()])
cal_ref = calendar_sorted_periods.index(max_historical_date)

# COMMAND ----------

## Using the above reference points 
## Correction code in case user enters a negative value
if HOLDOUT_PERIOD_LEN < 0:
  HOLDOUT_PERIOD_LEN = (HOLDOUT_PERIOD_LEN * (-1))
  print('Converted holdout period to positive integer to preserve below calculations')
  
HOLDOUT_RANGE = (calendar_sorted_periods[cal_ref - HOLDOUT_PERIOD_LEN + 1], calendar_sorted_periods[cal_ref])
print('Holdout Range = {}'.format(HOLDOUT_RANGE))

# COMMAND ----------

## Print data versions
print('DBA_MRD_version = {}'.format(DBA_MRD_version))
print('DBA_MODELIDS_version = {}'.format(DBA_MODELIDS_version))
print('DBA_MRD_CLEAN_version = {}'.format(DBA_MRD_CLEAN_version))
print('DBA_MRD_EXPLORATORY_data_version = {}'.format(DBA_MRD_EXPLORATORY_data_version))
print('DBO_OUTLIERS_version = {}'.format(DBO_OUTLIERS_version))
print('DBO_SEGMENTS_version = {}'.format(DBO_SEGMENTS_version))

## Set up MLflow parameter monitoring
## TODO - anything else we want to ad here? (Corey added some stuff on 7/19/2021)
mlflow.log_param('Target Variable', TARGET_VAR)
mlflow.log_param('Time Variable', TIME_VAR)
mlflow.log_param('Holdout Range', HOLDOUT_RANGE)
mlflow.log_param('Forecast Aggregation Level', MODEL_ID_HIER)
mlflow.log_param('DBA_MRD_version', DBA_MRD_version) #Log data version as parameter
mlflow.log_param('DBA_MODELIDS_version', DBA_MODELIDS_version) #Log data version as parameter
mlflow.log_param('DBA_MRD_CLEAN_version', DBA_MRD_CLEAN_version) #Log data version as parameter
mlflow.log_param('DBA_MRD_EXPLORATORY_data_version', DBA_MRD_EXPLORATORY_data_version) #Log data version as parameter
mlflow.log_param('DBO_OUTLIERS_version', DBO_OUTLIERS_version) #Log data version as parameter
mlflow.log_param('DBO_SEGMENTS_version', DBO_SEGMENTS_version) #Log data version as parameter

# COMMAND ----------

## Exit for egregious errors
if len(intersect_two_lists([TARGET_VAR], mrd_df.columns)) == 0:
  dbutils.notebook.exit("Target variable not in data, Exiting notebook")
else: print('Target Variable ({}) in dataframe!'.format(TARGET_VAR))
  
if len(intersect_two_lists([TIME_VAR], mrd_df.columns)) == 0:
  dbutils.notebook.exit("Time variable not in data, Exiting notebook")
else: print('Time Variable ({}) in dataframe!'.format(TIME_VAR))

# COMMAND ----------

# DBTITLE 1,Load Modeling Hyperparameters
## ANAND/PALAASH TODO - move these to ./configs
## Is there a reason we are retaining these here for now?

quantile_params = {
    'boosting_type': 'gbdt',\
    'objective': 'quantile',\
    'metric': {'quantile'},\
    'alpha': 0.50,\
    'num_leaves': 200,\
    'learning_rate': 0.10,\
    'feature_fraction': 0.65,\
    'bagging_fraction': 0.85,\
    'bagging_freq': 5,\
    'verbose': -1
}

catboost_params = {
   'depth': 8,\
   'learning_rate': 0.10,\
   'iterations': 200,\
   'subsample': 0.80,\
   'grow_policy': 'Depthwise',\
   'l2_leaf_reg': 5.0,\
}

rf_params = {
   'criterion': 'mse',\
   'n_estimators': 100,\
   'max_depth': 8,\
   'min_samples_split': 200,\
   'max_features': 0.65,\
   'max_samples': 0.85,\
   'n_jobs': -1
}

xgb_params = {
    'eta': 0.05,\
    'max_depth': 8,\
    'min_child_weight': 10,\
    'subsample': 0.80,\
    'gamma': 0.05,\
}

lgbm_params = {
    'boosting_type': 'gbdt',\
    'objective': 'regression',\
    'metric': {'l2', 'l1'},\
    'max_depth': 6,\
    'num_leaves': 100,\
    'learning_rate': 0.25,\
    'min_gain_to_split': 0.02,\
    'feature_fraction': 0.65,\
    'bagging_fraction': 0.85,\
    'bagging_freq': 5,\
    'verbose': -1
}

# COMMAND ----------

# DBTITLE 1,Modeling Setup
## This is needed since our params file has stage 1 and 2 parameters
## Earlier iterations only had stage 1 params
if 'stage' in model_params.columns:
  stage1_params = model_params[model_params['stage'] == 1].drop(columns='stage')
  stage2_params = model_params[model_params['stage'] == 2].drop(columns='stage')
else:
  stage1_params = model_params
  stage2_params = model_params

# COMMAND ----------

# DBTITLE 1,Stage1 & Stage2 Parameter Logic
## Note: adding params for additional quantiles in stage 1
## This is to help address the under-forecasting (likely the result of data sparsity)
quantile = stage1_params[stage1_params['train_func'] =='gbm_quantile_model']['params'].iloc[0]
quantile2 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_2) for key, value in eval(quantile).items()})
quantile3 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_3) for key, value in eval(quantile).items()})

## Append these parameters to our Stage 1 dictionary
stage1_params = stage1_params.append({'train_func':'gbm_quantile_model2', "params":quantile2}, ignore_index=True)
stage1_params = stage1_params.append({'train_func':'gbm_quantile_model3', "params":quantile3}, ignore_index=True)

## Note: adding params for additional quantiles in stage 2
## This is to set upper and lower bounds for our Stage 2 models (for bounds)
quantile_mid = str({key:(value if key != 'alpha' else MID_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_lb  = str({key:(value if key != 'alpha' else LOWER_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_ub  = str({key:(value if key != 'alpha' else UPPER_ALPHA_LEVEL) for key, value in eval(quantile).items()})

stage2_params = stage2_params.append({'train_func':'gbm_quantile_mid', "params":quantile_mid}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':'gbm_quantile_lower', "params":quantile_lb}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':'gbm_quantile_upper', "params":quantile_ub}, ignore_index=True)

display(stage1_params)
display(stage2_params)

# COMMAND ----------

# DBTITLE 1,Modeling UDF Setup
train_function_mappings = {
    'gbm_quantile_model'  : train_lightGBM,
    'gbm_quantile_model2' : train_lightGBM,
    'gbm_quantile_model3' : train_lightGBM,
    'gbm_quantile_lower'  : train_lightGBM,
    'gbm_quantile_upper'  : train_lightGBM,
    'gbm_quantile_mid'    : train_lightGBM,
    'catboost_model'      : train_catboost,
    'rforest_model'       : train_random_forest,
    'lightGBM_model'      : train_lightGBM,
}

pred_function_mappings = {
    'gbm_quantile_model'  : predict_lightGBM,
    'gbm_quantile_model2' : predict_lightGBM,
    'gbm_quantile_model3' : predict_lightGBM,
    'gbm_quantile_lower'  : predict_lightGBM,
    'gbm_quantile_upper'  : predict_lightGBM,
    'gbm_quantile_mid'    : predict_lightGBM,
    'catboost_model'      : predict_catboost,
    'rforest_model'       : predict_random_forest,
    'lightGBM_model'      : predict_lightGBM,
}

stage1_models = {
    'gbm_quantile_model'  : quantile_params,
    'gbm_quantile_model2' : quantile_params,
    'gbm_quantile_model3' : quantile_params,
    'lightGBM_model'      : lgbm_params,
    'catboost_model'      : catboost_params,
    'rforest_model'       : rf_params,
}

stage2_models = {
  'lightGBM_model'        : lgbm_params,
  'gbm_quantile_model'    : quantile_params,
  'gbm_quantile_lower'    : quantile_params,
  'gbm_quantile_upper'    : quantile_params,
}

## Removed the below from Stage2 for now
#   'gbm_quantile_mid'      : quantile_params,
#   'gbm_quantile_lower'    : quantile_params,
#   'gbm_quantile_upper'    : quantile_params,

# COMMAND ----------

# DBTITLE 1,UDF Definition
## ANAND/PALAASH TODO - can we move this to source code? Or leave here?

## Using decorator to make training pandasUDF dynamic
def training_generator(train_schema, stage_cls, hyperparams_df=None):
  @pandas_udf(train_schema, PandasUDFType.GROUPED_MAP)
  def training_udf(data):
      return parallelize_core_models(df=data, model_info=stage_cls, hyperparams_df=hyperparams_df) 
  return training_udf

## Using decorator to make prediction pandasUDF dynamic
def prediction_generator(predict_schema, model_info_object, model_objects):
  @pandas_udf(predict_schema, PandasUDFType.GROUPED_MAP)
  def prediction_udf(data):
      return score_forecast(df=data, model_info=model_info_object, OBJECTS_DICT=model_objects)
  return prediction_udf

# COMMAND ----------

# DBTITLE 0,Static Forecast
## Setup information for modeling
## Must group using FCST_START_DATE (and train_func) for our 'rolling' efforts

model_info_dict = dict(
    target               = TARGET_VAR,                           #Target variable that we want to predict
    train_id_field       = ['TRAIN_IND'],                        #Binary indicator if data is train/test (data is stacked for rolling holdout)
    group_level          = ["train_func", "FCST_START_DATE"],    #Column in data that represents model we want to run (data is stacked for multiple algorithms)
    train_func_map       = train_function_mappings,              #Dictionary of training function mapped to model names
    pred_func_map        = pred_function_mappings,               #Dictionary of prediction function mapped to model names
    hyperparams          = stage1_models,                        #Hyperparameters for stage 1 models
    pickle_encoding      = 'latin-1',                            #Encoding translation for pickle objects
    id_fields            = ["MODEL_ID", TIME_VAR]                #ID fields that we want to retain in prediction data 
)

stage1_cls = StageModelingInfoDict(**model_info_dict)
stage1_cls.target

## Defining class for stage2
model_info_dict_stage2 = model_info_dict.copy()
model_info_dict_stage2['hyperparams'] = stage2_models
stage2_cls = StageModelingInfoDict(**model_info_dict_stage2)

# COMMAND ----------

# DBTITLE 1,Schema Definition
## Train Schema
train_schema = StructType([
StructField("train_func", StringType()),
StructField("FCST_START_DATE", IntegerType()),    ## only needed for rolling runs
StructField("model_pick", StringType())])

## Predict Schema
predict_schema = StructType([
StructField("MODEL_ID", StringType()),
StructField(TIME_VAR, IntegerType()),
StructField("train_func", StringType()),
StructField("FCST_START_DATE", IntegerType()),    ## only needed for rolling runs
StructField("pred", DoubleType())])

# COMMAND ----------

# DBTITLE 1,Check parameter tuning table compatibility
## Check the availability of params for algos in ensemble
training_algos = set(list(stage1_models.keys()) + list(stage2_models.keys())) 
tuned_algos = set(stage1_params['train_func'].unique())

print("Tuned parameters available for: ", training_algos.intersection(tuned_algos))
print("Tuning is missing for:", training_algos.difference(tuned_algos))

## Check the param level is same is the prediction level
param_required = [x for x in stage1_cls.group_level if x not in ['FCST_START_DATE']]
param_available = [x for x in stage1_params.columns if x not in ['params', 'one_model_dummy_seg']]

if sorted(param_required) == sorted(param_available):
  print("Parameter tuning results will be used")
else:
  print("Using default params as the tuning level and prediction level do not match. Make sure the variable 'DBO_HYPERPARAMATER_version' is set correctly")

# COMMAND ----------

# MAGIC %md # TRAIN/TEST MODELING

# COMMAND ----------

stage1_models_list = list(stage1_models.keys())
stage2_models_list = list(stage2_models.keys())
algos_list = sorted(list(set(stage1_models_list + stage2_models_list)))

## Stack dataset for the algorithms we wish to run
for this_algo in algos_list:
  algo_df = mrd_df
  algo_df = algo_df.withColumn("train_func", lit(this_algo))
  
  if this_algo == algos_list[0]:
    stacked_data = algo_df
  else:
    stacked_data = stacked_data.union(algo_df)


## New approach as provided by Ricard on 7/22/2021    
all_months = mrd_df.select(TIME_VAR).distinct().rdd.map(lambda r: r[0]).collect()
if RUN_ROLLING_XVAL == False:
  ROLLING_START_DATE = HOLDOUT_RANGE[0]
  rolling_months = [month for month in sorted(all_months) if month>=ROLLING_START_DATE]
  rolling_months = rolling_months[:1]
else:
  rolling_months = [month for month in sorted(all_months) if month>=ROLLING_START_DATE]
  rolling_months = rolling_months[:PERIODS]
  
## Append rolling holdout samples
for start_period in rolling_months:
  this_roll = stacked_data
  this_roll = this_roll.withColumn("TRAIN_IND", when(col(TIME_VAR) < start_period, 1).otherwise(0))
  this_roll = this_roll.withColumn("FCST_START_DATE", lit(start_period))
  
  if start_period == ROLLING_START_DATE:
    rolling_df = this_roll
  else:
    rolling_df = rolling_df.union(this_roll)

rolling_df.cache()


## REMOVE THIS VERSION ONCE WE TEST NEW VERSION ABOVE
# ## Dynamic setting for the below
# ## When not using rolling, we treat as a simple Train/Test split
# if RUN_ROLLING_XVAL == False:
#   ROLLING_START_DATE = HOLDOUT_RANGE[0]
#   ROLLING_END_DATE = ROLLING_START_DATE + 1
# else:
#   ROLLING_END_DATE = ROLLING_START_DATE + PERIODS
  
# ## Append rolling holdout samples
# for start_period in range(ROLLING_START_DATE, ROLLING_END_DATE):
#   this_roll = stacked_data
#   this_roll = this_roll.withColumn("TRAIN_IND", when(col(TIME_VAR) < start_period, 1).otherwise(0))
#   this_roll = this_roll.withColumn("FCST_START_DATE", lit(start_period))
  
#   if start_period == ROLLING_START_DATE:
#     rolling_df = this_roll
#   else:
#     rolling_df = rolling_df.union(this_roll)
# rolling_df.cache()

## Print-out and review
print('Stage1 Model List = {}'.format(stage1_models_list))
print('Stage2 Model List = {}'.format(stage2_models_list))
print('Set of Model List = {}'.format(algos_list))

## Note - can remove this in production - because .count() calls take a while
# print('Stacking check: ', algo_df.count() * len(algos_list) * (ROLLING_END_DATE - ROLLING_START_DATE) == rolling_df.count())

# COMMAND ----------

# Subset dataset to max oldout range
rolling_df = rolling_df.filter(col(TIME_VAR)<=HOLDOUT_RANGE[1])

# COMMAND ----------

## Training + algo-filtering for Stage1
train_stage_udf = training_generator(train_schema, stage1_cls)
stage1_df = rolling_df.filter(col('train_func').isin(stage1_models_list))  ## filter stage1 dataset

## Training + algo-filtering for Stage2
train_stage2_udf = training_generator(train_schema, stage2_cls)
stage2_df = rolling_df.filter(col('train_func').isin(stage2_models_list))  ## filter stage2 dataset

## Setup for our loop - don't want this to run in each loop iteration
interim_periods = stage1_df.filter(col('TRAIN_IND') == 0).select(TIME_VAR).distinct().collect()
holdout_periods = sorted([row[0] for row in interim_periods])
comparison_lag_set = set(DYNAMIC_LAG_PERIODS)

## Print and review
print('Holdout Periods = {}'.format(holdout_periods))
print('Full Lag Set = {}'.format(comparison_lag_set))

# COMMAND ----------

stage1_model_check = sorted([i.train_func for i in stage1_df.select('train_func').distinct().collect()])
stage2_model_check = sorted([i.train_func for i in stage2_df.select('train_func').distinct().collect()])

print('Stage1 models to run = {}'.format(stage1_model_check))
print('Stage2 models to run = {}'.format(stage2_model_check))

# COMMAND ----------

# DBTITLE 1,Train/Test or Rolling Loop Setup
## Set the number of holdout periods being run
MODELING_PERIODS_HOLDOUT = np.arange(1, len(holdout_periods) + 1, 1)

## New from Ricard on 7/22/2021
## Cross-checks lag models to run against time periods forward
## This allows us to minimize the models run (ie, ideally not 1 model for each period)
cross_over_list = []
for each_holdout_period in MODELING_PERIODS_HOLDOUT:
  temp_list = [x for x in DYNAMIC_LAG_MODELS if x >= each_holdout_period]
  if len(temp_list) > 0:
    min_lag_model = np.min(temp_list)
    cross_over_list.append(min_lag_model)

## Validate the above
cross_over_list = sorted(list(set(cross_over_list)))
print('Lag models to be run = {}'. format(cross_over_list))
print('Holdout periods ({} in total) = {}'.format(np.max(MODELING_PERIODS_HOLDOUT), holdout_periods))

# COMMAND ----------

## QC/validation - Stage1 time validation - can likely delete after review

stage1_train_df = stage1_df.filter(col('TRAIN_IND') == 1)
stage1_train_periods = sorted([i[TIME_VAR] for i in stage1_train_df.select(TIME_VAR).distinct().collect()])
print('Training dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage1_train_periods), np.max(stage1_train_periods)))

stage1_hold_df = stage1_df.filter(col('TRAIN_IND') == 0)
stage1_hold_periods = sorted([i[TIME_VAR] for i in stage1_hold_df.select(TIME_VAR).distinct().collect()])
print('Holdout dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage1_hold_periods), np.max(stage1_hold_periods)))

# COMMAND ----------

## QC/validation - Stage2 time validation - can likely delete after review

stage2_train_df = stage2_df.filter(col('TRAIN_IND') == 1)
stage2_train_periods = sorted([i[TIME_VAR] for i in stage2_train_df.select(TIME_VAR).distinct().collect()])
print('Training dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage2_train_periods), np.max(stage2_train_periods)))

stage2_hold_df = stage2_df.filter(col('TRAIN_IND') == 0)
stage2_hold_periods = sorted([i[TIME_VAR] for i in stage2_hold_df.select(TIME_VAR).distinct().collect()])
print('Holdout dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage2_hold_periods), np.max(stage2_hold_periods)))

# COMMAND ----------

## Review before modeling loop
check = load_delta_info(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
display(check.history())

# COMMAND ----------

# DBTITLE 1,Modeling Effort - Loop For Each Dynamic-Lag Model
## TODO - can comment out print statements - using them for QA/QC

## Propsoed add from Ricard on 7/22/2021
aggregated_rolling_output = None

for each_time_period in cross_over_list:  
  
  ## Set-up to drop our 'leakage' lagged columns in dynamic way
  dynamic_lagged_periods = set([lag for lag in DYNAMIC_LAG_PERIODS if lag > each_time_period])
  lags_to_drop = comparison_lag_set - dynamic_lagged_periods
  
  ## Set retention columns based on the above
  filter_text_list = ['_lag' + str(each_dropped_lag) for each_dropped_lag in lags_to_drop]
  cols_to_keep = [col_nm for col_nm in stage1_df.columns if col_nm.endswith(tuple(filter_text_list)) == False]
  print(each_time_period, dynamic_lagged_periods, lags_to_drop)
  
  ## Set-up to control the 'volume' of lagged periods used for each model (should likely retain ~3-6)
  min_lag_period = np.max(list(lags_to_drop)) + 1
  lag_period_list_to_keep = np.arange(min_lag_period, min_lag_period + LAGS_TO_KEEP, 1)

  ## Set drop columns based on the above
  keep_lag_text_list = ['_lag' + str(each_kept_lag) for each_kept_lag in lag_period_list_to_keep]
  cols_to_drop = [col_nm for col_nm in stage1_df.columns if '_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False]
#   cols_to_drop = [col_nm for col_nm in stage1_df.columns if ('CASES_lag' not in col_nm) and ('_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False)]
  print(min_lag_period, lag_period_list_to_keep)
  
  ##################################################################################
  ##################################################################################
  
  ## Subset to appropriate lag columns to prevent data leakage
  print(len(stage1_df.columns))
  stage1_df_temp = stage1_df[cols_to_keep]
  print(len(stage1_df_temp.columns))
  stage1_df_temp = stage1_df_temp.drop(*cols_to_drop)
  print(len(stage1_df_temp.columns))
  
  ## Stage1: train and cache for using groupyBy Pandas UDF
  rolling_stage1_pickles = stage1_df_temp.groupBy(stage1_cls.group_level).apply(train_stage_udf)
  rolling_stage1_pickles.cache()
  print("running for lagged model {} // len of pickles = {}".format(each_time_period, rolling_stage1_pickles.count()))

  ## Saving pickle for decomp - only need to do so for Stage1 models
  pickle_save = rolling_stage1_pickles.withColumn('lag_model', F.lit(int(each_time_period)))
  save_df_as_delta(pickle_save, DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT, enforce_schema=False)
  
  ## Create model dictionary
  temp_models = get_model_id(rolling_stage1_pickles, "score_lookup", stage1_cls.group_level)
  rolling_stage1_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
  rolling_stage1_objects = {key:{key:rolling_stage1_objects[key]} for key in rolling_stage1_objects.keys()}
 
  ## Prediction: Stage1
  stage1_df_pred = get_model_id(stage1_df_temp, "score_lookup", stage1_cls.group_level).cache()
  predict_models = prediction_generator(predict_schema, stage1_cls, rolling_stage1_objects)
  stage1_rolling_preds = stage1_df_pred.groupBy(stage1_cls.group_level).apply(predict_models)
  stage1_rolling_preds.cache()
  ## print("count of rolling stage1 prediction", stage1_rolling_preds.count()) ## removing this count for runtime purposes

  ## Shape predictions from long to wide
  pivot_columns = [x for x in list(stage1_rolling_preds.columns) if x not in ['train_func', 'pred']]
  stage1_rolling_preds = stage1_rolling_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage1'))) ## adding substring to identify Stage1 output
  stage1_rolling_output = stage1_rolling_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()
  
  ##################################################################################
  ##################################################################################
  
  ## Subset to appropriate lag columns to prevent data leakage
  ## Mirrors the above approach for Stage 1
  print(len(stage2_df.columns))
  stage2_df_temp = stage2_df[cols_to_keep]
  print(len(stage2_df_temp.columns))
  stage2_df_temp = stage2_df_temp.drop(*cols_to_drop)
  print(len(stage2_df_temp.columns))
  
  ## Join the outputs of stage1 (for "stacking" ensemble)
  ## Must remember to join on the 'aggregate' output (union from looping)
  stage2_df_temp = stage2_df_temp.join(stage1_rolling_output, on=pivot_columns).cache()   
  print(len(stage2_df_temp.columns))
  
  ## Stage2: train and cache for using groupyBy Pandas UDF
  rolling_stage2_pickles = stage2_df_temp.groupBy(stage2_cls.group_level).apply(train_stage2_udf)
  rolling_stage2_pickles.cache()  
  print("running for lagged model {} // len of pickles = {}".format(each_time_period, rolling_stage2_pickles.count()))

  ## Create model dictionary
  temp_models = get_model_id(rolling_stage2_pickles, "score_lookup", stage2_cls.group_level)
  rolling_stage2_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
  rolling_stage2_objects = {key:{key:rolling_stage2_objects[key]} for key in rolling_stage2_objects.keys()}
 
  ## Prediction: Stage2
  stage2_df_pred = get_model_id(stage2_df_temp, "score_lookup", stage2_cls.group_level).cache()
  predict_models = prediction_generator(predict_schema, stage2_cls, rolling_stage2_objects)
  stage2_rolling_preds = stage2_df_pred.groupBy(stage2_cls.group_level).apply(predict_models)
  stage2_rolling_preds.cache()
  ## print('count of rolling stage2 prediction', stage2_rolling_preds.count()) ## removing this count for runtime purposes
   
  ## Unlog prediction of both the stages
  stage1_rolling_preds = stage1_rolling_preds.withColumn("pred", exp(col("pred"))-lit(1))
  stage2_rolling_preds = stage2_rolling_preds.withColumn("pred", exp(col("pred"))-lit(1))   

  ## Shape predcitions from long to wide
  pivot_columns = [x for x in list(stage1_rolling_preds.columns) if x not in ['train_func', 'pred']]
  stage2_rolling_preds = stage2_rolling_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage2'))) ## adding substring to identify Stage2 output
  stage1_rolling_output = stage1_rolling_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()
  stage2_rolling_output = stage2_rolling_preds.groupby(pivot_columns).pivot("train_func").avg("pred")
  
  ##################################################################################
  ##################################################################################
  
  ## Joining with stage 1 output
  rolling_output = stage1_rolling_output.join(stage2_rolling_output, on=pivot_columns).cache()

  ## Create new column 
  rolling_output = rolling_output.withColumn('lag_period', F.lit(int(each_time_period)))
  ## print("count of stage2 output", rolling_output.count())  ## removing this count for runtime purposes
  
  ## Aggregating elements within the FOR loop
  ## We are capturing the holdout periods, so this should not create a huge dataframe   
  
  ## Ricard add on 7/22/2021
  if aggregated_rolling_output == None:
    aggregated_rolling_output = rolling_output
  else:
    aggregated_rolling_output = aggregated_rolling_output.union(rolling_output)

  aggregated_rolling_output.cache()
  aggregated_rolling_output.count()
  
  ## Final printout
  print("LOOP COMPLETE! - aggregated for lagged model {}".format(each_time_period))
  print('\n')

# COMMAND ----------

## BUGFIX: compute max_forcast for each prediction
## to assit correct computation of max_forcast when year changes.
## 'forcast_periods': this parameter is necessary to handle edge cases like last week in leap year.

# TODO move this function to library
def get_maxforcast(start_date, period, forcast_periods):
  forcast_periods = [x for x in sorted(forcast_periods) if x>=start_date]
  
  if period>len(forcast_periods):
    return forcast_periods[-1]
  else:
    return forcast_periods[period-1]

## defining udf
all_periods = mrd_df.select(TIME_VAR).distinct().rdd.map(lambda x: x[0]).collect()  
maxforcastUDF = udf(lambda z: get_maxforcast(z[0], z[1], all_periods), IntegerType())

# Making sure that predicted periods is not more than lag period
aggregated_rolling_output = aggregated_rolling_output.withColumn('max_forcast', maxforcastUDF(struct('FCST_START_DATE', 'lag_period')))\
                                                     .filter(col(TIME_VAR) <= col('max_forcast'))

# QC: max_forcast
display(aggregated_rolling_output.groupby('FCST_START_DATE', 'lag_period' )\
            .agg(max(TIME_VAR).alias('max_prediction'))\
            .orderBy(['FCST_START_DATE', 'lag_period']))

# COMMAND ----------

## SABAH/COREY TODO - do we need to predict/score for all our training data?
## SABAH/COREY TODO - can cut this to trim down time and output - check with Jordan re: PowerBI

# COMMAND ----------

# DBTITLE 1,Post-Processing BEFORE Model Selection & Feature Review
## Join to get hierarchy details and actuals for review
mrd_join_df = mrd_df.select("MODEL_ID", TIME_VAR, TARGET_VAR)
mrd_join_df = mrd_join_df.withColumnRenamed(TARGET_VAR, TARGET_VAR + "_ORIG")
mrd_join_df = mrd_join_df.withColumn(TARGET_VAR + "_ORIG", exp(col(TARGET_VAR + "_ORIG"))-lit(1))
mrd_join_df = mrd_join_df.join(model_info, on=['MODEL_ID'])  ## contains hierarchy details

# COMMAND ----------

## BUGFIX: assists calculation of forcast lag. this is needed to handle edge cases like when year changes.
## forcast lag is computed only for holdout set, train set forcast is returned as -1
def get_forcast_lag(time_var, fcst_start_date, forcast_period):
  if time_var<fcst_start_date: #this signifies train period
    return -1
  else:
    time_range = [x for x in forcast_period if x>=fcst_start_date and x<time_var] #get time periods between fcst_start and time_var
    return len(time_range)

## defining udf
forcastlagUDF = udf(lambda z: get_forcast_lag(z[0], z[1], all_periods), IntegerType())

# COMMAND ----------

aggregated_rolling_output_review = aggregated_rolling_output.join(mrd_join_df, on=[TIME_VAR, 'MODEL_ID'])

# ## Add IS/OOS designations here (easier for downstream)
aggregated_rolling_output_review = aggregated_rolling_output_review.withColumn('fcst_periods_fwd', forcastlagUDF(struct(TIME_VAR, 'FCST_START_DATE')))
aggregated_rolling_output_review = aggregated_rolling_output_review.withColumn('sample', when(aggregated_rolling_output_review['fcst_periods_fwd'] >= 0, 'OOS').otherwise('IS'))

display(aggregated_rolling_output_review)

# COMMAND ----------

model_cols = [col_name for col_name in aggregated_rolling_output_review.columns if 'stage1' in col_name or 'stage2' in col_name]

## Correct our output via low-prediction updates and rounding (ie, no fractional cases)
for each_model_col in model_cols:
  aggregated_rolling_output_review = aggregated_rolling_output_review.withColumn(each_model_col, F.when(F.col(each_model_col) < 1, 0).otherwise(F.col(each_model_col)))  ## correct low predictions down
  aggregated_rolling_output_review = aggregated_rolling_output_review.withColumn(each_model_col, F.ceil(F.col(each_model_col)))  ## round predictions so we do not have fractional prediction output
  
display(aggregated_rolling_output_review)

# COMMAND ----------

# DBTITLE 1,Save Train-Test Split Output
try:
  save_df_as_delta(aggregated_rolling_output_review, DBO_FORECAST_TRAIN_TEST_SPLIT, enforce_schema=False)
  train_test_delta_info = load_delta_info(DBO_FORECAST_TRAIN_TEST_SPLIT)
  set_delta_retention(train_test_delta_info, '90 days')
  display(train_test_delta_info.history())
  
  ## Accuracy delta table version
  data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBO_FORECAST_TRAIN_TEST_SPLIT +"`)").collect()
  data_version = data_version[0][0]
  mlflow.log_param('Simple Accuracy Delta Version', data_version) 

except:
  print("Train-Test delta run not written")

# COMMAND ----------

# DBTITLE 1,Pull the "Diagonals" - ie, the least-dropped lag model for each line item
## TODO - have team validate this approach - might not need this if we aggreagte up
## Commenting out for now, as we are not using the below
## Note - will use Lag1 model for all IS line items using this set-up

# aggregated_rolling_output_diagonals = aggregated_rolling_output_review.filter(col('lag_period') > col('fcst_periods_fwd'))\
#                                                                       .orderBy(col('MODEL_ID').desc(), col(TIME_VAR).desc(), col('lag_period').asc())\
#                                                                       .dropDuplicates(['MODEL_ID', 'FCST_START_DATE', TIME_VAR])
# aggregated_rolling_output_diagonals.cache()
# aggregated_rolling_output_diagonals.count()
# display(aggregated_rolling_output_diagonals)

# COMMAND ----------

# DBTITLE 1,Model Selection by Hierarchy Level
print('Selecting OOS dataframe for our results')
model_selection_df = aggregated_rolling_output_review.filter(col('sample') == 'OOS')
## model_selection_df = aggregated_rolling_output_diagonals.filter(col('sample') == 'OOS')  ## alternative ... not needed for now

model_selection_df.cache()
model_selection_df.count()

display(model_selection_df)

# COMMAND ----------

if TARGET_VAR in model_selection_df.columns:
  actuals_to_use = TARGET_VAR

elif TARGET_VAR + '_ORIG' in model_selection_df.columns:
  actuals_to_use = TARGET_VAR + '_ORIG'

else: print('Must add historical actuals!!')

## Output to dictate model selection
best_model_df = select_best_model(model_selection_df, model_cols, actuals_to_use, select_hier=BEST_MODEL_SELECTION)
best_model_df.cache()
best_model_df.count()
display(best_model_df)

# COMMAND ----------

# DBTITLE 1,Feature Importance for Feature Subsetting
check = load_delta_info(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
display(check.history())

# COMMAND ----------

## Select a lag model to dictate what we reference
FEAT_IMP_LAG = np.max(cross_over_list)  ## will pick 'largest' lag model

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
pickles_pd = pickles.toPandas()

pickles_pd = pickles_pd[(pickles_pd['lag_model'] == FEAT_IMP_LAG) & (pickles_pd['train_func'] == FEAT_IMP_MODEL)]
pickles_pd = pickles_pd[['train_func', 'model_pick']]
pickles_dict = dict(zip(pickles_pd.train_func, pickles_pd.model_pick))

# COMMAND ----------

importance_list = {}

for model in [FEAT_IMP_MODEL]:
  obj = pickles_dict[model].encode('latin-1')
  obj = pickle.loads(obj)
  feature_importance_dict = dict(zip(obj.feature_name(), obj.feature_importance(importance_type='gain').T))
  importance_list[model] = feature_importance_dict
  
importance_pd = pd.DataFrame.from_dict(importance_list)\
                            .reset_index()\
                            .sort_values(by=FEAT_IMP_MODEL, ascending=False)

importance_pd.columns = ['feature_name', FEAT_IMP_MODEL]
importance_pd.head(10)

# COMMAND ----------

features_to_drop = list(importance_pd[importance_pd[FEAT_IMP_MODEL] == 0]['feature_name'].values)
print('Interim ## of Features to Drop: {}'.format(len(features_to_drop)))

# COMMAND ----------

indexed_cols = [col_name for col_name in features_to_drop if '_index' in col_name]
cols_to_retain = [TARGET_VAR, TIME_VAR, 'TRAIN_IND', 'FCST_START_DATE'] + BEST_MODEL_SELECTION + indexed_cols   ## add anything else we need to ensure we retain

final_features_to_drop = list(set(features_to_drop) - set(cols_to_retain))
print('Final ## of Features to Drop: {}'.format(len(final_features_to_drop)))

# COMMAND ----------

# MAGIC %md # FUTURE MODELING & PREDICTION

# COMMAND ----------

# DBTITLE 1,Selecting Backtesting Partition - as discussed re: driver predictions - to remove in full production version
mrd_df = load_delta(DBA_MRD, DBA_MRD_version)
mrd_df = mrd_df.filter(col('FCST_START_DATE')==FORECAST_PARTITION) #use this for forcasting

# COMMAND ----------

# DBTITLE 1,Re-Train on Full Dataset (Using Above)
## Pulling dates as references for downstream
## Our full dataset now contains historicals and future shell details
historicals = load_delta(DBA_MRD_CLEAN, DBA_MRD_CLEAN_version)
max_historical_date = historicals.select(max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

full_data = load_delta(DBA_MRD, DBA_MRD_version)
max_future_date = full_data.select(max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

print('Max historical date = {}'.format(max_historical_date))
print('Max full dataset date = {}'.format(max_future_date))

## Pulling our calendar - to handle edge cases for when crossing into another year
calendar_df = load_delta(DBI_CALENDAR)
calendar_pd = calendar_df.toPandas()

## To use as a reference (for edge cases) in cell below
calendar_sorted_periods = sorted([i[TIME_VAR] for i in calendar_df.select(TIME_VAR).distinct().collect()])
cal_ref = calendar_sorted_periods.index(max_historical_date)

# COMMAND ----------

## Correction code in case user enters a negative value
if TIME_PERIODS_FORWARD < 0:
  TIME_PERIODS_FORWARD = (TIME_PERIODS_FORWARD * (-1))
  print('Converted forward-looking period to positive integer to preserve below calculations')

FORWARD_RANGE = (calendar_sorted_periods[cal_ref + 1], calendar_sorted_periods[cal_ref + TIME_PERIODS_FORWARD])
print('Holdout Periods Used = {}'.format(FORWARD_RANGE))

# COMMAND ----------

## ANAND/PALAASH TODO - confirm that I am using the MRD correctly
## Since this includes future shell from the outset, should just be able to re-load as below
mrd_df_future = mrd_df

## Dropping columns flagged in the above Feature Importance effort
mrd_df_future = mrd_df_future.drop(*final_features_to_drop)
len(mrd_df.columns), len(mrd_df_future.columns)  

# COMMAND ----------

## Exit for egregious errors
if len(intersect_two_lists([TARGET_VAR], mrd_df_future.columns)) == 0:
  dbutils.notebook.exit("Target variable not in data, Exiting notebook")
else: print('Target Variable ({}) in dataframe!'.format(TARGET_VAR))
  
if len(intersect_two_lists([TIME_VAR], mrd_df_future.columns)) == 0:
  dbutils.notebook.exit("Time variable not in data, Exiting notebook")
else: print('Time Variable ({}) in dataframe!'.format(TIME_VAR))

# COMMAND ----------

## ANAND/PALAASH TODO - REMOVE FROM PRODUCTION VERSION
## ANAND/PALAASH TODO - CREATED AS WORKAROUND - TESTING WITHOUT FUTURE SHELL DATA
# FORWARD_RANGE = (202120, 202125)

# COMMAND ----------

# DBTITLE 1,Reset Our Modeling Details (group_level)
## Setup information for modeling
## Must group using FCST_START_DATE (and train_func) for our 'rolling' efforts

model_info_dict = dict(
    target               = TARGET_VAR,                           #Target variable that we want to predict
    train_id_field       = ['TRAIN_IND'],                        #Binary indicator if data is train/test (data is stacked for rolling holdout)
    group_level          = ["train_func"],                       #Column in data that represents model we want to run (data is stacked for multiple algorithms)
    train_func_map       = train_function_mappings,              #Dictionary of training function mapped to model names
    pred_func_map        = pred_function_mappings,               #Dictionary of prediction function mapped to model names
    hyperparams          = stage1_models,                        #Hyperparameters for stage 1 models
    pickle_encoding      = 'latin-1',                            #Encoding translation for pickle objects
    id_fields            = ["MODEL_ID", TIME_VAR]                #ID fields that we want to retain in prediction data 
)

stage1_cls = StageModelingInfoDict(**model_info_dict)
stage1_cls.target

## Defining class for stage2
model_info_dict_stage2 = model_info_dict.copy()
model_info_dict_stage2['hyperparams'] = stage2_models
stage2_cls = StageModelingInfoDict(**model_info_dict_stage2)

# COMMAND ----------

# DBTITLE 1,Reset Our Schema (Future != Rolling)
## Train Schema
train_schema = StructType([
StructField("train_func", StringType()),
## StructField("FCST_START_DATE", IntegerType()),   ## only needed for rolling runs
StructField("model_pick", StringType())])

## Predict Schema
predict_schema = StructType([
StructField("MODEL_ID", StringType()),
StructField(TIME_VAR, IntegerType()),
StructField("train_func", StringType()),
## StructField("FCST_START_DATE", IntegerType()),    ## only needed for rolling runs
StructField("pred", DoubleType())])

# COMMAND ----------

## Subset train/test variables
## Ensure using 'less than' to retain intended length of holdout period
mrd_df_future = mrd_df_future.withColumn("TRAIN_IND", when(col(TIME_VAR) < FORWARD_RANGE[0], 1).otherwise(0))

train_rows = mrd_df_future.filter(col("TRAIN_IND") == 1).count()
holdout_rows = mrd_df_future.filter(col("TRAIN_IND") == 0).count()

print('Rows in full training dataframe = {}'.format(train_rows))
print('Rows in forward-looking dataframe = {}'.format(holdout_rows))
print('Dataset split validation:', train_rows + holdout_rows == mrd_df_future.count())

# COMMAND ----------

stage1_models_list = list(stage1_models.keys())
stage2_models_list = list(stage2_models.keys())
algos_list = sorted(list(set(stage1_models_list + stage2_models_list)))

## Stack dataset for the algorithms we wish to run
for this_algo in algos_list:
  algo_df = mrd_df_future
  algo_df = algo_df.withColumn("train_func", lit(this_algo))
  
  if this_algo == algos_list[0]:
    stacked_data = algo_df
  else:
    stacked_data = stacked_data.union(algo_df)
    
## Print-out and review
print('Stage1 Model List = {}'.format(stage1_models_list))
print('Stage2 Model List = {}'.format(stage2_models_list))
print('Set of Model List = {}'.format(algos_list))

## Note - can remove this in production - because .count() calls take a while
print('Algo-based stacking check: ', algo_df.count() * len(algos_list) == stacked_data.count())

# COMMAND ----------

stacked_data.count(), mrd_df_future.count()

# COMMAND ----------

## Training + Algo-Filtering for Stage1
train_stage_udf = training_generator(train_schema, stage1_cls)
stage1_df = stacked_data.filter(col('train_func').isin(stage1_models_list)) # filter stage1 dataset

## Training + Algo-Filtering for Stage2
train_stage2_udf = training_generator(train_schema, stage2_cls)
stage2_df = stacked_data.filter(col('train_func').isin(stage2_models_list))  # filter stage2 dataset

## Setup for our loop - don't want this to run in each loop iteration
interim_periods = stage1_df.filter(col('TRAIN_IND') == 0).select(TIME_VAR).distinct().collect()
holdout_periods = sorted([row[0] for row in interim_periods])
comparison_lag_set = set(DYNAMIC_LAG_PERIODS)

## Print and review
print('Holdout Periods = {}'.format(holdout_periods))
print('Full Lag Set = {}'.format(comparison_lag_set))

# COMMAND ----------

stage1_model_check = sorted([i.train_func for i in stage1_df.select('train_func').distinct().collect()])
stage2_model_check = sorted([i.train_func for i in stage2_df.select('train_func').distinct().collect()])

print('Stage1 models to run = {}'.format(stage1_model_check))
print('Stage2 models to run = {}'.format(stage2_model_check))

# COMMAND ----------

## Set the number of forward periods being run
## References the configuration TIME_PERIODS_FORWARD
MODELING_PERIODS_FWD = list(np.arange(1, TIME_PERIODS_FORWARD + 1, 1))

## Cross-checks lag models to run against time periods forward
## This allows us to minimize the models run (ie, ideally not 1 model for each period)
cross_over_list = []
for each_holdout_period in MODELING_PERIODS_FWD:
  temp_list = [x for x in DYNAMIC_LAG_MODELS if x >= each_holdout_period]
  if len(temp_list) > 0:
    min_lag_model = np.min(temp_list)
    cross_over_list.append(min_lag_model)

## Validate the above
cross_over_list = sorted(list(set(cross_over_list)))
print('Lag models to be run: {}'. format(cross_over_list))

# COMMAND ----------

## QC/validation - Stage1 time validation - can likely delete after review

stage1_train_df = stage1_df.filter(col('TRAIN_IND') == 1)
stage1_train_periods = sorted([i[TIME_VAR] for i in stage1_train_df.select(TIME_VAR).distinct().collect()])
print('Training dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage1_train_periods), np.max(stage1_train_periods)))

stage1_hold_df = stage1_df.filter(col('TRAIN_IND') == 0)
stage1_hold_periods = sorted([i[TIME_VAR] for i in stage1_hold_df.select(TIME_VAR).distinct().collect()])
print('Holdout dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage1_hold_periods), np.max(stage1_hold_periods)))

# COMMAND ----------

## QC/validation - Stage2 time validation - can likely delete after review
## Should look the same as the above 

stage2_train_df = stage2_df.filter(col('TRAIN_IND') == 1)
stage2_train_periods = sorted([i[TIME_VAR] for i in stage2_train_df.select(TIME_VAR).distinct().collect()])
print('Training dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage2_train_periods), np.max(stage2_train_periods)))

stage2_hold_df = stage2_df.filter(col('TRAIN_IND') == 0)
stage2_hold_periods = sorted([i[TIME_VAR] for i in stage2_hold_df.select(TIME_VAR).distinct().collect()])
print('Holdout dataset TIME_VAR min = {} and TIME_VAR max = {}'.format(np.min(stage2_hold_periods), np.max(stage2_hold_periods)))

# COMMAND ----------

# DBTITLE 1,RETRAINING: Modeling Effort - Loop For Each Dynamic-Lag Model
## Only run the below loop in re-training
## Note - keeping 'RETRAIN' set to TRUE for now until alternative is working

if RETRAIN_FUTURE_MODELS and RUN_FUTURE_PREDICTION:

  for each_time_period in cross_over_list:  

    ## Set-up to drop our 'leakage' lagged columns in dynamic way
    dynamic_lagged_periods = set([lag for lag in DYNAMIC_LAG_PERIODS if lag > each_time_period])
    lags_to_drop = comparison_lag_set - dynamic_lagged_periods

    ## Set retention columns based on the above
    filter_text_list = ['_lag' + str(each_dropped_lag) for each_dropped_lag in lags_to_drop]
    cols_to_keep = [col_nm for col_nm in stage1_df.columns if col_nm.endswith(tuple(filter_text_list)) == False]
    print(each_time_period, dynamic_lagged_periods, lags_to_drop)

    ## Set-up to control the 'volume' of lagged periods used for each model
    ## Newly added with the configuration above at top of this cell
    ## Only usually need ~4 lagged periods for a feature (not 10+)
    min_lag_period = np.max(list(lags_to_drop)) + 1
    lag_period_list_to_keep = np.arange(min_lag_period, min_lag_period + LAGS_TO_KEEP, 1)

    ## Set drop columns based on the above
    keep_lag_text_list = ['_lag' + str(each_kept_lag) for each_kept_lag in lag_period_list_to_keep]
    cols_to_drop = [col_nm for col_nm in stage1_df.columns if '_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False]
#     cols_to_drop = [col_nm for col_nm in stage1_df.columns if ('CASES_lag' not in col_nm) and ('_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False)]
    print(min_lag_period, lag_period_list_to_keep)

    ## Subset to appropriate lag columns to prevent data leakage
    print(len(stage1_df.columns))
    stage1_df_temp = stage1_df[cols_to_keep]
    print(len(stage1_df_temp.columns))
    stage1_df_temp = stage1_df_temp.drop(*cols_to_drop)
    print(len(stage1_df_temp.columns))

    ## Stage1: train and cache for using groupyBy Pandas UDF
    static_stage1_pickles = stage1_df_temp.groupBy(stage1_cls.group_level).apply(train_stage_udf)
    static_stage1_pickles.cache()
    print("running for period {} // len of pickles ie stage1 models = {}".format(each_time_period, static_stage1_pickles.count()))

    ## Saving pickle for decomp - only need to do so for Stage1 models
    pickle_save = static_stage1_pickles.withColumn('lag_model', F.lit(int(each_time_period)))
    save_df_as_delta(pickle_save, DBO_PICKLE_STAGE1_FUTURE_PERIOD, enforce_schema=False)

    ## Create model dictionary
    temp_models = get_model_id(static_stage1_pickles, "score_lookup", stage1_cls.group_level)
    static_stage1_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
    static_stage1_objects = {key:{key:static_stage1_objects[key]} for key in static_stage1_objects.keys()}

    ## Prediction: Stage1
    stage1_df_pred = get_model_id(stage1_df_temp, "score_lookup", stage1_cls.group_level).cache()
    predict_models = prediction_generator(predict_schema, stage1_cls, static_stage1_objects)
    stage1_static_preds = stage1_df_pred.groupBy(stage1_cls.group_level).apply(predict_models)
    stage1_static_preds.cache()
    ## print("count of static stage1 prediction", stage1_static_preds.count()) ## removing this count for runtime purposes

    ## Shape predictions from long to wide
    pivot_columns = [x for x in list(stage1_static_preds.columns) if x not in ['train_func', 'pred']]
    stage1_static_preds = stage1_static_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage1'))) ## adding substring to identify Stage1 output
    stage1_static_output = stage1_static_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()

    ## Subset to appropriate lag columns to prevent data leakage
    ## Mirrors the above approach for Stage 1
    print(len(stage2_df.columns))
    stage2_df_temp = stage2_df[cols_to_keep]
    print(len(stage2_df_temp.columns))
    stage2_df_temp = stage2_df_temp.drop(*cols_to_drop)
    print(len(stage2_df_temp.columns))

    ## Join the outputs of stage1 (for "stacking" ensemble)
    ## Must remember to join on the 'aggregate' output (union from looping)
    stage2_df_temp = stage2_df_temp.join(stage1_static_output, on=pivot_columns).cache()   
    print(len(stage2_df_temp.columns))

    ## Stage2: train and cache for using groupyBy Pandas UDF
    static_stage2_pickles = stage2_df_temp.groupBy(stage2_cls.group_level).apply(train_stage2_udf)
    static_stage2_pickles.cache()  
    print("running for period {} // len of pickles ie stage2 models = {}".format(each_time_period, static_stage2_pickles.count()))

    ## Create model dictionary
    temp_models = get_model_id(static_stage2_pickles, "score_lookup", stage2_cls.group_level)
    static_stage2_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
    static_stage2_objects = {key:{key:static_stage2_objects[key]} for key in static_stage2_objects.keys()}

    ## Prediction: Stage2
    stage2_df_pred = get_model_id(stage2_df_temp, "score_lookup", stage2_cls.group_level).cache()
    predict_models = prediction_generator(predict_schema, stage2_cls, static_stage2_objects)
    stage2_static_preds = stage2_df_pred.groupBy(stage2_cls.group_level).apply(predict_models)
    stage2_static_preds.cache()
    ## print('count of static stage2 prediction', stage2_static_preds.count()) ## removing this count for runtime purposes

    ## Unlog prediction of both the stages
    stage1_static_preds = stage1_static_preds.withColumn("pred", exp(col("pred"))-lit(1))
    stage2_static_preds = stage2_static_preds.withColumn("pred", exp(col("pred"))-lit(1))   

    ## Shape predcitions from long to wide
    pivot_columns = [x for x in list(stage1_static_preds.columns) if x not in ['train_func', 'pred']]
    stage2_static_preds = stage2_static_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage2'))) ## adding substring to identify Stage2 output
    stage1_static_output = stage1_static_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()
    stage2_static_output = stage2_static_preds.groupby(pivot_columns).pivot("train_func").avg("pred")

    ## Joining with stage 1 output
    static_output = stage1_static_output.join(stage2_static_output, on=pivot_columns).cache()

    ## Create new column 
    static_output = static_output.withColumn('lag_period', F.lit(int(each_time_period)))
    ## print("count of stage2 output", static_output.count())  ## removing this count for runtime purposes

    ## Aggregating elements within the FOR loop
    ## We are capturing the holdout periods, so this should not create a huge dataframe 
    if each_time_period == cross_over_list[0]:
      aggregated_future_output = static_output
    else:
      aggregated_future_output = aggregated_future_output.union(static_output)

    aggregated_future_output.cache()
    aggregated_future_output.count()

    ## Final printout
    print("LOOP COMPLETE! - aggregated for lagged model {}".format(each_time_period))
    print('\n')

else: print('Not retraining models - see below cell!!')

# COMMAND ----------

# Quick Fix: adding forcast_start_date since it was not included in the model output, and is need for future computation
aggregated_future_output = aggregated_future_output.withColumn('FCST_START_DATE', lit(FORECAST_PARTITION))

# COMMAND ----------

## defining udf
all_periods = mrd_df.select(TIME_VAR).distinct().rdd.map(lambda x: x[0]).collect()  
maxforcastUDF = udf(lambda z: get_maxforcast(z[0], z[1], all_periods), IntegerType())

# Making sure that predicted periods is not more than lag period
aggregated_future_output = aggregated_future_output.withColumn('max_forcast', maxforcastUDF(struct('FCST_START_DATE', 'lag_period')))\
                                                     .filter(col(TIME_VAR) <= col('max_forcast'))

# QC: max_forcast
display(aggregated_future_output.groupby('FCST_START_DATE', 'lag_period' )\
            .agg(max(TIME_VAR).alias('max_prediction'))\
            .orderBy(['FCST_START_DATE', 'lag_period']))

# COMMAND ----------

## SABAH/COREY TODO - do we need to predict/score for all our training data?
## SABAH/COREY TODO - can cut this to trim down time and output
## display(aggregated_future_output)

# COMMAND ----------

# BUG FIX: re-creating "mrd_join_df" for the forcast partition. This was earlier initialised off the backtesting partition 
## Join to get hierarchy details and actuals for review
mrd_join_df = mrd_df.select("MODEL_ID", TIME_VAR, TARGET_VAR)
mrd_join_df = mrd_join_df.withColumnRenamed(TARGET_VAR, TARGET_VAR + "_ORIG")
mrd_join_df = mrd_join_df.withColumn(TARGET_VAR + "_ORIG", exp(col(TARGET_VAR + "_ORIG"))-lit(1))
mrd_join_df = mrd_join_df.join(model_info, on=['MODEL_ID'])  ## contains hierarchy details

# COMMAND ----------

# DBTITLE 1,Future Prediction Post-Processing
if RUN_FUTURE_PREDICTION:
  
  #BUG FIX: making this left join instead of inner
  aggregated_future_output_review = aggregated_future_output.join(mrd_join_df, on=[TIME_VAR, 'MODEL_ID'], how='left')

  ## defining udf
  forcastlagUDF = udf(lambda z: get_forcast_lag(z[0], z[1], all_periods), IntegerType())
  aggregated_future_output_review = aggregated_future_output_review.withColumn('fcst_periods_fwd', forcastlagUDF(struct(TIME_VAR, 'FCST_START_DATE')))

  ## Add IS/OOS designations here (easier for downstream)
  aggregated_future_output_review = aggregated_future_output_review.withColumn('sample', when(aggregated_future_output_review['fcst_periods_fwd'] >= 0, 'OOS').otherwise('IS'))

  ## display(aggregated_future_output_review)

# COMMAND ----------

if RUN_FUTURE_PREDICTION:
  
  model_cols = [col_name for col_name in aggregated_future_output_review.columns if 'stage1' in col_name or 'stage2' in col_name]

  ## Correct our output via low-prediction updates or rounding (ie, no fractional cases)
  for each_model_col in model_cols:
    aggregated_future_output_review = aggregated_future_output_review.withColumn(each_model_col, F.when(F.col(each_model_col) < 1, 0).otherwise(F.col(each_model_col)))  ## correct low predictions down
    aggregated_future_output_review = aggregated_future_output_review.withColumn(each_model_col, F.ceil(F.col(each_model_col)))  ## round predictions so we do not have fractional prediction output

  display(aggregated_future_output_review)

# COMMAND ----------

# DBTITLE 1,Use "Best" Model Based on Hierarchy Selection Elements [long runtime]
if RUN_FUTURE_PREDICTION:
  
  cols_to_keep = ['best_model'] + BEST_MODEL_SELECTION
  best_model_merge = best_model_df.select(cols_to_keep)
  best_model_review = aggregated_future_output_review.join(best_model_merge, on=BEST_MODEL_SELECTION).cache()

# COMMAND ----------

if RUN_FUTURE_PREDICTION:
  
  algos_list_stage1 = [col_name + '_stage1' for col_name in algos_list]
  algos_list_stage2 = [col_name + '_stage2' for col_name in algos_list]
  full_algos_list = list(set(algos_list_stage1 + algos_list_stage2))

  ## TODO - find dynamic way to handle the below - manual workaround for now
  ## Currently manually handling to accommodate any model that I know we have ... 
  final_prediction_pd = best_model_review
  final_prediction_pd = final_prediction_pd.withColumn('final_prediction_value',\
                                           when(final_prediction_pd['best_model'] == 'lightGBM_model_stage2', final_prediction_pd['lightGBM_model_stage2']).otherwise(\
                                           when(final_prediction_pd['best_model'] == 'gbm_quantile_model_stage2', final_prediction_pd['gbm_quantile_model_stage2']).otherwise(\
                                           when(final_prediction_pd['best_model'] == 'rforest_model_stage1', final_prediction_pd['rforest_model_stage1']).otherwise(\
                                           when(final_prediction_pd['best_model'] == 'catboost_model_stage1', final_prediction_pd['catboost_model_stage1']).otherwise(\
                                           when(final_prediction_pd['best_model'] == 'gbm_quantile_model2_stage1', final_prediction_pd['gbm_quantile_model2_stage1']).otherwise(\
                                           when(final_prediction_pd['best_model'] == 'gbm_quantile_model3_stage1', final_prediction_pd['gbm_quantile_model3_stage1']).otherwise(\
                                           final_prediction_pd['lightGBM_model_stage2'])))))))

# COMMAND ----------

# DBTITLE 1,Aggregate by Lagged Models (via User Config Flag)
if RUN_FUTURE_PREDICTION:
  
  if TARGET_VAR in model_selection_df.columns:
    actuals_to_use = TARGET_VAR

  elif TARGET_VAR + '_ORIG' in model_selection_df.columns:
    actuals_to_use = TARGET_VAR + '_ORIG'

  else: print('Must add historical actuals!!')

  agg_dict = {'final_prediction_value':'mean'}
  grouping_cols = ['MODEL_ID', TIME_VAR, actuals_to_use] + BEST_MODEL_SELECTION

  ## This aggregates predicted value for a MODEL ID and TIME across different lag models
  ## Review showed that this led to accuracy improvements 
  if AGGREGATE_LAGS_TO_PREDICT:
    agg_final_prediction_pd = final_prediction_pd.groupBy(grouping_cols).agg(agg_dict)
    agg_final_prediction_pd = agg_final_prediction_pd.withColumnRenamed('avg(final_prediction_value)', 'final_agg_prediction_value')
    final_prediction_pd = final_prediction_pd.join(agg_final_prediction_pd, on=grouping_cols)

# COMMAND ----------

# DBTITLE 1,Output Updates for PowerBI
if RUN_FUTURE_PREDICTION:
  
  if 'week' in TIME_VAR.lower():
    final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('Weekly'))

  elif 'month' in TIME_VAR.lower():
    final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('Monthly'))

  else: final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('User-Error'))

# COMMAND ----------

if RUN_FUTURE_PREDICTION:
  
  def translate(mapping):
      def translate_(col):
          return mapping.get(col)
      return udf(translate_, StringType())

  from calendar import monthrange
  calendar_df_ref = calendar_df.withColumn('Month', substring('Month_Of_Year', 5, 2))
  calendar_df_ref = calendar_df_ref.withColumn('Month', calendar_df_ref['Month'].cast(IntegerType()))

  ## Reference lists pulled from our existing PySpark df
  years_list = calendar_df_ref.select("Year").rdd.flatMap(lambda x: x).collect()
  months_list = calendar_df_ref.select("Month").rdd.flatMap(lambda x: x).collect()
  week_year_list = calendar_df_ref.select("Week_Of_Year").rdd.flatMap(lambda x: x).collect()
  month_year_list = calendar_df_ref.select("Month_Of_Year").rdd.flatMap(lambda x: x).collect()

  ## Pulling days in month based on month and year combination
  days_in_month_list = []
  if len(years_list) == len(months_list):
    for each_idx in np.arange(0, len(years_list) - 1, 1):
      temp_days = monthrange(years_list[each_idx], months_list[each_idx])[1]
      days_in_month_list.append(temp_days)
  else: print('User to re-run!!')

  ## Setting up dataframe to capture days in week/month
  calendar_df_ref = calendar_df_ref.withColumn('Days_In_Week', F.lit(7))
  calendar_month_dict = dict(zip(month_year_list, days_in_month_list))
  calendar_df_ref = calendar_df_ref.withColumn("Days_In_Month", translate(calendar_month_dict)("Month_Of_Year"))

  ## Setting duration == 7 for week
  if 'week' in TIME_VAR.lower():
    final_prediction_pd = final_prediction_pd.withColumn('Time_Duration', F.lit(7))
    final_prediction_pd = final_prediction_pd.join(calendar_df_ref.select(TIME_VAR, 'Week_start_date').distinct(), on =[TIME_VAR], how='left')

  ## Using dictionary mapping for duration for month
  elif 'month' in TIME_VAR.lower():
    calendar_df_join = calendar_df_ref.select(TIME_VAR, 'Days_In_Month')
    final_prediction_pd = final_prediction_pd.withColumn("Days_In_Month", translate(calendar_month_dict)("Month_Of_Year"))
    final_prediction_pd = final_prediction_pd.withColumnRenamed('Days_In_Month', 'Time_Duration')
    final_prediction_pd = final_prediction_pd.join(calendar_df_ref.select(TIME_VAR, 'Month_start_date').distinct(), on =[TIME_VAR], how='left')

# COMMAND ----------

# DBTITLE 1,Save Future Prediction Outputs
if RUN_FUTURE_PREDICTION:
  
  try:
    save_df_as_delta(final_prediction_pd, DBO_FORECAST_FUTURE_PERIOD, enforce_schema=False)
    future_period_delta_info = load_delta_info(DBO_FORECAST_FUTURE_PERIOD)
    set_delta_retention(future_period_delta_info, '90 days')
    display(future_period_delta_info.history())

  except:
    print("Future delta run not written")

# COMMAND ----------

mlflow.end_run()
print('end')

# COMMAND ----------

# DBTITLE 1,Graveyard Code - TO DELETE THE BELOW


# COMMAND ----------

# ## added 0708: to assit correct computation of max_forcast when year changes.
# ## 'forcast_periods': this parameter is necessary to handle edge cases like last week in leap year.

# def get_maxforcast(start_date, period, forcast_periods):
#   forcast_periods = sorted(forcast_periods)
#   forcast_periods = [x for x in forcast_periods if x>=start_date]
  
#   if period>len(forcast_periods):
#     return forcast_periods[-1]
#   else:
#     return forcast_periods[period-1]

# FORCAST_PERIODS = mrd_df.select(TIME_VAR).distinct()\
#                         .filter(col(TIME_VAR)>=ROLLING_START_DATE).rdd.map(lambda x: x[0]).collect()  
  
# ## defining udf
# maxforcastUDF = udf(lambda z: get_maxforcast(z[0], z[1], FORCAST_PERIODS), IntegerType())

# Making sure that predicted periods is not more than lag period
# aggregated_rolling_output = aggregated_rolling_output.withColumn('max_forcast', maxforcastUDF(struct('FCST_START_DATE', 'lag_period')))\
#                                                      .filter(col(TIME_VAR) <= col('max_forcast'))

# # print("count of aggregated rolling output", aggregated_rolling_output.count())

# # # QC
# # display(aggregated_rolling_output.groupby('lag_period', 'FCST_START_DATE')\
# #             .agg(max(TIME_VAR).alias('max_prediction'))\
# #             .orderBy(['FCST_START_DATE', 'lag_period'])

# COMMAND ----------

# ## added 0708: assists calculation of forcast lag. this is needed to handle edge cases like when year changes.
# ## forcast lag is computed only for holdout set, train set forcast is returned as -1
# def get_forcast_lag(time_var, fcst_start_date, forcast_period):
#   if time_var<fcst_start_date: #this signifies train period
#     return -1
#   else:
#     time_range = [x for x in forcast_period if x>=fcst_start_date and x<time_var] #get time periods between fcst_start and time_var
#     return len(time_range)

# ## defining udf
# forcastlagUDF = udf(lambda z: get_forcast_lag(z[0], z[1], FORCAST_PERIODS), IntegerType())

# aggregated_rolling_output = aggregated_rolling_output.withColumn(FCST_LAG_VAR[0], forcastlagUDF(struct(TIME_VAR, 'FCST_START_DATE')))

# COMMAND ----------



# ## Dedup model_info and join
# model_info = model_info.drop("Month_Of_Year", "STAT_CLUSTER", "ABC", "XYZ")
# model_info = model_info.dropDuplicates()

# # print("count1 of aggregated rolling output", aggregated_rolling_output.count())
# aggregated_rolling_output = aggregated_rolling_output.join(model_info, on="MODEL_ID", how="left")
# # print("count2 of aggregated rolling output", aggregated_rolling_output.count())
# aggregated_rolling_output = aggregated_rolling_output.join(mrd_clean_df.select(["MODEL_ID", TIME_VAR, TARGET_VAR_ORIG + "_orig"]), on=["MODEL_ID", TIME_VAR], how="left").cache()
# # print("count3 of aggregated rolling output", aggregated_rolling_output.count())

# ## Develop forecast lag variable
# aggregated_rolling_output = aggregated_rolling_output.withColumn(FCST_LAG_VAR[0], forcastlagUDF(struct(TIME_VAR, 'FCST_START_DATE')))
# # print(aggregated_rolling_output.count())

# COMMAND ----------

# DBTITLE 1,Experiment - to implement or delete - WIP
# ## testing averaging up our disparate lag models
# model_cols = [col_name for col_name in aggregated_rolling_output_review.columns if '_model' in col_name]
# agg_dict = {model_col:'mean' for model_col in model_cols}

# cnp_2 = aggregated_rolling_output_review.filter(col('sample') == 'OOS')
# cnp_2 = cnp_2.groupBy(['MODEL_ID', TIME_VAR, TARGET_VAR + '_ORIG'] + BEST_MODEL_SELECTION).agg(agg_dict)

# model_cols = [col_name for col_name in cnp_2.columns if 'avg' in col_name]
# cnp_best_model_test_2 = select_best_model(cnp_2, model_cols, TARGET_VAR + '_ORIG', select_hier=BEST_MODEL_SELECTION)
# display(cnp_best_model_test_2)
# ## Note - this DOES appear to help - to explore further

# COMMAND ----------

# ## testing averages across modeling columns
# cnp = aggregated_rolling_output_review.filter(col('sample') == 'OOS')
# cnp = cnp.withColumn('avg_model_col1', (cnp['catboost_model_stage1'] + cnp['rforest_model_stage1'] + cnp['gbm_quantile_model_stage2'] + cnp['lightGBM_model_stage2']) / 4 )
# cnp = cnp.withColumn('avg_model_col2', (cnp['rforest_model_stage1'] + cnp['gbm_quantile_model_stage2'] + cnp['lightGBM_model_stage2']) / 3 )
# cnp = cnp.withColumn('avg_model_col3', (cnp['gbm_quantile_model_stage2'] + cnp['lightGBM_model_stage2']) / 2 )
# cnp = cnp.withColumn('avg_model_col4', (cnp['gbm_quantile_model2_stage1'] + cnp['rforest_model_stage1'] + cnp['gbm_quantile_model_stage2'] + cnp['lightGBM_model_stage2']) / 4 )

# model_cols = [col_name for col_name in cnp.columns if '_model' in col_name]
# cnp_best_model_test = select_best_model(cnp, model_cols, TARGET_VAR + '_ORIG', select_hier=BEST_MODEL_SELECTION)
# display(cnp_best_model_test)
# ## Note - does not appear to be very much help

# COMMAND ----------

# DBTITLE 1,WIP - WITHOUT RETRAINING: Modeling Effort - Loop For Each Dynamic-Lag Model - issue with model consistency
# RETRAIN_FUTURE_MODELS = False

# ## Only run the below loop when NOT re-training the models for future
# if not RETRAIN_FUTURE_MODELS:

#   for each_time_period in cross_over_list:  

#     ## Set-up to drop our 'leakage' lagged columns in dynamic way
#     dynamic_lagged_periods = set([lag for lag in DYNAMIC_LAG_PERIODS if lag > each_time_period])
#     lags_to_drop = comparison_lag_set - dynamic_lagged_periods

#     ## Set retention columns based on the above
#     filter_text_list = ['_lag' + str(each_dropped_lag) for each_dropped_lag in lags_to_drop]
#     cols_to_keep = [col_nm for col_nm in stage1_df.columns if col_nm.endswith(tuple(filter_text_list)) == False]
#     print(each_time_period, dynamic_lagged_periods, lags_to_drop)

#     ## Set-up to control the 'volume' of lagged periods used for each model
#     ## Newly added with the configuration above at top of this cell
#     ## Only usually need ~4 lagged periods for a feature (not 10+)
#     min_lag_period = np.max(list(lags_to_drop)) + 1
#     lag_period_list_to_keep = np.arange(min_lag_period, min_lag_period + LAGS_TO_KEEP, 1)

#     ## Set drop columns based on the above
#     keep_lag_text_list = ['_lag' + str(each_kept_lag) for each_kept_lag in lag_period_list_to_keep]
#     cols_to_drop = [col_nm for col_nm in stage1_df.columns if '_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False]
#     print(min_lag_period, lag_period_list_to_keep)

#     ## Subset to appropriate lag columns to prevent data leakage
#     print(len(stage1_df.columns))
#     stage1_df_temp = stage1_df[cols_to_keep]
#     print(len(stage1_df_temp.columns))
#     stage1_df_temp = stage1_df_temp.drop(*cols_to_drop)
#     print(len(stage1_df_temp.columns))

#     ## Prediction: Stage1
#     stage1_df_pred = get_model_id(stage1_df_temp, "score_lookup", stage1_cls.group_level).cache()
#     predict_models = prediction_generator(predict_schema, stage1_cls, rolling_stage1_objects)      ## from our earlier run
#     stage1_future_preds = stage1_df_pred.groupBy(stage1_cls.group_level).apply(predict_models)
#     stage1_future_preds.cache()
#     ## print("count of future stage1 prediction", stage1_future_preds.count()) ## removing this count for runtime purposes

#     ## Shape predictions from long to wide
#     pivot_columns = [x for x in list(stage1_future_preds.columns) if x not in ['train_func', 'pred']]
#     stage1_future_preds = stage1_future_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage1'))) ## adding substring to identify Stage1 output
#     stage1_future_output = stage1_future_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()

#     ## Subset to appropriate lag columns to prevent data leakage
#     ## Mirrors the above approach for Stage 1
#     print(len(stage2_df.columns))
#     stage2_df_temp = stage2_df[cols_to_keep]
#     print(len(stage2_df_temp.columns))
#     stage2_df_temp = stage2_df_temp.drop(*cols_to_drop)
#     print(len(stage2_df_temp.columns))

#     ## Join the outputs of stage1 (for "stacking" ensemble)
#     ## Must remember to join on the 'aggregate' output (union from looping)
#     stage2_df_temp = stage2_df_temp.join(stage1_future_output, on=pivot_columns).cache()   
#     print(len(stage2_df_temp.columns))

#     ## Prediction: Stage2
#     stage2_df_pred = get_model_id(stage2_df_temp, "score_lookup", stage2_cls.group_level).cache()
#     predict_models = prediction_generator(predict_schema, stage2_cls, rolling_stage2_objects)      ## from our earlier run
#     stage2_future_preds = stage2_df_pred.groupBy(stage2_cls.group_level).apply(predict_models)
#     stage2_future_preds.cache()
#     ## print('count of future stage2 prediction', stage2_future_preds.count()) ## removing this count for runtime purposes

#     ## Unlog prediction of both the stages
#     stage1_future_preds = stage1_future_preds.withColumn("pred", exp(col("pred"))-lit(1))
#     stage2_future_preds = stage2_future_preds.withColumn("pred", exp(col("pred"))-lit(1))   

#     ## Shape predcitions from long to wide
#     pivot_columns = [x for x in list(stage1_future_preds.columns) if x not in ['train_func', 'pred']]
#     stage2_future_preds = stage2_future_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage2'))) ## adding substring to identify Stage2 output
#     stage1_future_output = stage1_future_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()
#     stage2_future_output = stage2_future_preds.groupby(pivot_columns).pivot("train_func").avg("pred")

#     ## Joining with stage 1 output
#     future_output = stage1_future_output.join(stage2_future_output, on=pivot_columns).cache()

#     ## Create new column 
#     future_output = future_output.withColumn('lag_period', F.lit(int(each_time_period)))
#     ## print("count of stage2 output", future_output.count())  ## removing this count for runtime purposes

#     ## Aggregating elements within the FOR loop
#     ## We are capturing the holdout periods, so this should not create a huge dataframe 
#     if each_time_period == MODELING_PERIODS_FWD[0]:
#       aggregated_future_output = future_output
#     else:
#       aggregated_future_output = aggregated_future_output.union(future_output)

#     aggregated_future_output.cache()
#     aggregated_future_output.count()

#     ## Final printout
#     print("LOOP COMPLETE! - aggregated for lagged model {}".format(each_time_period))
#     print('\n')

# else: print('Retrained models - see above cell!!')

# COMMAND ----------



# COMMAND ----------

