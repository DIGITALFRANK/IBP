# Databricks notebook source
# MAGIC %md ## All Neccessary Cmd

# COMMAND ----------

# MAGIC %run /Shared/PEP_Master_Pipeline/src/libraries

# COMMAND ----------

# MAGIC %run /Shared/PEP_Master_Pipeline/src/load_src_parallel

# COMMAND ----------

# MAGIC %run /Shared/PEP_Master_Pipeline/src/load_src

# COMMAND ----------

# MAGIC %run /Shared/PEP_Master_Pipeline/src/config

# COMMAND ----------

# MAGIC %md ## Using monthly ahead as an example 

# COMMAND ----------

# defining the parameters

# training versions
DBA_MRD_version =  44
DBA_MODELIDS_version = 34 
DBA_MRD_CLEAN_version =  38 
DBA_MRD_EXPLORATORY_data_version = 34
DBO_OUTLIERS_version = 17
DBO_SEGMENTS_version = 27
DBO_HYPERPARAMATER_version = 14 # monthly with drivers

## training configs
TARGET_VAR_ORIG = "CASES"
TARGET_VAR = "CASES"
TIME_VAR = "Month_Of_Year"
DBO_HYPERPARAMATER = 'dbfs:/mnt/adls/Tables/DBO_HYPERPARAMS'
HOLDOUT_RANGE = (202010, 202104)
LAGS_TO_KEEP = 4

aggregated_rolling_output = None
lag_period = 1
lags_to_drop = set([1])

# model configs
LOWER_ALPHA_LEVEL = 0.10
UPPER_ALPHA_LEVEL = 0.95
MID_ALPHA_LEVEL = 0.70
QUANTILE_ALPHA_2 = 0.65
QUANTILE_ALPHA_3 = 0.85

# COMMAND ----------

# MAGIC %md ## Getting Data

# COMMAND ----------

# load data
try:
  mrd_df = load_delta(DBA_MRD, DBA_MRD_version)
  delta_info = load_delta_info(DBA_MRD)
except:
  dbutils.notebook.exit("DBA_MRD load failed, Exiting notebook")
  
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

mrd_clean_df = mrd_df.select("MODEL_ID", TIME_VAR, TARGET_VAR)
mrd_clean_df = mrd_clean_df.withColumnRenamed(TARGET_VAR, TARGET_VAR + "_ORIG")
mrd_clean_df = mrd_clean_df.withColumn(TARGET_VAR + "_ORIG", exp(col(TARGET_VAR + "_ORIG"))-lit(1))

# COMMAND ----------

# MAGIC %md ## Modeling set up

# COMMAND ----------

# load hyperparameters
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

# setup stage1 and stage2 params
if 'stage' in model_params.columns:
  stage1_params = model_params[model_params['stage'] == 1].drop(columns='stage')
  stage2_params = model_params[model_params['stage'] == 2].drop(columns='stage')
else:
  stage1_params = model_params
  stage2_params = model_params
  
quantile = stage1_params[stage1_params['train_func'] =='gbm_quantile_model']['params'].iloc[0]
quantile2 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_2) for key, value in eval(quantile).items()})
quantile3 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_3) for key, value in eval(quantile).items()})
stage1_params = stage1_params.append({'train_func':'gbm_quantile_model2', "params":quantile2}, ignore_index=True)
stage1_params = stage1_params.append({'train_func':'gbm_quantile_model3', "params":quantile3}, ignore_index=True)

quantile_mid = str({key:(value if key != 'alpha' else MID_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_lb  = str({key:(value if key != 'alpha' else LOWER_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_ub  = str({key:(value if key != 'alpha' else UPPER_ALPHA_LEVEL) for key, value in eval(quantile).items()})
stage2_params = stage2_params.append({'train_func':'gbm_quantile_mid', "params":quantile_mid}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':'gbm_quantile_lower', "params":quantile_lb}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':'gbm_quantile_upper', "params":quantile_ub}, ignore_index=True)

# model mapping
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
}

## Defining class for stage1
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

## Defining class for stage2
model_info_dict_stage2 = model_info_dict.copy()
model_info_dict_stage2['hyperparams'] = stage2_models
stage2_cls = StageModelingInfoDict(**model_info_dict_stage2)

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

## Dynamic setting for the below
## When not using rolling, we treat as a simple Train/Test split
all_months = mrd_df.select(TIME_VAR).distinct().rdd.map(lambda r: r[0]).collect()
ROLLING_START_DATE = HOLDOUT_RANGE[0]
rolling_months = [month for month in sorted(all_months) if month>=ROLLING_START_DATE]
rolling_months = rolling_months[:1]

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

# COMMAND ----------

# MAGIC %md ## Pandas UDF

# COMMAND ----------

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

## Training + algo-filtering for Stage1
train_stage_udf = training_generator(train_schema, stage1_cls)
stage1_df = rolling_df.filter(col('train_func').isin(stage1_models_list))  ## filter stage1 dataset

## Training + algo-filtering for Stage2
train_stage2_udf = training_generator(train_schema, stage2_cls)
stage2_df = rolling_df.filter(col('train_func').isin(stage2_models_list))  ## filter stage2 dataset

# COMMAND ----------

# MAGIC %md ## Training loop 
# MAGIC SHAP decomp will be called here: after the stage1 predictions, we compute the performance of the model and apply decomp on the best stage1 model only

# COMMAND ----------

## Set retention columns based on the above
filter_text_list = ['_lag' + str(each_dropped_lag) for each_dropped_lag in lags_to_drop]
cols_to_keep = [col_nm for col_nm in stage1_df.columns if col_nm.endswith(tuple(filter_text_list)) == False]

## Set-up to control the 'volume' of lagged periods used for each model (should likely retain ~3-6)
min_lag_period = np.max(list(lags_to_drop)) + 1
lag_period_list_to_keep = np.arange(min_lag_period, min_lag_period + LAGS_TO_KEEP, 1)

## Set drop columns based on the above
keep_lag_text_list = ['_lag' + str(each_kept_lag) for each_kept_lag in lag_period_list_to_keep]
cols_to_drop = [col_nm for col_nm in stage1_df.columns if '_lag' in col_nm and col_nm.endswith(tuple(keep_lag_text_list)) == False]

## Subset to appropriate lag columns to prevent data leakage
stage1_df_temp = stage1_df[cols_to_keep]
stage1_df_temp = stage1_df_temp.drop(*cols_to_drop)

## Stage1: train and cache for using groupyBy Pandas UDF
rolling_stage1_pickles = stage1_df_temp.groupBy(stage1_cls.group_level).apply(train_stage_udf)
rolling_stage1_pickles.cache()
#print("running for lagged model {} // len of pickles = {}".format(each_time_period, rolling_stage1_pickles.count()))

## Saving pickle for decomp - only need to do so for Stage1 models
pickle_save = rolling_stage1_pickles.withColumn('lag_model', F.lit(int(lag_period)))
#save_df_as_delta(pickle_save, DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT, enforce_schema=False)

temp_models = get_model_id(rolling_stage1_pickles, "score_lookup", stage1_cls.group_level)
rolling_stage1_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
rolling_stage1_objects = {key:{key:rolling_stage1_objects[key]} for key in rolling_stage1_objects.keys()}

## Create model dictionary
temp_models = get_model_id(rolling_stage1_pickles, "score_lookup", stage1_cls.group_level)
rolling_stage1_objects = convertDFColumnsToDict(temp_models, "score_lookup", "model_pick") 
rolling_stage1_objects = {key:{key:rolling_stage1_objects[key]} for key in rolling_stage1_objects.keys()}

## Prediction: Stage1
stage1_df_pred = get_model_id(stage1_df_temp, "score_lookup", stage1_cls.group_level).cache()
predict_models = prediction_generator(predict_schema, stage1_cls, rolling_stage1_objects)
stage1_rolling_preds = stage1_df_pred.groupBy(stage1_cls.group_level).apply(predict_models)

## Shape predictions from long to wide
pivot_columns = [x for x in list(stage1_rolling_preds.columns) if x not in ['train_func', 'pred']]
stage1_rolling_preds = stage1_rolling_preds.withColumn('train_func', F.concat('train_func', F.lit('_stage1'))) 
## adding substring to identify Stage1 output
stage1_rolling_output = stage1_rolling_preds.groupby(pivot_columns).pivot("train_func").avg("pred").cache()

# COMMAND ----------

# MAGIC %md ## SHAP Decomposition

# COMMAND ----------

# MAGIC %run ./shap_decompostion

# COMMAND ----------

# need append cases to calculate best model
# not required if we load the final df
stg_data = stage1_rolling_output.join(mrd_df,['MODEL_ID', 'Month_of_Year'])

# COMMAND ----------

# compute best stage 1 model
best_stg_model = get_best_stg_model(stg_data, stage1_models_list)
print(f'The best model in Stage 1 is {best_stg_model}')

# COMMAND ----------

## Compute SHAP using pandas UDF
decomp_results = shap_decomposition(stage1_df_pred, best_stg_model, pickle_save)
display(decomp_results)

# COMMAND ----------


def create_decomp_schema():
  decomp_schema = StructType([
  StructField("MODEL_ID", StringType()),
  StructField("Base", DoubleType()),
  StructField("Cannibalization", DoubleType()), 
  StructField("Macroeconomics", DoubleType()),
  StructField("Media", DoubleType()),
  StructField("Pricing", DoubleType()), 
  StructField("Promotions", DoubleType()),
  StructField("Weather", DoubleType()),
  StructField("Month_Of_Year", IntegerType())])
  return decomp_schema

schema_test = create_decomp_schema()


# COMMAND ----------

