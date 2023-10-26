# Databricks notebook source
# MAGIC %md #08 - Modeling
# MAGIC 
# MAGIC This script develops the forecast on the mrd (model ready dataset).
# MAGIC This runs through a train/test period, uses the results for model selection/feature importance, and then runs on forward-looking period.

# COMMAND ----------

# DBTITLE 1,Imports
from typing import List, Dict, Tuple
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType, StringType, ArrayType
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from functools import reduce, partial
from itertools import product
from collections import namedtuple
from time import time
import json
import mlflow
import lightgbm
import ast

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

# DBTITLE 1,Functions
def get_lags(model_lag:int, n_lags:int):
  return list(range(model_lag + 1, model_lag + n_lags + 1))


def filter_by_lag(cols:List[str], lag:int, n_lags_to_keep:int, is_filter_cases:bool, col_target:str) -> List[str]:
  
  lags = get_lags(lag, n_lags_to_keep)
  cols_lags = [c for c in cols if any(map(lambda x: c.lower().endswith(f'_lag{x}'), lags)) or '_lag' not in c.lower() and c.lower() != col_target.lower()]
  
  if not is_filter_cases:
    cols_cases = [c for c in cols if c.lower().startswith('cases_lag') and int(c.lower().replace('cases_lag', '')) > lag]
    cols_lags = list(set(cols_lags + cols_cases))
    
  return cols_lags


def get_lag_model_threshold_top_features(lag_model:int, threshold:float, cols:List[str], is_filter_cases:bool, path_input:str):
  
  cols_top = (load_delta(path_input)
              .filter((F.col('LAG_MODEL') == lag_model) & (F.col('THRESHOLD') == threshold))
              .sort('THRESHOLD')
              .select('FEATS')
              .first()
              .FEATS)
  
  if not is_filter_cases:
    cols_cases = [c for c in cols if c.lower().startswith('cases_lag') and int(c.lower().replace('cases_lag', '')) > lag_model]
    cols_top = list(set(cols_top + cols_cases))
    
  return cols_top


def get_cols(cols_features:List[str],
             cols_utils:List[str],
             col_target:str,
             lag_model:int,
             n_lags_to_keep:int,
             is_filter_cases:bool,
             df_feature_importance:pd.DataFrame=None):

  cols_features_lag_model_all = filter_by_lag(cols_features,
                                              lag=lag_model,
                                              n_lags_to_keep=n_lags_to_keep,
                                              is_filter_cases=is_filter_cases,
                                              col_target=col_target)
  
  cols_features_lag_model = cols_features_lag_model_all
  
  if df_feature_importance is not None:
    print(f'Using top feature importance')
    df_feature_importance_lag_model = df_feature_importance.loc[df_feature_importance['LAG_MODEL'] == lag_model, 'FEATS']
    assert not df_feature_importance_lag_model.empty, f'lag_model {lag_model} not found in df_feature_importance.'
    cols_features_lag_model = df_feature_importance_lag_model.iloc[0].tolist()
    
  cols_all = cols_features_lag_model + cols_utils + [col_target]
  cols_all = list(set(cols_all))
  return cols_features_lag_model, cols_features_lag_model_all, cols_all


def get_feature_selection_dbfs_path(time_var:str, path_base:str='dbfs:/mnt/adls/Tables/exploration/feature_importance', table_name:str='df_top_features'):
  return path_base + ('_weekly' if time_var == 'Week_Of_Year' else '_monthly') + f'/{table_name}'

def get_lag_models_list(df:pd.DataFrame, col_time:str, lag_models:List[int]):
  ## Setup for our loop - don't want this to run in each loop iteration
  holdout_periods = df.filter(F.col('TRAIN_IND') == 0).select(col_time).distinct().rdd.map(lambda x: x[col_time]).collect()
  ## Set the number of holdout periods being run
  lag_models_filtered = []
  for each_holdout_period in np.arange(1, len(holdout_periods) + 1, 1):
    temp_list = [x for x in lag_models if x >= each_holdout_period]
    if len(temp_list) > 0:
      min_lag_model = np.min(temp_list)
      lag_models_filtered.append(min_lag_model)

  ## Validate the above
  lag_models_filtered = sorted(list(set(lag_models_filtered)))
  
  return lag_models_filtered


def stack_data_by_algorithm(df: DataFrame, algorithms:List[str]):
  return reduce(DataFrame.union, map(lambda x: df.withColumn('train_func', F.lit(x)), algorithms))

# COMMAND ----------

# DBTITLE 1,Feature importance
def train_lgbm_model(df:pd.DataFrame,
                                     lag_model:int,
                                     col_target:str,
                                     cols_features:List[str],
                                     cols_categorical:List[str],
                                     cols_utils:List[str],
                                     params:Dict[str, any]) -> pd.DataFrame:
  
  import gc
  
  print(f'Running feat importance analysis for lag_model: {lag_model}')
  
  assert(col_target not in cols_features + cols_utils)
  df_x = df.loc[:, cols_features + cols_utils]
  
  # lgbm categorical encoding
  if len(cols_categorical) > 0: df_x = df_x.astype(dict([(c, 'category') for c in cols_categorical]))
  
  # pd training datasets
  df_x_train = df_x.loc[df_x['TRAIN_IND']==1, :].drop(columns=cols_utils)
  ds_y_train = df.loc[df['TRAIN_IND']==1, col_target]
  
  # lgbm training dataset
  dtrain = lightgbm.Dataset(data=df_x_train, label=ds_y_train, feature_name='auto', categorical_feature='auto', free_raw_data=True)

  print(f'Training LightGBM model')
  model = lightgbm.train(params, dtrain)  
  
  del df_x
  del df_x_train
  del ds_y_train
  del dtrain
  gc.collect()
  
  return model


def get_model_feature_importance(model:any, lag_model:int, forecast_start_date:int) -> pd.DataFrame:
  
  import gc
  
  print('Computing feature importance')
  
  importance_list = list(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
  
  df_importance = (
    pd.DataFrame
    .from_records(importance_list, columns=['FEAT', 'IMPORTANCE'])
    .sort_values(by='IMPORTANCE', ascending=False)
    .reset_index(drop=True)
  )
  df_importance['IMPORTANCE_CUMSUM'] = df_importance['IMPORTANCE'].cumsum()
  df_importance['IMPORTANCE_PERC'] = df_importance['IMPORTANCE_CUMSUM'] / df_importance['IMPORTANCE_CUMSUM'].max()
  df_importance.drop(columns=['IMPORTANCE', 'IMPORTANCE_CUMSUM'], inplace=True)
  df_importance['LAG_MODEL'] = lag_model
  df_importance['FORECAST_START_DATE'] = forecast_start_date
  
  del importance_list
  gc.collect()
  
  return df_importance


def get_threshold_top_features(df:pd.DataFrame, threshold:float) -> pd.DataFrame:
  
  import gc
  
  cols_return = ['LAG_MODEL', 'THRESHOLD', 'N_FEATS', 'FEATS']
  df_results = pd.DataFrame([], columns=cols_return)
  
  print('Selecting top feature importance')
  
  assert(threshold is not None and threshold > .0)
  
  df_selected = df.loc[df['IMPORTANCE_PERC'] <= threshold]
  
  if not df_selected.empty: 
    df_results = (df_selected
                  .groupby(by=['LAG_MODEL'])
                  .apply(lambda x: 
                         pd.DataFrame([[
                           x['LAG_MODEL'].unique()[0],
                           threshold,
                           x['FEAT'].nunique(),
                           x['FEAT'].unique()
                         ]], columns=cols_return)
                        )
                  .reset_index(drop=True))
    
  del df_selected
  gc.collect()
  
  return df_results


def get_lightgbm_feature_importance(df:pd.DataFrame,
                                     lag_model:int,
                                    forecast_start_date:int,
                                     col_target:str,
                                     cols_features:List[str],
                                     cols_categorical:List[str],
                                     cols_utils:List[str],
                                     params:Dict[str, any],
                                   threshold:float) -> pd.DataFrame:
  
  import gc
  
  model = train_lgbm_model(df, lag_model, col_target, cols_features, cols_categorical, cols_utils, params)
  df_importance = get_model_feature_importance(model, lag_model, forecast_start_date)
  df_top = get_threshold_top_features(df_importance, threshold)
  
  del model
  del df_importance
  gc.collect()
  
  return df_top


def run_feature_importance_lag_models_training(df:pd.DataFrame,
                                               lag_models:List[int],
                                               cols_features:List[str],
                                               cols_utils:List[str],
                                               col_target:str,
                                               cols_categorical:List[str],
                                               col_time:int,
                                               n_lags_to_keep:int,
                                               is_filter_cases:bool,
                                               period:int,
                                               model_params:Dict[str, any],
                                               out_schema:any,
                                               threshold) -> pd.DataFrame:
  """
  Iterate over lag models, and get their top feature importances.
  """
  
  importance_dfs = []
  # iterate over lag_models
  for ix, lag_model in enumerate(lag_models):
    
    time_lag_model_start = time()
    
    print(f'Getting feature importance for lag model: {lag_model} ({ix+1}/{len(lag_models)})')
  
    # filter training columns
    cols_features_lag_model, _, cols_all_lag_model = get_cols(cols_features, cols_utils, col_target, lag_model, n_lags_to_keep, is_filter_cases)
    
    print(f'Using n features: {len(cols_features_lag_model)}')
  
    # just in case
    if len(cols_categorical) > 0: assert(all(map(lambda x: x in cols_features, cols_categorical)))
  
    # pandasUDF
    pd_udf = partial(get_lightgbm_feature_importance,
                      lag_model=lag_model,
                     forecast_start_date=period,
                      col_target=col_target,
                      cols_features=cols_features_lag_model,
                      cols_categorical=cols_categorical,
                      cols_utils=cols_utils,
                      params=model_params,
                     threshold=threshold)
    
    df_importance_lag_model = (df
                               .select(cols_all_lag_model)
                               .groupBy('train_func')
                               .applyInPandas(pd_udf, schema=out_schema))
    
    # the feat list is small, i will collect it on the driver
    importance_dfs.append(df_importance_lag_model.toPandas())
        
    print(f'Lag model took: {np.round((time() - time_lag_model_start) / 60, 2)}')
  
  df_importances = pd.concat(importance_dfs)
  
  return df_importances

# COMMAND ----------

# DBTITLE 1,Modelling
def predict(df, model_info, model_objects, predict_schema):
  df_data_models_predictions = get_model_id(df, "score_lookup", model_info.group_level)
  pd_udf = partial(score_forecast, model_info=model_info, OBJECTS_DICT=model_objects)
  df_predictions_models = (df_data_models_predictions
                           .groupby(model_info.group_level)
                           .applyInPandas(pd_udf, schema=predict_schema))
  return df_predictions_models


def filter_dataset_by_models(df:DataFrame, models:List[str], cols:List[str], is_train:bool):
  return (df
          .select(cols)
          .filter(
            (F.col('train_func').isin(models)) &
            (F.col('TRAIN_IND') == int(is_train))))


def train_and_predict(lag_model:int,
                      models:List[str],
                      n_concurrent_jobs:int,
                      cols_all:List[str],
                      df_data:DataFrame,
                      model_info:Dict[str, any],
                      model_params:any,
                      train_schema:any,
                      predict_schema:any):

  df_pickles, df_predictions_train, df_predictions_test = [None] * 3
  
  n_models = len(models)
  batch_size = np.min([n_models, n_concurrent_jobs])
  batches = list(enumerate(range(0, n_models - batch_size + 1, batch_size)))
  for ix, i in batches:
    
    print(f'---- ML models run: ({ix+1}/{len(batches)})')
    # subset models / training dataset
    df_data_models_train = filter_dataset_by_models(df_data, models, cols_all, is_train=True)
    df_data_models_test = filter_dataset_by_models(df_data, models, cols_all, is_train=False)
    
    df_data_models_train = df_data_models_train.cache()
    _ = df_data_models_train.count()
    df_data_models_test = df_data_models_test.cache()
    _ = df_data_models_test.count()
    
    # asserts
    train_shapes = df_data_models_train.groupby('train_func').count().select('count').distinct().rdd.map(lambda x: x['count']).collect()
    test_shapes = df_data_models_test.groupby('train_func').count().select('count').distinct().rdd.map(lambda x: x['count']).collect()
    assert_msg = 'Review {dataset_type} stacked dataframe. Different number of samples among algorithms.'
    assert len(train_shapes) == 1, assert_msg.format(dataset_type='training')
    assert len(test_shapes) == 1, assert_msg.format(dataset_type='testing')
    
    print(f'df_data_models_train.count: {train_shapes[0]}')
    print(f'df_data_models_test.count: {test_shapes[0]}')

    # train
    print(f'---- Training...')
    time_start = time()
    pd_udf = partial(parallelize_core_models, model_info=model_info, hyperparams_df=model_params)
    df_pickles_models = (df_data_models_train
                         .groupby(model_info.group_level)
                         .applyInPandas(pd_udf, schema=train_schema))
    df_pickles = df_pickles.union(df_pickles_models) if df_pickles is not None else df_pickles_models
    df_pickles = df_pickles.cache()
    print(df_pickles.show())
    print(f'---- training took: {np.round((time() - time_start) / 60, 2)}')

    # predict
    print(f'---- Predicting on training set...')
    time_start = time()
    df_pickles_score_lookup = get_model_id(df_pickles_models, "score_lookup", model_info.group_level)
    model_objects = convertDFColumnsToDict(df_pickles_score_lookup, "score_lookup", "model_pick")
    model_objects = {key: {key: model_objects[key]} for key in model_objects.keys()}
    
    df_predictions_models_train = predict(df_data_models_train, model_info, model_objects, predict_schema)
    
    df_predictions_train = df_predictions_train.union(df_predictions_models_train) if df_predictions_train is not None else df_predictions_models_train
    df_predictions_train = df_predictions_train.cache()
    df_predictions_train.count()
    
    print(f'---- predicting on training set took: {np.round((time() - time_start) / 60, 2)}')
    
    print(f'---- Predicting on testing set...')
    time_start = time()
    df_predictions_models_test = predict(df_data_models_test, model_info, model_objects, predict_schema)
    df_predictions_test = df_predictions_test.union(df_predictions_models_test) if df_predictions_test is not None else df_predictions_models_test
    df_predictions_test = df_predictions_test.cache()
    df_predictions_test.count()
    print(f'---- predicting on testing set took: {np.round((time() - time_start) / 60, 2)}')
  
  df_data_models_train = df_data_models_train.unpersist()
  df_data_models_test = df_data_models_test.unpersist()
  
  return df_pickles, df_predictions_train, df_predictions_test


def train_lag_models(df_data:DataFrame, lag_models:List[int],
    cols_features_all:List[str], cols_utils:List[str], col_target:str,
    n_lags_to_keep:int, is_filter_cases:bool, df_importances:pd.DataFrame,
    stage1_models_list:List, stage1_cls:any, stage1_params:any,
    stage2_models_list:List, stage2_cls:any, stage2_params:any,
    train_schema:List, predict_schema:List,
    output_path_dbfs_tmp_table:str,
    is_forward_run:bool,
    n_concurrent_jobs:int):
  
  # asserts
  assert df_data.count() > 0 # performance: this will require the data to be cached beforehand.
  assert len(lag_models) > 0
  assert len(cols_features_all) > 0
  assert len(cols_utils) > 0
  assert col_target is not None and col_target != ''
  assert output_path_dbfs_tmp_table is not None and output_path_dbfs_tmp_table != ''
  assert n_concurrent_jobs > 0

  # run loop
  for ix_lag_model, lag_model in enumerate(lag_models):
    
    time_lag_model_start = time()
    
    print(f'##--- LAG MODEL: {lag_model} ({ix_lag_model+1}/{len(lag_models)})')

  #   im doing this way cause i wanna see the print below
    cols_features_lag_model, cols_features_lag_model_all, cols_all = get_cols(cols_features_all, cols_utils, 
                                                                              col_target, int(lag_model), n_lags_to_keep, 
                                                                              is_filter_cases, df_importances)
  
    print(f'---- N FEATS: ({len(cols_features_lag_model)}/{len(cols_features_lag_model_all)})')

    print('---- Running models stage 1')
    df_stage1_pickles, df_stage1_predictions_train, df_stage1_predictions_test = train_and_predict(lag_model,
                                                                                                   stage1_models_list,
                                                                                                   n_concurrent_jobs,
                                                                                                   cols_all, 
                                                                                                   df_data,
                                                                                                   stage1_cls,
                                                                                                   stage1_params,
                                                                                                   train_schema,
                                                                                                   predict_schema)
  
    # append train + test preds
    df_stage1_predictions = df_stage1_predictions_train.union(df_stage1_predictions_test)
    df_stage1_predictions_pivot, cols_pivot = pivot_model_predictions(df_stage1_predictions, stage=1)
    
    df_stage1_predictions_pivot = df_stage1_predictions_pivot.cache()
    df_stage1_predictions_pivot.count()
  
    # unpersist
    df_stage1_pickles = df_stage1_pickles.unpersist(True)
    df_stage1_predictions_train = df_stage1_predictions_train.unpersist(True)
    df_stage1_predictions_test = df_stage1_predictions_test.unpersist(True)
    
    # join predictions stage 1
    df_data_stage2 = df_data.select(cols_all).filter(F.col('train_func').isin(stage2_models_list))
    df_data_stage2_with_stage1_predictions = df_data_stage2.join(df_stage1_predictions_pivot, on=cols_pivot)
    df_data_stage2_with_stage1_predictions = df_data_stage2_with_stage1_predictions.cache()
    df_data_stage2_with_stage1_predictions.count()
  
    print('---- Running models stage 2')
    df_stage2_pickles, df_stage2_predictions_train, df_stage2_predictions_test = train_and_predict(lag_model,
                                                                                                   stage2_models_list,
                                                                                                   n_concurrent_jobs,
                                                                                                   cols_all,
                                                                                                   df_data_stage2_with_stage1_predictions,
                                                                                                   stage2_cls,
                                                                                                   stage2_params,
                                                                                                   train_schema,
                                                                                                   predict_schema)
    
    df_stage2_predictions_train = df_stage2_predictions_train.cache()
    df_stage2_predictions_train.count()
    df_stage2_predictions_test = df_stage2_predictions_test.cache()
    df_stage2_predictions_test.count()
  
    df_stage2_predictions = df_stage2_predictions_train.union(df_stage2_predictions_test) if is_forward_run else df_stage2_predictions_test
    df_stage2_predictions_pivot, _ = pivot_model_predictions(df_stage2_predictions, stage=2)
  
    ## Joining with stage 1 output
    df_predictions_lag_model = df_stage1_predictions_pivot.join(df_stage2_predictions_pivot, on=cols_pivot)
    ## Create new column 
    df_predictions_lag_model = df_predictions_lag_model.withColumn('lag_period', F.lit(int(lag_model)))

    # save intermediary results
    print(f'---- Saving lag model results to tmp table: {output_path_dbfs_tmp_table}')
    save_df_as_delta(df_predictions_lag_model, output_path_dbfs_tmp_table, enforce_schema=False)
  
    # unpersist
    df_stage2_pickles = df_stage2_pickles.unpersist(True)
    df_stage2_predictions_train = df_stage2_predictions_train.unpersist(True)
    df_stage2_predictions_test = df_stage2_predictions_test.unpersist(True)
    df_stage1_predictions_pivot = df_stage1_predictions_pivot.unpersist(True)
    df_data_stage2_with_stage1_predictions = df_data_stage2_with_stage1_predictions.unpersist(True)
    df_stage2_predictions_train = df_stage2_predictions_train.unpersist(True)
    df_stage2_predictions_test = df_stage2_predictions_test.unpersist(True)
    df_predictions_lag_model = df_predictions_lag_model.unpersist(True)
    
    print(f'--- Lag model took: {np.round((time() - time_lag_model_start) / 60, 2)}')


def pivot_model_predictions(df_predictions:DataFrame, stage:int):
  # Shape predictions from long to wide
  cols_pivot = [x for x in df_predictions.columns if x not in ['train_func', 'pred']]
  df_predictions_renamed = df_predictions.withColumn('train_func', F.concat('train_func', F.lit(f'_stage{stage}')))
  df_predictions_pivot = df_predictions_renamed.groupby(cols_pivot).pivot("train_func").avg("pred")
  return df_predictions_pivot, cols_pivot


def get_tmp_table(col_time:str, path_base:str):
  assert path_base is not None and path_base != ''
  return f'{path_base}/tmp_{int(time() * 10000)}'


def read_training_loop_output_from_dbfs(input_path:str):
  versions = load_delta_info(input_path).history().rdd.map(lambda x: x.version).collect()
  assert(len(versions) > 0)
  return reduce(DataFrame.union, [load_delta(input_path, v) for v in versions])


def delete_tmp_table(path_:str):
  dbutils.fs.rm(path_, recurse=True)

# COMMAND ----------

# DBTITLE 1,Modelling postprocessing
## BUGFIX: assists calculation of forcast lag. this is needed to handle edge cases like when year changes.
## forcast lag is computed only for holdout set, train set forcast is returned as -1
def get_forcast_lag(time_var, fcst_start_date, forcast_period):
  if time_var<fcst_start_date: #this signifies train period
    return -1
  else:
    time_range = [x for x in forcast_period if x>=fcst_start_date and x<time_var] #get time periods between fcst_start and time_var
    return len(time_range)
  
  
def create_aggregated_output(output_path_dbfs_tmp_table:str, maxforcastUDF:any, forcastlagUDF:any, col_time:str, mrd_join_df:DataFrame, join_type:str):
  df = (read_training_loop_output_from_dbfs(output_path_dbfs_tmp_table)
        .withColumn('max_forcast', maxforcastUDF(F.struct('FCST_START_DATE', 'lag_period')))
        .filter(F.col(col_time) <= F.col('max_forcast'))
        .join(mrd_join_df, on=[col_time, 'MODEL_ID'], how=join_type)
        .withColumn('fcst_periods_fwd', forcastlagUDF(F.struct(col_time, 'FCST_START_DATE')))
        .withColumn('sample', F.when(F.col('fcst_periods_fwd') >= 0, 'OOS').otherwise('IS'))
       )

  cols_model = [c for c in df.columns if 'stage1' in c or 'stage2' in c]

  for c in cols_model:
    df = (df
          .withColumn(c,
                      F.when(F.col(c) < 1, 0)
                      .otherwise( F.ceil( F.col(c)))) 
         )
  return df

# COMMAND ----------

# DBTITLE 1,Cross validation
class Splitter(object):
  
  def __init__(self, n_splits:int):
    self.n_splits = n_splits
    
  def split(self):
    raise NotImplementedError('Subclass should implement this')
    
  def get_latest_split(self, periods:List[str]):
    # TODO: hacky. Im using a generator but listing it here anyways.
    return list(self.split(periods))[-1]
  
  def get_n_splits(self, periods:List[str]):
    # TODO: hacky. Im using a generator but listing it here anyways.
    return len(list(self.split(periods)))
    
  def plot(self, periods:List[str], col_time:str, figsize:Tuple[int,int]=(10,5), linewidth:int=10, labelrotation:int=90, xticks_frequency:int=4):

    periods = sorted(periods)

    # create splits
    ys = []
    for ix, (period_train_start, period_train_end, period_test_start, period_test_end) in enumerate(self.split(periods)):
      y = [1] * np.sum(np.array(periods) <= period_train_end)
      y += [np.nan] * (get_periods_difference(period_train_end, period_test_start, col_time) - 1)
      y += [0] * np.sum( (np.array(periods) >= period_test_start) & (np.array(periods) <= period_test_end) )
      y += [np.nan] * np.sum(np.array(periods) > period_test_end)
      ys.append(pd.Series(y))

    df_tmp = pd.concat(ys, axis=1)

    df_tmp.index = map(lambda x: period_to_datetime(x, col_time), periods)

    # plot splits
    fig, ax = plt.subplots(figsize=figsize)
    for ix in df_tmp.columns:
      sns.scatterplot(x='index', y='y', marker='_', linewidth=linewidth, color='green', data=df_tmp.loc[df_tmp[ix]==1, ix].reset_index().assign(y=ix+1), ax=ax)
      sns.scatterplot(x='index', y='y', marker='_', linewidth=linewidth, color='blue', data=df_tmp.loc[df_tmp[ix]==0, ix].reset_index().assign(y=ix+1), ax=ax)
    _ = ax.set_xlim((df_tmp.index.min(), df_tmp.index.max()))
    _ = ax.set_xticks(list(df_tmp.index)[::xticks_frequency])
    _ = ax.tick_params(axis='x', labelrotation=labelrotation)
    _ = ax.set_xlabel('')
    _ = ax.set_ylabel('')
    _ = ax.set_yticks([])
    _ = ax.set_yticklabels('')
    _ = ax.set_ylim((0, len(df_tmp.columns)+1))


class BacktestingSplitter(Splitter):
  
  def __init__(self, n_splits:int, test_start:int, test_end:int, gap:int):
    super().__init__(n_splits=n_splits)
    self.test_start = test_start
    self.test_end = test_end
    self.gap = gap

  def split(self, periods:List[int]):
    periods = np.sort(np.array(periods))
    periods_rolling = periods[periods >= self.test_start][-self.gap::-self.gap][:self.n_splits][::-1]
    train_start = np.min(periods)
    for test_start in periods_rolling:
      train_end = np.max(periods[periods < test_start])
      yield(train_start, train_end, test_start, self.test_end)


class TimeseriesSplitter(Splitter):
  
  def __init__(self, n_splits:int=5, max_train_size:int=None, test_size:int=None, gap:int=0):
    super().__init__(n_splits=n_splits)
    self.max_train_size = max_train_size
    self.test_size = test_size
    self.gap = gap

  def split(self, periods:List[int]):
    periods = sorted(periods)
    n_samples = len(periods)
    n_folds = self.n_splits + 1
    test_size = self.test_size if self.test_size is not None else n_samples // n_folds

    # Make sure we have enough samples for the given split parameters
    if n_folds > n_samples:
        raise ValueError(
            f"Cannot have number of folds={n_folds} greater"
            f" than the number of samples={n_samples}."
        )
    if n_samples - self.gap - (test_size * self.n_splits) <= 0:
        raise ValueError(
            f"Too many splits={self.n_splits} for number of samples"
            f"={n_samples} with test_size={self.test_size} and gap={self.gap}."
        )

#     indices = np.arange(n_samples)
    ix_train_start = 0
    ix_test_starts = range(n_samples - self.n_splits * test_size, n_samples, test_size)
    
    for ix_test_start in ix_test_starts:
        ix_train_end = ix_test_start - self.gap - 1
        # if no test_size, then take all samples till end as test samples
        ix_test_end = ix_test_start + test_size - 1 if self.test_size is not None else n_samples
        yield (periods[ix_train_start], periods[ix_train_end], periods[ix_test_start], periods[ix_test_end])


def period_to_datetime(period:int, col_time:str):
  return datetime.strptime(str(period) + '1', "%G%V%u" if col_time.lower() == 'week_of_year' else '%Y%m%u')


def get_periods_difference(p1:int, p2:int, col_time:str):
  d1 = period_to_datetime(p1, col_time)
  d2 = period_to_datetime(p2, col_time)
  if col_time.lower() == 'week_of_year':
    diff = int((d2 - d1).days / 7)
  else:
    diff = (d2.year - d1.year) * 12 + d2.month - d1.month
  return diff

# COMMAND ----------

# DBTITLE 1,MLFlow
class MLFlowTracker(object):
  
  def __init__(self, experiment_name:str, experiment_path:str):
    self._experiment = self._create_experiment(experiment_name, experiment_path)
    self._run_id = self._start_run()

  @staticmethod
  def _create_experiment(name:str, path:str):
    try:
      experiment_id = mlflow.create_experiment(name, artifact_location=path)
      return mlflow.get_experiment(experiment_id)
    except mlflow.exceptions.RestException as e:
      if e.error_code == 'RESOURCE_ALREADY_EXISTS':
        mlflow.set_experiment(name)
        return mlflow.get_experiment_by_name(name)
    
  def _start_run(self):
    with mlflow.start_run(experiment_id=self._experiment.experiment_id) as mlflow_run:
      return mlflow_run.info.run_id
    
  def _resume_run(self):
    return mlflow.start_run(run_id=self._run_id, experiment_id=self._experiment.experiment_id)
  
  def log_param(self, key:str, val:any):
    with self._resume_run() as mlflow_run:
      mlflow.log_param(key, val)
      
  def log_artifact(self, local_path:str):
    with self._resume_run() as mlflow_run:
      mlflow.log_artifact(local_path)
      
  def log_metrics(self, metrics:Dict[str, float]):
    with self._resume_run() as mlflow_run:
      mlflow.log_metrics(metrics)
      
  def log_dict(self, _dict:any, artifact_path:str):
    with self._resume_run() as mlflow_run:
      mlflow.log_dict(_dict, artifact_path)
    
  def __str__(self):
    return f'Experiment_id: {self._experiment.experiment_id} Run_id: {self._run_id}'

# COMMAND ----------

## Check configurations exist for this script
required_configs = [DBA_MRD, DBA_MODELIDS, DBO_FORECAST, DBO_FORECAST_ROLLING, RUN_TYPE, TIME_VAR]
print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

# DBTITLE 1,Dataset version config
## Find way for us to set this more easily (alternative is to simply have as ./configs)
# ## TODO - delete from production version - no over-rides
# ## PALAASH/ANAND - weekly meta
if TIME_VAR == "Week_Of_Year":
  DBA_MRD_version =  2
  DBA_MODELIDS_version =  0
  DBA_MRD_CLEAN_version =  0
  DBA_MRD_EXPLORATORY_data_version = 0
  DBO_OUTLIERS_version =  0
  DBO_SEGMENTS_version = 0
  DBO_HYPERPARAMATER_version = 0   ## 0 is latest run with full weekly // 1 is Pingo Doce weekly with drivers

## PALAASH/ANAND - monthly meta
if TIME_VAR == "Month_Of_Year":
  DBA_MRD_version =  6
  DBA_MRD_SUB_version = 3
  DBA_MODELIDS_version = 1
  DBA_MRD_CLEAN_version =  1 
  DBA_MRD_EXPLORATORY_data_version = 1
  DBO_OUTLIERS_version = 1
  DBO_SEGMENTS_version = 1
  DBO_HYPERPARAMATER_version = 0 # monthly with drivers

# COMMAND ----------

# DBTITLE 1,Run config
# ------- RUN DEFINITION: start ------- #
# Modalities:
#  1. run a backtest & no forward.
#  2. run a forward chaining xvalidation & forward.
RUN_BACKTEST = False
RUN_FORWARD = True

# you either run a backtest or a future forecast
assert(RUN_BACKTEST != RUN_FORWARD)
# ------- RUN DEFINITION: end ------- #


## User to dictate if we roll-up different lagged models to a single value
## Experiments indicated this led to a benefit in accuracy when tested OOS
AGGREGATE_LAGS_TO_PREDICT = True

## Defines the length of our OOS holdout for the Train-Test split part of this notebook
## Default should be to define this as 8-12 weeks from latest date
HOLDOUT_PERIOD_LEN = 8


# ------- FEATURE IMPORTANCE: start ------- #
# do you want to run the feature importance analysis and train on a subset of the features?
IS_RUN_FEATURE_IMPORTANCE = True
# which feature importance threshold should be used?
# e.g. ".99" would mean: take features that account for the top 99% cumsum importance.
FEATURE_IMPORTANCE_THRESHOLD = .99
# ------- FEATURE IMPORTANCE: end ------- #


## Allows user to control 'volume' of retained lag values - cut down width and noise of FEATURES that are lagged
## Eg, if this value = 4 then Lag6 model will retain features lagged 7, 8, 9, 10 (ie, the 4 closest periods we can use)
LAGS_TO_KEEP = 4

# do you want to train on the full set of target_lag features? or keep the ones according to the "LAGS_TO_KEEP" var?
# "IS_FILTER_CASES = False" would mean to train on all (past) target_cases
# e.g. for a lag 2 model, train using ['cases_lag_3', 'cases_lag_4', 'cases_lag_5', ... , 'cases_lag_14', ...]
IS_FILTER_CASES = False

# define how many concurrent PandasUDFs you want to launch
N_CONCURRENT_JOBS = 5

if TIME_VAR == 'Month_Of_Year':
  ## Note - these will only be used if RUN_BACKTEST = True 
  ROLLING_START_DATE = 202009
  PERIODS = 2
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 18  
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [4, 6, 12, 18] # Manu
#   DYNAMIC_LAG_MODELS = [4] # TODO: for debugging
  
  # define xvalidation params
  if RUN_FORWARD:
    XVALIDATION_N_SPLITS = 1 # nr of splits to create.
    XVALIDATION_GAP = 0 # periods between train & validation sets.
    XVALIDATION_HOLDOUT_SIZE = 3 # length of validation set. If "None" then take all till end of dataset.
  else:
    XVALIDATION_N_SPLITS = 2 # max nr of splits to create.
    XVALIDATION_GAP = 1 # periods between validation set start n & validation set start n+1
  
else: # WEEKLY
  ## Note - these will only be used if RUN_BACKTEST = True 
  ROLLING_START_DATE = 202101
  PERIODS = 20
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 16
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [3, 7, 13, 17] # you need to add 1 to each lag_model (in this case, it would be lag_models [2, 6, 12, 16])
#   DYNAMIC_LAG_MODELS = [3] # TODO: for debugging
  
  # define xvalidation params
  if RUN_FORWARD:
    XVALIDATION_N_SPLITS = 2 # nr of splits to create.
    XVALIDATION_GAP = 0 # periods between train & validation sets.
    XVALIDATION_HOLDOUT_SIZE = 12 # length of validation set. If "None" then take all till end of dataset.
  else:
    XVALIDATION_N_SPLITS = 15 # max nr of splits to create.
    XVALIDATION_GAP = 4 # periods between validation set start n & validation set start n+1

# COMMAND ----------

# bookeeping structure to loop over and delete at the end
dbfs_tmp_tables_to_delete = []

# COMMAND ----------

# DBTITLE 1,MLFlow tracker instantiation
mlflow_tracker = MLFlowTracker(**MLFLOW_CONFIGS['modelling']._asdict())
print(mlflow_tracker)

# COMMAND ----------

# DBTITLE 1,Load Data
try:
  mrd_df = load_delta(DBA_MRD, DBA_MRD_version)
#   mrd_df = load_delta(DBA_MRD_SUB, DBA_MRD_SUB_version) # sampled dataset: for quick testing.
  delta_info = load_delta_info(DBA_MRD)
  display(delta_info.history())
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

# COMMAND ----------

# cache & trigger
mrd_df = mrd_df.cache()
mrd_df.count()

# COMMAND ----------

# create this df used later on for model selection.
# done here so we can del mrd_df afterwards.
# also avoiding duplicated backtest/forward duplicated code.
mrd_join_df = (mrd_df
               .select('MODEL_ID', TIME_VAR, TARGET_VAR)
               .withColumnRenamed(TARGET_VAR, TARGET_VAR + '_ORIG')
               .join(model_info, on=['MODEL_ID'])
               .distinct())

# cache & trigger
mrd_join_df = mrd_join_df.cache()
mrd_join_df.count()

# COMMAND ----------

## Capturing pared-down MRD version for downstream joining/merging 
## MODEL ID will be the unique identifier for downstream merging

mrd_clean_df = (mrd_df
                .select("MODEL_ID", TIME_VAR, TARGET_VAR)
                .withColumnRenamed(TARGET_VAR, TARGET_VAR + "_ORIG")
                )
print(len(mrd_df.columns), mrd_df.count())
print(len(mrd_clean_df.columns), mrd_clean_df.count())  ## only 3 cols as specified above

# COMMAND ----------

# DBTITLE 1,Set Train/Test Holdout Period Using Historical Dates
## Pulling dates as references for downstream
## Our full dataset now contains historicals and future shell details
historicals = load_delta(DBA_MRD_CLEAN, DBA_MRD_CLEAN_version)
max_historical_date = historicals.select(F.max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

full_data = load_delta(DBA_MRD, DBA_MRD_version)
max_future_date = full_data.select(F.max(TIME_VAR)).collect()[0].asDict()['max(' + TIME_VAR + ')']

print('Max historical date = {}'.format(max_historical_date))
print('Max full dataset date = {}'.format(max_future_date))

## Pulling our calendar - to handle edge cases for when crossing into another year
calendar_df = load_delta(DBI_CALENDAR)
calendar_pd = calendar_df.toPandas()

## To use as a reference (for edge cases) in cell below
calendar_sorted_periods = sorted([i[TIME_VAR] for i in calendar_df.select(TIME_VAR).distinct().collect()])
cal_ref = calendar_sorted_periods.index(max_historical_date)

# COMMAND ----------

# HOLDOUT_RANGE defines the timespan of the training set of the forward run
# FORWARD_RANGE defines the timestmap of the testing set of the forward run


# Correction code in case user enters a negative value
if HOLDOUT_PERIOD_LEN < 0:
  HOLDOUT_PERIOD_LEN = (HOLDOUT_PERIOD_LEN * (-1))
  print('Converted holdout period to positive integer to preserve below calculations')
  
HOLDOUT_RANGE = (calendar_sorted_periods[cal_ref - HOLDOUT_PERIOD_LEN + 1], calendar_sorted_periods[cal_ref])
print('Holdout Range = {}'.format(HOLDOUT_RANGE))


if RUN_FORWARD:
  ## Correction code in case user enters a negative value
  if TIME_PERIODS_FORWARD < 0:
    TIME_PERIODS_FORWARD = (TIME_PERIODS_FORWARD * (-1))
    print('Converted forward-looking period to positive integer to preserve below calculations')

  FORWARD_RANGE = (calendar_sorted_periods[cal_ref + 1], calendar_sorted_periods[cal_ref + TIME_PERIODS_FORWARD])
  print('Holdout Periods Used = {}'.format(FORWARD_RANGE))
  
  mlflow_tracker.log_param('FORWARD_RANGE', FORWARD_RANGE)

# COMMAND ----------

# create backtest dataset
mrd_df_backtest = mrd_df.filter( F.col(TIME_VAR) <= HOLDOUT_RANGE[1] )

# cache & trigger
mrd_df_backtest = mrd_df_backtest.cache()
mrd_df_backtest.count()


if RUN_FORWARD:
  # create forward dataset
  mrd_df_future = (mrd_df
                   .filter(
                     (F.col('FCST_START_DATE') == FORWARD_RANGE[0]) &
                     (F.col(TIME_VAR) <= FORWARD_RANGE[1]))
                   .withColumn('TRAIN_IND',
                               F.when(F.col(TIME_VAR) < FORWARD_RANGE[0], 1)
                               .otherwise(0))
                  )
  
  # cache & trigger
  mrd_df_future = mrd_df_future.cache()
  mrd_df_future.count()

# COMMAND ----------

# get all dates
dates = [row[0] for row in mrd_df_backtest.select(TIME_VAR).distinct().sort(TIME_VAR).collect()]

# Add a snapshot date to compute model selection
if RUN_BACKTEST:
  
  first_snapshot = dates.index(ROLLING_START_DATE)
  ROLLING_START_DATE = dates[first_snapshot-1]  
  PERIODS += 1

# COMMAND ----------

# DBTITLE 1,Delete unused objects
# "mrd_df" should be used anymore after this point.
# unpersist & delete.
# this object shouldnt be used anymore. Reference mrd_df_backtest or mrd_df_forward.
mrd_df = mrd_df.unpersist(True)
del mrd_df

# this one is also not used anymore.
# unpersist if necessary.
del model_info

# COMMAND ----------

## Print data versions
print('DBA_MRD_version = {}'.format(DBA_MRD_version))
print('DBA_MODELIDS_version = {}'.format(DBA_MODELIDS_version))
print('DBA_MRD_CLEAN_version = {}'.format(DBA_MRD_CLEAN_version))
print('DBA_MRD_EXPLORATORY_data_version = {}'.format(DBA_MRD_EXPLORATORY_data_version))
print('DBO_OUTLIERS_version = {}'.format(DBO_OUTLIERS_version))
print('DBO_SEGMENTS_version = {}'.format(DBO_SEGMENTS_version))

## Set up MLflow parameter monitoring
#Global Parameters
mlflow_tracker.log_param('Target Variable', TARGET_VAR)
mlflow_tracker.log_param('Time Variable', TIME_VAR)
mlflow_tracker.log_param('Forecast Aggregation Level', MODEL_ID_HIER)
mlflow_tracker.log_param('Holdout Range', HOLDOUT_RANGE)
mlflow_tracker.log_param('HOLDOUT_PERIOD_LEN', HOLDOUT_PERIOD_LEN)
mlflow_tracker.log_param('RUN_FORWARD', RUN_FORWARD)


#Data Versions
mlflow_tracker.log_param('DBA_MRD_version', DBA_MRD_version) 
mlflow_tracker.log_param('DBA_MODELIDS_version', DBA_MODELIDS_version) 
mlflow_tracker.log_param('DBA_MRD_CLEAN_version', DBA_MRD_CLEAN_version) 
mlflow_tracker.log_param('DBA_MRD_EXPLORATORY_data_version', DBA_MRD_EXPLORATORY_data_version) 
mlflow_tracker.log_param('DBO_OUTLIERS_version', DBO_OUTLIERS_version) 
mlflow_tracker.log_param('DBO_SEGMENTS_version', DBO_SEGMENTS_version) 
mlflow_tracker.log_param('DBO_HYPERPARAMATER_version', DBO_HYPERPARAMATER_version) 

# COMMAND ----------

## Exit for egregious errors
if len(intersect_two_lists([TARGET_VAR], mrd_df_backtest.columns)) == 0:
  dbutils.notebook.exit("Target variable not in data, Exiting notebook")
else: print('Target Variable ({}) in dataframe!'.format(TARGET_VAR))
  
if len(intersect_two_lists([TIME_VAR], mrd_df_backtest.columns)) == 0:
  dbutils.notebook.exit("Time variable not in data, Exiting notebook")
else: print('Time Variable ({}) in dataframe!'.format(TIME_VAR))

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

key_lightgbm = 'lightGBM_model'
key_quantile_1 = 'gbm_quantile_model'
key_quantile_2 = 'gbm_quantile_model2'
key_quantile_3 = 'gbm_quantile_model3'
key_quantile_lower = 'gbm_quantile_lower'
key_quantile_middle = 'gbm_quantile_mid'
key_quantile_upper = 'gbm_quantile_upper'
key_catboost = 'catboost_model'
key_random_forest = 'rforest_model'

# COMMAND ----------

# DBTITLE 1,Stage1 & Stage2 Parameter Logic
## Note: adding params for additional quantiles in stage 1
## This is to help address the under-forecasting (likely the result of data sparsity)
quantile = stage1_params[stage1_params['train_func'] == key_quantile_1]['params'].iloc[0]
quantile2 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_2) for key, value in eval(quantile).items()})
quantile3 = str({key:(value if key != 'alpha' else QUANTILE_ALPHA_3) for key, value in eval(quantile).items()})

## Append these parameters to our Stage 1 dictionary
stage1_params = stage1_params.append({'train_func':key_quantile_2, "params":quantile2}, ignore_index=True)
stage1_params = stage1_params.append({'train_func':key_quantile_3, "params":quantile3}, ignore_index=True)

## Note: adding params for additional quantiles in stage 2
## This is to set upper and lower bounds for our Stage 2 models (for bounds)
quantile_mid = str({key:(value if key != 'alpha' else MID_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_lb  = str({key:(value if key != 'alpha' else LOWER_ALPHA_LEVEL) for key, value in eval(quantile).items()})
quantile_ub  = str({key:(value if key != 'alpha' else UPPER_ALPHA_LEVEL) for key, value in eval(quantile).items()})

stage2_params = stage2_params.append({'train_func':key_quantile_lower, "params":quantile_lb}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':key_quantile_middle, "params":quantile_mid}, ignore_index=True)
stage2_params = stage2_params.append({'train_func':key_quantile_upper, "params":quantile_ub}, ignore_index=True)

display(stage1_params)
display(stage2_params)

# COMMAND ----------

#Log hyperparameters to mlflow as yaml
stage1_dict = convertDFColumnsToDict(spark.createDataFrame(stage1_params), "train_func", "params")
for i in stage1_dict:
  stage1_dict[i] = ast.literal_eval(stage1_dict[i])
mlflow_tracker.log_dict(stage1_dict, "stage1_dict.yml")

stage2_dict = convertDFColumnsToDict(spark.createDataFrame(stage2_params), "train_func", "params")
for i in stage2_dict:
  stage2_dict[i] = ast.literal_eval(stage2_dict[i])
mlflow_tracker.log_dict(stage2_dict, "stage2_dict.yml")

# COMMAND ----------

# DBTITLE 1,Modeling UDF Setup
train_function_mappings = {
    key_quantile_1: train_lightGBM,
    key_quantile_2: train_lightGBM,
    key_quantile_3: train_lightGBM,
    key_quantile_lower: train_lightGBM,
    key_quantile_upper: train_lightGBM,
    key_quantile_middle: train_lightGBM,
    key_catboost: train_catboost,
    key_random_forest: train_random_forest,
    key_lightgbm: train_lightGBM,
}

pred_function_mappings = {
    key_quantile_1: predict_lightGBM,
    key_quantile_2: predict_lightGBM,
    key_quantile_3: predict_lightGBM,
    key_quantile_lower: predict_lightGBM,
    key_quantile_upper: predict_lightGBM,
    key_quantile_middle: predict_lightGBM,
    key_catboost: predict_catboost,
    key_random_forest: predict_random_forest,
    key_lightgbm: predict_lightGBM,
}

stage1_models = {
    key_lightgbm: lgbm_params,
    key_quantile_1: quantile_params,
    key_quantile_2: quantile_params,
    key_quantile_3: quantile_params,
    key_catboost: catboost_params,
    # key_random_forest: rf_params,
}

stage2_models = {
  key_lightgbm: lgbm_params,
  key_quantile_1: quantile_params,
  key_quantile_lower: quantile_params,
  key_quantile_upper: quantile_params,
}

# COMMAND ----------

# DBTITLE 0,Static Forecast
## Setup information for modeling
## Must group using FCST_START_DATE (and train_func) for our 'rolling' efforts

model_info_dict = dict(
    target               = TARGET_VAR,                           #Target variable that we want to predict
    train_id_field       = ['TRAIN_IND'],                        #Binary indicator if data is train/test (data is stacked for rolling holdout)
    group_level          = ['train_func', 'FCST_START_DATE'],    #Column in data that represents model we want to run (data is stacked for multiple algorithms)
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
  StructField("model_pick", StringType())
])

## Predict Schema
predict_schema = StructType([
  StructField("MODEL_ID", StringType()),
  StructField(TIME_VAR, IntegerType()),
  StructField("train_func", StringType()),
  StructField("FCST_START_DATE", IntegerType()),    ## only needed for rolling runs
  StructField("pred", DoubleType())
])

# COMMAND ----------

# DBTITLE 1,Check parameter tuning table compatibility
## Check the availability of params for algos in ensemble
stage1_models_list = list(stage1_models.keys())
stage2_models_list = list(stage2_models.keys())
training_algos = list( set( stage1_models_list + stage2_models_list ) )
tuned_algos = set(stage1_params['train_func'].unique())

print("Tuned parameters available for: ", set(training_algos).intersection(tuned_algos))
print("Tuning is missing for:", set(training_algos).difference(tuned_algos))

## Check the param level is same is the prediction level
param_required = [x for x in stage1_cls.group_level if x not in ['FCST_START_DATE']]
param_available = [x for x in stage1_params.columns if x not in ['params', 'one_model_dummy_seg']]

if sorted(param_required) == sorted(param_available):
  print("Parameter tuning results will be used")
  mlflow_tracker.log_param('TUNED_PARAMS_USED', True)
else:
  print("Using default params as the tuning level and prediction level do not match. Make sure the variable 'DBO_HYPERPARAMATER_version' is set correctly")
  mlflow_tracker.log_param('TUNED_PARAMS_USED', False)
  
mlflow_tracker.log_param('STAGE1_ALGOS', stage1_models.keys())
mlflow_tracker.log_param('STAGE2_ALGOS', stage2_models.keys())

# COMMAND ----------

# DBTITLE 1,Instantiate splitter
# HOLDOUT_RANGE defines the timespan of the training set of the forward run
# FORWARD_RANGE defines the timestmap of the testing set of the forward run

periods_all = mrd_df_backtest.select(TIME_VAR).distinct().sort(TIME_VAR).rdd.map(lambda r: r[0]).collect()

if RUN_BACKTEST:
  # intended for running backtesting & no forward run
  cv_splitter = BacktestingSplitter(n_splits=XVALIDATION_N_SPLITS, test_start=ROLLING_START_DATE, test_end=HOLDOUT_RANGE[1], gap=XVALIDATION_GAP)
else:
  # intended for running xvalidation & forward run
  cv_splitter = TimeseriesSplitter(n_splits=XVALIDATION_N_SPLITS, test_size=XVALIDATION_HOLDOUT_SIZE, gap=XVALIDATION_GAP)

# plot
cv_splitter.plot(periods_all, TIME_VAR)

# COMMAND ----------

# DBTITLE 1,Feature Importance
schema_importance = StructType([
  StructField('LAG_MODEL', IntegerType()),
  StructField('THRESHOLD', DoubleType()),
  StructField('N_FEATS', IntegerType()),
  StructField('FEATS', ArrayType(StringType()))
])

cols_features_all = mrd_df_backtest.columns
col_target = TARGET_VAR
cols_utils = ['train_func', 'TRAIN_IND', 'FCST_START_DATE', 'MODEL_ID']
cols_categorical = [f.name for f in mrd_df_backtest.schema if f.dataType == StringType() and f.name not in cols_utils]
_, _, period_train_test_feature_importance, _ = cv_splitter.get_latest_split(periods_all) # get latest

# in case it is not run
df_importances = None

if IS_RUN_FEATURE_IMPORTANCE and FEATURE_IMPORTANCE_THRESHOLD is not None:
  
  is_exists_partition = period_train_test_feature_importance in (mrd_df_backtest
                             .select('FCST_START_DATE')
                             .distinct()
                             .rdd
                             .map(lambda x: x.FCST_START_DATE)
                             .collect())

# period_train_test_feature_importance

  assert is_exists_partition, f'Driver forecast partition {period_train_test_feature_importance} not found in backtest dataset.'
  
  # get lightgbm data
  df_data_feature_importance = (mrd_df_backtest
                                  .filter( F.col('FCST_START_DATE') == int(period_train_test_feature_importance) )
                                  .filter( F.col(TIME_VAR) < int(period_train_test_feature_importance) )
                                  .withColumn('TRAIN_IND', F.lit(1))
                                  .withColumn('train_func', F.lit('dummy'))
                               )
  # cache and trigger
  df_data_feature_importance.cache()
  df_data_feature_importance.count()
  
  assert df_data_feature_importance.count() > 0, 'No training samples found.'
  
  print(f'Training on {df_data_feature_importance.count()} samples')
  
  df_importances = run_feature_importance_lag_models_training(df=df_data_feature_importance,
                                                              lag_models=DYNAMIC_LAG_MODELS,
                                                              cols_features=cols_features_all,
                                                              cols_utils=cols_utils,
                                                              col_target=col_target,
                                                              cols_categorical=cols_categorical,
                                                              col_time=TIME_VAR,
                                                              n_lags_to_keep=LAGS_TO_KEEP,
                                                              is_filter_cases=IS_FILTER_CASES,
                                                              period=period_train_test_feature_importance,
                                                              model_params=lgbm_params,
                                                              out_schema=schema_importance,
                                                              threshold=FEATURE_IMPORTANCE_THRESHOLD)
  df_data_feature_importance.unpersist(True)
  
  # Log feature importance to Mlflow
  path_output = 'df_importances.csv'
  df_importances.to_csv(path_output)
  mlflow_tracker.log_artifact(path_output)
  
  df_importances

# COMMAND ----------

# DBTITLE 1,Modelling
# TODO: lqrz. Saving the models pickles crashes. Figure out whats the ideal way of handling model saving. 

output_path_dbfs_tmp_table = get_tmp_table(TIME_VAR, path_base='dbfs:/mnt/adls/Tables/modelling')

dbfs_tmp_tables_to_delete.append(output_path_dbfs_tmp_table)

n_splits = cv_splitter.get_n_splits(periods_all)

for ix_period, (period_train_start, period_train_end, period_test_start, period_test_end) in enumerate(cv_splitter.split(sorted(periods_all))):
  
  time_period_start = time()

  print(f'#--- PERIOD: {period_train_start}-{period_train_end} {period_test_start}-{period_test_end} ({ix_period+1}/{n_splits})')
  
  # select specific driver forecast partition
  mrd_df_backtest_period = mrd_df_backtest.filter(F.col('FCST_START_DATE') == int(period_test_start))

  # stack by algorithm (so as to parallelise model training)
  stacked_data = stack_data_by_algorithm(mrd_df_backtest_period, training_algos)
  
  df_rolling = (stacked_data
                .withColumn('TRAIN_IND',
                            F.when( (F.col(TIME_VAR) >= int(period_train_start)) & (F.col(TIME_VAR) <= int(period_train_end)), 1)
                            .when( (F.col(TIME_VAR) >= int(period_test_start)) & (F.col(TIME_VAR) <= int(period_test_end)), 0))
                .filter(F.col('TRAIN_IND').isNotNull()))
  
  df_rolling = df_rolling.cache()
  
  
  lag_models = get_lag_models_list(df_rolling, TIME_VAR, DYNAMIC_LAG_MODELS)
  print(f'Lag models to be run: {lag_models}')
  
  # run lag models training
  train_lag_models(df_data=df_rolling,
                   lag_models=lag_models,
                   cols_features_all=cols_features_all,
                   cols_utils=cols_utils,
                   col_target=col_target,
                   n_lags_to_keep=LAGS_TO_KEEP, is_filter_cases=IS_FILTER_CASES,
                   df_importances=df_importances,
                   stage1_models_list=stage1_models_list, stage1_cls=stage1_cls, stage1_params=stage1_params,
                   stage2_models_list=stage2_models_list, stage2_cls=stage2_cls, stage2_params=stage2_params,
                   train_schema=train_schema, predict_schema=predict_schema,
                   output_path_dbfs_tmp_table=output_path_dbfs_tmp_table,
                   is_forward_run=RUN_FORWARD,
                   n_concurrent_jobs=N_CONCURRENT_JOBS)
    
  df_rolling = df_rolling.unpersist(True)
  print(f'--- Period took: {np.round((time() - time_period_start) / 60, 2)}')

# COMMAND ----------

maxforcastUDF = udf(lambda z: get_maxforcast(z[0], z[1], dates), IntegerType())
forcastlagUDF = udf(lambda z: get_forcast_lag(z[0], z[1], dates), IntegerType())

aggregated_rolling_output_review = create_aggregated_output(output_path_dbfs_tmp_table, maxforcastUDF, forcastlagUDF, TIME_VAR, mrd_join_df, join_type='inner')

# cache & trigger
aggregated_rolling_output_review.cache()
aggregated_rolling_output_review.count()

aggregated_rolling_output_review.display()

# COMMAND ----------

# DBTITLE 1,Backtest model selection
#TODO: lqrz. Duplicated code in here.

if RUN_BACKTEST:
  
  # Generating list of dates to iterate through
  dates = [row[0] for row in aggregated_rolling_output_review.select('FCST_START_DATE').distinct().sort('FCST_START_DATE').collect()]
  # Creating placeholder for last element which will not participate in model selection
  dates = dates + [dates[-1] + 1]
  
  # Function to select next snapshot date
  def next_time_var(date, dates):
    date_index = dates.index(date)
    return dates[date_index+1]
  
  next_time_varUDF = udf(lambda z:next_time_var(z, dates), IntegerType())
  
  # Select best model per snapshot
  model_selection_df = aggregated_rolling_output_review.filter(F.col('sample') == 'OOS')
  
  # cache & trigger
  model_selection_df.cache()
  model_selection_df.count()
  
  if TARGET_VAR in model_selection_df.columns:
    actuals_to_use = TARGET_VAR
  elif TARGET_VAR + '_ORIG' in model_selection_df.columns:
    actuals_to_use = TARGET_VAR + '_ORIG'
  else:
    print('Must add historical actuals!!')

  ## Output to dictate model selection
  
  model_cols = [c for c in model_selection_df.columns if (c.endswith('_stage1') or c.endswith('_stage2')) and c != 'gbm_quantile_lower_stage2']
  best_model_df = (select_best_model(model_selection_df.filter(F.col('fcst_periods_fwd') == 1),
                                    model_cols,
                                    actuals_to_use,
                                    select_hier=BEST_MODEL_SELECTION + ['FCST_START_DATE'])
                   .withColumn('FCST_START_DATE', next_time_varUDF(F.col('FCST_START_DATE')))
                  )
  
  best_model_df.cache()
  best_model_df.count()

  # For null results, use the most frequent best model
  max_best_model = (best_model_df
                    .groupBy("best_model")
                    .count()
                    .orderBy(F.col("count").desc())
                    .collect()[0][0]
                    )
  
  # Merge it
  cols_to_keep = ['best_model', 'FCST_START_DATE'] + BEST_MODEL_SELECTION
  best_model_merge = best_model_df.select(cols_to_keep)
  
  best_model_review = (aggregated_rolling_output_review
                       .join(best_model_merge, on=BEST_MODEL_SELECTION + ['FCST_START_DATE'], how="left")
                       .fillna(max_best_model, subset="best_model")
                      )
  
  # cache & trigger
  best_model_review.cache()
  best_model_review.count()
  
  full_algos_list = [x for x in best_model_review.columns if '_stage' in x]
  
  final_prediction_pd_subset = best_model_review.select(['MODEL_ID', TIME_VAR, 'lag_period', 'best_model','FCST_START_DATE'] + full_algos_list)
  
  melted_df = (melt_df_pyspark(final_prediction_pd_subset,
                              id_vars=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model', 'FCST_START_DATE'],
                              value_vars=full_algos_list,
                              value_name='final_prediction_value')
               .filter(F.col('best_model') == F.col('variable'))
               .drop('variable')
              )
  
  final_prediction_pd = (best_model_review
                         .join(melted_df, on=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model', 'FCST_START_DATE'], how='left')
                        )
  
  # cache & trigger
  final_prediction_pd.cache()
  final_prediction_pd.count()
  
  # Check that everything has been built properly
  if final_prediction_pd.count() != aggregated_rolling_output_review.count(): raise Exception("Number of rows not matching")
  
  #Log unaggregated accuracies to Mlflow for experiment comparison
  accuracy_log = aggregate_data(model_selection_df
                                .select(BEST_MODEL_SELECTION + [CATEGORY_FIELD])
                                .join(best_model_df, on=BEST_MODEL_SELECTION),
                                [CATEGORY_FIELD], ["max_accuracy"], [F.mean])
  accuracy_dict = convertDFColumnsToDict(accuracy_log, CATEGORY_FIELD, "mean_max_accuracy")
  mlflow_tracker.log_metrics(accuracy_dict)

# COMMAND ----------

# DBTITLE 1,Save Train-Test Split Output
try:
  df_to_save = final_prediction_pd if RUN_BACKTEST else aggregated_rolling_output_review
  save_df_as_delta(df_to_save, DBO_FORECAST_TRAIN_TEST_SPLIT, enforce_schema=False)

  train_test_delta_info = load_delta_info(DBO_FORECAST_TRAIN_TEST_SPLIT)
  set_delta_retention(train_test_delta_info, '90 days')
  display(train_test_delta_info.history())

  ## Accuracy delta table version
  data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBO_FORECAST_TRAIN_TEST_SPLIT +"`)").collect()
  data_version = data_version[0][0]
  mlflow_tracker.log_param('Simple Accuracy Delta Version', data_version) 

except:
  print("Train-Test delta run not written")

# COMMAND ----------

# DBTITLE 1,Save Best Model
try:
  save_df_as_delta(best_model_df, DBO_BEST_MODEL_SELECTION, enforce_schema=False)
  delta_info = load_delta_info(DBO_BEST_MODEL_SELECTION)
  set_delta_retention(delta_info, '90 days')
  display(delta_info.history())
  
  # Save Delta Version
  data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBO_BEST_MODEL_SELECTION +"`)").collect()
  data_version = data_version[0][0]
  
  #Save accuracy information to Mlflow
  mlflow_tracker.log_param('BEST_MODEL_SELECTION', BEST_MODEL_SELECTION) #Global config, move up
  mlflow_tracker.log_param('DBO_BEST_MODEL_SELECTION_version', data_version) 

except:
  print("Best model selection delta run not written")

# COMMAND ----------

if RUN_BACKTEST:
  # remove from cache
  aggregated_rolling_output_review.unpersist(True)
  model_selection_df.unpersist(True)
  best_model_df.unpersist(True)
  best_model_review.unpersist(True)
  final_prediction_pd.unpersist(True)

# COMMAND ----------

# DBTITLE 1,Model Selection by Hierarchy Level
#TODO: lqrz. Duplicated code in here.

if RUN_FORWARD:
  print('Selecting OOS dataframe for our results')

  #Use model selection snapshot in case of rolling backtest
  model_selection_df = aggregated_rolling_output_review.filter(F.col('sample') == 'OOS')
  
  # cache & trigger
  model_selection_df.cache()
  model_selection_df.count()
  
  if TARGET_VAR in model_selection_df.columns:
    actuals_to_use = TARGET_VAR
  elif TARGET_VAR + '_ORIG' in model_selection_df.columns:
    actuals_to_use = TARGET_VAR + '_ORIG'
  else: print('Must add historical actuals!!')
    
  # compute best model across 4 different levels
  model_cols = [c for c in aggregated_rolling_output_review.columns if (c.endswith('_stage1') or c.endswith('_stage2')) and c != 'gbm_quantile_lower_stage2']
  best_model_df_level1 = select_best_model(model_selection_df, model_cols, actuals_to_use, select_hier=BEST_MODEL_SELECTION_level1)
  best_model_df_level2 = select_best_model(model_selection_df, model_cols, actuals_to_use, select_hier=BEST_MODEL_SELECTION_level2)
  best_model_df_level3 = select_best_model(model_selection_df, model_cols, actuals_to_use, select_hier=BEST_MODEL_SELECTION_level3)
  best_model_df_level4 = select_best_model(model_selection_df, model_cols, actuals_to_use, select_hier=BEST_MODEL_SELECTION_level4)

  best_model_df_level1.cache()
  best_model_df_level1.count()
  best_model_df_level2.cache()
  best_model_df_level2.count()
  best_model_df_level3.cache()
  best_model_df_level3.count()
  best_model_df_level4.cache()
  best_model_df_level4.count()
  
  # compute best model by: SUBBRND_SHRT_NM>BRND_NM>SRC_CTGY_1_NM
  # the best model is chosen based on SUBBRND_SHRT_NM, if null then by BRND_NM, if null then by SRC_CTGY_1_NM
  BEST_MODEL_SELECTION = list(set(BEST_MODEL_SELECTION_level1 + BEST_MODEL_SELECTION_level2 + BEST_MODEL_SELECTION_level3))
  best_model_merge = (aggregated_rolling_output_review
                      .select(BEST_MODEL_SELECTION)
                      .distinct()
                      .join(best_model_df_level1.select(BEST_MODEL_SELECTION_level1+['best_model']),
                            on=BEST_MODEL_SELECTION_level1, how='left')
                      .join(best_model_df_level2.select(BEST_MODEL_SELECTION_level2+['best_model']),
                            on=BEST_MODEL_SELECTION_level2, how='left')
                      .join(best_model_df_level3.select(BEST_MODEL_SELECTION_level3+['best_model']),
                            on=BEST_MODEL_SELECTION_level3, how='left')
                      .join(best_model_df_level4.select(BEST_MODEL_SELECTION_level4+['best_model']),
                            on=BEST_MODEL_SELECTION_level4, how='left')
                      .withColumn('best_model_F',
                                  F.coalesce( *[best_model_df_level1.best_model,
                                                best_model_df_level2.best_model,
                                                best_model_df_level3.best_model,
                                                best_model_df_level4.best_model
                                               ]))
                      .select(BEST_MODEL_SELECTION + ['best_model_F'])
                      .withColumnRenamed('best_model_F', 'best_model')
                     )

  # join best model with forcasting output
  best_model_review = aggregated_rolling_output_review.join(best_model_merge, on=BEST_MODEL_SELECTION, how='left')
  
  # cache & trigger
  best_model_review = best_model_review.cache()
  best_model_review.count()
  best_model_review.display()

  # select the best model(determined by backtesting) from forcast result
  full_algos_list = [x for x in best_model_review.columns if '_stage' in x]
  
  final_prediction_pd_subset = best_model_review.select(['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'] + full_algos_list)
  
  melted_df = (melt_df_pyspark(final_prediction_pd_subset,
                               id_vars=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'],
                               value_vars=full_algos_list,
                               value_name='final_prediction_value')
               .filter(F.col('best_model')== F.col('variable'))
               .drop('variable')
              )
  
  final_prediction_pd = best_model_review.join(melted_df, on=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'], how='left')
  
  # cache & trigger
  final_prediction_pd.cache()
  final_prediction_pd.count()
  
  final_prediction_pd.display()

# COMMAND ----------

# DBTITLE 1,Compute Interval based on prediction error
# TODO: lqrz. This is very expensive, performance shoulb be improved.

if RUN_FORWARD:
  # The limit is specified as %tile in config. dividing it by 5 to map to a decile_rank
  lower_limit_adjusted = int(LOWER_LIMIT/5)
  upper_limit_adjusted = int(UPPER_LIMIT/5)

  error_pd = final_prediction_pd.withColumn('error', F.abs(F.col('final_prediction_value') - F.col('CASES_ORIG')) / F.col('CASES_ORIG'))\
                                .filter(F.col('error').isNotNull())
  error_pd = error_pd.select(*error_pd.columns, 
                            F.ntile(20).over(Window.partitionBy(CONFIDENCE_LEVEL).orderBy('error'))
                                       .alias("decile_rank"))
  error_pd = error_pd.groupBy(*CONFIDENCE_LEVEL,'decile_rank').agg(F.min('error').alias('error'))
  error_pd = error_pd.filter((F.col('decile_rank') == lower_limit_adjusted) | (F.col('decile_rank') == upper_limit_adjusted))
  error_pd = error_pd.groupBy(CONFIDENCE_LEVEL).pivot("decile_rank").sum("error")
  
  error_pd = error_pd.cache()
  error_pd.count()
  error_pd.display()

# COMMAND ----------

if RUN_FORWARD:
  # Removing from cache (no longer used or overriden in the future).
  aggregated_rolling_output_review.unpersist(True)
  model_selection_df.unpersist(True)
  best_model_review.unpersist(True)
  final_prediction_pd.unpersist(True)

# COMMAND ----------

# MAGIC %md # FORWARD RUN

# COMMAND ----------

if RUN_FORWARD:
  ## Exit for egregious errors
  if len(intersect_two_lists([TARGET_VAR], mrd_df_future.columns)) == 0:
    dbutils.notebook.exit("Target variable not in data, Exiting notebook")
  else: print('Target Variable ({}) in dataframe!'.format(TARGET_VAR))

  if len(intersect_two_lists([TIME_VAR], mrd_df_future.columns)) == 0:
    dbutils.notebook.exit("Time variable not in data, Exiting notebook")
  else: print('Time Variable ({}) in dataframe!'.format(TIME_VAR))

# COMMAND ----------

# DBTITLE 1,RETRAINING: Modeling Effort - Loop For Each Dynamic-Lag Model
# TODO: Saving the models pickles crashes. Figure out whats the ideal way of handling model saving. 

## Only run the below loop in re-training
## Note - keeping 'RETRAIN' set to TRUE for now until alternative is working

if RUN_FORWARD:

  output_path_dbfs_tmp_table = get_tmp_table(TIME_VAR, path_base='dbfs:/mnt/adls/Tables/modelling')
  dbfs_tmp_tables_to_delete.append(output_path_dbfs_tmp_table)
  
  lag_models = sorted(get_lag_models_list(mrd_df_future, TIME_VAR, DYNAMIC_LAG_MODELS))

  # stack mrd by algorithm
  stacked_data = (stack_data_by_algorithm(mrd_df_future, training_algos)
                  .withColumn('FCST_START_DATE', F.lit(FORWARD_RANGE[0])))
  # cache & trigger
  stacked_data.cache()
  stacked_data.count()
  
  # run lag models training
  train_lag_models(df_data=stacked_data,
                   lag_models=lag_models,
                   cols_features_all=cols_features_all,
                   cols_utils=cols_utils,
                   col_target=col_target,
                   n_lags_to_keep=LAGS_TO_KEEP, is_filter_cases=IS_FILTER_CASES,
                   df_importances=df_importances,
                   stage1_models_list=stage1_models_list, stage1_cls=stage1_cls, stage1_params=stage1_params,
                   stage2_models_list=stage2_models_list, stage2_cls=stage2_cls, stage2_params=stage2_params,
                   train_schema=train_schema, predict_schema=predict_schema,
                   output_path_dbfs_tmp_table=output_path_dbfs_tmp_table,
                   is_forward_run=RUN_FORWARD,
                   n_concurrent_jobs=N_CONCURRENT_JOBS)
  
  print(f'--- Period took: {np.round((time() - time_period_start) / 60, 2)}')

# COMMAND ----------

if RUN_FORWARD:
  all_periods = mrd_df_future.select(TIME_VAR).distinct().rdd.map(lambda x: x[0]).collect()  
  maxforcastUDF = udf(lambda z: get_maxforcast(z[0], z[1], all_periods), IntegerType())
  forcastlagUDF = udf(lambda z: get_forcast_lag(z[0], z[1], all_periods), IntegerType())

  aggregated_future_output_review = create_aggregated_output(output_path_dbfs_tmp_table, maxforcastUDF, forcastlagUDF, TIME_VAR, mrd_join_df, join_type='left')

  # cache & trigger
  aggregated_future_output_review.cache()
  aggregated_future_output_review.count()

  aggregated_future_output_review.display()

# COMMAND ----------

# DBTITLE 1,Use "Best" Model Based on Hierarchy Selection Elements [long runtime]
#TODO: lqrz. Duplicated code in here.

if RUN_FORWARD:  
  # compute best model by: SUBBRND_SHRT_NM>BRND_NM>SRC_CTGY_1_NM
  # the best model is chosen based on SUBBRND_SHRT_NM, if null then by BRND_NM, if null then by SRC_CTGY_1_NM
  BEST_MODEL_SELECTION = list(set(BEST_MODEL_SELECTION_level1 + BEST_MODEL_SELECTION_level2 + BEST_MODEL_SELECTION_level3))
  
  best_model_merge = (aggregated_future_output_review
                      .select(BEST_MODEL_SELECTION)
                      .distinct()
                      .join(best_model_df_level1.select(BEST_MODEL_SELECTION_level1+['best_model']),
                            on=BEST_MODEL_SELECTION_level1, how='left')
                      .join(best_model_df_level2.select(BEST_MODEL_SELECTION_level2+['best_model']), 
                            on=BEST_MODEL_SELECTION_level2, how='left')
                      .join(best_model_df_level3.select(BEST_MODEL_SELECTION_level3+['best_model']), 
                            on=BEST_MODEL_SELECTION_level3, how='left')
                      .join(best_model_df_level4.select(BEST_MODEL_SELECTION_level4+['best_model']),
                            on=BEST_MODEL_SELECTION_level4, how='left')
                      .withColumn('best_model_F',
                                  F.coalesce(*[best_model_df_level1.best_model,
                                               best_model_df_level2.best_model,
                                               best_model_df_level3.best_model,
                                               best_model_df_level4.best_model
                                              ]))
                      .select(BEST_MODEL_SELECTION + ['best_model_F'])
                      .withColumnRenamed('best_model_F', 'best_model')
                     )
  
  # join best model with forcasting output
  best_model_review = aggregated_future_output_review.join(best_model_merge, on=BEST_MODEL_SELECTION, how='left')
  
  # cache & trigger
  best_model_review.cache()
  best_model_review.count()
  
  # select the best model(determined by backtesting) from forcast result
  full_algos_list = [x for x in best_model_review.columns if '_stage' in x]
  
  final_prediction_pd_subset = best_model_review.select(['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'] + full_algos_list)
  
  melted_df = (melt_df_pyspark(final_prediction_pd_subset,
                            id_vars=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'],
                            value_vars=full_algos_list,
                            value_name='final_prediction_value')
             .filter( F.col('best_model') == F.col('variable'))
             .drop('variable')
            )
  
  final_prediction_pd = best_model_review.join(melted_df, on=['MODEL_ID', TIME_VAR, 'lag_period', 'best_model'], how='left')
  
  # cache & trigger
  final_prediction_pd.cache()
  final_prediction_pd.count()
  
  final_prediction_pd.display()

# COMMAND ----------

# DBTITLE 1,Aggregate by Lagged Models (via User Config Flag) - Diagonal version
if RUN_FORWARD:
  
  if TARGET_VAR in model_selection_df.columns:
    actuals_to_use = TARGET_VAR

  elif TARGET_VAR + '_ORIG' in model_selection_df.columns:
    actuals_to_use = TARGET_VAR + '_ORIG'

  else: print('Must add historical actuals!!')
  
  
  #Obtain for which categories we are meant to use the max (coming from config)
  categ_max = fcst_categ_max[TIME_VAR]
  
  # Define parameters for maxing
  agg_dict = {'final_prediction_value':'max'}
  grouping_cols = ['MODEL_ID', TIME_VAR, actuals_to_use] + BEST_MODEL_SELECTION
  

  ## This aggregates predicted value for a MODEL ID and TIME across different lag models
  ## Review showed that this led to accuracy improvements 
  if (AGGREGATE_LAGS_TO_PREDICT):
    # Choose max/diagonal depending on config condition
    if len(categ_max) > 0:
      # First we filter the dataset where max is supposed to happen by selecting the filters indicated in the categ_max dictionary
      k = 0 # Control integer
      for item in categ_max:
        if k == 0:
          temp = final_prediction_pd.filter( F.col(item).isin(categ_max[item]) )
        else:
          temp = temp.filter( F.col(item).isin(categ_max[item]) )
        k = k + 1
      
      # Aggregate with max for temp dataset
      agg_final_prediction_pd = temp.groupBy(grouping_cols).agg(agg_dict).withColumnRenamed('max(final_prediction_value)', 'final_agg_prediction_value')
      # For the rest of model ids use the diagonal: we make an anti_join to keep only the MODEL_ID where the max has not been applied
      model_id_list = agg_final_prediction_pd.select("MODEL_ID").distinct()
      diag = final_prediction_pd.join(model_id_list, on = "MODEL_ID", how = "left_anti")
      # We concatenate the diagonal
      agg_final_prediction_pd = (agg_final_prediction_pd
                                 .union(diag
                                        .filter( F.col('lag_period') > col('fcst_periods_fwd'))
                                        .orderBy(F.col('MODEL_ID').desc(), F.col(TIME_VAR).desc(), F.col('lag_period').asc())
                                        .dropDuplicates(['MODEL_ID', TIME_VAR])
                                        .select(grouping_cols + ['final_prediction_value'])
                                        .distinct()
                                        .withColumnRenamed('final_prediction_value', 'final_agg_prediction_value')))
    else:
      # If we are meant to use only the diagonal, it goes into the else clause
      agg_final_prediction_pd = (final_prediction_pd
                                 .filter( F.col('lag_period') > F.col('fcst_periods_fwd'))
                                 .orderBy(F.col('MODEL_ID').desc(), F.col(TIME_VAR).desc(), F.col('lag_period').asc())
                                 .dropDuplicates(['MODEL_ID', TIME_VAR])
                                 .select(grouping_cols + ['final_prediction_value'])
                                 .distinct()
                                 .withColumnRenamed('final_prediction_value', 'final_agg_prediction_value'))
    
    # Make merge of final predictions
    final_prediction_pd = final_prediction_pd.join(agg_final_prediction_pd, on=grouping_cols)
  else:
    raise Exception("Aggregation is not happening")

# COMMAND ----------

# DBTITLE 1,Output Updates for PowerBI
if RUN_FORWARD:
  
  if 'week' in TIME_VAR.lower():
    final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('Weekly'))

  elif 'month' in TIME_VAR.lower():
    final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('Monthly'))

  else: final_prediction_pd = final_prediction_pd.withColumn('Demand_Flag', F.lit('User-Error'))

# COMMAND ----------

if RUN_FORWARD:
  
  def translate(mapping):
      def translate_(col):
          return mapping.get(col)
      return udf(translate_, StringType())

  from calendar import monthrange
  calendar_df_ref = calendar_df.withColumn('Month', F.substring('Month_Of_Year', 5, 2))
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

# DBTITLE 1,Compute CI for final prediction
if RUN_FORWARD:
  final_prediction_pd = final_prediction_pd.join(error_pd, on=CONFIDENCE_LEVEL, how='left')

  # get upper bound and lower bound
  final_prediction_pd = (final_prediction_pd
                         .withColumn('upper_bound',
                                     F.col('final_prediction_value') + (F.col('final_prediction_value') * F.col(str(lower_limit_adjusted))))
                         .withColumn('lower_bound',
                                     F.col('final_prediction_value') - (F.col('final_prediction_value') * F.col(str(upper_limit_adjusted)))))
  
  final_prediction_pd = final_prediction_pd.withColumn('lower_bound', F.when( F.col('lower_bound') < 0, 0).otherwise( F.col('lower_bound') ) )

# COMMAND ----------

# DBTITLE 1,Save Future Prediction Outputs
if RUN_FORWARD:
  
  try:
    save_df_as_delta(final_prediction_pd, DBO_FORECAST_FUTURE_PERIOD, enforce_schema=False)
    future_period_delta_info = load_delta_info(DBO_FORECAST_FUTURE_PERIOD)
    set_delta_retention(future_period_delta_info, '90 days')
    display(future_period_delta_info.history())
    
    # Save final forecast version number to Mlflow
    data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBO_FORECAST_FUTURE_PERIOD +"`)").collect()
    data_version = data_version[0][0]
    mlflow_tracker.log_param('DBO_FORECAST_FUTURE_PERIOD_version', data_version) 

  except:
    print("Future delta run not written")



if RUN_FORWARD:
  # remove from cache
  stacked_data.unpersist(True)
  aggregated_future_output_review.unpersist(True)
  best_model_df_level1.unpersist(True)
  best_model_df_level2.unpersist(True)
  best_model_df_level3.unpersist(True)
  best_model_df_level4.unpersist(True)
  best_model_review.unpersist(True)
  model_selection_df.unpersist(True)
  final_prediction_pd.unpersist(True)

# COMMAND ----------

# DBTITLE 1,Cleanup
# remove dbfs tmp tables
_ = [delete_tmp_table(p) for p in dbfs_tmp_tables_to_delete]

mlflow.end_run()

# COMMAND ----------

