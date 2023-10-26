# Databricks notebook source
# MAGIC %run ../src/parallel/utilities

# COMMAND ----------

TIME_VAR = 'Week_Of_Year' # weekly
lag_models = [3, 5, 7, 13, 17] # weekly

# TIME_VAR = 'Month_Of_Year' # montly
# lag_models = [5, 6, 7, 13, 16, 18] # monthly

n_concurrent_jobs = 5
n_lags_to_keep = 4


feature_thresholds = [.99]
# feature_thresholds = np.arange(.9, 1., .01).round(2)

# COMMAND ----------

OUTPUT_PATH_DELTA_BASE = 'dbfs:/mnt/adls/Tables/exploration/feature_importance' + ('_weekly' if TIME_VAR == 'Week_Of_Year' else '_monthly')

# COMMAND ----------

input_path_rolling_df = f'{OUTPUT_PATH_DELTA_BASE}/df_rolling_lightgbm'
rolling_df = load_delta(input_path_rolling_df)

# COMMAND ----------

import lightgbm
from typing import Dict, List, Callable
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType, BooleanType
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
import pandas as pd
from functools import partial
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from time import time
sns.set_style('whitegrid')
# import shutil
# from tempfile import NamedTemporaryFile
# from pathlib import Path

# COMMAND ----------

rolling_df_lgb = rolling_df.filter(F.col('train_func')=='lightGBM_model')
periods_all = (rolling_df_lgb
               .groupBy('TRAIN_IND', 'FCST_START_DATE')
               .agg(
                 F.min(TIME_VAR).alias('PERIOD_START'),
                 F.max(TIME_VAR).alias('PERIOD_END')
               ).collect())

# COMMAND ----------

(rolling_df_lgb
 .groupBy('FCST_START_DATE', 'TRAIN_IND')
 .agg(
   F.min(TIME_VAR).alias('PERIOD_START'),
   F.max(TIME_VAR).alias('PERIOD_END'),
   F.count(TIME_VAR).alias('COUNT'),
 ).display())

# COMMAND ----------

def save_pd_df_as_delta(df:pd.DataFrame, table_name:str):
  global OUTPUT_PATH_DELTA_BASE
  return save_df_as_delta(spark.createDataFrame(df), f'{OUTPUT_PATH_DELTA_BASE}/{table_name}')

# COMMAND ----------

def get_specific_train_test_period(period:int, periods:List):
  data = [p for p in periods if p.FCST_START_DATE == period]
  train = [(p.PERIOD_START, p.PERIOD_END) for p in data if p.TRAIN_IND == 1][0]
  test = [(p.PERIOD_START, p.PERIOD_END) for p in data if p.TRAIN_IND == 0][0]
  return list(train) + list(test)

# COMMAND ----------

def get_lags(model_lag:int, n_lags:int): return list(range(model_lag + 1, model_lag + n_lags + 1))

def filter_by_lag(cols:List[str], lag:int, n_lags_to_keep:int, is_filter_cases:bool, col_target:str) -> List[str]:
  lags = get_lags(lag, n_lags_to_keep)
  cols_lags = [c for c in cols if any(map(lambda x: c.lower().endswith(f'_lag{x}'), lags)) or '_lag' not in c.lower() and c.lower() != col_target.lower()]
  if not is_filter_cases:
    cols_cases = [c for c in cols if c.lower().startswith('cases_lag') and int(c.lower().replace('cases_lag', '')) > lag]
    cols_lags = list(set(cols_lags + cols_cases))
  return cols_lags

# COMMAND ----------

def _train_lgbm(df:pd.DataFrame, 
                 col_target:str,
                 cols_features:List[str],
                 cols_categorical:List[str],
                 cols_utils:List[str],
                params:Dict[str, any]
               ) -> any:
  assert(col_target not in cols_features + cols_utils)
  df_x = df.loc[:, cols_features + cols_utils]
  if len(cols_categorical) > 0: df_x = df_x.astype(dict([(c, 'category') for c in cols_categorical]))
  df_x_train = df_x.loc[df_x['TRAIN_IND']==1, :].drop(columns=cols_utils)
  df_x_test = df_x.loc[df_x['TRAIN_IND']==0, :].drop(columns=cols_utils)
  ds_y_train = df.loc[df['TRAIN_IND']==1, col_target]
  ds_y_test = df.loc[df['TRAIN_IND']==0, col_target]
  dtrain = lightgbm.Dataset(data=df_x_train, label=ds_y_train, feature_name='auto', categorical_feature='auto', free_raw_data=True)
  model = lightgbm.train(params, dtrain)
  y_preds = model.predict(df_x_test)
  mse = mean_squared_error(y_true=ds_y_test.values, y_pred=y_preds)
  mae = mean_absolute_error(y_true=ds_y_test.values, y_pred=y_preds)
  del df_x
  del df_x_train
  del df_x_test
  del ds_y_train
  del ds_y_test
  del dtrain
  del y_preds
  gc.collect()
  return model, mse, mae


def _get_lgbm_feature_importance(df:pd.DataFrame,
                                  lag_model:int,
                                  col_target:str,
                                  cols_features:List[str],
                                  cols_categorical:List[str],
                                  cols_utils:List[str],
                                  params:Dict[str, any]
                                ) -> any:
  
  forecast_start_date = df['FCST_START_DATE'].unique()[0]
  model, mse, mae = _train_lgbm(df, col_target, cols_features, cols_categorical, cols_utils, params)
#   mlflow.lightgbm.save_model(model, f'/dbfs/mnt/adls/models/lqrz_test/feature_selection/model_{lag_model}_{forecast_start_date}')
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
  df_importance['FORECAST_START_DATE'] = forecast_start_date
  df_importance['LAG_MODEL'] = lag_model
  df_importance['MSE'] = mse
  df_importance['MAE'] = mae
  del forecast_start_date
  del model
  del mse
  del mae
  del importance_list
  gc.collect()
  return df_importance
  

# defined in src/core/modeling
def _train_and_get_lgbm_metrics(df:pd.DataFrame,
                    lag_model:int,
                    col_target:str,
                    cols_features:List[str],
                    cols_categorical:List[str],
                    cols_utils:List[str],
                    params:Dict[str, any],
                    threshold:float
                  ) -> pd.DataFrame:
    
  forecast_start_date = df['FCST_START_DATE'].unique()[0]
  
  n_feat = len(cols_features)
  
  _, mse, mae = _train_lgbm(df, col_target, cols_features, cols_categorical, cols_utils, params)
  
  df_return = pd.DataFrame([[lag_model,
                             threshold,
                             n_feat,
                             forecast_start_date,
                             mse,
                             mae]], columns=['LAG_MODEL',
                                             'THRESHOLD',
                                             'N_FEAT',
                                             'FORECAST_START_DATE',
                                             'MSE',
                                             'MAE'])
  del forecast_start_date
  del n_feat
  del mse
  del mae
  gc.collect()
  return df_return

# COMMAND ----------

def run_lag_model_training(df:pd.DataFrame, lag_models:List[int], f_cols_features:Callable, cols_utils:List[str], col_target:str, forecast_start_dates:List[int], n_concurrent_jobs:int, lgbm_params:Dict[str, any], get_f:Callable, f_out_schema:any, threshold:float=None) -> pd.DataFrame:
  
#   return
  importance_dfs = []

  for lag_model in lag_models:
    
    time_start_lag_model = time()
    
#     get features for specific lag_model
    cols_features, cols_categorical = f_cols_features(lag_model=lag_model)
  
#     just in case
    if len(cols_categorical) > 0: assert(all(map(lambda x: x in cols_features, cols_categorical)))
    
#     do not include a "+ cols_categorical" separatedly in this list.
    cols_all = cols_features + cols_utils + [col_target]

    batch_size = np.min([len(forecast_start_dates), n_concurrent_jobs])
    
    for ix, i in enumerate(range(0, len(forecast_start_dates) - batch_size + 1, batch_size)):
      
      
      print(f'# lag_model: {lag_model} ix: {ix}')
      
      dates = forecast_start_dates[i : i + batch_size]

      pd_udf = get_f(lag_model=lag_model,
                      col_target=col_target,
                      cols_features=cols_features,
                      cols_categorical=cols_categorical,
                      cols_utils=cols_utils,
                      params=lgbm_params,
                      threshold=threshold)
      
      time_start_train = time()
      
      df_lag_results = (df
                        .select(cols_all)
                        .filter(F.col('FCST_START_DATE').isin(dates))
                        .groupBy('FCST_START_DATE')
                        .applyInPandas(pd_udf, schema=f_out_schema))
      
      importance_dfs.append(df_lag_results.toPandas())
      
      print(f'Training took: {np.round((time() - time_start_train) / 60, 2)}')
  
    print(f'Lag model took: {np.round((time() - time_start_lag_model) / 60, 2)}')
    
  df_return = pd.concat(importance_dfs)
  
  del cols_features
  del cols_categorical
  del cols_all
  del batch_size
  del dates
  del pd_udf
  del df_lag_results
  del importance_dfs
  gc.collect()
  
  return df_return

# COMMAND ----------

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'max_depth': 6,
    'num_leaves': 100,
    'learning_rate': 0.25,
    'min_gain_to_split': 0.02,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1
}


col_target = 'CASES'
cols_utils = ['train_func', 'TRAIN_IND', 'FCST_START_DATE']
cols_excluded = ['MODEL_ID']
cols_features = [c for c in rolling_df_lgb.columns if c not in [col_target] + cols_utils + cols_excluded]
cols_categorical = []
# path_output_model = 'dbfs:/mnt/adls/models/lqrz_test/feature_selection'


schema_importance = StructType([
  StructField('FORECAST_START_DATE', IntegerType()),
  StructField('LAG_MODEL', IntegerType()),
  StructField('FEAT', StringType()),
  StructField('IMPORTANCE_PERC', DoubleType()),
  StructField('MSE', DoubleType()),
  StructField('MAE', DoubleType())
])


forecast_start_dates = list(set([p.FCST_START_DATE for p in periods_all]))


def get_f_importance(lag_model:int, col_target:str, cols_features:List[str], cols_categorical:List[str], cols_utils:List[str], params:Dict[str, any], *args, **kwargs) -> Callable:
  return partial(_get_lgbm_feature_importance,
                        lag_model=lag_model,
                        col_target=col_target,
                        cols_features=cols_features,
                        cols_categorical=cols_categorical,
                        cols_utils=cols_utils,
                        params=params)


def get_lag_model_features_all(cols_features:List[str], lag_model:int, n_lags_to_keep:int, cols_categorical:List[str], col_target:str):
  features_all = filter_by_lag(cols_features, lag=lag_model, n_lags_to_keep=n_lags_to_keep, is_filter_cases=False, col_target=col_target)
  features_categorical = [c for c in cols_categorical if c in features_all]
  return features_all, features_categorical


f_get_lag_model_features_all = partial(get_lag_model_features_all, cols_features=cols_features, n_lags_to_keep=n_lags_to_keep, cols_categorical=cols_categorical, col_target=col_target)

df_baseline_results = run_lag_model_training(rolling_df_lgb, lag_models, f_get_lag_model_features_all, cols_utils, col_target, forecast_start_dates, n_concurrent_jobs, lgbm_params, get_f_importance, schema_importance)

save_pd_df_as_delta(df_baseline_results, 'df_baseline_results')

# COMMAND ----------

# df_baseline_results.loc[(df_baseline_results.LAG_MODEL==2) & (df_baseline_results.FORECAST_START_DATE==202116)].iloc[:20]

# COMMAND ----------

def get_baseline_metrics(df:pd.DataFrame, cols_metrics:List[str]) -> pd.DataFrame:
  return (df
            .loc[:, cols_metrics]
            .groupby(by=['LAG_MODEL', 'FORECAST_START_DATE'])
            .mean()
            .reset_index())


cols_metrics = ['LAG_MODEL', 'FORECAST_START_DATE', 'MSE', 'MAE']

df_baseline_metrics = get_baseline_metrics(df_baseline_results, cols_metrics)

save_pd_df_as_delta(df_baseline_metrics, 'df_baseline_metrics')

# COMMAND ----------

def get_threshold_top_features(df:pd.DataFrame, threshold:float) -> pd.DataFrame:
  cols_return = ['LAG_MODEL', 'THRESHOLD', 'N_FEATS', 'FEATS']
  df_selected = df.loc[df['IMPORTANCE_PERC'] < threshold]
  if df_selected.shape[0] == 0: return pd.DataFrame([], columns=cols_return)
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
  return df_results

def get_all_top_features(df:pd.DataFrame, feature_thresholds:List[float]) -> pd.DataFrame:
  return pd.concat([get_threshold_top_features(df, th) for th in feature_thresholds])


df_top_features = get_all_top_features(df_baseline_results, feature_thresholds)

save_pd_df_as_delta(df_top_features, 'df_top_features')

# COMMAND ----------

feature_mapping = (spark
 .read.option("header", "true")
 .option("delimiter", ",")
 .csv('dbfs:/FileStore/tables/weekly_mapping.csv')
 .select('feature', 'New_Driver')
 .withColumnRenamed('feature', 'FEATURE')
 .withColumnRenamed('New_Driver', 'DRIVER')
 .toPandas())

def get_driver_mapping_metrics(df, feature_mapping):
  return feature_mapping.groupby('DRIVER').apply(lambda x: pd.DataFrame(
    [[
      len(set(x.FEATURE.values).intersection(set(df.FEATS.iloc[0]))),
      x.FEATURE.shape[0],
      len(set(x.FEATURE.values).intersection(set(df.FEATS.iloc[0]))) / x.FEATURE.shape[0],
    ]],
    columns=['N_PRESENT', 'N_TOTAL', 'PERC']
                                                )).T
  
df_top_features.loc[:, ['LAG_MODEL', 'THRESHOLD', 'FEATS']].groupby(['LAG_MODEL', 'THRESHOLD']).apply(get_driver_mapping_metrics, feature_mapping)

# COMMAND ----------

schema_train = StructType([
  StructField('LAG_MODEL', IntegerType()),
  StructField('THRESHOLD', DoubleType()),
  StructField('N_FEAT', IntegerType()),
  StructField('FORECAST_START_DATE', IntegerType()),
  StructField('MSE', DoubleType()),
  StructField('MAE', DoubleType())
])

# lag_models defined above
# cols_categorical defined above
# cols_utils defined above
# col_target defined above
# forecast_start_dates defined above
# n_concurrent_jobs defined above
# lgbm_params defined above
# feature_thresholds defined above

def get_lag_model_threshold_feature(df_top_features:pd.DataFrame, lag_model:int, threshold:float, cols_categorical:List[str], col_target:str) -> List[str]:
  features_all = (df_top_features
                  .loc[(df_top_features['LAG_MODEL']==lag_model) & (df_top_features['THRESHOLD']==threshold), 'FEATS']
                  .iloc[0]
                  .tolist())
  assert(col_target not in features_all)
  features_categorical = [c for c in cols_categorical if c in features_all]
  return features_all, features_categorical



def get_f_train_and_get_lgbm_metrics(lag_model:int, col_target:str, cols_features:List[str], cols_categorical:List[str], cols_utils:List[str], params:Dict[str, any], threshold:float, *args, **kwargs) -> Callable:
  return partial(_train_and_get_lgbm_metrics,
                          lag_model=lag_model,
                          col_target=col_target,
                          cols_features=cols_features,
                          cols_categorical=cols_categorical,
                          cols_utils=cols_utils,
                          params=params,
                          threshold=threshold)


# hacky
threshold_metrics_dfs = []
for th in feature_thresholds:
  
  print(f'# -- Threshold:{th}')
  
#   get col features by threshold func
  f_get_lag_model_threshold_features = partial(get_lag_model_threshold_feature, df_top_features=df_top_features, threshold=th, cols_categorical=cols_categorical, col_target=col_target)
  
  df_threshold_metrics = run_lag_model_training(rolling_df_lgb, lag_models, f_get_lag_model_threshold_features, cols_utils, col_target, forecast_start_dates, n_concurrent_jobs, lgbm_params, get_f_train_and_get_lgbm_metrics, schema_train, threshold=th)
  
  threshold_metrics_dfs.append(df_threshold_metrics)

df_threshold_metrics = pd.concat(threshold_metrics_dfs)

save_pd_df_as_delta(df_threshold_metrics, 'df_threshold_metrics')

# COMMAND ----------

df_comparison = df_threshold_metrics.merge(df_baseline_metrics, on=['LAG_MODEL', 'FORECAST_START_DATE'], suffixes=['_THRESHOLD', '_BASELINE'])
df_comparison['MSE_DIFF'] = df_comparison['MSE_THRESHOLD'] - df_comparison['MSE_BASELINE']
df_comparison['MAE_DIFF'] = df_comparison['MAE_THRESHOLD'] - df_comparison['MAE_BASELINE']

save_pd_df_as_delta(df_comparison, 'df_comparison')

# COMMAND ----------

def plot_metric_comparison(df:pd.DataFrame, ax:any, is_plot_mae:bool, is_plot_mse:bool):
  assert(any([is_plot_mse, is_plot_mae]))
  lag_model = df['LAG_MODEL'].unique()[0]
  color_baseline = 'green'
  color_threshold = 'blue'
  if is_plot_mse:
    _ = sns.lineplot(x='THRESHOLD', y='MSE_BASELINE', data=df, ax=ax, label='MSE_BASELINE', color=color_baseline)
    _ = sns.lineplot(x='THRESHOLD', y='MSE_THRESHOLD', data=df, ax=ax, label='MSE_THRESHOLD', color=color_threshold)
  if is_plot_mae:
    _ = sns.lineplot(x='THRESHOLD', y='MAE_BASELINE', data=df, ax=ax, label='MAE_BASELINE', color=color_baseline)
    _ = sns.lineplot(x='THRESHOLD', y='MAE_THRESHOLD', data=df, ax=ax, label='MAE_THRESHOLD', color=color_threshold)
  _ = ax.set_ylabel('TEST SET METRIC')
  _ = ax.set_xlabel('FEAT IMPORTANCE THRESHOLD')
  

def plot_feature_count(df:pd.DataFrame, ax:any):
  _ = sns.barplot(x='THRESHOLD', y='N_FEAT', data=df, ax=ax, palette='Blues')
  _ = ax.set_ylabel('N FEATURES')
  _ = ax.set_xlabel('FEAT IMPORTANCE THRESHOLD')


for lag_model in df_comparison['LAG_MODEL'].unique():
  f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,5), sharex=False, sharey=False)
  f.suptitle(f'LAG_MODEL: {lag_model}')
  df_comparison_lag_model = df_comparison.loc[df_comparison['LAG_MODEL']==lag_model, :]
  plot_metric_comparison(df_comparison_lag_model, ax0, is_plot_mae=False, is_plot_mse=True)
  plot_metric_comparison(df_comparison_lag_model, ax1, is_plot_mae=True, is_plot_mse=False)
  plot_feature_count(df_comparison_lag_model, ax2)

# COMMAND ----------

# mlflow.lightgbm.save_model(model, '/dbfs/mnt/adls/models/lqrz_test/feature_selection/test')

# COMMAND ----------

# dbutils.fs.mkdirs("/mnt/adls/models/lqrz_test/feature_selection")
# dbutils.fs.rm("/mnt/adls/models/lqrz_test/", True)
display(dbutils.fs.ls(OUTPUT_PATH_DELTA_BASE))

# COMMAND ----------

# mlflow.lightgbm.load_model(path_output_model)

# COMMAND ----------

# DBTITLE 1,Inspect results
load_delta(f'{OUTPUT_PATH_DELTA_BASE}/df_top_features').display()
# load_delta(f'{OUTPUT_PATH_DELTA_BASE}/df_top_features').display()

# COMMAND ----------

