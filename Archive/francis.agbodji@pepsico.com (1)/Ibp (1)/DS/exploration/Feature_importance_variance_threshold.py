# Databricks notebook source
# MAGIC %run ../src/parallel/utilities

# COMMAND ----------

TIME_VAR = 'Week_Of_Year'

# COMMAND ----------

rolling_df = load_delta('dbfs:/mnt/adls/Tables/lqrz_test/rolling_df')

# COMMAND ----------

import lightgbm
from typing import Dict, List
from pyspark.sql.functions import col, min, max
from pyspark.sql.types import StructType, StructField, IntegerType, LongType, DoubleType, StringType, BooleanType
from pyspark.ml.feature import VectorAssembler, VarianceThresholdSelector, StandardScaler, MinMaxScaler
from pyspark.ml import Pipeline
import pandas as pd
from functools import partial
import numpy as np
import mlflow
# from time import time
# import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
# import shutil
# from tempfile import NamedTemporaryFile
# from pathlib import Path

# COMMAND ----------

rolling_df_lgb = rolling_df.filter(col('train_func')=='lightGBM_model')
periods_all = rolling_df_lgb.groupBy('TRAIN_IND', 'FCST_START_DATE').agg(min(TIME_VAR).alias('PERIOD_START'), max(TIME_VAR).alias('PERIOD_END')).collect()

# COMMAND ----------

def get_lags(model_lag:int, n_lags:int): return list(range(model_lag + 1, model_lag + n_lags + 1))

# COMMAND ----------

rolling_df_lgb.groupBy('FCST_START_DATE', 'TRAIN_IND').agg(min(TIME_VAR).alias('PERIOD_START'), max(TIME_VAR).alias('PERIOD_END')).display()

# COMMAND ----------

def get_specific_train_test_period(period:int, periods:List):
  data = [p for p in periods if p.FCST_START_DATE == period]
  train = [(p.PERIOD_START, p.PERIOD_END) for p in data if p.TRAIN_IND == 1][0]
  test = [(p.PERIOD_START, p.PERIOD_END) for p in data if p.TRAIN_IND == 0][0]
  return list(train) + list(test)

# COMMAND ----------

def get_selected_features(df:pd.DataFrame, cols:List[str], variance_threshold:float=.0, dtypes_numeric:List=[IntegerType, LongType, DoubleType]) -> List[str]:
  cols_numeric = [c.name for c in df.schema.fields if c.name in cols and any(map(lambda d: isinstance(c.dataType, d), dtypes_numeric))]
  col_output_vector_assembler = 'features'
  vector_assembler = VectorAssembler().setInputCols(cols_numeric).setOutputCol(col_output_vector_assembler)
  df_vectorised = vector_assembler.transform(df)
  col_output_scaler = f'{col_output_vector_assembler}_scaled'
  scaler = MinMaxScaler(inputCol=col_output_vector_assembler, outputCol=col_output_scaler)#, withStd=True, withMean=False)
  col_output_variance_selector = f'{col_output_scaler}_selected'
  variance_selector = VarianceThresholdSelector(varianceThreshold=variance_threshold, featuresCol=col_output_scaler, outputCol=col_output_variance_selector)
  pipeline = Pipeline(stages=[scaler, variance_selector])
  pipeline = pipeline.fit(df_vectorised)
  variance_selector_ixs = pipeline.stages[1].selectedFeatures
  cols_selected = list(np.array(cols_numeric)[variance_selector_ixs])
  return cols_numeric, cols_selected, pipeline, df_vectorised, variance_selector_ixs

# COMMAND ----------

for forecast_start_date in list(set([p.FCST_START_DATE for p in periods_all]))[:1]:
  period_start_train, period_end_train, period_start_test, period_end_test = get_specific_train_test_period(period=forecast_start_date, periods=periods_all)
  for lag in [2]:
    for is_feat_selection in [True]:
      for variance_threshold in [.2]:
#         period_train_start, period_train_end, forecast_start_date = period_train
#         period_test_start, period_test_end, _ = period_test
        df_data = rolling_df_lgb.filter(col('FCST_START_DATE') == forecast_start_date)
      #   df_train = df_data.filter(col('TRAIN_IND') == 1)
      #   df_test = df_data.filter(col('TRAIN_IND') == 0)
      #   df_train_output = df_train.groupBy('TRAIN_IND').applyInPandas(f_train, schema=schema_train)
        cols_lag = filter_by_lag(cols_features, lag=lag, n_lags_to_keep=n_lags_to_keep)
        cols_numeric, cols_selected, pipeline, df_vectorised, variance_selector_ixs = get_selected_features(df=df_data, cols=cols_lag, variance_threshold=variance_threshold) if is_feat_selection else cols_lag

# COMMAND ----------

variance_selector_ixs

# COMMAND ----------

cols_numeric

# COMMAND ----------

cols_selected

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, DoubleType

def split_array_to_list(col):
    def to_list(v):
        return v.toArray().tolist()
    return F.udf(to_list, ArrayType(DoubleType()))(col)

# df_vectorised.select(split_array_to_list(F.col("features")).alias("split_features"))\
#     .select([F.col("split_features")[i] for i in range(139)]).describe().display()

pipeline.transform(df_vectorised)\
.select(split_array_to_list(F.col("features_scaled")).alias("split_features"))\
    .select([F.col("split_features")[i].alias(c) for c, i in zip(cols_numeric, range(len(cols_numeric)))]).describe().display()

# COMMAND ----------

ds_del = df_data.select('Life_expectancy_at_birth_total_years_').toPandas()

# COMMAND ----------

def filter_by_lag(cols:List[str], lag:int, n_lags_to_keep:int) -> List[str]:
  lags = get_lags(lag, n_lags_to_keep)
  return [c for c in cols if any(map(lambda x: c.lower().endswith(f'_lag{x}'), lags)) or '_lag' not in c.lower()]

# COMMAND ----------

# defined in src/core/modeling
def train_lightGBM(df:pd.DataFrame,
                    forecast_start_date:int,
                    period_start_train:int,
                    period_end_train:int,
                    period_start_test:int,
                    period_end_test:int,
                    lag:int,
                    is_feat_selection:bool,
                    variance_threshold:float,
                   params:Dict[str, any],
                   col_target:str,
                   cols_features:List[str],
                   cols_categorical:List[str],
                   cols_utils:List[str],
                   perc_feats_of_total:float,
#                    description:str,
#                    path_output_model:str,
                   *args, **kwargs) -> pd.DataFrame:
  
  n_features_total, n_feature_categorical = len(cols_features), len(cols_categorical)
  
  df_x = df.loc[:, cols_features + cols_utils]
  if len(cols_categorical) > 0: df_x = df_x.astype(dict([(c, 'category') for c in cols_categorical]))
  df_x_train = df_x.loc[df_x['TRAIN_IND']==1, :].drop(columns=cols_utils)
  df_x_test = df_x.loc[df_x['TRAIN_IND']==0, :].drop(columns=cols_utils)
  
  ds_y_train = df.loc[df['TRAIN_IND']==1, col_target]
  ds_y_test = df.loc[df['TRAIN_IND']==0, col_target]
  
  dtrain = lightgbm.Dataset(data=df_x_train, label=ds_y_train, feature_name='auto', categorical_feature=cols_categorical, free_raw_data=True)
#   dtest = lightgbm.Dataset(data=df_x_test, label=ds_y_test, feature_name='auto', categorical_feature=cols_categorical, free_raw_data=True)
  
#   model = None
  model = lightgbm.train(params, dtrain, *args, **kwargs)
#   model_fs_name = f'{path_output_model.replace("dbfs:", "/dbfs")}/{int(time() * 1000)}'
#   mlflow.lightgbm.save_model(model, model_fs_name)
#   model_str = pickle.dumps(model).decode('latin-1')
  y_preds = model.predict(df_x_test)
  mse = mean_squared_error(y_true=ds_y_test.values, y_pred=y_preds)
  mae = mean_absolute_error(y_true=ds_y_test.values, y_pred=y_preds)
  df_return = pd.DataFrame([[
                            forecast_start_date,
                            period_start_train,
                            period_end_train,
                            period_start_test,
                            period_end_test,
                            lag,
                            is_feat_selection,
                            variance_threshold,
                             n_features_total,
                            perc_feats_of_total,
                             n_feature_categorical,
                             mse,
                             mae]], columns=['FORECAST_START_DATE',
                                             'PERIOD_START_TRAIN',
                                             'PERIOD_END_TRAIN',
                                             'PERIOD_START_TEST',
                                             'PERIOD_END_TEST',
                                             'LAG',
                                             'IS_FEAT_SELECTION',
                                             'VARIANCE_THRESHOLD',
                                             'N_FEATS',
                                             'PERC_FEATS_OF_TOTAL',
                                             'N_FEAT_CATEGORICAL',
                                             'MSE',
                                             'MAE'])
#   df_return = pd.DataFrame([['', 1., 1.]], columns=['MODEL', 'MSE', 'MAE'])
  return df_return

# COMMAND ----------

# ADD LAGS TO KEEP AND TRAIN A SPECIFIC LAG MODEL

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
cols_features = [c for c in rolling_df_lgb.columns if c not in [col_target] + cols_utils]
cols_categorical = ['MODEL_ID']
# path_output_model = 'dbfs:/mnt/adls/models/lqrz_test/feature_selection'

n_lags_to_keep = 4

lags = [2, 6, 12]
variance_thresholds = [.0, .1, .7, 1.]
feat_selections = [False, True]
lags = [2]
variance_thresholds = [.0, .1, .2, .3, .4]
feat_selections = [True]

# schema_train = StructType([StructField('MODEL_STR', StringType())])
schema_train = StructType([
  StructField('FORECAST_START_DATE', IntegerType()),
  StructField('PERIOD_START_TRAIN', IntegerType()),
  StructField('PERIOD_END_TRAIN', IntegerType()),
  StructField('PERIOD_START_TEST', IntegerType()),
  StructField('PERIOD_END_TEST', IntegerType()),
  StructField('LAG', IntegerType()),
  StructField('IS_FEAT_SELECTION', BooleanType()),
  StructField('VARIANCE_THRESHOLD', DoubleType()),
  StructField('N_FEATS', IntegerType()),
  StructField('PERC_FEATS_OF_TOTAL', DoubleType()),
  StructField('N_FEAT_CATEGORICAL', IntegerType()),
  StructField('MSE', DoubleType()),
  StructField('MAE', DoubleType())
])

df_train_outputs = []

# for period_train, period_test in train_test_periods:
# for forecast_start_date in set([p.FCST_START_DATE for p in periods_all]):
for forecast_start_date in list(set([p.FCST_START_DATE for p in periods_all]))[:1]:
  period_start_train, period_end_train, period_start_test, period_end_test = get_specific_train_test_period(period=forecast_start_date, periods=periods_all)
  for lag in lags:
    for is_feat_selection in feat_selections:
      for variance_threshold in variance_thresholds:
#         period_train_start, period_train_end, forecast_start_date = period_train
#         period_test_start, period_test_end, _ = period_test
        df_data = rolling_df_lgb.filter(col('FCST_START_DATE') == forecast_start_date)
      #   df_train = df_data.filter(col('TRAIN_IND') == 1)
      #   df_test = df_data.filter(col('TRAIN_IND') == 0)
      #   df_train_output = df_train.groupBy('TRAIN_IND').applyInPandas(f_train, schema=schema_train)
        cols_lag = filter_by_lag(cols_features, lag=lag, n_lags_to_keep=n_lags_to_keep)
        _, cols_selected, _, _, _ = get_selected_features(df=df_data, cols=cols_lag, variance_threshold=variance_threshold) if is_feat_selection else cols_lag
        cols_selected_categorical = [c for c in cols_categorical if c in cols_selected]
        
        perc_feats_of_total = len(cols_selected) / len(cols_lag)
        
        description = f'fcst_{forecast_start_date}_lags_{lag}_select_{is_feat_selection}_thres_{variance_threshold}'
        print(description)
#         print(len(cols_selected), len(cols_selected_categorical))
#         print(cols_selected)
#         print(cols_selected_categorical)
        f_train = partial(train_lightGBM,
                          forecast_start_date=forecast_start_date,
                          period_start_train=period_start_train,
                          period_end_train=period_end_train,
                          period_start_test=period_start_test,
                          period_end_test=period_end_test,
                          lag=lag,
                          is_feat_selection=is_feat_selection,
                          variance_threshold=variance_threshold,
                          col_target=col_target,
                          params=lgbm_params,
                          cols_features=cols_selected,
                          cols_categorical=cols_selected_categorical,
                          cols_utils=cols_utils,
                         perc_feats_of_total=perc_feats_of_total)
#                           , description=description,
#                           path_output_model=path_output_model)
        df_train_outputs.append(df_data.groupBy('FCST_START_DATE').applyInPandas(f_train, schema=schema_train).toPandas())

df_results = pd.concat(df_train_outputs)
df_results

# COMMAND ----------

# DBTITLE 1,Save results as delta
# save_df_as_delta(spark.createDataFrame(df_results), 'dbfs:/mnt/adls/Tables/lqrz_test/feat_selection_variance_results')

# COMMAND ----------

# dbutils.fs.mkdirs("/mnt/adls/models/lqrz_test/feature_selection")
# dbutils.fs.rm("/mnt/adls/models/lqrz_test/feature_selection", True)
# display(dbutils.fs.ls("dbfs:/mnt/adls/models/lqrz_test/feature_selection/"))

# COMMAND ----------

# mlflow.lightgbm.load_model(path_output_model)