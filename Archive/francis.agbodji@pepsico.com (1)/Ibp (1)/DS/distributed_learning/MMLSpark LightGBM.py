# Databricks notebook source
from mmlspark.lightgbm import LightGBMRegressor
from pyspark.ml.feature import VectorAssembler
import pyspark.sql.functions as F
import pandas as pd

# COMMAND ----------

def load_delta(path, version=None):
    """Loads delta table as Spark Dataframe"""
    if version is None:
      latest_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + path +"`)").collect()
      df = spark.read.format("delta").option("versionAsOf", latest_version[0][0]).load(path)
    else:
      df = spark.read.format("delta").option("versionAsOf", version).load(path)
    return(df)

# COMMAND ----------

# LAUTARO_TABLE = "dbfs:/mnt/adls/Tables/lqrz_test/201842_202109_202137"
LAUTARO_TABLE = "dbfs:/mnt/adls/Tables/lqrz_test/201842_202120_202137"

# save_df_as_delta(stage1_df, LAUTARO_TABLE, enforce_schema=False)
stage1_df = load_delta(LAUTARO_TABLE)

df_data = stage1_df.filter( F.col('FCST_START_DATE') == 202120)

# COMMAND ----------

col_time = 'Week_Of_Year'
col_target = 'CASES'
cols_utils = ['train_func', 'TRAIN_IND', 'FCST_START_DATE', 'MODEL_ID']
cols_features = [c for c in df_data.columns if c not in [col_time] + [col_target] + cols_utils]

# COMMAND ----------

featurizer = VectorAssembler(
    inputCols=cols_features,
    outputCol='features'
)

df_data_transformed = featurizer.transform(df_data)
df_data_transformed.cache()
df_data_transformed.count()

# COMMAND ----------

lgbm_params = {
    'boostingType': 'gbdt',
    'objective': 'regression',
#     'metric': {'l2', 'l1'},\
    'maxDepth': 6,
    'numLeaves': 100,
    'learningRate': 0.25,
#     'minGainToSplit': 0.02,
    'featureFraction': 0.65,
    'baggingFraction': 0.85,
    'baggingFreq': 5,
#     'verbose': -1
}

model = LightGBMRegressor(featuresCol='features', labelCol=col_target, **lgbm_params).fit(df_data_transformed)

# COMMAND ----------

df_preds = model.transform(df_data_transformed)
display(df_preds)

# COMMAND ----------

df_importances = pd.DataFrame(list(zip(cols_features, model.getFeatureImportances())), columns=['FEATURE', 'IMPORTANCE']).sort_values(by='IMPORTANCE', ascending=False)

df_importances

# COMMAND ----------

