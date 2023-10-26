# Databricks notebook source
# MAGIC %md
# MAGIC ##07 - Hyperparameter Tuning
# MAGIC 
# MAGIC This script uses Bayesian Optimization (and Gridsearch) to tune hyperparameters of certain algorithms for Stage1 and Stage2 models.  
# MAGIC * The search is performed individually by segment (HYPERPARAM_LEVEL) and leverages pyspark to parallelize segment search across workers.  If no segment is given, a dummy segment is developed to run one search for the entire dataset.  
# MAGIC * The optimal parameters are stored as dictionary yaml files to mlflow and picked up in the modeling script.

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql.functions import udf
from typing import Iterable
import json
import mlflow
import pyspark.sql.functions as F
import pickle

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

#GLOBALS
# TODO: Push this to config
HYPERPARAM_LEVEL = None
STAGE_1_ALGOS = ["gbm_quantile_model","catboost_model","lightGBM_model", "rforest_model"] 

if TIME_VAR == "Month_Of_Year":
  DBA_MRD_version =  6
  PARTITION = 202009
  NUM_FWD_FRCST = 6
  LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] 
else:
  DBA_MRD_version =  3
  PARTITION = 202114
  NUM_FWD_FRCST = 4
  LAG_PERIODS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 52, 104] 
  
## Allows user to control 'volume' of retained lag values - cut down width and noise of FEATURES that are lagged
## Eg, if this value = 4 then Lag6 model will retain features lagged 7, 8, 9, 10 (ie, the 4 closest periods we can use)
LAGS_TO_KEEP = 4

# COMMAND ----------

#Check configurations exist for this script
required_configs = [STAGE_1_ALGOS, HYPERPARAM_LEVEL]
print(json.dumps(required_configs, indent=4))
if required_configs.count(None) >1 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Setup Mlflow
#For experiment tracking
try:
  mlflow.set_experiment("/Ibp/DS/Experiments/PEP Hyperparam")
  mlflow.start_run()
  experiment = mlflow.get_experiment_by_name("/Ibp/DS/Experiments/PEP Hyperparam")
  print("Experiment_id: {}".format(experiment.experiment_id))
  print("Artifact Location: {}".format(experiment.artifact_location))
  print("Tags: {}".format(experiment.tags))
  print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
except:
  print("Mlflow run EXCEPTION")

# COMMAND ----------

# Log configs
mlflow.log_param('TARGET_VAR',TARGET_VAR)
mlflow.log_param('HYPERPARAM_LEVEL', HYPERPARAM_LEVEL) 
mlflow.log_param('STAGE_1_ALGOS', STAGE_1_ALGOS) 
mlflow.log_dict(STAGE_1_GRIDS, "STAGE_1_GRIDS.yml")

# COMMAND ----------

# DBTITLE 1,Load Data
try:
  mrd = load_delta(DBA_MRD, DBA_MRD_version)
  delta_info = load_delta_info(DBA_MRD)
  display(delta_info.history())
except:
  dbutils.notebook.exit("DBA_MRD load failed, Exiting notebook")
  
#Create dummy segment for "one model" if segment doesn't exist within data
if HYPERPARAM_LEVEL == None or len(intersect_two_lists(HYPERPARAM_LEVEL, mrd.columns))==0 :
  mrd = mrd.withColumn("one_model_dummy_seg",(F.lit(1)))
  HYPERPARAM_LEVEL = ["one_model_dummy_seg"]
'''  
elif len(intersect_two_lists(HYPERPARAM_LEVEL, mrd.columns))==0:
  mrd = mrd.withColumn("one_model_dummy_seg",(F.lit(1)))
  HYPERPARAM_LEVEL = ["one_model_dummy_seg"]
'''  
print(TARGET_VAR)
print(HYPERPARAM_LEVEL)

# COMMAND ----------

# getting the holdout end date
all_months = mrd.select(TIME_VAR).distinct().rdd.map(lambda r: r[0]).collect()
houldout_end = [x for x in sorted(all_months) if x>=PARTITION][NUM_FWD_FRCST]

# selecting the backtesting partition for parameter tuning
mrd = mrd.filter(F.col('FCST_START_DATE')==PARTITION)
mrd = mrd.filter(F.col(TIME_VAR)<houldout_end)

# COMMAND ----------

#Moved to cmd 12
#print(DBA_MRD)
#load_delta_info(DBA_MRD).history().display()

# COMMAND ----------

# adjusting the lag features
# making sure we use closest 'n' lags for training, defined by LAGS_TO_KEEP
selected_lagged_periods = set([lag for lag in LAG_PERIODS if lag > NUM_FWD_FRCST and lag<=(NUM_FWD_FRCST+LAGS_TO_KEEP)])
lags_to_drop = set(LAG_PERIODS) - selected_lagged_periods

filter_text_list = ['_lag' + str(each_dropped_lag) for each_dropped_lag in lags_to_drop]
cols_to_drop = [col_nm for col_nm in mrd.columns if col_nm.endswith(tuple(filter_text_list)) == True]
cols_to_keep = [col_nm for col_nm in mrd.columns if col_nm.endswith(tuple(filter_text_list)) == False]

## Subset to appropriate lag columns to prevent data leakage
mrd = mrd[cols_to_keep]

# COMMAND ----------

mrd.cache()
mrd.count()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Stage 1 Hyperparameter Tuning (Bayesian)
#Setup information for stage 1 hyperparameter tuning
#Schema for pandas udf
def set_up_tunning(HYPERPARAM_LEVEL:list, mrd:F.DataFrame):
  auto_schema = mrd.select(HYPERPARAM_LEVEL).limit(1)
  auto_schema = auto_schema.withColumn('hyper_pick', F.lit('StringType'))
  auto_schema = auto_schema.schema

  #Model information class
  model_info_dict = dict(
      group_level        = HYPERPARAM_LEVEL,
      start_week         = PARTITION,
      holdout_duration   = NUM_FWD_FRCST,
      parameter_grid     = None,
      error_metric_func  = calc_RMSE,
      pickle_encoding    = 'latin-1',
      algo_func          = bayesian_opt_xgb_params

  )
  return ParallelBayesHyperparam(**model_info_dict), auto_schema

# COMMAND ----------

#Update pandas udf parameters
#parallel_stage1_hyperparam is in -> /src/load_src_parallel -> /parallel/hyperparameter -> hyperparameter
#STAGE_1_MODELS is in /src/config
def hyperparam_tune_one_model(this_model:str, result_master:list, mrd, HYPERPARAM_LEVEL, get_string):
  print("Tuning: ", this_model)
  hyperparam_cls, auto_schema = set_up_tunning(HYPERPARAM_LEVEL, mrd)

  #Update parameter grid and algorithm function name for this algorithm
  hyperparam_cls.set_param(**{'parameter_grid' : STAGE_1_GRIDS.get(this_model)})
  hyperparam_cls.set_param(**{'algo_func' : STAGE_1_FUNCS.get(this_model)})
  
  @F.pandas_udf(auto_schema, F.PandasUDFType.GROUPED_MAP)
  def parallel_hyperparam_udf(data):
    return parallel_stage1_hyperparam(data, hyperparam_cls, **STAGE_1_MODELS.get(this_model))
  
  #Run hyper parameter tuning
  this_dict_df = mrd.groupBy(HYPERPARAM_LEVEL).apply(parallel_hyperparam_udf)
  this_dict_df = this_dict_df.withColumn('params', get_string('hyper_pick')).withColumn('Model', F.lit(this_model))
  this_dict_df.cache()
  this_dict_df.count()
  return this_dict_df, hyperparam_cls

# COMMAND ----------

def hyperparameter_loop(encoding:str, list_algorithms:list, mrd, HYPERPARAM_LEVEL):
  get_string = udf(lambda x: str(pickle.loads(x[:len(x)].encode(encoding))), F.StringType())
  result_master = []
  
  #Loop through algos, parallelize across segments  
  for this_model in list_algorithms:
    this_dict_df, hyperparam_cls = hyperparam_tune_one_model(this_model, result_master, mrd, HYPERPARAM_LEVEL, get_string)
    result_master.append(this_dict_df)

  print("combining all params")
  params_final = reduce(DataFrame.union, result_master)
  
  #TODO check why they only take the last one hyperparam_cls
  params_final = params_final.select(['Model'] + hyperparam_cls.group_level + ['params'])
  return params_final

# COMMAND ----------

params_final = hyperparameter_loop('latin-1', STAGE_1_ALGOS, mrd, HYPERPARAM_LEVEL)

# COMMAND ----------

# saving to mlflow
params_final.toPandas().to_csv('/dbfs/FileStore/tables/temp/parameters.csv', index=False)
mlflow.log_artifact('/dbfs/FileStore/tables/temp/parameters.csv')

# COMMAND ----------

save_df_as_delta(params_final, DBO_HYPERPARAMATER, enforce_schema=False)
test_load = load_delta_info(DBO_HYPERPARAMATER)
set_delta_retention(test_load, '90 days')
display(test_load.history())

#Accuracy delta table version
data_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + DBO_HYPERPARAMATER +"`)").collect()
data_version = data_version[0][0]
mlflow.log_param('Hyperparameter version',data_version) 

# end run
mlflow.end_run()

# COMMAND ----------

display(params_final)
mrd.unpersist(True)