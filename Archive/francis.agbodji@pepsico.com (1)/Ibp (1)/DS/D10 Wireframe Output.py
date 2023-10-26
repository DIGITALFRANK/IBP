# Databricks notebook source
# MAGIC %md
# MAGIC #10 - Wireframe output
# MAGIC 
# MAGIC Outputs tables to the silver layer for wireframe pickup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instantiate with Notebook Imports

# COMMAND ----------

# DBTITLE 0,Instantiate with NoImportstebook 
# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Load Data

# COMMAND ----------

weekly_forecast = load_delta(DBO_FORECAST_FUTURE_PERIOD, 37)  #Palaash to update
monthly_forecast = load_delta(DBO_FORECAST_FUTURE_PERIOD, 36) #10

weekly_forecast = weekly_forecast.withColumnRenamed("Week_Of_Year","TIME_PERIOD")
weekly_forecast = weekly_forecast.withColumnRenamed("Week_start_date","START_DATE")
monthly_forecast = monthly_forecast.withColumnRenamed("Month_Of_Year","TIME_PERIOD")
monthly_forecast = monthly_forecast.withColumnRenamed("Month_start_Date","START_DATE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter Output
# MAGIC Reduce to needed output for recieving apps 

# COMMAND ----------

print("pre filter ",weekly_forecast.count())
agg_window  = Window.partitionBy('MODEL_ID','FCST_START_DATE','TIME_PERIOD')
weekly_forecast=weekly_forecast.withColumn("min_lag_period", min(col("lag_period")).over(agg_window))
weekly_forecast=weekly_forecast.withColumn('min_lag_unique',F.when(weekly_forecast.min_lag_period == weekly_forecast.lag_period, 1).otherwise(0)).drop('min_lag_period')
weekly_forecast=weekly_forecast.filter(col('min_lag_unique')==1).drop('min_lag_unique')
print("post filter ",weekly_forecast.count())

# COMMAND ----------

print("pre filter ",monthly_forecast.count())
agg_window  = Window.partitionBy('MODEL_ID','FCST_START_DATE','TIME_PERIOD')
monthly_forecast=monthly_forecast.withColumn("min_lag_period", min(col("lag_period")).over(agg_window))
monthly_forecast=monthly_forecast.withColumn('min_lag_unique',F.when(monthly_forecast.min_lag_period == monthly_forecast.lag_period, 1).otherwise(0)).drop('min_lag_period')
monthly_forecast=monthly_forecast.filter(col('min_lag_unique')==1).drop('min_lag_unique')
print("post filter ",monthly_forecast.count())

# COMMAND ----------

#colp_dict:={'needed column from DB':'associated silver layer name'}
col_dict={'MODEL_ID': 'modl_id',
          'TIME_PERIOD': 'tm_prd',
          'START_DATE': 'strt_dt',
          'CASES_ORIG': 'cases_orignl',
          'HRCHY_LVL_3_NM' : 'hrchy_lvl_3_nm',
          'SUBBRND_SHRT_NM' : 'subbrnd_shrt_nm',
          'lag_period': 'lag_prd',
          'DMDUNIT' : 'prod_cd',
          'FLVR_NM' : 'flvr_nm',
          'XYZ':'xyz',
          'STAT_CLUSTER': 'sttstc_clstr',
          'BRND_NM' : 'brnd_nm',
          'ABC':'abc',
          'PLANG_CUST_GRP_VAL': 'cust_grp',
          'SRC_CTGY_1_NM':'stc_ctgy_1_am',
          'PLANG_PROD_KG_QTY': 'prod_kg_qty',
          'PCK_CNTNR_SHRT_NM': 'pck_cntr_shrt_nm',
          'LOC': 'loc',
          'PLANG_MTRL_EA_PER_CASE_CNT': 'mtrl_ea_per_case_cnt',
          'fcst_periods_fwd': 'fcst_prds_fwd',
          'sample': 'smpl',
          'best_model': 'best_modl',
          'final_prediction_value': 'fnl_prdctn_val',
          'final_agg_prediction_value': 'fnl_aggrd_val',
          'Demand_Flag': 'dmnd_flg',
          'Time_Duration':'tm_durtn',
          'FCST_START_DATE':'fcst_start_date'
          }

# COMMAND ----------

keep_columns = []
for DB_col in col_dict:
  keep_columns.append(DB_col)


weekly_forecast = weekly_forecast.select(keep_columns)
monthly_forecast = monthly_forecast.select(keep_columns)

final_forecast = weekly_forecast.union(monthly_forecast)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rename columms
# MAGIC For expected silver layer payload

# COMMAND ----------

for DB_col in col_dict:
  final_forecast=final_forecast.withColumnRenamed(DB_col,col_dict[DB_col])

# COMMAND ----------

# %run /Ibp/DE/ntbk_ibp_poc_adls_cred (credentials works for all clusters)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output

# COMMAND ----------

targetContainer = 'supplychain-ibp'
targetStorageAccount = 'cdodevadls2'
targetPath = 'silver/iberia/ibp-poc/DBO_FORECAST_FUTURE_PERIOD/'
silverPath = get_target_path(targetContainer, targetStorageAccount, targetPath)

save_df_as_delta(final_forecast, silverPath, False)
delta_info = load_delta_info(silverPath)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())