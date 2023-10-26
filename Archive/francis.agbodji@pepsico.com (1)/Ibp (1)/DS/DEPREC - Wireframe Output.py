# Databricks notebook source
# MAGIC %md
# MAGIC ##08 - Wireframe output
# MAGIC 
# MAGIC Outputs tables to the silver layer for wireframe pickup

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

# DBTITLE 1,Load Data
weekly_forecast = load_delta(DBO_FORECAST_FUTURE_PERIOD, 6)  
monthly_forecast = load_delta(DBO_FORECAST_FUTURE_PERIOD, 5)

weekly_forecast = weekly_forecast.withColumnRenamed("Week_Of_Year","TIME_PERIOD")
monthly_forecast = monthly_forecast.withColumnRenamed("Month_Of_Year","TIME_PERIOD")

# COMMAND ----------

keep_columns = ['MODEL_ID',
 'TIME_PERIOD',
 'CASES_ORIG',
 'HRCHY_LVL_3_NM',
 'SUBBRND_SHRT_NM',
 'lag_period',
 'DMDUNIT',
 'FLVR_NM',
 'XYZ',
 'STAT_CLUSTER',
 'BRND_NM',
 'ABC',
 'PLANG_CUST_GRP_VAL',
 'SRC_CTGY_1_NM',
 'PLANG_PROD_KG_QTY',
 'PCK_CNTNR_SHRT_NM',
 'LOC',
 'PLANG_MTRL_EA_PER_CASE_CNT',
 'fcst_periods_fwd',
 'sample',
 'best_model',
 'final_prediction_value',
 'final_agg_prediction_value',
 'Demand_Flag',
 'Time_Duration']
weekly_forecast = weekly_forecast.select(keep_columns)
monthly_forecast = monthly_forecast.select(keep_columns)

final_forecast = weekly_forecast.union(monthly_forecast)


# COMMAND ----------

targetContainer = 'supplychain-ibp'
targetStorageAccount = 'cdodevadls2'
targetPath = 'silver/iberia/ibp-poc/DBO_FORECAST_FUTURE_PERIOD'
silverPath = get_target_path(targetContainer, targetStorageAccount, targetPath)

save_df_as_delta(final_forecast, silverPath, False)
delta_info = load_delta_info(silverPath)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

display(monthly_forecast)