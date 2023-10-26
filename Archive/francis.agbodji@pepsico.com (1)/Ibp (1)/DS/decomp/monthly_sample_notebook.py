# Databricks notebook source
# MAGIC %run ../src/libraries

# COMMAND ----------

# MAGIC %run ../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./shap_decompostion

# COMMAND ----------

# lag_period     DBO_PICKLE_STAGE1_FUTURE_PERIOD_v
# 1	             163
# 2	             164
# 4	             165
# 8	             166
# 12	         167
# 16	         168
# 18	         169

# lag version vs period forward
lag_period = {1:[202105], 2:[202106], 4:[202107, 202108], 8:[202109, 202110, 202111, 202112], 
             12:[202201,202202, 202203, 202204], 16:[202205, 202206, 202207, 202208], 18:[202209, 202210]}
DBO_PICKLE_STAGE1_FUTURE_PERIOD_list = [163, 164, 165, 166, 167, 168, 169]

# COMMAND ----------

#load all the models
# model pickle version
DBO_PICKLE_STAGE1_FUTURE_PERIOD = 'dbfs:/mnt/adls/Tables/DBO_PICKLE_STAGE1_FUTURE_PERIOD'
model_pd = pd.DataFrame()
for DBO_PICKLE_STAGE1_FUTURE_PERIOD_v in DBO_PICKLE_STAGE1_FUTURE_PERIOD_list:
  model_save = load_delta(DBO_PICKLE_STAGE1_FUTURE_PERIOD, DBO_PICKLE_STAGE1_FUTURE_PERIOD_v).toPandas()
  model_pd = model_pd.append(model_save)

# COMMAND ----------

# data version
DBA_MRD = 'dbfs:/mnt/adls/Tables/DBA_MRD'
DBA_MRD_v = 58

# forecast results
DBO_FORECAST_FUTURE_PERIOD = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_FUTURE_PERIOD'
DBO_FORECAST_FUTURE_PERIOD_v = 36

# start date should be 202105
FCST_START_DATE = 202105
TIME_VAR = 'Month_Of_Year'

# COMMAND ----------

# MAGIC %md ## Load data, model pickle and predictions

# COMMAND ----------

mrd_df = load_delta(DBA_MRD, DBA_MRD_v) 
mrd_df = mrd_df.filter(mrd_df.FCST_START_DATE>=FCST_START_DATE)
pred = load_delta(DBO_FORECAST_FUTURE_PERIOD, DBO_FORECAST_FUTURE_PERIOD_v)

# COMMAND ----------

feature_categories = get_drivers(TIME_VAR)
feature_categories.Driver = feature_categories.apply(lambda x: 'Pricing' if x.feature =='Discount_Depth' or x.feature =='Dollar_Discount' else x.Driver, axis = 1)

# COMMAND ----------

#get mapping and schema
driver_list = feature_categories.dropna().Driver.unique().tolist()
decomp_schema = create_decomp_schema(TIME_VAR)
final_schema = StructType([
  StructField(TIME_VAR, IntegerType()),
  StructField("MODEL_ID", StringType()),
  StructField("final_prediction_value", DoubleType()),
  StructField("Base", DoubleType()),
  StructField("Cannibalization", DoubleType()), 
  StructField("Macroeconomics", DoubleType()),
  StructField("Media", DoubleType()),
  StructField("Pricing", DoubleType()), 
  StructField("Weather", DoubleType())])

shap_decomposition_result_final = spark.createDataFrame(data=[],schema=final_schema)
#lag_period = {1:[202105]}

for this_lag in lag_period:
  print(f'this lag is:{this_lag}')
  print(f'months covered: {lag_period[this_lag]}')

  lag_model_pd = model_pd[model_pd.lag_model==this_lag]

  # find the best model
  pred_before_forcasting_date = pred.filter((pred.lag_period==this_lag)&(pred[TIME_VAR]<FCST_START_DATE))
  stage1_models_list = lag_model_pd.train_func.to_list()
  best_stg_model = get_best_stg_model(pred_before_forcasting_date, stage1_models_list)
  print(f'The best model in Stage 1 is {best_stg_model}')
  #print(np.sort(pred_before_forcasting_date.toPandas()[TIME_VAR].unique()))

  # load the model and get model features
  model = load_model(lag_model_pd, best_stg_model)
  if best_stg_model == 'lightGBM_model':
      model.params['objective'] = 'regression'
  model_features = get_model_features(model)

  # data for decomp
  stg_data = mrd_df.filter(mrd_df[TIME_VAR].isin(lag_period[this_lag]))
  stg_data = stg_data.withColumn("TRAIN_IND", lit(1))
  #print(np.sort(stg_data.toPandas()[TIME_VAR].unique()))

  # calculate decomp
  decomp_models_udf = decomp_udf_wrapper(decomp_schema, model, model_features, feature_categories,TIME_VAR)
  shap_decomposition_result = stg_data.groupBy([TIME_VAR]).apply(decomp_models_udf)
  #print(shap_decomposition_result.count())
  
  # join with predict column
  final_pred = pred.filter((pred.lag_period==this_lag)&(pred[TIME_VAR].isin(lag_period[this_lag])))
  final_pred = final_pred.select(TIME_VAR, 'MODEL_ID', 'final_prediction_value')
  #final_pred = final_pred.select(TIME_VAR, 'MODEL_ID', best_stg_model+'_stage1', #'final_prediction_value').withColumnRenamed(best_stg_model+'_stage1',"best_stage1_model")
  final_pred = final_pred.join(shap_decomposition_result,[TIME_VAR,'MODEL_ID'])
  #display(final_pred)
  shap_decomposition_result_final=shap_decomposition_result_final.unionByName(final_pred)
  #display(shap_decomposition_result_final)
  
shap_decomposition_result_final = get_driver_decomp(shap_decomposition_result_final, driver_list)
print(shap_decomposition_result_final.select(TIME_VAR).distinct().collect())
#display(shap_decomposition_result_final)

# COMMAND ----------

DBO_DECOMP = "dbfs:/mnt/adls/Tables/RF_test/DBO_DCOMP"
save_df_as_delta(shap_decomposition_result_final, DBO_DECOMP, enforce_schema=False)

# COMMAND ----------

# MAGIC %md ##Feature Importance

# COMMAND ----------

importance_table = pd.DataFrame()
for index,row in model_pd.iterrows():
  obj = row.model_pick.encode("latin-1")
  model = pickle.loads(obj)
  if hasattr(model, 'feature_names_'):
    #model_features = model.feature_names_
    importance_list = list(zip(model.feature_names_, model.get_feature_importance()))
  else:
    #model_features = model.feature_name()
    importance_list = list(zip(model.feature_name(), model.feature_importance(importance_type='gain')))
  
  importance_pd = pd.DataFrame(importance_list)
  importance_pd['model_name'] = row.train_func
  importance_pd['lag'] = row.lag_model
  importance_table = importance_table.append(importance_pd)
  #print(test)
  
importance_table = importance_table.rename(columns={0:'feature', 1:'importance'})
importance_table = importance_table.merge(feature_categories, how='left', on='feature')
importance_table.Driver = importance_table.Driver.apply(lambda x: x if isinstance(x, str) else 'Base')
importance_table.lag.unique()

# COMMAND ----------

DBO_FEATURE_IMPORTANCE = "dbfs:/mnt/adls/Tables/RF_test/DBO_FEATURE_IMPORTANCE"
importance_table =  spark.createDataFrame(importance_table)
save_df_as_delta(importance_table, DBO_FEATURE_IMPORTANCE, enforce_schema=False)

# COMMAND ----------

feat_imp_monthly = load_delta("dbfs:/mnt/adls/Tables/RF_test/DBO_FEATURE_IMPORTANCE")
feat_imp_monthly.select('lag').distinct().show()

# COMMAND ----------

