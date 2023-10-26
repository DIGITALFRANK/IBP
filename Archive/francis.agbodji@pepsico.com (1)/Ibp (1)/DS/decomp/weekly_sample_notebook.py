# Databricks notebook source
# MAGIC %run ../src/libraries

# COMMAND ----------

# MAGIC %run ../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./shap_decompostion

# COMMAND ----------

# lag_period     DBO_PICKLE_STAGE1_FUTURE_PERIOD_v
#1	             176
#2	             177
#4	             178
#8	             179
#12	             180
#16	             181

# lag version vs period forward
lag_period = {1:[202122], 2:[202123], 4:[202124, 202125], 8:[202126, 202127, 202128, 202129], 
             12:[202130, 202131, 202132, 202133], 16:[202134, 202135, 202136, 202137]}
DBO_PICKLE_STAGE1_FUTURE_PERIOD_list = [176, 177, 178, 179, 180, 181]

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
DBA_MRD_v = 59

# forecast results
DBO_FORECAST_FUTURE_PERIOD = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_FUTURE_PERIOD'
DBO_FORECAST_FUTURE_PERIOD_v = 37

# start date 
FCST_START_DATE = 202122
TIME_VAR = 'Week_Of_Year'

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

#get schema
driver_list = feature_categories.dropna().Driver.unique().tolist()
decomp_schema = create_decomp_schema(TIME_VAR)
final_schema = StructType([
  StructField(TIME_VAR, IntegerType()),
  StructField("MODEL_ID", StringType()),
  StructField("final_prediction_value", DoubleType()),
  StructField("Base", DoubleType()),
  StructField("Cannibalization", DoubleType()), 
  StructField("Media", DoubleType()),
  StructField("Pricing", DoubleType()), 
 # StructField("Promotions", DoubleType()),
  StructField("Weather", DoubleType())])

# COMMAND ----------

shap_decomposition_result_final = spark.createDataFrame(data=[],schema=final_schema)

for this_lag in lag_period:
  print(f'this lag is:{this_lag}')
  print(f'weeks covered: {lag_period[this_lag]}')

  lag_model_pd = model_pd[model_pd.lag_model==this_lag]

  # find the best model
  pred_before_forcasting_date = pred.filter((pred.lag_period==this_lag)&(pred[TIME_VAR]<FCST_START_DATE))
  stage1_models_list = lag_model_pd.train_func.to_list()
  best_stg_model = get_best_stg_model(pred_before_forcasting_date, stage1_models_list)
  print(f'The best model in Stage 1 is {best_stg_model}')

  # load the model and get model features
  model = load_model(lag_model_pd, best_stg_model)
  if best_stg_model == 'lightGBM_model':
      model.params['objective'] = 'regression'
  model_features = get_model_features(model)

  # data for decomp
  stg_data = mrd_df.filter(mrd_df[TIME_VAR].isin(lag_period[this_lag]))
  stg_data = stg_data.withColumn("TRAIN_IND", lit(1))
 # print(stg_data.select(TIME_VAR).distinct().collect())
  
  # calculate decomp
  decomp_models_udf = decomp_udf_wrapper(decomp_schema, model, model_features, feature_categories, TIME_VAR)
  shap_decomposition_result = stg_data.groupBy([TIME_VAR]).apply(decomp_models_udf)
 
  # join with predict column
  final_pred = pred.filter((pred.lag_period==this_lag)&(pred[TIME_VAR].isin(lag_period[this_lag])))
  final_pred = final_pred.select(TIME_VAR, 'MODEL_ID', 'final_prediction_value')
  final_pred = final_pred.join(shap_decomposition_result,[TIME_VAR,'MODEL_ID'])
  shap_decomposition_result_final=shap_decomposition_result_final.unionByName(final_pred)
  #display(shap_decomposition_result_final)
  
shap_decomposition_result_final = get_driver_decomp(shap_decomposition_result_final, driver_list)

# COMMAND ----------

DBO_DECOMP_weekly = "dbfs:/mnt/adls/Tables/RF_test/DBO_DECOMP_weekly"
save_df_as_delta(shap_decomposition_result_final, DBO_DECOMP_weekly, enforce_schema=False)

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
  importance_pd['model_name'] = row.train_func #+str(row.lag_model) 
  importance_pd['lag'] = row.lag_model
  importance_table = importance_table.append(importance_pd)
  #print(test)
#importance_table.lag.unique()
importance_table = importance_table.rename(columns={0:'feature', 1:'importance'})
importance_table = importance_table.merge(feature_categories, how='left', on='feature')
importance_table.Driver = importance_table.Driver.apply(lambda x: x if isinstance(x, str) else 'Base')
importance_table.lag.unique()

# COMMAND ----------

DBO_FEATURE_IMPORTANCE_weekly = "dbfs:/mnt/adls/Tables/RF_test/DBO_FEATURE_IMPORTANCE_weekly"
importance_table =  spark.createDataFrame(importance_table)
save_df_as_delta(importance_table, DBO_FEATURE_IMPORTANCE_weekly, enforce_schema=False)

# COMMAND ----------

feat_imp_weekly = load_delta("dbfs:/mnt/adls/Tables/RF_test/DBO_FEATURE_IMPORTANCE_weekly")
feat_imp_weekly.select('lag').distinct().show()

# COMMAND ----------

