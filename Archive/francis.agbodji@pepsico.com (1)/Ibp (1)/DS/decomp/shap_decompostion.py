# Databricks notebook source
from pyspark.sql.types import *

# COMMAND ----------

def get_best_stg_model(data, model_list):
  acc=[]
  for m in model_list:
    aux= data.select([m+'_stage1', 'CASES_ORIG' ])
    aux = aux.withColumn("AbsError", abs(F.col(m+'_stage1')-F.col("CASES_ORIG")))
    accuracy=1-aux.agg(F.sum("AbsError")).collect()[0][0]/aux.agg(F.sum("CASES_ORIG")).collect()[0][0]
    acc+=[accuracy]
  results=pd.DataFrame(zip(model_list,acc),columns=['model','accuracy'])
  best_stg_model=results.iloc[results.accuracy.argmax(), 0]
  return best_stg_model

# COMMAND ----------

def load_model(pickles_pd, best_stg_model):
  #pickles_pd = model_pickle.toPandas()
  obj = pickles_pd.loc[(pickles_pd.train_func==best_stg_model),'model_pick'].values[0].encode("latin-1")
  model = pickle.loads(obj)
  return model

# COMMAND ----------

def get_model_features(model):
  if hasattr(model, 'feature_names_'):
      model_features = model.feature_names_
  else:
      model_features = model.feature_name()
  return model_features

# COMMAND ----------

def get_drivers(TIME_VAR):
  if TIME_VAR=='Month_Of_Year':
    feat_categories= spark.read.option("header","true").option("delimiter",",").csv("dbfs:/FileStore/tables/DBP_DECOMP-4.csv").toPandas()
    feat_categories=feat_categories[['Variable','New_Driver']].rename(columns={"Variable": "feature", "New_Driver": "Driver"})
  else:
    feat_categories= spark.read.option("header","true").option("delimiter",";").csv("dbfs:/FileStore/tables/Drivers_Weekly_pingo.csv").toPandas()
    feat_categories=feat_categories[['Variable','Driver']].rename(columns={"Variable": "feature"})
    feat_categories.feature=feat_categories.feature.str.strip()
  return feat_categories

# COMMAND ----------

def create_decomp_schema(TIME_VAR):
  if TIME_VAR == 'Month_Of_Year':
    return StructType([
    StructField("MODEL_ID", StringType()),
    StructField("Base", DoubleType()),
    StructField("Cannibalization", DoubleType()), 
    StructField("Macroeconomics", DoubleType()),
    StructField("Media", DoubleType()),
    StructField("Pricing", DoubleType()), 
#    StructField("Promotions", DoubleType()),
   # StructField("Holidays", DoubleType()),
    StructField("Weather", DoubleType()),
    StructField(TIME_VAR, IntegerType())])
  else:
    return StructType([
    StructField("MODEL_ID", StringType()),
    StructField("Base", DoubleType()),
    StructField("Cannibalization", DoubleType()), 
    StructField("Media", DoubleType()),
    StructField("Pricing", DoubleType()), 
 #   StructField("Promotions", DoubleType()),
    StructField("Weather", DoubleType()),
    StructField(TIME_VAR, IntegerType())])

# COMMAND ----------

def decomp_udf_wrapper(decomp_schema, model, model_features, feature_categories, TIME_VAR):
  @pandas_udf(decomp_schema, PandasUDFType.GROUPED_MAP)
  def decomp_udf(data):
    return calculate_decomp(df=data, model=model, model_features=model_features, feature_categories=feature_categories, TIME_VAR=TIME_VAR)
  return decomp_udf

# COMMAND ----------

def calculate_decomp(df, model, model_features, feature_categories, TIME_VAR):
  #df = df.toPandas()
  time_var = df[TIME_VAR].iloc[0]
  #print(month_of_year)
  # get shap values
  explainer_tree = shap.TreeExplainer(model)
  shap_values_tree = explainer_tree.shap_values(df[model_features])
  
  # mean values
  expected_value=explainer_tree.expected_value
  
  # Convert to spark dataframe
  shap_values_df = pd.DataFrame(shap_values_tree, columns=model_features)
  
  # aggregate features to driver 
  shap_values_df = shap_values_df.join(df['MODEL_ID'])
  
  # Make long and add type
  shap_long = pd.melt(shap_values_df, id_vars=['MODEL_ID'], value_vars=model_features, var_name='feature', value_name='shap')
  
  shap_long= shap_long.merge(feature_categories, how='left', on='feature')
  shap_long["Driver"].fillna('Base',inplace=True)
  
  # Sum by MODEL_ID
  shap_by_modelID=shap_long.groupby("MODEL_ID")['shap'].sum().reset_index()
  shap_by_modelID=shap_by_modelID.rename(columns={"shap":"shap_MODEL_ID"})
 
  # Sum by MODEL_ID AND DRIVER
  shap_by_modelID_driver=shap_long.groupby(["MODEL_ID","Driver"])['shap'].sum().reset_index()
  shap_by_modelID_driver=shap_by_modelID_driver.rename(columns={"shap": "shap_MODEL_ID_Driver"})
  
  
  # why are we adding mean to the base value???
  shap_by_modelID_driver['shap_MID_Driver'] = shap_by_modelID_driver.apply(lambda x: x.shap_MODEL_ID_Driver + expected_value if x.Driver == 'Base' else x.shap_MODEL_ID_Driver, axis =1)
  shap_by_modelID['shap_MODEL_ID']=shap_by_modelID.shap_MODEL_ID + expected_value

  #Extract base
 # temp_base = shap_by_modelID_driver[shap_by_modelID_driver['Driver']=='Base'].drop(columns=['Driver', 'shap_MODEL_ID_Driver']).rename(columns={"shap_MID_Driver": "Base_shap_MODELID"})
  
  # Combine all and add base shap
  shap_final=shap_by_modelID_driver.merge(shap_by_modelID, how='left', on='MODEL_ID')
 # shap_final=shap_final.merge(temp_base, how='left', on='MODEL_ID')
  
  # Compute decomp ratio (with respect to total) and lift ratio (with respect to base)
  shap_final['Decomp_ratio'] = shap_final.shap_MID_Driver / shap_final.shap_MODEL_ID
  # shap_final['Lift_ratio'] = shap_final.shap_MID_Driver / shap_final.Base_shap_MODELID
  
  # pivot
  pivot_shap = shap_final.pivot(index='MODEL_ID', columns='Driver', values='Decomp_ratio').reset_index()
  pivot_shap[TIME_VAR] = time_var
  return(pivot_shap)

# COMMAND ----------

def get_driver_decomp(data, driver_list):
  for driver in driver_list:
    if driver in data.columns:
      data = data.withColumn(driver, data[driver] * data['final_prediction_value'])
  return data