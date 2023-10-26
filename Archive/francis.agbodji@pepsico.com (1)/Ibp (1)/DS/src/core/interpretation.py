# Databricks notebook source
# DBTITLE 1,Pandas Interpret Error Metrics
## NET NEW VERSION - COREY ADDED 'additional_model_list' input on 5/11/2021
## This allows additional columns to be included in metrics calcs without renaming using the model designator

def calc_error_metrics(pd_df, tar_var, error_func_list, model_designator="_model_", additional_model_list=[], additional_columns_tuple=None, replace_infinities=True):
  """
  Calculate error metrics from a pandas DataFrame of predictions
  """  
  
  error_cols = []
  
  model_names = [col for col in list(pd_df.columns) if model_designator in col]
  filt_model_names = [col for col in model_names if "calc" not in col]
  filt_model_names = filt_model_names + additional_model_list
  
  final_pd = pd_df.copy()
  
  # calculate error metrics
  for model_name in filt_model_names:
    for error_func in error_func_list:
      error_name = model_name + "_" + error_func.__name__
      final_pd[error_name] = error_func(final_pd[tar_var], final_pd[model_name])
      
  if additional_columns_tuple:
    add_tar_var = additional_columns_tuple[0]
    add_column_names = additional_columns_tuple[1]
    
    for model_name in add_column_names:
      for error_func in error_func_list:
        error_name = model_name + "_" + error_func.__name__
        final_pd[error_name] = error_func(final_pd[add_tar_var], final_pd[model_name])
      
  # replace infinities (often caused by dividing by zero)
  if replace_infinities:
    final_pd = final_pd.replace([np.inf, -np.inf], np.nan)
  
  return final_pd


### DEPRECATED VERSION OF THIS FUNCTION BELOW
### KEEPING FOR NOW, UNTIL WE CAN 
# def calc_error_metrics(pd_df, tar_var, error_func_list, model_designator="_model_", additional_columns_tuple=None, replace_infinities=True):
#   """
#   Calculate error metrics from a pandas DataFrame of predictions
#   """  
  
#   error_cols = []
  
#   model_names = [col for col in list(pd_df.columns) if model_designator in col]
#   filt_model_names = [col for col in model_names if "calc" not in col]
  
#   final_pd = pd_df.copy()
  
#   # calculate error metrics
#   for model_name in filt_model_names:
#     for error_func in error_func_list:
#       error_name = model_name + "_" + error_func.__name__
#       final_pd[error_name] = error_func(final_pd[tar_var], final_pd[model_name])
      
#   if additional_columns_tuple:
#     add_tar_var = additional_columns_tuple[0]
#     add_column_names = additional_columns_tuple[1]
    
#     for model_name in add_column_names:
#       for error_func in error_func_list:
#         error_name = model_name + "_" + error_func.__name__
#         final_pd[error_name] = error_func(final_pd[add_tar_var], final_pd[model_name])
      
#   # replace infinities (often caused by dividing by zero)
#   if replace_infinities:
#     final_pd = final_pd.replace([np.inf, -np.inf], np.nan)
  
#   return final_pd
      

def print_error_metrics(prediction_pd):
  """
  Wrapper to print summary stats for in-sample and out-of-sample error metrics
  """

  columns = [key for key in list(prediction_pd.columns) if "calc" in key] + ["sample"]

  if is_dask_df(prediction_pd):
    summary_stat_pd = prediction_pd[columns].groupby("sample")\
                                            .apply(lambda x: x.describe())
  else:
    summary_stat_pd = prediction_pd[columns].groupby("sample")\
                                            .describe()
  
  final_pd = summary_stat_pd.transpose().reset_index()
  final_pd.columns = ["Models", "Aggregation", "In-Sample", "Out-of-Sample"]
  
  print(final_pd)
  
  return summary_stat_pd


def calc_SE(actuals, predictions):
    """
    Calculate the squared error (SE) given predicted and actual values.
    """     
    return ((actuals-predictions)**2)


def calc_RMSE(actuals, predictions):
    """
    Calculate the root mean squared error (RMSE) given predicted and actual values.
    """     
    return (calc_SE(actuals, predictions).mean())**(1/2)


def calc_RWMSE(actuals, predictions):
    """
    Calculate the root weighted mean squared error (RWMSE) given predicted and actual values.
    """     
    return np.sqrt(np.sum(calc_SE(actuals,predictions) * actuals) / np.sum(actuals))
    

def calc_AE(actuals, predictions):
  """
  Calculate absolute error given predicted and actual values.
  """
  return (np.fabs(actuals - predictions))

  
def calc_MAE(actuals, predictions):
    """
    Calculate mean absolute error (MAE) given predicted and actual values.
    """
    return calc_AE(actuals, predictions).mean()
  
  
def calc_APE(actuals, predictions):
  """
  Calculate absolute percent error given predicted and actual values.
  """
  return (np.fabs(actuals - predictions)/actuals)


def calc_APA(actuals, predictions):
  """
  Calculate absolute percent accuracy given predicted and actual values.
  """
  return (1 - (np.fabs(actuals - predictions)/actuals))


def calc_MAPE(actuals, predictions):
    """
    Calculate mean absolute percent error (MAPE) given predicted and actual values.
    """
    return calc_APE(actuals, predictions).mean()
  
  
def calc_MAPA(actuals, predictions):
    """
    Calculate mean absolute percent accuracy (1-MAPE) given predicted and actual values.
    """
    return calc_APA(actuals, predictions).mean()
  
  
def calc_DPA(actuals, predictions):
  """
  Calculate the interim DPA error metric (Nestle metric)
  """
  return 1-(np.fabs(predictions - actuals)/predictions)


def calc_APPE(actuals, predictions):
  """
  Calculate "absolute percent prediction error" (Nestle metric) given predicted and actual values.
  """
  return (np.fabs(predictions - actuals)/predictions)


def calc_MAPPE(actuals, predictions):
  """
  Calculate "absolute percent prediction error" (Nestle metric) given predicted and actual values.
  """
  return calc_APPE(actuals, predictions).mean()


## Corey updated Bias calculation on 5/18/2021 to conform with PEP standards
def calc_Bias(actuals, predictions):
  """
  Calculate the "bias" error metric to better gauge under/over prediction propsensity.
  Previous calculation: ((actuals - predictions)/actuals)
  """
  return ((predictions - actuals)/actuals)


def calc_bias(actuals, predictions):
  """
  Calculate the "bias" error metric (from old client definition).
  """
  return ((predictions - actuals)/predictions)


def calc_classification_accuracy(actuals, predicted):
  from sklearn.metrics import accuracy_score

  return accuracy_score(y_true=actuals, y_pred=predicted)

  
def aggregate_error_metrics(prediction_pd, prod_lvl, business_lvl, agg_func, ind_cols_list=None):
  """
  Return a dataframe of aggregated error metrics by levels of the product and business hierarchies
  """
    
  if not ind_cols_list:
    ind_cols_list = []
  
  # handle the exception where the TARGET_VAR has already been converted to the base_target_var name
  base_target_var = get_base_target_var() 
  if base_target_var in prediction_pd.columns:
    tar_var = base_target_var
  else: 
    tar_var = TARGET_VAR
    
  if not prod_lvl:
    prod_columns = []
  else:
    prod_columns = PRODUCT_HIER[:prod_lvl]
    
  if not business_lvl:
     business_columns = []
  else:
    business_columns = BUSINESS_HIER[:business_lvl]
    
  if agg_func == "weighted_mean":
    agg_func = get_weighted_average_func(prediction_pd, tar_var)
    
  ## use set to remove duplicates in case the user wants to indicate by a level of the hierarchy
  grouping_cols = list(set(business_columns + prod_columns + ["sample"] + ind_cols_list))

  model_dict = {column:np.sum for column in list(prediction_pd.columns) if "_model" in column}
  error_dict = {column:agg_func for column in list(prediction_pd.columns) if "_calc" in column}
  actuals_dict = {tar_var:np.sum}
  
  new_error_names_dict = {old:old.replace("_calc", "_aggcalc") for old in error_dict.keys()}
  
  agg_dict = {**model_dict, **error_dict, **actuals_dict}      
  agg_pd = prediction_pd.groupby(grouping_cols).agg(agg_dict)
  final_pd = agg_pd.rename(new_error_names_dict, axis=1)
  
  return final_pd.reset_index()


def plot_aggregated_error_metrics(aggregated_error_metrics_pd, sample, other_ind=None, viz_function=plot_violin_plot, metric_type_list=["calc_APE"], model_substr_list=["_model_"], multiply_values_by_100=False, **kwargs):
  """
  Plot aggregated error metrics by levels of the product and business hierarchies
  """
  
  # Force users to pick a sample so they don't confuse in-sample with out-of-sample metrics 
  input_pd = aggregated_error_metrics_pd[aggregated_error_metrics_pd["sample"] == sample]
      
  model_filter = [column for column in list(input_pd.columns) if any(model_name in column for model_name in model_substr_list)]
  
  filtered_input_pd = input_pd[model_filter]

  x_axis_column_list = [column for column in list(filtered_input_pd.columns) if any(metric_type in column for metric_type in metric_type_list)]
  melted_pd = pd.melt(filtered_input_pd[x_axis_column_list], id_vars=other_ind)
  
  if multiply_values_by_100:
    melted_pd["value"] = melted_pd["value"] * 100

  viz_output = viz_function(melted_pd, 'variable', 'value', **kwargs)
  
  return viz_output

# COMMAND ----------

# DBTITLE 1,Pandas Transformation Functions
def untransform_target_var(pd_df, pred_array=None):
  """
  Return the untransformed y-value for use in calculating error metrics.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
  
  pred_array : array-like, optional (default = None)
      Array of predicted values (used in place of transformed TARGET_VAR column in pd_df).
  
  Returns
  -------
  untransformed_target_var : pandas or koalas DataFrame
      Untransformed values of TARGET_VAR.
  
  TODO investigate ways to make this scaler more efficient while still taking advantage of the sklearn scalers
  """
  operation = TARGET_VAR[-4:] 
  updated_target_var_name = TARGET_VAR[:-4]
  
  def printOperation():
    print("%s operation found; returning transformed column" % operation)
    
  # This is ugly but necessary if we want to use the sklearn transformers
  if pred_array is not None:
    pd_df[TARGET_VAR] = pred_array

  if  operation == "_std":
    printOperation()
    return destandardize_columns(pd_df)[updated_target_var_name]

  if  operation == "_nrm":
    printOperation()
    return denormalize_df(pd_df)[updated_target_var_name]

  if  operation == "_log":
    printOperation()
    return delog_df(pd_df)[updated_target_var_name]
  
  else:
    print("Transformation not recognized. Returning original target variable.")
    return pd_pd[TARGET_VAR]

# COMMAND ----------

# DBTITLE 1,Pandas Hierarchical Regression Coefficients
def save_hier_regression_coefs(model_obj, grouping_column, path_ind=""):
  """
  Save hierarchical regression coefficients
  """
  
  file_name_ending_in_dot_npy = get_pipeline_name() + "_label_dict" + ".npy" 
  random_effects = model_obj.random_effects
  random_effects_pd = pd.DataFrame.from_dict(random_effects, orient="index").rename({"Group":"Intercept_Adjustment"}, axis=1).reset_index()
  
  ## Corey updated - .npy adjustment
  if "_index" in grouping_column:
    random_effects_pd = deindex_features(random_effects_pd.rename({"index":grouping_column}, axis=1), grouping_column)\
                       .rename({grouping_column.replace("_index", ""):"Grouping_Feature"}, axis=1)
    
  fixed_effect_pd = pd.DataFrame(model_obj.fe_params).transpose()
  fixed_effect_pd = fixed_effect_pd.loc[fixed_effect_pd.index.repeat(len(random_effects_pd))]
  fixed_effect_pd.reset_index(inplace=True) # reset index so that it's monotonically increasing
  fixed_effect_pd.drop("index", inplace=True, axis=1)

  joined_pd = fixed_effect_pd.join(random_effects_pd)
  joined_pd["Adj_Intercept"] = joined_pd["Intercept"] + joined_pd["Intercept_Adjustment"] 
  
  cols_to_sum = [col for col in list(joined_pd.columns) if col not in ["Intercept", "Intercept Adjustment", "Grouping_Feature"]]
  joined_pd["Total"] = joined_pd[cols_to_sum].sum(axis=1)

  final_df = pd.melt(joined_pd, id_vars="Grouping_Feature").rename({"variable":"Feature", "value":"Coefficient"},axis=1)
  
  if path_ind:
    path_ind = "_" + path_ind
  else:
    path_ind = ""
  
  save_df_as_csv(final_df, OUTBOUND_PATH + get_pipeline_name() + "_hierarchical_regression_coefficients" + path_ind)

  return final_df


# COMMAND ----------

# DBTITLE 1,Pandas Elasticities
def calc_elasticity(pd_df, column_list, model, prediction_function, adjustment_percentage=0.01):
  """
  Get elasticities for a given training dataset and model.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      Training dataset.
  
  column_list : list
      List of columns for which to calculate elasticities.

  model : generic model type
      Fit model object that corresponds with prediction_function.
  
  prediction_function : function (default = predict_xgboost)
      Function that generates predictions for the provided model and training data.
  
  adjustment_percentage : float (default = 0.01)
      Adjustment percentage to use when calculating adjusted predictions (to avoid division by 0 errors).
  
  Returns
  -------
  perc_change_df : pandas DataFrame
      Elasticities for the provided training dataset and model. 
  """
  #TODO transform _std, _nrm
  
  index_cols = get_hierarchy()
  transformed_index_cols = [col + "_index" for col in index_cols]
      
  # get initial prediction
  mean_pd = pd_df.groupby(transformed_index_cols).mean().reset_index()
  mean_pd[TIME_VAR] = pd_df[TIME_VAR].max()
  initial_predictions = prediction_function(mean_pd[list(train_pd.columns)], model)
  
  # get adjusted predictions
  adj_mean_pd = mean_pd.copy()
  adj_mean_pd[column_list] = adj_mean_pd[column_list] * (1 + adjustment_percentage)
  adj_predictions = prediction_function(adj_mean_pd[list(train_pd.columns)], model)
  
  perc_change = (adj_predictions - initial_predictions)/(initial_predictions + .00000001) # add a small number to avoid divide by zero errors; TODO explore whether this causes small changes to blow up
  
  return pd.DataFrame(perc_change)

# COMMAND ----------

# DBTITLE 1,Pandas Feature Importance Plots & SHAP Values
def save_booster_feature_importance_scores(importance_type="weight", path_indicator=None):
  """
  Save the feature importance scores from XGBoost and LightGBM that are stored in MODEL_DICT
  
  "type" can be:
    - ‘weight’: the number of times a feature is used to split the data across all trees.
    - ‘gain’: the average gain across all splits the feature is used in.
    - ‘cover’: the average coverage across all splits the feature is used in.
    - ‘total_gain’: the total gain across all splits the feature is used in.
    - ‘total_cover’: the total coverage across all splits the feature is used in.
  """
  xgb_models = [model for model in MODEL_DICT if "xgb" in model]
  lgbm_models = [model for model in MODEL_DICT if "lightGBM" in model]
      
  result_list = {}
  for model in xgb_models:
    feature_importance_dict = MODEL_DICT[model].get_score(importance_type=importance_type)
    result_list[model] = feature_importance_dict
    
  #NOTE: LightGBM feature importance is always calculated using 'split' (same thing as 'weight' for XGB). Must be adjusted during fitting.
  for model in lgbm_models:
    lightGBM_importance_array = MODEL_DICT[model].feature_importance()
    feature_names = MODEL_DICT[model].feature_name()
    
    feature_importance_dict = dict(zip(feature_names, lightGBM_importance_array.T))
    result_list[model] = feature_importance_dict

  final_pd = pd.DataFrame.from_dict(result_list).reset_index()
  
  if path_indicator:
    path_ind = "_" + path_indicator
  else:
    path_ind = ""
    
  save_df_as_csv(final_pd, OUTBOUND_PATH + get_pipeline_name() + "_feature_importance_scores_" + importance_type + path_ind)

  return final_pd


def save_SHAP_feature_importance_scores(shap_dict, model_name, pd_df, path_indicator):
  """
  Save dataframe of SHAP feature importance values
  """
  
  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)
  
  shap_values = shap_dict[model_name]["Values"]
  
  scores = np.abs(shap_values).mean(0)

  feature_importance_pd = pd.DataFrame(list(zip(filtered_pd.columns, scores)), columns=['feature','SHAP_feature_importance_vals'])
  feature_importance_pd.sort_values(by=['SHAP_feature_importance_vals'], ascending=False, inplace=True)
  
  if path_indicator:
    path_ind = "_" + path_indicator
  else:
    path_ind = ""
    
  save_csv(feature_importance_pd, OUTBOUND_PATH + get_pipeline_name() + get_pipeline_name() + "_SHAP_feature_importance_scores_" + model_name + path_ind)
  
  return feature_importance_pd


def calc_SHAP_values(pd_df, approximate_xgb=True, tree_limit=None, path_indicator=None):
  """
  Save the SHAP values for all model
  """
  
  if approximate_xgb:
    print("NOTE: Approximating XGB SHAP values.")
  
# Shreya updating model_dict to MODEL_DICT 05/21
  xgb_models = [model for model in MODEL_DICT if "xgb_model_stage1" in model]
  lgbm_models = [model for model in MODEL_DICT if "lightGBM_model_stage1" in model]
  
  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)

  shap_dict = {}
  
  for model in xgb_models:
    print("Calculating SHAP values for %s ..." % model)
    print(MODEL_DICT[model])
    explainer = shap.TreeExplainer(MODEL_DICT[model])
    shap_values = explainer.shap_values(filtered_pd,
                                        tree_limit=tree_limit,
                                        approximate=approximate_xgb)
    
    dict_entry = {"Values":shap_values, "Explainer":explainer}
    shap_dict[model] = dict_entry

    
  for model in lgbm_models:
    print("Calculating SHAP values for %s ..." % model)
    explainer = shap.TreeExplainer(MODEL_DICT[model])
    shap_values = explainer.shap_values(filtered_pd,
                                        tree_limit=tree_limit)
  
    dict_entry = {"Values":shap_values, "Explainer":explainer}
    shap_dict[model] = dict_entry
    
    
  print("Saving SHAP feature importance scores ...")
  for model in xgb_models + lgbm_models:
    save_SHAP_feature_importance_scores(shap_dict, model, pd_df, path_indicator)
  
  print("Done!")
  
  return shap_dict


def plot_SHAP_feature_summary(shap_dict, model_name, pd_df, plot_type="violin", plot_size="auto"):
  """
  Plot a feature importance plot using SHAP values
  plot_type: ["violin", "bar"]
  """
  shap_values = shap_dict[model_name]["Values"]
  
  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)
  
  return shap.summary_plot(shap_values, filtered_pd, plot_type=plot_type, plot_size=plot_size)


def plot_SHAP_dependency_plot(column_name_or_rank_func, shap_dict, model_name, pd_df, interaction_index=None):
  """
  Plot dependency plot using SHAP values
  """
  shap_values = shap_dict[model_name]["Values"]

  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)

  if not interaction_index:
    # Approximate most useful feature to explore for dependency
    inds = shap.approximate_interactions(column_name_or_rank_func, shap_values, filtered_pd)
    interaction_index = inds[0]

  return shap.dependence_plot(column_name_or_rank_func, shap_values, filtered_pd, interaction_index=interaction_index)



# COMMAND ----------

# DBTITLE 1,Pandas Classification Error Metrics
def print_confusion_matrix(actuals, predicted):
  from sklearn.metrics import confusion_matrix
  print(confusion_matrix(y_true=actuals, y_pred=predicted, normalize='true'))

# COMMAND ----------

# DBTITLE 1,PySpark Error Metrics
def calc_RMSE_pyspark(df, tar_var):
    """
    Calculates the root mean squared error (RMSE) from a Spark dataframe containing both a) the tar_var column and b) 
    """    
    error_dict = df.select([sqrt(mean((col(tar_var)-col(model_name))**2)).alias(model_name + "_RMSE") for model_name in MODEL_DICT])\
                  .toPandas()\
                  .to_dict('r')[0]

    return error_dict
  

def calc_MSE_pyspark(df, tar_var):
  """
  Calculates the mean squared error (MSE) when given predicted and actual values
  """    

  error_dict = df.select([mean((col(tar_var)-col(model_name))**2).alias(model_name + "_MSE") for model_name in MODEL_DICT])\
                  .toPandas()\
                  .to_dict('r')[0]
  return error_dict

  
def calc_MAE_pyspark(df, tar_var):
  """
  Calculates the mean absolute error (MAE) when given predicted and actual values
  """    

  error_dict = df.select([mean(abs(col(tar_var)-col(model_name))).alias(model_name + "_MAE") for model_name in MODEL_DICT])\
                  .toPandas()\
                  .to_dict('r')[0]
  
  return error_dict
    
  
def calc_MAPE_pyspark(df, tar_var):
  """
  Calculates the mean absolute percent error (MAPE) when given predicted and actual values
  """    
  error_dict = df.select([mean(abs((col(tar_var)-col(model_name))/col(tar_var))).alias(model_name + "_MAPE") for model_name in MODEL_DICT])\
                  .toPandas()\
                  .to_dict('r')[0]
  
  return error_dict
  
  
def calc_error_metrics_pyspark(df, error_func_list):
  """
  Print key error metrics for a given prediction DataFrame.
  #TODO is there a way to do this using vector math?
  """
  from collections import ChainMap
  
  base_target_var = get_base_target_var()
  
  error_list = [*(error_func(df, base_target_var) for error_func in error_func_list)]
  consolidated_error_list = dict(ChainMap(*error_list))

  return consolidated_error_list


def print_error_metrics_pyspark(is_prediction_df, oos_prediction_df, error_func_list = [calc_MAPE_pyspark, calc_MAE_pyspark, calc_RMSE_pyspark, calc_MSE_pyspark]):
  """
  Wrapper to print in-sample and out-of-sample error idctionaries
  """
  is_error_dict = calc_error_metrics_pyspark(is_prediction_df, error_func_list)
  oos_error_dict = calc_error_metrics_pyspark(oos_prediction_df, error_func_list)
  
  consolidated_error_dict = {"In-sample": is_error_dict, "Out-of-sample": oos_error_dict}
  
  final_df = pd.DataFrame.from_dict(consolidated_error_dict, orient="columns")
  final_df.sort_index(inplace=True)
    
  return final_df


def calc_elastic_net_params(model_name): 
  """
  Provide a summary of output for the trained and cross-validated Elastic Net model from sklearn.
  """
  EN_params = MODEL_DICT[model_name].get_params()

  nonzero_coefs = coefs[coefs['value'] != 0]
  
  return {"Params":EN_params, "Non-zero params":nonzero_coefs}

# COMMAND ----------

# DBTITLE 1,Workbench
def plot_SHAP_force_plot(shap_dict, model_name, pd_df, pred_row):
  """
  Create a force plot using SHAP values
  #TODO wasn't producing a useful output
  """
  explainer = shap_dict[model_name]["Explainer"] 
  shap_values = shap_dict[model_name]["Values"]
  
  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)
  
  return shap.force_plot(explainer.expected_value, shap_values[model_name][pred_row,:], filtered_pd[pred_row,:])



def melt_error_df(input_pd, time_ind, metric_type_list=["calc_APE"], model_substr_list=["_model_"], id_col_override=None):
  """
  Melt a dataframe of prediction errors to give the following columns:
    Error Metric 	New Column - Example: "DPA, APA"
    Error 	New Column - ".132"
    Algorithm	New Column - Example: "XGBoost"
    MonthYear	New Column - Example: "Jan 2019"
    MaterID	New Column
    MaterID_Description	New Column
    PL2_Business_Description	New Column
    PL3_Category_Description	New Column
    PL5_Segment_Description	New Column
  """
  
  if not id_col_override: 
    id_cols = get_hierarchy() + [time_ind]
  else:
    id_cols = id_col_override
    
  model_filter = [column for column in list(input_pd.columns) if any(model_name in column for model_name in model_substr_list)]
  filtered_input_pd = input_pd[model_filter + id_cols]
  
  column_list = [column for column in list(filtered_input_pd.columns) if any(metric_type in column for metric_type in metric_type_list)] + id_cols
  
  melted_pd = pd.melt(filtered_input_pd[column_list], id_vars=id_cols)\
                .rename({"variable":"Error Metric", "value":"Error"}, axis=1)
  
  melted_pd["Algorithm"] = melted_pd["Error Metric"].str.replace("_model_", " ").str.split("_").str[0]
  melted_pd["Error_Flag"] = np.where(melted_pd['Error Metric'].str.contains("calc_APA"), "APA", "DPA")

  
  return melted_pd

# COMMAND ----------

# DBTITLE 1,Demand Sensing Functions - weekly 'disaggregation' effort
def exponential_decay(input_pd, grouping_cols, day_cols, decay_rate, periods_to_shift_data=1):
  '''
  Uses exponential smoothing to fill day-of-week columns with percentage breakdown of the weekly order 
  Ref Doc: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
  '''
  day_list = list(day_cols)
  grouping_list = list(grouping_cols)
  sorting_list = [*grouping_list + [TIME_VAR]]
  decay_fct = lambda rolling_decay: rolling_decay.ewm(com=decay_rate).mean()
  
  print('Days:', day_list)
  print('Grouping vars:', grouping_list)
  print('Sorting vars:', sorting_list)
  
  ## sorting so EWM moves from historical (ie, smaller Week_Of_Year)
  temp_pd = input_pd.sort_values(sorting_list, ascending=True) 
  
  temp_pd['Totals'] = temp_pd[day_list].sum(axis=1)
  temp_pd[day_list] = temp_pd[day_list].div(temp_pd.Totals, axis=0)
  
  ## shift so that it does NOT take current week into account
  temp_pd[day_list] = temp_pd.groupby(by=grouping_list)[day_list].transform(lambda x: x.shift(periods_to_shift_data))
  temp_pd[day_list] = temp_pd.groupby(by=grouping_list)[day_list].transform(decay_fct)

  final_pd = temp_pd.drop(columns='Totals')
  final_pd['Weekly_PercentTotal'] = final_pd[day_list].sum(axis=1)

  return final_pd


def moving_average(input_pd, grouping_cols, day_cols, window_size,\
                   min_time_periods=1, window_type=None, periods_to_shift_data=1):
  '''
  Uses rolling window moving average to fill day-of-week columns with percentage breakdown of the weekly order 
  Ref Doc: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rolling.html
  '''
  day_list = list(day_cols)
  grouping_list = list(grouping_cols)
  sorting_list = [*grouping_list + [TIME_VAR]]
  moving_avg_fct = lambda mov_avg: mov_avg.rolling(window=window_size, min_periods=min_time_periods, win_type=window_type).mean()
 
  print('Days:', day_list)
  print('Grouping vars:', grouping_list)
  print('Sorting vars:', sorting_list)
  
  ## sorting so moving average moves from historical (ie, smaller Week_Of_Year)
  temp_pd = input_pd.sort_values(sorting_list, ascending=True) 
  
  temp_pd['Totals'] = temp_pd[day_list].sum(axis=1)
  temp_pd[day_list] = temp_pd[day_list].div(temp_pd.Totals, axis=0)
  
  ## shift so that it does NOT take current week into account
  temp_pd[day_list] = temp_pd.groupby(by=grouping_list)[day_list].transform(lambda x: x.shift(periods_to_shift_data))
  temp_pd[day_list] = temp_pd.groupby(by=grouping_list)[day_list].transform(moving_avg_fct)
 
  final_pd = temp_pd.drop(columns='Totals')
  final_pd['Weekly_PercentTotal'] = final_pd[day_list].sum(axis=1)

  return final_pd


def prediction_disaggregation(input_pd, model_col_to_use, day_cols):
  '''
  Spreads weekly prediction totals down to the day level
  '''
  day_list = list(day_cols)
  final_pd = input_pd
  final_pd[day_list] = (final_pd[day_list].multiply(final_pd[model_col_to_use], axis=0)).round(0)
  
  return final_pd


def melt_daily_values_into_tidy_DF(df, value_cols=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],\
                                   var_name='Day_Of_Week', value_name='OrderQy'):
  """
  Melt a list of daily prediction columns into just one TARGET_VAR
  """
  id_cols = [col for col in df if col not in value_cols]
  return pd.melt(df, id_vars=id_cols, value_vars=value_cols, var_name=var_name, value_name=value_name)



def calc_metrics_for_multiple_cols(input_pd, metrics_func_list, actuals_col_list,\
                                   pred_designator='_pred', replace_infinities=True):
  '''
  Allows customization of metrics calculated from prediction columns
  (and provided actuals as baseline)
  '''
  
  ## function set-up
  metrics_func_list = ensureIsList(metrics_func_list) 
  actuals_col_list = ensureIsList(actuals_col_list)
  final_pd = input_pd.copy()
  
  ## calculate error metrics
  for actuals_col in actuals_col_list:
    for metrics_func in metrics_func_list:
      metrics_name = actuals_col + '_' + metrics_func.__name__
      final_pd[metrics_name] = metrics_func(final_pd[actuals_col], final_pd[actuals_col + pred_designator])
  
  ## replace infinities (often caused by dividing by zero)
  if replace_infinities:
    final_pd = final_pd.replace([np.inf, -np.inf], np.nan)
  
  return final_pd


def calc_weighted_avg(pd_df, actuals_cols, metrics_cols):
  '''
  Custom function for weighted average across multiple columns
  Output is a dictionary with weight of each calculation represented as each value
  '''
  
  actuals_cols = ensureIsList(actuals_cols)
  metrics_cols = ensureIsList(metrics_cols)
  
  len_a = len(actuals_cols)
  len_m = len(metrics_cols)
  assert len_a == len_m

  index_list = range(0, len_a, 1)
  weighted_metric_dict = {}
        
  for each in index_list:
    wted_metric = ((pd_df[actuals_cols[each]] * pd_df[metrics_cols[each]]).sum()) / (pd_df[actuals_cols[each]].sum())
    weighted_metric_dict[each] = wted_metric
  
  return weighted_metric_dict

# COMMAND ----------

## COREY ADDED ON 7/20/2021 - from parallel source code

def select_best_model(df, models, actual_var, select_hier = None):
  """
  Selects the best model for some level of the hierarchy

  Parameters
  ----------                                                                                                                                                                                                                                                                       
  df : PySpark Dataframe
     Scored predictions
  models : List
      List of column names that contain the model predictions
  actual_var : String
      Column name of actual demand
  select_hier : List
      Level at which to select best model (e.g., country, MODEL_ID).  If left blank, one model will be selected for entire
      population

  Returns
  -------
  gof_report : PySpark Dataframe
      Weighted accuracies at select_hier level and best model selection
  """
  
  if select_hier is None:
    df = df.withColumn("dummy_seg", F.lit("dummy_seg"))
    select_hier = ["dummy_seg"]
    
  #Calculate accuracies at specified level
  gof_nested_dict = {}
  for i in models:
    this_acc_dict = {'error_func' : calculate_wtd_accuracy,
                     'parameters': [df, actual_var, i, select_hier]
                    }
    gof_nested_dict[i + "_ACC"] = this_acc_dict

  gof_report = calculate_gof(df, select_hier, gof_nested_dict)
  
  #Find maximum accuracy at specified level
  cols = add_suffix_to_list(models, "_ACC")
  cond = "F.when" + ".when".join(["(F.col('" + c + "') == F.col('max_accuracy'), F.lit('" + c + "'))" for c in gof_report.columns])
  gof_report = gof_report.withColumn("max_accuracy", F.greatest(*cols))\
      .withColumn("best_model", eval(cond))
  gof_report = gof_report.withColumn('best_model', F.regexp_replace('best_model', '_ACC', ''))
  
  return gof_report

# COMMAND ----------

## Added by Corey on 7/21/2021 - to include here from 'parallel' codebase

def select_best_model(df, models, actual_var, select_hier = None):
  """
  Selects the best model for some level of the hierarchy

  Parameters
  ----------
  df : PySpark Dataframe
     Scored predictions
  models : List
      List of column names that contain the model predictions
  actual_var : String
      Column name of actual demand
  select_hier : List
      Level at which to select best model (e.g., country, MODEL_ID).  If left blank, one model will be selected for entire
      population

  Returns
  -------
  gof_report : PySpark Dataframe
      Weighted accuracies at select_hier level and best model selection
  """
  
  if select_hier is None:
    df = df.withColumn("dummy_seg", F.lit("dummy_seg"))
    select_hier = ["dummy_seg"]
    
  #Calculate accuracies at specified level
  gof_nested_dict = {}
  for i in models:
    this_acc_dict = {'error_func' : calculate_wtd_accuracy,
                     'parameters': [df, actual_var, i, select_hier]
                    }
    gof_nested_dict[i + "_ACC"] = this_acc_dict

  gof_report = calculate_gof(df, select_hier, gof_nested_dict)
  
  #Find maximum accuracy at specified level
  cols = add_suffix_to_list(models, "_ACC")
  cond = "F.when" + ".when".join(["(F.col('" + c + "') == F.col('max_accuracy'), F.lit('" + c + "'))" for c in gof_report.columns])
  gof_report = gof_report.withColumn("max_accuracy", F.greatest(*cols))\
      .withColumn("best_model", eval(cond))
  gof_report = gof_report.withColumn('best_model', F.regexp_replace('best_model', '_ACC', ''))
  
  return gof_report

# COMMAND ----------

