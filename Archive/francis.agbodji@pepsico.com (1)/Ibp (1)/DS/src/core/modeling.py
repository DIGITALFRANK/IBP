# Databricks notebook source
# DBTITLE 1,Imports
import os

# COMMAND ----------

# DBTITLE 1,Pandas Modeling
def train_hierarchical_regression(pd_df, formula, grouping_column, random_effects_formula=None, variance_components_formula=None):
  """
  Trains hierarchical regression
  - grouping_column: adds a random intercept for each group in this column. should be a column name (e.g., "RetailerName")
  - random_effects_formula: specifies a column by which your slope can vary. should be a column name with a tilda in front (e.g., "~WeekOfYear"). if grouping_column = "RetailerName" and this variable = "~WeekOfYear", \
    you're allowing your intercept to vary by RetailerName and your demand to grow at a different rate for each retailer (i.e., the effect of WeekOfYear varies by retailer).
    run
  Resources:
    -https://web.stanford.edu/class/psych252/section/Mixed_models_tutorial.html
    -https://rlbarter.github.io/Practical-Statistics/2017/03/03/fixed-mixed-and-random-effects/
  """
  
  return smf.mixedlm(formula, pd_df, groups=pd_df[grouping_column], re_formula=random_effects_formula, vc_formula=variance_components_formula).fit()


def predict_hierarchical_regression(pd_df, statsmodel_obj):
  """
  Generate predictions using statsmodel hierarchical regression object
  """
  
  predictions = statsmodel_obj.predict(pd_df)
  
  return predictions


def train_random_forest(input_pd, params, *args, **kwargs):
  '''
  Using SKLearn's implementation of Random Forest for training RF model
  Inputs require a dataframe (pandas) and the training parameters to use (dictionary)
  '''
    
  from sklearn.ensemble import RandomForestRegressor
    
  y = input_pd[TARGET_VAR]
  X = input_pd.drop([TARGET_VAR], axis=1, inplace=False)
    
  X_one_hot = pd.get_dummies(X)
       
  rf_model = RandomForestRegressor(**params)
  #rf_model.set_params(*args, **kwargs)
  rf_model.set_params(**kwargs)
  rf_model.fit(X_one_hot, y.ravel()) 
    
  return rf_model


def predict_random_forest(input_pd, rf_model, *args, **kwargs):
  '''
  Using instantiated model for predictions
  Inputs require dataframe (pandas) and already-trained model
  '''
    
  X = input_pd.drop([TARGET_VAR], axis=1, inplace=False)
  X_one_hot = pd.get_dummies(X)
  rf_predictions = rf_model.predict(X_one_hot, *args, **kwargs)
   
  return rf_predictions


def train_gbm(input_pd, params, *args, **kwargs):
  """
  Fit sklearn GBM model on training data using 'quantile' ceiling
  Pass parameters as a dictionary -- parallel sturcture to other models in pipeline
  """

  from sklearn.ensemble import GradientBoostingRegressor

  y = input_pd[TARGET_VAR].values
  X = input_pd.drop([TARGET_VAR], axis=1, inplace=False)

  gbm_model = GradientBoostingRegressor(**params)
  gbm_model.set_params(*args, **kwargs)
  gbm_model.fit(X,y)

  return gbm_model


def predict_gbm(input_pd, gbm_model, *args, **kwargs):
  """
  Predict TARGET_VAR using fitted sklearn GBM model
  Returns predicted values of TARGET_VAR using data from input dataframe
  """
    
  X = input_pd.drop([TARGET_VAR], axis=1, inplace=False)
  gbm_predictions = gbm_model.predict(X, *args, **kwargs)
    
  return gbm_predictions


def train_lightGBM(df, params, *args, **kwargs):
  """
  Fit LightGBM model on training data.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
  
  params : dict
      Parameter-value pairs for LightGBM model parameters.
      
  nround : int
      Number of boosting rounds (num_boost_round parameter for LightGBM).
      
  convert : bool (default = True)
      If True, convert pd_df into a LightGBM Dataset. If False, use global variables df and test_df.
  
  Returns
  -------
  lgbm_model : LGBMModel
      Fit LightGBM model.
  """
  import lightgbm
  import os
  import gc
  
  X, y = None, None
  if is_pandas_df(df):
    y = df[TARGET_VAR].values
    X = df.drop([TARGET_VAR], axis=1)
    
    dtrain = lightgbm.Dataset(X, label=y)

    # save as binary and load back to avoid dtype explosion during conversaion (see this link: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/53773)
#     file_path = get_temp_path() + "_lgbm{}.bin".format(np.random.randint(1, 5000)*np.random.randint(1, 100)+np.random.randint(1, 5000))
#     try:
#       os.remove(file_path)
#     except OSError:
#       pass
#     dtrain.save_binary(file_path)
#     dtrain = lightgbm.Dataset(file_path)
    
  else:
    dtrain = df
    
  lgbm_model = lightgbm.train(params, dtrain, *args, **kwargs)

  del dtrain
  del X
  del y
  gc.collect()
  
  return lgbm_model


def predict_lightGBM(pd_df, lightGBM_model, *args, **kwargs):
  """
  Predict TARGET_VAR using LightGBM model.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with testing data (X and y).
  
  lightGBM_model : LGBMModel
      Fit LightGBM model.
      
  convert : bool, optional (default = True)
      If True, convert pd_df into a LightGBM Dataset. If False, use global variables df and test_df.

  Returns
  -------
  predictions : array-like
      Predicted values of TARGET_VAR using data from pd_df.
  """
  import lightgbm
  X = pd_df.drop([TARGET_VAR], axis=1).values

  predictions = lightGBM_model.predict(X, *args, **kwargs)
  
  return predictions


def train_elastic_net_cv(pd_df, num_folds=5, l1_ratio_grid = [0.1, 0.5, 0.7, 0.8, 0.9, 0.95], eps_val = 0.0001, verbose = False, *args, **kwargs):
  """
  Fit a cross-validated Elastic Net model on training data.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
  
  num_folds : int (default = 5)
      Number of cross-validation folds to perform.
  
  l1_ratio_grid : list (default = [.1, .5, .7, .8, .9, .95])
      Grid of L1 ratios (l1_ratio parameter for ElasticNetCV).
  
  eps_val : float (default = 0.0001)
      Epsilon value (eps parameter for ElasticNetCV).
  
  verbose : bool (default = False)
      Binary indicator to print key parameters for model fitting.
      
  Returns
  -------
  ENCV : ElasticNetCV
      Fit ElasticNetCV model.

  TODO: see if there is a better way to run CV for time-series
  """
  
  from sklearn.linear_model import ElasticNetCV

  y = pd_df[TARGET_VAR]
  X = pd_df.drop([TARGET_VAR], axis=1)

  # Run ENCV
  ENCV = ElasticNetCV(l1_ratio = l1_ratio_grid, 
                    eps=eps_val, 
                    n_alphas=100, 
                    alphas=None, 
                    fit_intercept=True, 
                    normalize=False, 
                    max_iter=10000, 
                    tol=0.0001,
                    cv = num_folds,
                    *args, **kwargs
                  )

  ENCV.fit(X, y.ravel())

  if verbose:
    # Print key params
    print("ElasticNet chose an alpha of %f and a lambda of %f \n" % (ENCV.l1_ratio_, ENCV.alpha_))
    print("Selected Coefficients: ")
    print(" + ".join(getCoefficients(ENCV, X)))

  return ENCV


def get_elastic_net_coefficients(model, X_pd):
    """
    Return selected coefficients from a cross-validated Elastic Net model.
    
    Parameters
    ----------
    model : ElasticNetCV
        Fit ElasticNetCV model.
    
    X_pd : pandas or koalas DataFrame
        Training data for fit ElasticNetCV model.
        
    Returns
    -------
    coefficients: list
        Non-zero coefficients from fit ElasticNetCV model.
    """ 
    coefficients = pd.DataFrame(model.coef_.transpose(), index = X_pd.columns, columns = ["value"])
    non_zero_coefficients = coefficients[coefficients["value"] != 0]
    return non_zero_coefficients.index.tolist()


def train_xgboost(df, params, *args, **kwargs):
  """
  Fit a XGBoost model on training data.

  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
  
  params : dict
      Parameter-value pairs for XGBoost model parameters.
      
  convert : bool (default = True)
      If True, convert pd_df into a XGBoost DMatrix. If False, use global variable df.
      
  Returns
  -------
  xgb_model : XGBModel
      Fit XGBoost model.
  """
  if is_pandas_df(df):
    y = df[TARGET_VAR]
    X = df.drop([TARGET_VAR], axis=1)
    dtrain = xgb.DMatrix(X, label=y)

  else:
    dtrain = df

  xgb_model = xgb.train(params, dtrain, *args, **kwargs)

  return xgb_model


def predict_xgboost(pd_df, xgb_model, *args, **kwargs):
  """
  Predict TARGET_VAR using XGBoost model.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
  
  xgb_model : XGBModel
      Fit XGBoost model.
      
  convert : bool (default = True)
      If True, convert pd_df into a XGBoost DMatrix. If False, use pd_df.

  Returns
  -------
  predictions : array-like
      Predicted values of TARGET_VAR using data from pd_df.
  """
  
  if is_pandas_df(pd_df):
    y = pd_df[TARGET_VAR]
    X = pd_df.drop([TARGET_VAR], axis=1)
    dtrain = xgb.DMatrix(X, label=y)

  else:
    dtrain = pd_df

  predictions = xgb_model.predict(dtrain, *args, **kwargs)
  
  return predictions


def train_catboost(pd_df, params, *args, **kwargs):
  """
  Fit CatBoost model on training data.
  
  Parameters
  ----------
  pd_df : pandas DataFrame
      DataFrame with training data (X and y).
  
  params : dict
      Parameter-value pairs for CatBoost model parameters.
  
  Returns
  -------
  catboost_model : CatBoostRegressor
      Fit CatBoost model.
  """
  from catboost import CatBoostRegressor, Pool

  assert is_pandas_df(pd_df), 'Only pandas DataFrames allowed'
  
  y = pd_df[TARGET_VAR].values
  X = pd_df.drop([TARGET_VAR], axis=1)
  
  train_pool = Pool(data=X, label=y)
  
  catboost_model = CatBoostRegressor(**params)
  
  # had to add set_params() call because you can't unzip both **params and **kwargs in a single function call
  catboost_model.set_params(**kwargs)
  catboost_model.fit(train_pool)
  
  return catboost_model


def predict_catboost(pd_df, catboost_model, *args, **kwargs):
  """
  Predict TARGET_VAR using CatBoost model.
  
  Parameters
  ----------
  pd_df : pandas DataFrame
      DataFrame with testing data (X and y).
  
  catboost_model : CatBoostRegressor
      Fit CatBoost model.

  Returns
  -------
  predictions : array-like
      Predicted values of TARGET_VAR using data from pd_df.
  """
  from catboost import Pool
  
  assert is_pandas_df(pd_df), 'Only pandas DataFrames allowed'
  
  X = pd_df.drop([TARGET_VAR], axis=1).values

  X_pool = Pool(X)
  
  predictions = catboost_model.predict(X, *args, **kwargs)
  
  return predictions


def train_naive(df, lag_period = "52"):
  """
  Tells the MODEL_DICT not to worry about training a naive forecasting model
  """
  return lag_period


def predict_naive(df, lag_period):
  """
  Generate naive predictions using lag from prior period
  """
  lag_column_name = TARGET_VAR + "_lag_" + str(lag_period)
  
  if lag_column_name not in list(df.columns):
    raise ValueError("Missing a lag column for your predict_naive column: %s" % lag_column_name)
  
  predictions = df[lag_column_name]
  
  return predictions


def predict_sklearn_model(df, sk_model):
  """
  Predict TARGET_VAR using an sklearn object (e.g., ElasticNetCV).
  
  Parameters
  ----------
  df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
  
  sk_model : sklearn.base.RegressorMixin
      Fit sklearn model.

  Returns
  -------
  predictions : array-like
      Predicted values of TARGET_VAR using data from df.
  """
  X = df.drop([TARGET_VAR], axis=1)
  
  predictions = sk_model.predict(X)
  
  return predictions


def add_predictions_to_df(pd_df, model):
  '''
  Add predictions from stage 1 model to DataFrame for stage 2 model.
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with training data (X and y).
      
  model : sklearn.base.RegressorMixin
      Fit sklearn model.
  
  Returns
  -------
  modified_df : pandas or koalas DataFrame
      DataFrame with one additional column (prediction from model on pd_df).
  '''
   
  modified_df = pd_df.copy()
  
  modified_df[model.__class__.__name__] = model.predict(modified_df.drop(TARGET_VAR, axis=1))
  
  return modified_df


def run_modeling_pipeline(train_pd, test_pd, stage1_dict, stage2_dict = None):
  """
  Fit all stage 1 and stage 2 models and output predictions
  
  Parameters
  ----------
  train_pd : pandas DataFrame
      In-sample records
  test_pd : pandas DataFrame
      Out-of-sample records   
  stage1_dict : dict
      Dictionary with keys as stage 1 model name and values as tuple of ((train function, parameters), predict function)
  stage2_dict : dict (default = None)
      Dictionary with keys as stage 2 model name and values as tuple of ((train function, parameters), predict function)

  Returns
  -------
  predictions_pd : pandas DataFrame
  """

  label_file_name_ending_in_dot_npy = get_pipeline_name() + "_label_dict.npy" 

  in_sample_predictions_pd = fit_modeling_pipeline(train_pd, stage1_dict, stage2_dict)
  out_of_sample_predictions_pd = predict_modeling_pipeline(test_pd, stage1_dict, stage2_dict)
  
  # stack dataframes
  predictions_pd = concatenate_dfs(in_sample_predictions_pd, out_of_sample_predictions_pd)
  
  return predictions_pd


def zero_negative_predictions(pd_df, model_designator="_model_", threshold=0):
  """
  Zero-out any negative predictions for all forecast columns
  """
  model_names = [col for col in list(pd_df.columns) if model_designator in col]
  filt_model_names = [col for col in model_names if "calc" not in col]
  
  pd_df[filt_model_names] = pd_df[filt_model_names].mask(pd_df[filt_model_names] < threshold, 0)
    
  return pd_df


def run_models(train_pd, test_pd, stage1_dict, stage2_dict=None, descale=True, zero_negative_preds=True, 
               suppressErrorMetrics=False, print_errors=False, deindex=True, pipeline_func=run_modeling_pipeline, *args, **kwargs):
  """
  Run models and transform predictions for better interpretation
  
  Parameters
  ----------
  train_pd : pandas DataFrame
      In-sample records
  test_pd : pandas DataFrame
      Out-of-sample records   
  stage1_dict : dict
      Dictionary with keys as stage 1 model name and values as tuple of ((train function, parameters), predict function)
  stage2_dict : dict (default = None)
      Dictionary with keys as stage 2 model name and values as tuple of ((train function, parameters), predict function)
  descale : bool (default = True)
      Option to undo scaling operations on results
  suppressErrorMetrics : bool (default = True)
      Option to suppress error metrics when running
  print_errors : bool (default = True)
      Option to print errors when running
  deindex : bool (default = True)
      Option to deindex variables on results
      
  Returns
  -------
  final_df : pandas DataFrame
  """
  
  predictions_pd = pipeline_func(train_pd=train_pd, test_pd=test_pd, stage1_dict=stage1_dict, stage2_dict=stage2_dict, *args, **kwargs)
    
  if deindex:
    deindex_trans_predictions_pd = deindex_features(predictions_pd)
  else:
    deindex_trans_predictions_pd = predictions_pd
    
  final_df = untransform_prediction_df(deindex_trans_predictions_pd, descale=descale, suppressErrorMetrics=suppressErrorMetrics, print_errors=print_errors)
  
  if zero_negative_preds:
    final_df = zero_negative_predictions(final_df)
  
  return final_df
  

def fit_modeling_pipeline(train_pd, stage1_dict, stage2_dict = None):
  """  
  Train stage1 and stage2 specified models and assign the trained model
  objects to the global variable MODEL_DICT
  
  Parameters
  --------
  train_pd : DataFrame
      In-sample records
  stage1_dict : dict
      Dictionary with keys as stage 1 model name and values as tuple of ((train function, parameters), predict function)
  stage2_dict : dict (default = None)
      Dictionary with keys as stage 2 model name and values as tuple of ((train function, parameters), predict function)
  
  Returns
  --------
  in_sample_predictions_pd : DataFrame
      Predictions from stage1 and stage2 trained models for in sample/training set 
  """
    
  global MODEL_DICT
  MODEL_DICT = {}
   
  in_sample_stage1_pred_dict = {TARGET_VAR : train_pd[TARGET_VAR]}
  
  # append predictions to training data
  staged_train_pd = train_pd.copy()
 
  # stage 1 models
  for model in stage1_dict:
    stage1_model_name = model + "_stage1" 
    
    # fit model
    stage1_model_fit = train_pd.pipe(*stage1_dict[model][0])   
    MODEL_DICT[stage1_model_name] = stage1_model_fit
    
    # run predict function on dataframe
    in_sample_stage1_pred_dict[stage1_model_name] = (
      train_pd.pipe(stage1_dict[model][1], stage1_model_fit)
    )
    
    # add predictions to dataframe for stage 2
    staged_train_pd[stage1_model_name] = in_sample_stage1_pred_dict[stage1_model_name]

  in_sample_stage2_pred_dict = {TARGET_VAR : staged_train_pd[TARGET_VAR]}
  
  predictions_train_pd = staged_train_pd.copy()
    
  if stage2_dict:
    # stage 2 models
    for model in stage2_dict:
      stage2_model_name = model + "_stage2" 
     
      # train
      stage2_model_fit = staged_train_pd.pipe(*stage2_dict[model][0])    
      MODEL_DICT[stage2_model_name] = stage2_model_fit
      
      # predict
      in_sample_stage2_pred_dict[stage2_model_name] = staged_train_pd.pipe(stage2_dict[model][1], stage2_model_fit)
      
      # add predictions to dataframe
      predictions_train_pd[stage2_model_name] = in_sample_stage2_pred_dict[stage2_model_name]
      
    # create merged error dictionaries
    in_sample_pred_dict = {**in_sample_stage1_pred_dict, **in_sample_stage2_pred_dict}
  else:
    in_sample_pred_dict = in_sample_stage1_pred_dict
  
  # only select columns that denotes predictions and add designator 
  in_sample_predictions_pd = predictions_train_pd.assign(sample="IS")
    
  return in_sample_predictions_pd


def predict_modeling_pipeline(test_pd, stage1_dict, stage2_dict = None):
  """  
  Predict the target variable on out of sample records in test_pd using stage1 and stage2 trained models.
  
  Parameters
  --------
  test_pd : DataFrame
      Out of sample records
  stage1_dict : dict
      Dictionary with keys as stage 1 model name and values as tuple of ((train function, parameters), predict function)
  stage2_dict : dict (default = None)
      Dictionary with keys as stage 2 model name and values as tuple of ((train function, parameters), predict function)
  Returns
  --------
  out_of_sample_predictions_pd : DataFrame
      Predictions from stage1 and stage2 trained models for out of sample/test set 
  """

  out_of_sample_stage1_pred_dict = {TARGET_VAR : test_pd[TARGET_VAR]}
  
  # append predictions to training data
  staged_test_pd = test_pd.copy()
  
  # stage 1 models
  for model in stage1_dict:
    stage1_model_name = model + "_stage1" 
    
    # run predict function on dataframe
    out_of_sample_stage1_pred_dict[stage1_model_name] = (
      test_pd.pipe(stage1_dict[model][1], MODEL_DICT[stage1_model_name])
    )
    
    # add predictions to dataframe for stage 2
    staged_test_pd[stage1_model_name] = out_of_sample_stage1_pred_dict[stage1_model_name]

  out_of_sample_stage2_pred_dict = {TARGET_VAR : staged_test_pd[TARGET_VAR]}
  
  predictions_test_pd = staged_test_pd.copy()

  if stage2_dict:
    # stage 2 models
    for model in stage2_dict:
      stage2_model_name = model + "_stage2" 

      # predict
      out_of_sample_stage2_pred_dict[stage2_model_name] = (
        staged_test_pd.pipe(stage2_dict[model][1], MODEL_DICT[stage2_model_name])
      )
              
      # add predictions to dataframe
      predictions_test_pd[stage2_model_name] = out_of_sample_stage2_pred_dict[stage2_model_name]
    
    out_of_sample_pred_dict = {**out_of_sample_stage1_pred_dict, **out_of_sample_stage2_pred_dict}
  else:
    out_of_sample_pred_dict = out_of_sample_stage1_pred_dict 
    
  # only select columns that denotes predictions and add designator 
  out_of_sample_predictions_pd = predictions_test_pd.assign(sample="OOS")
  
  return out_of_sample_predictions_pd


def untransform_prediction_df(predictions_pd, descale=True, suppressErrorMetrics=False, 
                              print_errors=False, error_func_list=None):
  """
  Clean up a prediction dataframe by deindexing, descaling, and calculating error metrics
  """
  
  if not error_func_list:
    error_func_list = [calc_RMSE, calc_RWMSE, calc_APA]
  
  prediction_cols = [model for model in list(predictions_pd.columns) if "_stage" in model]

  if descale == True:
    tar_var = get_base_target_var()
    columns_to_descale = prediction_cols + [TARGET_VAR]
    trans_predictions_pd = predictions_pd.copy() # note: don't delete this .copy otherwise you'll edit the input dataframe
        
    trans_predictions_pd[columns_to_descale] = undo_scaling_operations(trans_predictions_pd[columns_to_descale])
    trans_predictions_pd = trans_predictions_pd.rename(columns={TARGET_VAR : tar_var})

  else:
    tar_var = TARGET_VAR
    trans_predictions_pd = predictions_pd.copy()
    
    
  if suppressErrorMetrics:
    return trans_predictions_pd
      
  else:
    # add error metrics to dataframe
    final_pred_pd = calc_error_metrics(trans_predictions_pd, tar_var, error_func_list = error_func_list)

    if print_errors:
      print_error_metrics(final_pred_pd)

    return final_pred_pd


def run_rolling_window_cv(indexed_pd, holdout_time_start, stage1_dict, stage2_dict=None, holdout_time_end=None, time_ind=None, insample_periods_to_report=4, *args, **kwargs):
  """
  Run rolling-window (ie, dynamic train/holdout sets) predictions for comparison to health checks data.
  """
    
  if not time_ind:
    time_ind = TIME_VAR
  
  sorted_time_list = sorted(indexed_pd.loc[indexed_pd[time_ind] >= holdout_time_start, time_ind].unique())
  
  if holdout_time_end:
    sorted_time_list = [time for time in sorted_time_list if time <= holdout_time_end]
  
  final_pd = pd.DataFrame()
    
  for time_period in sorted_time_list:
    temp_train_pd = indexed_pd[indexed_pd[time_ind] < time_period]
    temp_holdout_pd = indexed_pd[indexed_pd[time_ind] == time_period]
    
    print("Running models for %s - Train Size: %s, Test Size: %s" % (time_period, temp_train_pd.shape, temp_holdout_pd.shape))

    temp_pred_pd = run_models(train_pd=temp_train_pd, test_pd=temp_holdout_pd, stage1_dict=stage1_dict, stage2_dict=stage2_dict, *args, **kwargs)
    
    # filter output size to save memory
    temp_pred_pd = temp_pred_pd[temp_pred_pd[time_ind] >= (time_period - insample_periods_to_report)]
    
    final_pd = final_pd.append(temp_pred_pd, ignore_index=True)
  
  return final_pd

  
def run_walkforward_crossvalidation(indexed_pd, holdout_time_start, stage1_model_dict, stage2_model_dict, time_ind=None, descale=True, suppressErrorMetrics=False, print_errors=False):
  """
  Run sliding-window (ie, dynamic train/holdout sets) predictions for comparison to health checks data.
  #TODO
  """

  raise ValueError("Not implemented yet. Please use run_rolling_window_cv")


def calc_elastic_net_metrics(in_sample_pd, out_of_sample_pd, num_folds=5, l1_ratio_grid = [.1, .5, .7, .8, .9, .95], eps_val = .0001):
  """
  Run ElasticNet model, calculate error params, and return both in-sample and out-of-sample params as list.
  
  Parameters
  ----------
  in_sample_pd : pandas or koalas DataFrame
      Training data for ElasticNet model.
  
  out_of_sample_pd : pandas or koalas DataFrame
      Testing data for ElasticNet model.
  
  num_folds : int (default = 5)
      Number of cross-validation folds to perform.
  
  l1_ratio_grid : list (default = [.1, .5, .7, .8, .9, .95])
      Grid of L1 ratios (l1_ratio parameter for ElasticNetCV).
  
  eps_val : float (default = 0.0001)
      Epsilon value (eps parameter for ElasticNetCV).
  
  verbose : bool (default = False)
      Binary indicator to print key parameters for model fitting.
      
  Returns
  -------
  error_metrics : list
      List of in-sample (train) and out-of-sample (test) error metrics.
  """
  ENCV = train_elastic_net_cv(in_sample_pd, num_folds, l1_ratio_grid, eps_val)
  
  IS_predictions = ENCV.predict(in_sample_pd.drop(TARGET_VAR, axis=1))
  OOS_predictions = ENCV.predict(out_of_sample_pd.drop(TARGET_VAR, axis=1))
  
  IS_error_metrics = calc_error_metrics(in_sample_pd, IS_predictions)
  OOS_error_metrics = calc_error_metrics(out_of_sample_pd, OOS_predictions)

  return (IS_error_metrics + OOS_error_metrics)


def gridsearch_params_cv(indexed_pd, holdout_time_start, stage1_dict, param_grid, model_to_gridsearch = "lightgbm", 
                         time_ind=None, insample_periods_to_report=4, *args, **kwargs):
  """
  Run a param gridsearch using rolling window cross-validation 
  """
  import itertools as it
  
  def get_key_to_gridsearch(model_dict):
    models_to_gridsearch = [key for key in list(model_dict.keys()) if model_to_gridsearch in key.lower()]
    assert len(models_to_gridsearch) == 1, "You have two or more keys in your modeling dictionary containing %s" % model_to_gridsearch

    return models_to_gridsearch[0]

  model_key = get_key_to_gridsearch(stage1_dict)

  sorted_names = sorted(param_grid)
  combinations = list(it.product(*(param_grid[name] for name in sorted_names)))

  starter_params = stage1_dict[model_key][0][1].copy()

  print("\n Starting cross-validated param gridsearch, filtering in-sample error metrics to only include the last %s periods." % insample_periods_to_report)
  final_pd = pd.DataFrame()
  
  pbar = ProgressBar()
  for param_combination in pbar(combinations):
    # update param dictionary for target model
    param_update_dict = {sorted_names[index]:param_combination[index] for index in range(len(sorted_names))}
    new_params = starter_params.copy()
    new_params.update(param_update_dict)
    stage1_dict[model_key][0][1] = new_params
    
    temp_gridsearch_preds = run_rolling_window_cv(indexed_pd=indexed_pd, 
                                                  holdout_time_start=holdout_time_start, 
                                                  stage1_dict=stage1_dict, 
                                                  insample_periods_to_report=insample_periods_to_report, 
                                                  *args, **kwargs)

          
    columns_to_keep = [col for col in temp_gridsearch_preds if "_model_" in col] + ["sample"]
    
    results = temp_gridsearch_preds.groupby(['sample'])[columns_to_keep].agg(['mean', 'median'])
    results.columns = list(map('_'.join, list(results.columns)))
    results = results.reset_index()

    results.insert(loc=0, column='feature_set', value=str(param_update_dict))
    
    final_pd = final_pd.append(results, ignore_index=True)
    
  return final_pd


def gridsearch_params(training_pd, testing_pd, stage1_dict, param_grid, model_to_gridsearch = "lightgbm", time_ind=None, *args, **kwargs):
  """
  Run a param gridsearch using a single training and holdout dataframe 
  """
  import itertools as it
  
  def get_key_to_gridsearch(model_dict):
    models_to_gridsearch = [key for key in list(model_dict.keys()) if model_to_gridsearch in key.lower()]
    assert len(models_to_gridsearch) == 1, "You have two or more keys in your modeling dictionary containing %s" % model_to_gridsearch

    return models_to_gridsearch[0]

  model_key = get_key_to_gridsearch(stage1_dict)

  sorted_names = sorted(param_grid)
  combinations = list(it.product(*(param_grid[name] for name in sorted_names)))

  starter_params = stage1_dict[model_key][0][1].copy()

  final_pd = pd.DataFrame()
    
  pbar = ProgressBar()
  for param_combination in pbar(combinations):
    # update param dictionary for target model
    param_update_dict = {sorted_names[index]:param_combination[index] for index in range(len(sorted_names))}
    new_params = starter_params.copy()
    new_params.update(param_update_dict)
    stage1_dict[model_key][0][1] = new_params
    
    temp_predictions = run_models(train_pd=training_pd, 
                                  test_pd=testing_pd, 
                                  stage1_dict=stage1_dict, 
                                  print_errors=False,
                                  *args, **kwargs)
          
    columns_to_keep = [col for col in temp_predictions if "_model_" in col] + ["sample"]

    results = temp_predictions.groupby(['sample'])[columns_to_keep].agg(['mean', 'median', 'std'])
    results.columns = list(map('_'.join, list(results.columns)))
    results = results.reset_index()

    results.insert(loc=0, column='feature_set', value=str(param_update_dict))
    
    final_pd = final_pd.append(results, ignore_index=True)
    
  return final_pd


# COMMAND ----------

# DBTITLE 1,Pandas Bayesian Optimization
# MAGIC %python
# MAGIC 
# MAGIC def convert_XGB_params_to_LGBM_params(param_dictionary):
# MAGIC   
# MAGIC   """
# MAGIC   Convert an XGB param_grid dictionary into a LightGBM dictionary
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   param_dictionary : Python dictionary 
# MAGIC       XGB parameter dictionary to be converted to a LightGBM dictionary
# MAGIC       Key:Value = XGB Parameter: (Min param value, Max param value)
# MAGIC   """
# MAGIC   
# MAGIC   replacement_dict = {
# MAGIC     "gamma":"min_gain_to_split",
# MAGIC     "num_machines":"num_workers",
# MAGIC     'max_depth': 'max_depth',
# MAGIC     'gamma':'min_gain_to_split',
# MAGIC     'learning_rate' : 'learning_rate',
# MAGIC     'min_child_weight': 'min_child_weight',
# MAGIC     'subsample': 'subsample',
# MAGIC     'colsample_bytree': 'colsample_bytree',
# MAGIC     'max_leaves': 'max_leaves'
# MAGIC   }
# MAGIC   
# MAGIC   return {replacement_dict[key]:value for key, value in param_dictionary.items()}
# MAGIC 
# MAGIC 
# MAGIC def save_best_LGBM_params(pd_df, start_week, parameter_grid=None, error_metric_func = calc_RWMSE, \
# MAGIC                                boost_rounds=1000, early_stop_rounds=100, init_points=3, n_iter=40):
# MAGIC   """
# MAGIC   Run Bayesian optimization over param grid to find best parameters for LightGBM.
# MAGIC   WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       Training data for XGBoost model.
# MAGIC   
# MAGIC   start_week: int
# MAGIC       Week of year after which rolling training/
# MAGIC       testing is performed. 
# MAGIC   
# MAGIC   error_metric_function: function
# MAGIC       Function used to calculate error to measure 
# MAGIC       hyperparameter/training performance.
# MAGIC   
# MAGIC   boost_rounds : int (default = 200)
# MAGIC       Number of boosting rounds to perform.
# MAGIC   
# MAGIC   early_stop_rounds : int (default = 20)
# MAGIC       Number of early stopping rounds for LightGBM model fitting.
# MAGIC       
# MAGIC   init_points: int
# MAGIC       Number of time Bayesian Optimization is called - 
# MAGIC       higher the better.
# MAGIC   
# MAGIC   n_iter: int
# MAGIC       Number of random parameters sets to pick from while 
# MAGIC       maximizing target.
# MAGIC 
# MAGIC   path : string, optional
# MAGIC       Filepath to save best LightGBM parameters.
# MAGIC       
# MAGIC   """
# MAGIC   
# MAGIC   if not parameter_grid:
# MAGIC     parameter_grid = convert_XGB_params_to_LGBM_params(BAYES_OPT_PARAM_GRID)
# MAGIC   
# MAGIC   path = get_temp_path() + '_lgbm_cv_params'
# MAGIC   
# MAGIC   best_params = optimize_LGBM_params(pd_df=pd_df, start_week=start_week, error_metric_func=error_metric_func, parameter_grid=parameter_grid, \
# MAGIC                                            boost_rounds=boost_rounds, early_stop_rounds=early_stop_rounds, init_points=init_points, n_iter=n_iter)['params']
# MAGIC   
# MAGIC   params = {'metric': 'l2, l1',
# MAGIC             'max_depth': int(best_params['max_depth']),
# MAGIC             'gamma': best_params['min_gain_to_split'],
# MAGIC             'learning_rate': best_params['learning_rate'],
# MAGIC             'min_child_weight': best_params['min_child_weight'],
# MAGIC             'subsample': best_params['subsample'],
# MAGIC             'colsample_bytree': best_params['colsample_bytree'],
# MAGIC             'max_leaves': best_params['max_leaves'],
# MAGIC             'objective': 'regression',
# MAGIC             'num_machines':2,
# MAGIC             'nthread': 32,
# MAGIC             'num_boost_round': boost_rounds,
# MAGIC             'verbose': 100
# MAGIC            }
# MAGIC   save_dict_as_csv(params, path)
# MAGIC 
# MAGIC   
# MAGIC def optimize_LGBM_params(pd_df, start_week, error_metric_func, parameter_grid, boost_rounds,\
# MAGIC                                early_stop_rounds, init_points, n_iter):
# MAGIC   """
# MAGIC   Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       Training data for LightGBM model.
# MAGIC       
# MAGIC   start_week: int
# MAGIC       Week of year after which rolling training/
# MAGIC       testing is performed. 
# MAGIC   
# MAGIC   error_metric_function: function
# MAGIC       Function used to calculate error to measure 
# MAGIC       hyperparameter/training performance.
# MAGIC   
# MAGIC   param_grid : dict
# MAGIC       Dictionary of parameter-value pairs.
# MAGIC   
# MAGIC   boost_rounds : int (default = 200)
# MAGIC       Number of boosting rounds to perform.
# MAGIC   
# MAGIC   early_stop_rounds : int (default = 20)
# MAGIC       Number of early stopping rounds for LightGBM model fitting.
# MAGIC 
# MAGIC   init_points: int
# MAGIC       Number of time Bayesian Optimization is called - 
# MAGIC       higher the better.
# MAGIC   
# MAGIC   n_iter: int
# MAGIC       Number of random parameters sets to pick from while 
# MAGIC       maximizing target.
# MAGIC 
# MAGIC   path : string, optional
# MAGIC       Filepath to save best LightGBM parameters.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   max : dict
# MAGIC       Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
# MAGIC   """
# MAGIC 
# MAGIC   def evaluate_LGBM(max_depth, min_gain_to_split, learning_rate, \
# MAGIC                        min_child_weight, subsample, colsample_bytree, max_leaves):  
# MAGIC 
# MAGIC       params = {'metric': 'rmse',
# MAGIC                 'max_depth': int(max_depth),
# MAGIC                 'min_gain_to_split': min_gain_to_split,
# MAGIC                 'learning_rate': learning_rate,
# MAGIC                 'min_child_weight': min_child_weight,
# MAGIC                 'subsample': subsample,
# MAGIC                 'colsample_bytree': colsample_bytree,
# MAGIC                 'max_leaves': int(max_leaves)
# MAGIC                 } 
# MAGIC       
# MAGIC       sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
# MAGIC       error_list = []
# MAGIC       
# MAGIC       for time_period in sorted_time_list:
# MAGIC         dtrain = pd_df[pd_df[TIME_VAR] < time_period]
# MAGIC         holdout_pd = pd_df[pd_df[TIME_VAR] == time_period]
# MAGIC         
# MAGIC         lgbm_model = train_lightGBM(dtrain, params, boost_rounds)
# MAGIC         
# MAGIC         y = holdout_pd[TARGET_VAR].values
# MAGIC         predictions = predict_lightGBM(holdout_pd, lgbm_model)
# MAGIC         
# MAGIC         error_calc = error_metric_func(y, predictions)
# MAGIC     
# MAGIC         error_list.append(error_calc)
# MAGIC         
# MAGIC       np_errors = np.asarray(error_list)
# MAGIC       
# MAGIC       return -1.0 * np.mean(np_errors)
# MAGIC   
# MAGIC   lightGBM_bo = BayesianOptimization(evaluate_LGBM, parameter_grid)
# MAGIC 
# MAGIC   lightGBM_bo.maximize(init_points, n_iter)
# MAGIC   
# MAGIC   return lightGBM_bo.max
# MAGIC 
# MAGIC 
# MAGIC def save_best_XGB_params(pd_df, start_week, parameter_grid=None, error_metric_func=calc_RWMSE,  boost_rounds=1000,\
# MAGIC                           early_stop_rounds=100, init_points=3, n_iter=30):
# MAGIC   
# MAGIC   """
# MAGIC   Run Bayesian optimization over param grid to find best parameters for XGBoost.
# MAGIC   WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       Training data for LightGBM model.
# MAGIC       
# MAGIC   start_week: int
# MAGIC       Week of year after which rolling training/
# MAGIC       testing is performed. 
# MAGIC   
# MAGIC   error_metric_function: function
# MAGIC       Function used to calculate error to measure 
# MAGIC       hyperparameter/training performance.
# MAGIC   
# MAGIC   param_grid : dict
# MAGIC       Dictionary of parameter-value pairs.
# MAGIC   
# MAGIC   boost_rounds : int (default = 200)
# MAGIC       Number of boosting rounds to perform.
# MAGIC   
# MAGIC   early_stop_rounds : int (default = 20)
# MAGIC       Number of early stopping rounds for LightGBM model fitting.
# MAGIC  
# MAGIC  init_points: int
# MAGIC       Number of time Bayesian Optimization is called - 
# MAGIC       higher the better.
# MAGIC   
# MAGIC   n_iter: int
# MAGIC       Number of random parameters sets to pick from while 
# MAGIC       maximizing target.
# MAGIC   """
# MAGIC   
# MAGIC   if not parameter_grid:
# MAGIC     parameter_grid = BAYES_OPT_PARAM_GRID
# MAGIC     
# MAGIC   path = get_temp_path() + '_xgb_cv_params' 
# MAGIC   best_params = optimize_XGB_params(pd_df=pd_df, start_week=start_week, error_metric_func=error_metric_func, parameter_grid=parameter_grid, \
# MAGIC                                           boost_rounds=boost_rounds, early_stop_rounds=early_stop_rounds, init_points=init_points, n_iter=n_iter)['params']
# MAGIC   
# MAGIC   params = {'metric': 'l2, l1',
# MAGIC             'max_depth': int(best_params['max_depth']),
# MAGIC             'gamma': best_params['gamma'],
# MAGIC             'learning_rate': best_params['learning_rate'],
# MAGIC             'min_child_weight': best_params['min_child_weight'],
# MAGIC             'subsample': best_params['subsample'],
# MAGIC             'colsample_bytree': best_params['colsample_bytree'],
# MAGIC             'max_leaves': best_params['max_leaves'],
# MAGIC             'objective': 'regression',
# MAGIC             'num_machines':2,
# MAGIC             'nthread': 32,
# MAGIC             'num_boost_round': boost_rounds,
# MAGIC             'verbose': 100
# MAGIC            }
# MAGIC   save_dict_as_csv(params, path)
# MAGIC 
# MAGIC 
# MAGIC def optimize_XGB_params(pd_df, start_week, error_metric_func,parameter_grid, boost_rounds, \
# MAGIC                           early_stop_rounds, init_points, n_iter):
# MAGIC   
# MAGIC   """
# MAGIC   Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       Training data for LightGBM model.
# MAGIC       
# MAGIC   start_week: int
# MAGIC       Week of year after which rolling training/
# MAGIC       testing is performed. 
# MAGIC   
# MAGIC   error_metric_function: function
# MAGIC       Function used to calculate error to measure 
# MAGIC       hyperparameter/training performance.
# MAGIC   
# MAGIC   param_grid : dict
# MAGIC       Dictionary of parameter-value pairs.
# MAGIC   
# MAGIC   boost_rounds : int (default = 1000)
# MAGIC       Number of boosting rounds to perform.
# MAGIC   
# MAGIC   early_stop_rounds : int (default = 100)
# MAGIC       Number of early stopping rounds for LightGBM model fitting.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   max : dict
# MAGIC       Parameters on grid that maximize chosen error metric of XGBoost model per Bayesian optimization.
# MAGIC   
# MAGIC   """
# MAGIC   
# MAGIC   def evaluate_XGB(max_depth, gamma, learning_rate, min_child_weight, subsample, colsample_bytree, max_leaves):
# MAGIC     params = {  'max_depth': int(max_depth),
# MAGIC                 'gamma': gamma,
# MAGIC                 'learning_rate': learning_rate,
# MAGIC                 'min_child_weight': min_child_weight,
# MAGIC                 'subsample': subsample,
# MAGIC                 'colsample_bytree': colsample_bytree,
# MAGIC                 'max_leaves': int(max_leaves)
# MAGIC                 } 
# MAGIC     
# MAGIC     sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
# MAGIC     error_list = []
# MAGIC     
# MAGIC     for time_period in sorted_time_list:
# MAGIC       
# MAGIC       dtrain = pd_df[pd_df[TIME_VAR] < time_period]
# MAGIC       holdout_pd = pd_df[pd_df[TIME_VAR] == time_period]
# MAGIC       
# MAGIC       xgb_model = train_xgboost(dtrain, params, boost_rounds)
# MAGIC       
# MAGIC       predictions = predict_xgboost(holdout_pd, xgb_model)
# MAGIC       y = holdout_pd[TARGET_VAR].values
# MAGIC       
# MAGIC       error = error_metric_func(y, predictions)
# MAGIC       
# MAGIC       error_list.append(error)
# MAGIC       
# MAGIC     np_errors = np.asarray(error_list)
# MAGIC     
# MAGIC     return -1.0 * np.mean(np_errors)
# MAGIC 
# MAGIC   xgb_bo = BayesianOptimization(evaluate_XGB, parameter_grid)
# MAGIC 
# MAGIC   xgb_bo.maximize(init_points, n_iter)
# MAGIC   
# MAGIC   return xgb_bo.max

# COMMAND ----------

# DBTITLE 1,Dask Modeling
def train_xgboost_dask(dask_df, params):
  """
  Fit a XGBoost model on distributed training data stored in a Dask dataframe.
  """  
  assert dask_client, "No dask client has been defined globally."

  y = dask_df[TARGET_VAR]
  X = dask_df.drop([TARGET_VAR], axis=1)
    
  xgb_model = dxgb.train(dask_client, params, X, y)

  return xgb_model


def predict_xgboost_dask(dask_df, distributed_xgb_model):
  """
  Predict TARGET_VAR using distributed XGBoost model built with Dask
  """
  assert dask_client, "No dask client has been defined globally."
  X = dask_df.drop([TARGET_VAR], axis=1)

  predictions = dxgb.predict(dask_client, distributed_xgb_model, X)
  
  return predictions

# COMMAND ----------

# DBTITLE 1,PySpark Modeling
# MAGIC %python
# MAGIC 
# MAGIC 
# MAGIC def run_models_pyspark(train_df, test_df, stage1_dict, stage2_dict=None, descale=True):
# MAGIC   """
# MAGIC   Fit all stage 1 and stage 2 models and output predictions in PySpark
# MAGIC   """
# MAGIC   
# MAGIC   global model_dict #
# MAGIC   model_dict = {} #
# MAGIC 
# MAGIC   # initialize error dicts ##
# MAGIC   is_stage1_error_dict = {TARGET_VAR : train_df[TARGET_VAR]} #
# MAGIC   oos_stage1_error_dict = {TARGET_VAR : test_df[TARGET_VAR]} #
# MAGIC 
# MAGIC   # vectorize dataframes ##
# MAGIC   vectorized_train_df = vectorize_df_pyspark(train_df) #
# MAGIC   vectorized_test_df = vectorize_df_pyspark(test_df) #
# MAGIC 
# MAGIC   # stage 1 models #
# MAGIC   staged_train_df = train_df #
# MAGIC   staged_test_df = test_df #
# MAGIC   
# MAGIC   for model in stage1_dict: 
# MAGIC     stage1_model_name = model + "_stage1" ##
# MAGIC 
# MAGIC     # fit model #
# MAGIC     stage1_model_fit = vectorized_train_df.pipe(*stage1_dict[model][0]) #
# MAGIC     model_dict[stage1_model_name] = stage1_model_fit #
# MAGIC 
# MAGIC     # run predict function on dataframe ##
# MAGIC     is_stage1_error_dict[stage1_model_name] = vectorized_train_df.pipe(stage1_dict[model][1], stage1_model_fit) #
# MAGIC     oos_stage1_error_dict[stage1_model_name] = vectorized_test_df.pipe(stage1_dict[model][1], stage1_model_fit) #
# MAGIC 
# MAGIC     # add predictions to dataframe for stage 2 #
# MAGIC     staged_train_df = columnwise_union_pyspark(staged_train_df, is_stage1_error_dict[stage1_model_name]\
# MAGIC                                                .withColumnRenamed("prediction", stage1_model_name).select(stage1_model_name)) #
# MAGIC     staged_test_df = columnwise_union_pyspark(staged_test_df, oos_stage1_error_dict[stage1_model_name]\
# MAGIC                                               .withColumnRenamed("prediction", stage1_model_name).select(stage1_model_name)) #
# MAGIC     
# MAGIC   if stage2_dict: ##
# MAGIC 
# MAGIC     # re-vectorize new columns ##
# MAGIC     vectorized_train_df = vectorize_df_pyspark(staged_train_df) #
# MAGIC     vectorized_test_df = vectorize_df_pyspark(staged_test_df) #
# MAGIC 
# MAGIC     # initialize stage 2 error dicts #
# MAGIC     is_stage2_error_dict = {TARGET_VAR : staged_train_df[TARGET_VAR]} #
# MAGIC     oos_stage2_error_dict = {TARGET_VAR : staged_test_df[TARGET_VAR]} #
# MAGIC 
# MAGIC     # stage 2 models
# MAGIC     for model in stage2_dict: ##
# MAGIC       stage2_model_name = model + "_stage2"  ##
# MAGIC 
# MAGIC       # train stage 2 models
# MAGIC       stage2_model_fit = vectorized_train_df.pipe(stage2_dict[model][0]) #   
# MAGIC       model_dict[stage2_model_name] = stage2_model_fit #
# MAGIC 
# MAGIC       # predict ##
# MAGIC       is_stage2_error_dict[stage2_model_name] = vectorized_train_df.pipe(stage2_dict[model][1], stage2_model_fit) #
# MAGIC       oos_stage2_error_dict[stage2_model_name] = vectorized_test_df.pipe(stage2_dict[model][1], stage2_model_fit) #
# MAGIC 
# MAGIC       # create prediction dataframes ##
# MAGIC       predictions_train_df = columnwise_union_pyspark(staged_train_df, is_stage2_error_dict[stage2_model_name]\
# MAGIC                                                       .withColumnRenamed("prediction", stage2_model_name).select(stage2_model_name)) #
# MAGIC       predictions_test_df = columnwise_union_pyspark(staged_test_df, oos_stage2_error_dict[stage2_model_name]\
# MAGIC                                              .withColumnRenamed("prediction", stage2_model_name).select(stage2_model_name)) #
# MAGIC   else: #
# MAGIC     predictions_train_df = staged_train_df #
# MAGIC     predictions_test_df = staged_test_df
# MAGIC       
# MAGIC   # only select columns that denote predictions ##
# MAGIC   is_predictions_df = predictions_train_df.select([TARGET_VAR] + [column for column in model_dict]) #
# MAGIC   oos_predictions_df = predictions_test_df.select([TARGET_VAR] + [column for column in model_dict]) #
# MAGIC 
# MAGIC   if descale == True: ##
# MAGIC   # unscale columns (e.g., delog Sales) ##
# MAGIC     final_is_pred_df = undo_scaling_operations_pyspark(is_predictions_df) #
# MAGIC     final_oos_pred_df = undo_scaling_operations_pyspark(oos_predictions_df) #
# MAGIC   else: ##
# MAGIC     final_is_pred_df = is_predictions_df #
# MAGIC     final_oos_pred_df = oos_predictions_df #
# MAGIC     
# MAGIC   # Print error metrics #
# MAGIC   print_error_metrics_pyspark(final_is_pred_df, final_oos_pred_df) #
# MAGIC 
# MAGIC   return {"In-sample" : final_is_pred_df, #
# MAGIC           "Out-of-sample" : final_oos_pred_df} #
# MAGIC 
# MAGIC 
# MAGIC def get_coefficients_pyspark(model, feature_list):
# MAGIC   """
# MAGIC   Return selected coefficients from a cross-validated Elastic Net model.
# MAGIC   """ 
# MAGIC   return pd.DataFrame(model.coefficients.tolist(), index=feature_list, columns=["value"])
# MAGIC 
# MAGIC 
# MAGIC def train_GBM_pyspark(vectorized_df, param_dict=None, max_iter=100):
# MAGIC   """
# MAGIC   Trains Spark's GBM regressor
# MAGIC   """
# MAGIC    
# MAGIC   GBM_model = GBTRegressor(featuresCol = 'features', labelCol=TARGET_VAR)
# MAGIC   
# MAGIC   # Set modeling object parameters using dictionary
# MAGIC   if param_dict:
# MAGIC     for key, value in param_dict.items():
# MAGIC       GBM_model.set(eval("GBM_model." + key), value)
# MAGIC 
# MAGIC   GBM_model_fit = GBM_model.fit(vectorized_df)    
# MAGIC   
# MAGIC   return GBM_model_fit
# MAGIC 
# MAGIC 
# MAGIC def train_elastic_net_pyspark(vectorized_df, param_dict=None, max_iter=10, reg_param=.0001, elastic_net_param=.8):
# MAGIC   """
# MAGIC   Trains cross-validated elastic net using PySpark
# MAGIC   """
# MAGIC   
# MAGIC   EN_model = LinearRegression(featuresCol = 'features', 
# MAGIC                               labelCol=TARGET_VAR, 
# MAGIC                               maxIter=max_iter, 
# MAGIC                               regParam=reg_param, 
# MAGIC                               elasticNetParam=elastic_net_param)
# MAGIC   
# MAGIC   # Set modeling object parameters using dictionary
# MAGIC   if param_dict:
# MAGIC     for key, value in param_dict.items():
# MAGIC       EN_model.set(eval("EN_model." + key), value)
# MAGIC   
# MAGIC   EN_model_fit = EN_model.fit(vectorized_df)
# MAGIC 
# MAGIC   return EN_model_fit
# MAGIC 
# MAGIC 
# MAGIC def train_rf_pyspark(vectorized_df, param_dict=None, max_iter=100):
# MAGIC   """
# MAGIC   Trains Spark's GBM regressor
# MAGIC   """
# MAGIC      
# MAGIC   RF_model = RandomForestRegressor(featuresCol = 'features', labelCol=TARGET_VAR)
# MAGIC   
# MAGIC   # Set modeling object parameters using dictionary
# MAGIC   if param_dict:
# MAGIC     for key, value in param_dict.items():
# MAGIC       RF_model.set(eval("RF_model." + key), value)
# MAGIC   
# MAGIC   RF_model_fit = RF_model.fit(vectorized_df)
# MAGIC 
# MAGIC   return RF_model_fit
# MAGIC 
# MAGIC 
# MAGIC def vectorize_df_pyspark(df):
# MAGIC   """
# MAGIC   Returns a vectorized copy of a dataframe for processing in PySpark
# MAGIC   """
# MAGIC   feature_list = getFeatureList(df)
# MAGIC   
# MAGIC   vectorized_df = (VectorAssembler(inputCols = feature_list, outputCol = 'features')\
# MAGIC                    .transform(df)\
# MAGIC                    .select(['features', TARGET_VAR]))
# MAGIC 
# MAGIC   return vectorized_df
# MAGIC 
# MAGIC 
# MAGIC def predict_pyspark_model(df, fitted_model):
# MAGIC   """
# MAGIC   Predict TARGET_VAR using an elastic net model and inputs from a PySpark dataframe
# MAGIC   """
# MAGIC   return fitted_model.transform(df)

# COMMAND ----------

# DBTITLE 1,PySpark Modeling Tuning
# MAGIC %python
# MAGIC 
# MAGIC 
# MAGIC def get_default_params_pyspark(model_obj):
# MAGIC   """
# MAGIC   Get a default param grid to grid-search over prior to modeling
# MAGIC   Param explainations: https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegressionModel.evaluate
# MAGIC   """
# MAGIC   default_param_grids = {GBTRegressor: ParamGridBuilder() \
# MAGIC                          .addGrid(GBTRegressor.maxDepth, [1, 6, 12]) \
# MAGIC                          .addGrid(GBTRegressor.maxBins, [60, 70]) \
# MAGIC                          .addGrid(GBTRegressor.maxIter, [50]) \
# MAGIC                          .addGrid(GBTRegressor.subsamplingRate, [.8, 1]) \
# MAGIC                          .addGrid(GBTRegressor.minInfoGain, [0]) \
# MAGIC                          .addGrid(GBTRegressor.minInstancesPerNode, [1, 2, 5]) \
# MAGIC                          .addGrid(GBTRegressor.stepSize, [.1, .001]),
# MAGIC 
# MAGIC                          RandomForestRegressor: ParamGridBuilder() \
# MAGIC                          .addGrid(RandomForestRegressor.maxDepth, [1, 6, 12]) \
# MAGIC                          .addGrid(RandomForestRegressor.maxBins, [60, 70]) \
# MAGIC                          .addGrid(RandomForestRegressor.numTrees, [500]) \
# MAGIC                          .addGrid(RandomForestRegressor.subsamplingRate, [.8, .9, 1.0]) \
# MAGIC                          .addGrid(RandomForestRegressor.minInfoGain, [0]) \
# MAGIC                          .addGrid(RandomForestRegressor.minInstancesPerNode, [1, 2, 5]),\
# MAGIC 
# MAGIC                         LinearRegression: ParamGridBuilder() \
# MAGIC                          .addGrid(LinearRegression.elasticNetParam, [0, .1, .3, .5, .7, .9, .1]) \
# MAGIC                          .addGrid(LinearRegression.regParam, [.1, .5, 1, 2]),
# MAGIC                         }
# MAGIC     
# MAGIC   selected_grid = default_param_grids[model_obj].build()
# MAGIC 
# MAGIC   return selected_grid
# MAGIC 
# MAGIC 
# MAGIC def save_best_params_pyspark(model_obj, df, param_grid=None, n_folds=5): 
# MAGIC   """
# MAGIC   Uses PySpark's cross-validation pipeline to grid search over a param grid to select the possible combination of parameters
# MAGIC   """
# MAGIC 
# MAGIC   if not param_grid:
# MAGIC     param_grid = get_default_params_pyspark(model_obj) 
# MAGIC 
# MAGIC   feature_list = getFeatureList(df)
# MAGIC   vectorizer = VectorAssembler(inputCols = feature_list, outputCol = 'features')
# MAGIC   model = model_obj(featuresCol = 'features', labelCol=TARGET_VAR)
# MAGIC 
# MAGIC   pipeline = Pipeline(stages=[vectorizer, model])
# MAGIC 
# MAGIC   cross_validator = CrossValidator(estimator=pipeline,
# MAGIC                             estimatorParamMaps=param_grid,
# MAGIC                             evaluator=RegressionEvaluator(labelCol=TARGET_VAR),
# MAGIC                             numFolds=n_folds)
# MAGIC 
# MAGIC   cv_fit = cross_validator.fit(df)
# MAGIC   best_params = cv_fit.getEstimatorParamMaps()[ np.argmax(cv_fit.avgMetrics) ]
# MAGIC 
# MAGIC   # Conver to a normal string, value dictionary so that it can be easily saved 
# MAGIC   param_dict = {}   
# MAGIC   for key, value in best_params.items():
# MAGIC     param_dict[(key.name)] = value
# MAGIC     
# MAGIC   print("Chosen parameters are as follows: %s \n" % param_dict)
# MAGIC   
# MAGIC   save_dict_as_csv(param_dict, get_filestore_path() + get_pipeline_name() + "_" + model_obj.__name__)
# MAGIC   
# MAGIC   return param_dict

# COMMAND ----------

# DBTITLE 1,Pandas Recursive Modeling Pipeline
def run_recursive_pipeline(train_pd, test_pd, number_of_periods_per_forecast, stage1_dict, stage2_dict=None, \
                                     pred_col_name='xgb_model_stage2', time_unit='weeks', datetime_format="%Y%U-%w", resample_format='W-MON'):
  """
  Do recursive predictions per Hyndman's paper (https://robjhyndman.com/publications/rectify/)
  Summary of approach available here: https://machinelearningmastery.com/multi-step-time-series-forecasting/
  
  Basic idea: train a model on the training set, then make step-by-step predictions for each future timestep so you can recalculate lags inbetween.
    prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
    prediction(t+2) = model(prediction(t+1), obs(t-1), ..., obs(t-n))
    ...
    
  NOTE: This function implicitly trims your in-sample observations to save on memory during the loop process. 
  """
  
  # define helper functions
  def filter_columns_from_df2 (df1, df2):
    repeated_columns = [col for col in set((df1.columns).intersection(df2.columns))]
    drop_cols = [col for col in repeated_columns if col not in hierarchy + [TIME_VAR]]
    df2 = df2.drop(drop_cols, axis =1)
    return df2

  def drop_and_rename_lagged_cols(lagged_df, number_of_periods_per_forecast, pred_col_name):
    
    # drop unwanted columns from a 'predictions df' to be a 'holdout df'
    tar_var_lags  = [(TARGET_VAR + '_lag_' + str(lag_num)) for lag_num in lag_weeks]
    drop_cols = tar_var_lags + list(MODEL_DICT.keys()) + ['sample']
    lagged_df = lagged_df.drop(drop_cols, axis =1)
  
    # rename lags
    pred_lags = [('col_to_lag' + '_lag_' + str(lag_num)) for lag_num in lag_weeks]
    rename_dict = dict(zip(pred_lags, tar_var_lags))
    lagged_df = lagged_df.rename(columns=rename_dict) 
    return lagged_df 

  def get_prepped_train_data(train_preds, number_of_periods_per_forecast, time_unit, datetime_format):
    relevant_train_data = filter_recent_time_periods(train_preds, number_of_periods_per_forecast, time_unit, datetime_format)
    relevant_train_data['col_to_lag'] = relevant_train_data[TARGET_VAR]
    return relevant_train_data

  def filter_recent_time_periods(train_data, number_of_periods_per_forecast, time_unit, datetime_format):
    last_train_period = max(train_data[TIME_VAR])
    req_start_time = update_time(last_train_period, (-1*number_of_periods_per_forecast), time_unit=time_unit, datetime_format=datetime_format)
    relevant_train_data = train_data[train_data[TIME_VAR] > int(req_start_time)]
    return relevant_train_data
    
  hierarchy = correct_suffixes_in_list(train_pd, get_hierarchy())
    
  train_preds = fit_modeling_pipeline(train_pd, stage1_dict, stage2_dict)
  
  sorted_time_list = sorted(test_pd[TIME_VAR].unique())
  first_holdout_week = sorted_time_list.pop(0)
  
  lagged_df_for_predict = test_pd[test_pd[TIME_VAR] == first_holdout_week]

  # add last few weeks of train data to get actual lags for initial holdout
  relevant_train_pd = filter_recent_time_periods(train_pd, number_of_periods_per_forecast, time_unit, datetime_format)
  holdout_pd_with_train = concatenate_dfs(relevant_train_pd, test_pd)
    
  # train data prepped to ensure predictions contain actuals 
  stacked_prev_preds = get_prepped_train_data(train_preds, number_of_periods_per_forecast, time_unit, datetime_format)
  
  # clear RAM
  del train_preds, relevant_train_pd

  # get the lag periods that need to be calculated at each timestep
  lag_weeks = [int(col.split("_")[-1]) for col in train_pd.columns if col.startswith(TARGET_VAR + "_lag_")]
  lag_weeks = [num for num in lag_weeks if num <= number_of_periods_per_forecast]
  
  pbar = ProgressBar()
  for time_period in pbar(sorted_time_list):
            
    # ---------PREDICTING------------
    
    # predict for current time period using the lagged preds from previous weeks and keep stacking 
    curr_preds = predict_modeling_pipeline(lagged_df_for_predict, stage1_dict, stage2_dict)
    
    #create a temp column for creating lags
    curr_preds['col_to_lag'] = curr_preds[pred_col_name]
    
    stacked_prev_preds = concatenate_dfs(stacked_prev_preds, curr_preds)
    
    # limit the holdout_df to only have necessary time periods of data for merge & lagging
    curr_holdout = holdout_pd_with_train[(holdout_pd_with_train[TIME_VAR] <= time_period) & 
                                         (holdout_pd_with_train[TIME_VAR] >= (time_period - number_of_periods_per_forecast))]
      
    # merge with stacked preds to get predictions for t-1 weeks to be used in lag calculation
    stacked_prev_preds_for_merge = filter_columns_from_df2(curr_holdout, stacked_prev_preds)

    pred_df_to_lag = curr_holdout.merge(stacked_prev_preds_for_merge, how = 'left', on = hierarchy + [TIME_VAR])
    
    # ----------LAGGING-------------
    
    # calculate lagged features using stacked predictions
    lagged_df_for_predict = get_lagged_features(pred_df_to_lag, ['col_to_lag'], lag_weeks, datetime_format=datetime_format, resample_format=resample_format)
    
    # replace target_var lags with prediction lags and remove unwanted columns to match holdout columns
    lagged_df_for_predict = drop_and_rename_lagged_cols(lagged_df_for_predict, number_of_periods_per_forecast, pred_col_name)
    
    # save only this time period's data to make preds in next loop
    lagged_df_for_predict = lagged_df_for_predict[lagged_df_for_predict[TIME_VAR] == time_period]
    
    # reindex features to match the order of the original test set 
    lagged_df_for_predict= lagged_df_for_predict.reindex(test_pd.columns, axis=1)
  
  # last holdout week's predictions
  curr_preds = predict_modeling_pipeline(lagged_df_for_predict, stage1_dict, stage2_dict)
  stacked_prev_preds = concatenate_dfs(stacked_prev_preds, curr_preds)
  
  stacked_prev_preds = stacked_prev_preds.drop('col_to_lag', axis=1)
      
  return stacked_prev_preds

# COMMAND ----------

# DBTITLE 1,Pandas Hurdle Model
# MAGIC %python
# MAGIC 
# MAGIC def run_hurdle_models(train_pd, test_pd, hurdle_dict, stage1_dict, stage2_dict=None, untransformDF = True, *args, **kwargs):
# MAGIC   """
# MAGIC   Similar to run_models, but uses a hurdle model to treat the zeroes differently from the other values
# MAGIC   """
# MAGIC   
# MAGIC   def replace_target_with_boolean(df):
# MAGIC     """
# MAGIC     Convert the target_row into a boolean value
# MAGIC     """
# MAGIC     df_copy = df.copy()
# MAGIC     df_copy[TARGET_VAR] = (df_copy[TARGET_VAR] > 0).astype(int)
# MAGIC     
# MAGIC     return df_copy
# MAGIC   
# MAGIC   
# MAGIC   def print_hurdle_model_proportions(train_act, test_act, train_pred, test_pred):
# MAGIC     """
# MAGIC     Prints helpful metrics to inform the user on what proportion of rows are being treated as zero-predictions in the test and training sets
# MAGIC     """
# MAGIC     print("Jaccard Accuracy: In-Sample = %s%%; OOS = %s%% \n" % (
# MAGIC       np.round(calc_classification_accuracy(train_act, train_pred),4)*100,
# MAGIC       np.round(calc_classification_accuracy(test_act, test_pred), 4)*100
# MAGIC     ))
# MAGIC 
# MAGIC     print("Percent of Non-Zero Rows: In-Sample = %s%%; OOS = %s%% \n" % (
# MAGIC       np.round(train_pred.sum()/len(train_pred), 4)*100,
# MAGIC       np.round(test_pred.sum()/len(test_pred), 4)*100
# MAGIC     ))
# MAGIC 
# MAGIC   assert len(hurdle_dict) == 1, "Please only use one model in your hurdle_dict"
# MAGIC   
# MAGIC   print("Double-check that your hurdle_dict params are set up for binary predictions (e.g., objective='binary:logistic', eval_metrics='auc')! \n")
# MAGIC 
# MAGIC   bin_train_pd, bin_test_pd = (replace_target_with_boolean(train_pd), 
# MAGIC                                replace_target_with_boolean(test_pd))
# MAGIC     
# MAGIC    # even though this dict only contains one key, I'm leaving it as a dict so that it mirrors the stage1 and stage2 dicts 
# MAGIC   for model in hurdle_dict:
# MAGIC     hurdle_model_name = model + "_hurdle"
# MAGIC     
# MAGIC     hurdle_model_fit = train_pd.pipe(*hurdle_dict[model][0])   
# MAGIC         
# MAGIC     bin_train_index = bin_train_pd.pipe(hurdle_dict[model][1], hurdle_model_fit)
# MAGIC     bin_test_index = bin_test_pd.pipe(hurdle_dict[model][1], hurdle_model_fit)
# MAGIC     
# MAGIC     print_hurdle_model_proportions(train_act=bin_train_pd[TARGET_VAR], test_act=bin_test_pd[TARGET_VAR], 
# MAGIC                                 train_pred=bin_train_index, test_pred=bin_test_index)
# MAGIC         
# MAGIC   # filter training data to only include non-zeroes
# MAGIC   filtered_train_pd = train_pd[bin_train_index.astype(bool)]
# MAGIC   filtered_test_pd = test_pd[bin_test_index.astype(bool)]
# MAGIC   
# MAGIC   # run models on filtered data
# MAGIC   predictions_pd = run_modeling_pipeline(train_pd=filtered_train_pd, test_pd=filtered_test_pd, stage1_dict=stage1_dict, stage2_dict=stage2_dict)
# MAGIC   
# MAGIC   # append predictions for rows where actuals is expected to equal zero
# MAGIC   modeling_columns_to_fill_with_zeroes = [col for col in list(predictions_pd.columns) if "_stage" in col]
# MAGIC   zero_dict = {col:0 for col in modeling_columns_to_fill_with_zeroes}
# MAGIC   
# MAGIC   append_train_pd = train_pd[~bin_train_index.astype(bool)]\
# MAGIC                       .assign(sample = "IS")
# MAGIC   append_test_pd = test_pd[~bin_test_index.astype(bool)]\
# MAGIC                       .assign(sample = "OOS")
# MAGIC   
# MAGIC   append_train_pd = append_train_pd.assign(**zero_dict)
# MAGIC   append_test_pd = append_test_pd.assign(**zero_dict)
# MAGIC 
# MAGIC   final_predictions_pd = pd.concat([predictions_pd, append_train_pd, append_test_pd])
# MAGIC   
# MAGIC   
# MAGIC   if untransformDF:
# MAGIC     return untransform_prediction_df(final_predictions_pd, **kwargs)
# MAGIC   else:
# MAGIC     return final_predictions_pd

# COMMAND ----------

# DBTITLE 1,PySpark Modeling on Partitions
# MAGIC %python
# MAGIC 
# MAGIC def create_schema_object_list(column_names, data_type):
# MAGIC   """
# MAGIC   Creates a list of struct objects given column names and a data type.
# MAGIC   """
# MAGIC   
# MAGIC   return [StructField(each, data_type, True) for each in column_names]
# MAGIC   
# MAGIC   
# MAGIC def create_prediction_schema(error_func_list, descale, suppressErrorMetrics, print_errors):
# MAGIC   """
# MAGIC   Creates the schema to be appended to the training data's schema for running partitioned models. 
# MAGIC   """
# MAGIC   
# MAGIC   stage1_model = ["".join((model, "_stage1")) for model in stage1_models]
# MAGIC   stage2_model = ["".join((model, "_stage2")) for model in stage2_models]
# MAGIC   models = stage1_model + stage2_model
# MAGIC   
# MAGIC   final_structs = create_schema_object_list(models, DoubleType())
# MAGIC   
# MAGIC   if suppressErrorMetrics==False:      
# MAGIC     errors = ["".join((model + "_" + error_func.__name__)) for model in models for error_func in error_func_list]
# MAGIC     
# MAGIC     final_structs = final_structs + create_schema_object_list(errors, DoubleType())
# MAGIC         
# MAGIC   if descale:
# MAGIC     TARGET_VAR = get_base_target_var()
# MAGIC   else:
# MAGIC     TARGET_VAR = get_base_target_var() + "_log"
# MAGIC   
# MAGIC   final_structs = final_structs + create_schema_object_list([TARGET_VAR], DoubleType())\
# MAGIC                                 + create_schema_object_list(['sample'],StringType())
# MAGIC                                      
# MAGIC   return StructType(final_structs)
# MAGIC   
# MAGIC   
# MAGIC def combine_schemas(list_of_SchemaLists):
# MAGIC   """
# MAGIC   Given a list of schemas, combines it into one StuctType Schema.
# MAGIC   """
# MAGIC   
# MAGIC   struct_list = list({schema_object for schema_list in list_of_SchemaLists for schema_object in schema_list})
# MAGIC   return struct_list
# MAGIC   
# MAGIC 
# MAGIC def descale_schema(final_schemalist):
# MAGIC   """
# MAGIC   Given the final schema list, ensures correct output target variable.
# MAGIC   """
# MAGIC   
# MAGIC   final_schemalist.remove(StructField(get_base_target_var() + "_log", DoubleType(), True))
# MAGIC   return StructType(final_schemalist)
# MAGIC   
# MAGIC   
# MAGIC def run_models_on_partitions (final_df, group_column, training_date_end, holdout_start_beg=None, untransform = False, 
# MAGIC                           suppressErrorMetrics=False, descale = True, print_errors=False, error_func_list = [calc_RMSE, calc_RWMSE, calc_APA]):
# MAGIC   """
# MAGIC   Run models on a dataset partitioned by a given column.
# MAGIC   """
# MAGIC   
# MAGIC   final_df = ensure_is_pyspark_df(final_df)
# MAGIC   
# MAGIC   if not holdout_start_beg:
# MAGIC     holdout_start_beg = int(update_time(training_date_end, 1))
# MAGIC   
# MAGIC   result_schema = descale_schema(combine_schemas([create_prediction_schema(error_func_list = error_func_list, \
# MAGIC                                                                                                descale = descale, \
# MAGIC                                                                                                suppressErrorMetrics=suppressErrorMetrics, \
# MAGIC                                                                                                print_errors=print_errors), \
# MAGIC                                                             final_df.schema]))
# MAGIC   
# MAGIC   
# MAGIC   @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
# MAGIC   def run_single_partition(pdf):     
# MAGIC     dtrain = pdf[pdf[TIME_VAR] <= training_date_end]
# MAGIC     holdout = pdf[pdf[TIME_VAR] > holdout_start_beg]
# MAGIC     
# MAGIC     partition_pred = run_modeling_pipeline(dtrain, holdout, stage1_models, stage2_models)
# MAGIC     if untransform:
# MAGIC       partition_pred = untransform_prediction_df(partition_pred, descale=descale, \
# MAGIC                                                suppressErrorMetrics=suppressErrorMetrics, print_errors=print_errors)
# MAGIC       
# MAGIC     return partition_pred
# MAGIC   
# MAGIC   final_pred_df = final_df.groupby(group_column).apply(run_single_partition)
# MAGIC   return final_pred_df

# COMMAND ----------

# DBTITLE 1,Workbench - General + ToDos
# MAGIC %python
# MAGIC 
# MAGIC #TODO throwing errors if you don't include eval_metrics. could be updated to only use one matching dictionary
# MAGIC def get_LGBM_params_from_XGB_params(xgb_params):
# MAGIC   """
# MAGIC   Get LightGBM parameters from a dictionary of XGBoost parameters.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   xgb_params : dict
# MAGIC       Dictionary of parameter-value pairs for XGBoost.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   lightGBM_params : dict
# MAGIC       Dictionary of parameter-value pairs for LightGBM.
# MAGIC   """
# MAGIC 
# MAGIC   lightGBM_params = {key.replace('gamma', 'min_gain_to_split'):value for key, value in param_grid.items()}
# MAGIC   lightGBM_params = {key.replace('num_workers','num_machines'):value for key, value in param_grid.items()}
# MAGIC   lightGBM_params['objective'] = 'regression'
# MAGIC   lightGBM_params = lightGBM_params.pop('eval_metrics')
# MAGIC   lightGBM_params['metric'] = {'l2','l1'}
# MAGIC 
# MAGIC   return lightGBM_params

# COMMAND ----------

# DBTITLE 1,Workbench - Run Models on Partitions w/o Untransform.
# MAGIC %python
# MAGIC 
# MAGIC #TODO update to match naming conventions
# MAGIC def createSchemaObjectList(column_names, data_type):
# MAGIC   """
# MAGIC   Creates a list of struct objects given column names and a data type.
# MAGIC   """
# MAGIC   
# MAGIC   return [StructField(each, data_type, True) for each in column_names]
# MAGIC   
# MAGIC   
# MAGIC def createPredictionSchema():
# MAGIC   """
# MAGIC   Creates the schema to be appended to the training data's schema for running partitioned models. 
# MAGIC   """
# MAGIC   
# MAGIC   stage1_model = ["".join((model, "_stage1")) for model in stage1_models]
# MAGIC   stage2_model = ["".join((model, "_stage2")) for model in stage2_models]
# MAGIC   models = stage1_model + stage2_model
# MAGIC   
# MAGIC   final_structs = [] + createSchemaObjectList(models, DoubleType())\
# MAGIC                      + createSchemaObjectList(['sample'],StringType())
# MAGIC                                      
# MAGIC   return StructType(final_structs)
# MAGIC 
# MAGIC   
# MAGIC def combineSchemas(list_of_SchemaLists):
# MAGIC   """
# MAGIC   Given a list of schemas, combines it into one StuctType Schema.
# MAGIC   """
# MAGIC   
# MAGIC   struct_list = list({schema_object for schema_list in list_of_SchemaLists for schema_object in schema_list})
# MAGIC   return StructType(struct_list)
# MAGIC   
# MAGIC 
# MAGIC def runModelsOnPartitions(final_df, group_column, training_date_end, holdout_start_beg=None):
# MAGIC   """
# MAGIC   Run models on a dataset partitioned by a given column. Input and output should both be Spark Dataframes
# MAGIC   """
# MAGIC   
# MAGIC   final_df = ensure_is_pyspark_df(final_df)
# MAGIC   
# MAGIC   if not holdout_start_beg:
# MAGIC     holdout_start_beg = int(update_time(training_date_end, 1))
# MAGIC   
# MAGIC   result_schema = combineSchemas([createPredictionSchema(), final_df.schema])
# MAGIC   
# MAGIC   @pandas_udf(result_schema, PandasUDFType.GROUPED_MAP)
# MAGIC   def run_single_partition(pdf):    
# MAGIC     dtrain = pdf[pdf[TIME_VAR] <= training_date_end]
# MAGIC     holdout = pdf[pdf[TIME_VAR] >= holdout_start_beg]
# MAGIC     
# MAGIC     partition_pred = run_modeling_pipeline(dtrain, holdout, stage1_models, stage2_models)
# MAGIC     
# MAGIC     return partition_pred
# MAGIC   
# MAGIC   final_pred_df = final_df.groupby(group_column).apply(run_single_partition)
# MAGIC   
# MAGIC   return final_pred_df

# COMMAND ----------

# MAGIC %python 
# MAGIC 
# MAGIC def prepare_for_prophet(pd_df):
# MAGIC   """
# MAGIC   Prepares a pandas df to be ready for prophet modeling. 
# MAGIC   """
# MAGIC   
# MAGIC   proph_pd = pd_df.copy()
# MAGIC   proph_pd = ensure_datetime_in_pd(proph_pd)
# MAGIC   proph_pd = proph_pd.rename(columns={TARGET_VAR:'y', 'datetime':'ds'})
# MAGIC   return proph_pd  
# MAGIC 
# MAGIC 
# MAGIC def train_prophet_model (pd_df, params, cols_to_include=None, **kwargs):
# MAGIC   """
# MAGIC   Fit Prophet model on training data.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas DataFrame
# MAGIC       DataFrame with training data (X and y).
# MAGIC   
# MAGIC   cols_to_include : list of strings
# MAGIC       List of feature name strings that need to be included in training besides the hierarchy columns. 
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   prophet_model : ProphetModel
# MAGIC       Fit Prophet model.
# MAGIC       
# MAGIC   Additional params to tune
# MAGIC   -------
# MAGIC   growth_inp : string
# MAGIC       Can either be set to 'linear' or 'logistic'. Logistic can be used if you know what the upper and lower
# MAGIC       'saturation' limits of the TARGET_VAR.
# MAGIC       
# MAGIC   seasonality_mode : string (default = additive)
# MAGIC       Can either be set to 'additive' or 'multiplicative'. Can use multiplicative if seasonality trends have stronger 
# MAGIC       effects in later time periods.
# MAGIC   
# MAGIC   daily_seasonality : Boolean (default = False)
# MAGIC       Set to true if data is at daily level.
# MAGIC   
# MAGIC   weekly_seasonality : string (default = False)
# MAGIC       Set to true if data is at weekly level and has weekly trends.
# MAGIC   
# MAGIC   yearly_seasonlity : string (default = False)
# MAGIC       Set to true if data has yearly trends.
# MAGIC   """
# MAGIC 
# MAGIC   
# MAGIC   pd_df = prepare_for_prophet(pd_df)
# MAGIC     
# MAGIC   pro_regressor = Prophet(**params, **kwargs)
# MAGIC   
# MAGIC   indexed_hierarchy =   [col + "_index" for col in get_hierarchy()]
# MAGIC 
# MAGIC   if cols_to_include:
# MAGIC     addl_regressors = indexed_hierarchy + cols_to_include
# MAGIC   else:
# MAGIC     addl_regressors = indexed_hierarchy
# MAGIC     
# MAGIC   for each in addl_regressors:
# MAGIC       pro_regressor.add_regressor(each)
# MAGIC     
# MAGIC   prophet_model = pro_regressor.fit(pd_df)
# MAGIC   
# MAGIC   return prophet_model
# MAGIC 
# MAGIC 
# MAGIC def predict_prophet_model (pd_df, prophet_model, keep_bounds = False):
# MAGIC   """
# MAGIC   Predict TARGET_VAR using Prophet model.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas DataFrame
# MAGIC       DataFrame with training data (X and y).
# MAGIC   
# MAGIC   prophet_model : ProphetModel
# MAGIC       Fit Prophet model.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   predictions : array-like
# MAGIC       Predicted values of TARGET_VAR using data from pd_df.
# MAGIC   """
# MAGIC   
# MAGIC   pd_df = prepare_for_prophet(pd_df)
# MAGIC   
# MAGIC   prediction_df = prophet_model.predict(pd_df)
# MAGIC   
# MAGIC   predictions = prediction_df['yhat']
# MAGIC   
# MAGIC   if keep_bounds:
# MAGIC     predictions = prediction_df[['yhat','yhat_lower', 'yhat_upper']]
# MAGIC   
# MAGIC   return predictions

# COMMAND ----------

# DBTITLE 1,Workbench - Corey add on 2/15/2020 - MLR for coefficient impact - "interpretation"
# MAGIC 
# MAGIC %python
# MAGIC 
# MAGIC #TODO naming conventions, too many comments
# MAGIC def trainMLR(pd, x_vars_as_list, y_var):
# MAGIC   import random
# MAGIC   import statsmodels.api as sm
# MAGIC   
# MAGIC   ## Define X and y for fit/predict
# MAGIC   random.shuffle(x_vars_as_list)
# MAGIC   y = pd[y_var]
# MAGIC   X = pd[x_vars_as_list]
# MAGIC   X = sm.add_constant(X)  ## must add a constant
# MAGIC 
# MAGIC   ## Fit model + predict based on model
# MAGIC   model = sm.OLS(y, X).fit()
# MAGIC   predictions = model.predict(X)  ## Note: not used at this time
# MAGIC 
# MAGIC   ## Pull parameters and tvalues - [1:] to drop constant term
# MAGIC   model_params = model.params[1:]
# MAGIC   model_tvals = model.tvalues[1:]
# MAGIC 
# MAGIC   ## Zip into dictionary - variable with coeff and associated t-value
# MAGIC   model_output = dict(zip(mlr_x_cols, zip(model_params, model_tvals)))
# MAGIC 
# MAGIC   ## Sort dictionary by abs(t-value) - provide sense of RELATIVE impact on y
# MAGIC   sorted_model_output = sorted(model_output.items(), key=lambda x: np.abs(x[1][1]), reverse=True) 
# MAGIC   return sorted_model_output
# MAGIC 
# MAGIC def corrMatrixReview(pd, x_vars_as_list):
# MAGIC   temp_pd = pd[x_vars_as_list]
# MAGIC   corr_mat = temp_pd.corr()
# MAGIC   return corr_mat
# MAGIC 
# MAGIC ## Note - must use display() function to show plots in Databricks
# MAGIC def corrMatrixHeatmap(pd, x_vars_as_list):
# MAGIC   import seaborn as sns
# MAGIC   heatmap = sns.heatmap(corrMatrixReview(pd, x_vars_as_list), annot=True)
# MAGIC   return heatmap

# COMMAND ----------

# DBTITLE 1,Workbench - Methods for PyTorch NN (DJMP) - COREY COMMENTING OUT
# MAGIC %python
# MAGIC 
# MAGIC # def create_sequences(input_data, n_steps):
# MAGIC #   sequences = []
# MAGIC #   input_length = len(input_data)
# MAGIC #   for i in range(input_length - n_steps):
# MAGIC #     seq = input_data[i:i + n_steps]
# MAGIC #     value = input_data[i + n_steps:i + n_steps + 1]
# MAGIC #     sequences.append((seq, value))
# MAGIC #   return sequences 
# MAGIC 
# MAGIC # def train_LSTM(train_sequences, 
# MAGIC #                learning_rate, n_steps, hidden_layer_size, dropout,
# MAGIC #                torch_loss_function, torch_optimizer, num_epochs, 
# MAGIC #                verbose = False):
# MAGIC 
# MAGIC #   class db_LSTM(nn.Module):
# MAGIC #     def __init__(self, input_size = 1, hidden_layer_size = 100, output_size = 1):
# MAGIC #       super().__init__()
# MAGIC #       self.hidden_layer_size = hidden_layer_size
# MAGIC #       self.lstm = nn.LSTM(input_size,
# MAGIC #                           hidden_layer_size,
# MAGIC #                           dropout = dropout)
# MAGIC #       self.linear = nn.Linear(hidden_layer_size, output_size)
# MAGIC #       self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
# MAGIC #                           torch.zeros(1,1,self.hidden_layer_size))
# MAGIC 
# MAGIC #     def forward(self, input_sequence):
# MAGIC #       lstm_out, self.hidden_cell = self.lstm(input_sequence.view(len(input_sequence), 1, -1), self.hidden_cell)
# MAGIC #       predictions = self.linear(lstm_out.view(len(input_sequence), -1))
# MAGIC #       return predictions[-1]
# MAGIC 
# MAGIC #   model = db_LSTM(hidden_layer_size = hidden_layer_size)
# MAGIC #   optimizer = torch_optimizer(model.parameters(), lr = learning_rate)
# MAGIC   
# MAGIC #   warnings.filterwarnings("ignore")
# MAGIC 
# MAGIC #   for i in range(num_epochs):
# MAGIC #     for seq, value in train_onecombo_sequences:
# MAGIC #       optimizer.zero_grad()
# MAGIC #       model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
# MAGIC #                            torch.zeros(1, 1, model.hidden_layer_size))
# MAGIC #       y_pred = model(seq)
# MAGIC #       single_loss = torch_loss_function(y_pred, train_onecombo_target[i])
# MAGIC #       single_loss.backward()
# MAGIC #       optimizer.step()
# MAGIC #     if verbose and (i % 2 == 0): print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
# MAGIC 
# MAGIC #   return model  
# MAGIC       
# MAGIC # def predict_LSTM(predict_sequences,
# MAGIC #                  LSTM_model):
# MAGIC   
# MAGIC #   predictions = [] 
# MAGIC #   for i in range(len(predict_sequences)):
# MAGIC #     seq, target = predict_sequences[i]
# MAGIC #   with torch.no_grad():
# MAGIC #     LSTM_model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
# MAGIC #                          torch.zeros(1, 1, model.hidden_layer_size))
# MAGIC #     predictions.append(LSTM_model(seq).tolist()[0])
# MAGIC   
# MAGIC #   return predictions

# COMMAND ----------

# DBTITLE 1,Cross-Validated Direct Recursive Hybrid Modeling
def run_cv_recursive_hybrid_pipeline(train_pd, test_pd, number_of_periods_per_forecast, retrain_interval, stage1_dict, stage2_dict=None, cv=False, hybrid=True, \
                                     pred_col_name='xgb_model_stage2', time_unit='weeks', datetime_format="%Y%U-%w", resample_format='W-MON'):
  """
  Do recursive predictions per Hyndman's paper (https://robjhyndman.com/publications/rectify/)
  Summary of approach available here: https://machinelearningmastery.com/multi-step-time-series-forecasting/
  
  Basic idea: train a model on the training set, then make step-by-step predictions for each future timestep so you can recalculate lags inbetween.
    prediction(t+1) = model(obs(t-1), obs(t-2), ..., obs(t-n))
    prediction(t+2) = model(prediction(t+1), obs(t-1), ..., obs(t-n))
    ...
    
  NOTE: This function implicitly trims your in-sample observations to save on memory during the loop process. 
  """
  
  # define helper functions
  def filter_columns_from_df2 (df1, df2):
    repeated_columns = [col for col in set((df1.columns).intersection(df2.columns))]
    drop_cols = [col for col in repeated_columns if col not in hierarchy + [TIME_VAR]]
    df2 = df2.drop(drop_cols, axis =1)
    return df2

  def drop_and_rename_lagged_cols(lagged_df, number_of_periods_per_forecast, pred_col_name):
    
    # drop unwanted columns from a 'predictions df' to be a 'holdout df'
    tar_var_lags  = [(TARGET_VAR + '_lag_' + str(lag_num)) for lag_num in lag_weeks]
    drop_cols = tar_var_lags + list(MODEL_DICT.keys()) + ['sample']
    lagged_df = lagged_df.drop(drop_cols, axis =1)
  
    # rename lags
    pred_lags = [('col_to_lag' + '_lag_' + str(lag_num)) for lag_num in lag_weeks]
    rename_dict = dict(zip(pred_lags, tar_var_lags))
    lagged_df = lagged_df.rename(columns=rename_dict) 
    return lagged_df 

  def get_prepped_train_data(train_preds, number_of_periods_per_forecast, time_unit, datetime_format):
    relevant_train_data = filter_recent_time_periods(train_preds, number_of_periods_per_forecast, time_unit, datetime_format)[0]
    relevant_train_data['col_to_lag'] = relevant_train_data[TARGET_VAR]
    return relevant_train_data

  def filter_recent_time_periods(train_data, number_of_periods_per_forecast, time_unit, datetime_format):
    last_train_period = max(train_data[TIME_VAR])
    req_start_time = update_time(last_train_period, (-1*number_of_periods_per_forecast), time_unit=time_unit, datetime_format=datetime_format)
    relevant_train_data = train_data[train_data[TIME_VAR] > int(req_start_time)]
    set_train_data = train_data[train_data[TIME_VAR] <= int(req_start_time)]
    return relevant_train_data,set_train_data
  
  def get_retrain_weeks(test_data, number_of_periods_per_forecast, retrain_interval):
    test_weeks = sorted(test_data[TIME_VAR].unique())
    retrain_weeks = test_weeks[::retrain_interval][1:]
    return retrain_weeks

  # helper function for cv
  def recalc_train_lags(prelagged_train_data, stacked_prev_preds, number_of_periods_per_forecast, pred_col_name, \
                        curr_time_period, lag_weeks, time_unit, datetime_format,resample_format):

    # modify stacked prev preds 
    train_data_actuals = stacked_prev_preds[stacked_prev_preds[TIME_VAR] <= (curr_time_period-number_of_periods_per_forecast)-1]
    train_data_preds = stacked_prev_preds[stacked_prev_preds[TIME_VAR] > (curr_time_period-number_of_periods_per_forecast)-1]

    train_data_actuals['col_to_lag'] = train_data_actuals[TARGET_VAR]
    train_data_preds['col_to_lag'] = train_data_preds[pred_col_name]

    to_relag_train_data = pd.concat([train_data_actuals, train_data_preds])
    
    # ensure this is an int. Q: why would this change?
    to_relag_train_data[TIME_VAR] = to_relag_train_data[TIME_VAR].apply(np.int64)
    lagged_df = get_lagged_features(to_relag_train_data, ['col_to_lag'], lag_weeks, datetime_format=datetime_format, resample_format=resample_format)

    relagged_train_data = drop_and_rename_lagged_cols(lagged_df, number_of_periods_per_forecast, pred_col_name)

    # concat with already calculated lags
    total_lagged_train_data = pd.concat([prelagged_train_data, relagged_train_data])
    total_lagged_train_data.drop('col_to_lag', axis=1, inplace=True)
    return total_lagged_train_data  

  from progressbar import ProgressBar
  
  hierarchy = correct_suffixes_in_list(train_pd, get_hierarchy())
    
  train_preds = fit_modeling_pipeline(train_pd, stage1_dict, stage2_dict)
  
  sorted_time_list = sorted(test_pd[TIME_VAR].unique())
  first_holdout_week = sorted_time_list.pop(0)
  
  lagged_df_for_predict = test_pd[test_pd[TIME_VAR] == first_holdout_week]
  
  #for Hybrid only
  stacked_train_pd = train_pd

  # add last few weeks of train data to get actual lags for initial holdout
  train_splits = filter_recent_time_periods(train_pd, number_of_periods_per_forecast, time_unit, datetime_format)
  relevant_train_pd = train_splits[0]
  holdout_pd_with_train = pd.concat([relevant_train_pd, test_pd])
    
  # train data prepped to ensure predictions contain actuals 
  stacked_prev_preds = get_prepped_train_data(train_preds, number_of_periods_per_forecast, time_unit, datetime_format)
  
  # clear RAM
  del train_preds, relevant_train_pd

  # get the lag periods that need to be calculated at each timestep
  lag_weeks = [int(col.split("_")[-1]) for col in train_pd.columns if col.startswith(TARGET_VAR + "_lag_")]
  lag_weeks = [num for num in lag_weeks if num <= number_of_periods_per_forecast]
  
  retrain_weeks = get_retrain_weeks(test_pd, number_of_periods_per_forecast, retrain_interval)

  
  pbar = ProgressBar()
  for time_period in pbar(sorted_time_list):
            
    # ---------PREDICTING------------
    
    # predict for current time period using the lagged preds from previous weeks and keep stacking 
    
    print("Testing on time period number: " + str(time_period - 1))
    curr_preds = predict_modeling_pipeline(lagged_df_for_predict, stage1_dict, stage2_dict)
  
    
    #create a temp column for creating lags
    curr_preds['col_to_lag'] = curr_preds[pred_col_name]
  
    stacked_prev_preds = pd.concat([stacked_prev_preds, curr_preds])
    
    #stack on train after predicting for previous week so model is retrained for next week
    if hybrid:
      stacked_train_pd = pd.concat([stacked_train_pd, lagged_df_for_predict])
      if time_period in retrain_weeks:
        if cv:
          prelagged_train = train_splits[1]
          updated_train = recalc_train_lags(prelagged_train, stacked_prev_preds, number_of_periods_per_forecast, pred_col_name,\
                                            time_period, lag_weeks, time_unit, datetime_format, resample_format)
          print("Retraining model including actuals until time period: " + str(time_period - number_of_periods_per_forecast - 1))
          fit_modeling_pipeline(updated_train, stage1_dict, stage2_dict)
        else:
          print("Retraining model including time period: " + str(time_period - 1))
          fit_modeling_pipeline(stacked_train_pd, stage1_dict, stage2_dict)
    
    # limit the holdout_df to only have necessary time periods of data for merge & lagging
    curr_holdout = holdout_pd_with_train[(holdout_pd_with_train[TIME_VAR] <= time_period) & 
                                         (holdout_pd_with_train[TIME_VAR] >= (time_period - number_of_periods_per_forecast))]
      
    # merge with stacked preds to get predictions for t-1 weeks to be used in lag calculation
    stacked_prev_preds_for_merge = filter_columns_from_df2(curr_holdout, stacked_prev_preds)

    pred_df_to_lag = curr_holdout.merge(stacked_prev_preds_for_merge, how = 'left', on = hierarchy + [TIME_VAR])
    
    # ----------LAGGING-------------
    
    # calculate lagged features using stacked predictions
    lagged_df_for_predict = get_lagged_features(pred_df_to_lag, ['col_to_lag'], lag_weeks, datetime_format=datetime_format, resample_format=resample_format)
    
    # replace target_var lags with prediction lags and remove unwanted columns to match holdout columns
    lagged_df_for_predict = drop_and_rename_lagged_cols(lagged_df_for_predict, number_of_periods_per_forecast, pred_col_name)
    
    # save only this time period's data to make preds in next loop
    lagged_df_for_predict = lagged_df_for_predict[lagged_df_for_predict[TIME_VAR] == time_period]
    
    # reindex features to match the order of the original test set 
    lagged_df_for_predict= lagged_df_for_predict.reindex(test_pd.columns, axis=1)
  
  # last holdout week's predictions   
  curr_preds = predict_modeling_pipeline(lagged_df_for_predict, stage1_dict, stage2_dict)
  stacked_prev_preds = pd.concat([stacked_prev_preds, curr_preds])
  
  stacked_prev_preds.drop('col_to_lag', axis=1, inplace=True)
      
  return stacked_prev_preds

# COMMAND ----------

# DBTITLE 1,Shreya Scratch
# TODO: include below logic for CV 
# regular 
# each time you re-train including week 't' preds in train data lags, you will have access to week t-number_of_periods_per_forecasts+1 actuals 
# write helper to create another 'col_to_lag' that contains predictions for all but t-num_periods, and actuals for time t-num_periods + 1 
# recalc lags for training data and rename
# train using updated data with actuals

# need to be able to update actuals every 4 weeks 
# create a list of weeks to swap actuals and predictions, if time_var is in list

# regular version:
# eg. 4 week lag
# if you are in 201848, then you have actuals for 201843, so update that and retrain before predicting for 201848

# in discussion w/ Nick 
# suppose they only retrain only every 4 weeks, so in 201848 for ex. they retrained in 201846 (w/ actuals in 201841), then used 201846 model to predict till 201850. but then in 201851, update lags (until 201846)