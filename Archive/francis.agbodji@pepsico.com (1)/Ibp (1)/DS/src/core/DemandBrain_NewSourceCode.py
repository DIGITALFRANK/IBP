# Databricks notebook source
########################################################################
## Added during development of SDK on 6/04/2021 ###############
########################################################################

def bayesian_opt_quantile_params(pd_df, start_week, holdout_duration, parameter_grid=None, error_metric_func=calc_APE, boost_rounds=500, early_stop_rounds=100, init_points=5, n_iter=10):
  
  """
  Run Bayesian optimization over param grid to find best parameters for LightGBM.
  WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  - init_points: int, Number of time Bayesian Optimization is called - higher the better.
  - n_iter: int, Number of random parameters sets to pick from while maximizing target.
  
  Returns
  -------
  - params : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  if not parameter_grid:
    parameter_grid = BAYES_OPT_PARAM_GRID
  
  best_params = optimize_quantile_params(pd_df=pd_df,                                     start_week=start_week,                                     holdout_duration=holdout_duration,                                     error_metric_func=error_metric_func,                                     parameter_grid=parameter_grid,                                     boost_rounds=boost_rounds,                                     early_stop_rounds=early_stop_rounds,                                     init_points=init_points,                                     n_iter=n_iter)['params']
    
  params = {
      'metric': 'quantile',
      'objective': 'quantile',
      'alpha': 0.75,
#       'num_boost_round': boost_rounds,
    
      'max_depth': int(best_params['max_depth']),
      'num_leaves': int(best_params['num_leaves']),\
      'learning_rate': best_params['learning_rate'],\
      'feature_fraction': best_params['feature_fraction'],\
      'bagging_fraction': best_params['bagging_fraction']
    }
  
  print(params)
  return params


def optimize_quantile_params(pd_df, start_week, holdout_duration, error_metric_func, parameter_grid,boost_rounds, early_stop_rounds, init_points, n_iter):
  
  """
  Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  
  Returns
  -------
  - max : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  def evaluate_lgbm(max_depth, num_leaves, learning_rate, feature_fraction, bagging_fraction):
 

    params = {      
      'metric': 'quantile',
      'objective': 'quantile',
      'alpha': 0.75,
      
      'max_depth': int(max_depth),
      'num_leaves': int(num_leaves),
      'learning_rate': learning_rate,
      'feature_fraction': feature_fraction,
      'bagging_fraction': bagging_fraction
     } 
    
    n = holdout_duration
    sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
    holdout_period_list = [sorted_time_list[i * n:(i + 1) * n] for i in                            range((len(sorted_time_list) + n - 1) // n)]
    
    if len({len(i) for i in holdout_period_list}) != 1:
        print('Set start date and holdout duration so all sublists have the same number of periods!')
        print('Holdout Set List:', holdout_period_list)
        return
    
    error_list = []
    
    for time_period in holdout_period_list:
        
        train_pd = pd_df[pd_df[TIME_VAR] < time_period[0]]
        holdout_pd = pd_df[(pd_df[TIME_VAR] >= time_period[0]) & (pd_df[TIME_VAR] <= time_period[len(time_period)-1])]
      
        lgbm_model = train_lightGBM(train_pd, params)
        predictions = predict_lightGBM(holdout_pd, lgbm_model)
        y = holdout_pd[TARGET_VAR].values
      
        error = error_metric_func(y, predictions)
        error_list.append(error)
      
    np_errors = np.asarray(error_list)    
    return -1.0 * np.mean(np_errors)

  lgbm_bo = BayesianOptimization(evaluate_lgbm, parameter_grid)
  lgbm_bo.maximize(init_points, n_iter)
  
  return lgbm_bo.max



# COMMAND ----------

########################################################################
## Added during development of SDK on 6/04/2021 ###############
########################################################################

def bayesian_opt_catboost_params(pd_df, start_week, holdout_duration, parameter_grid=None, error_metric_func=calc_APE, init_points=5, n_iter=10):
  
  """
  Run Bayesian optimization over param grid to find best parameters for LightGBM.
  WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - init_points: int, Number of time Bayesian Optimization is called - higher the better.
  - n_iter: int, Number of random parameters sets to pick from while maximizing target.
  
  Returns
  -------
  - params : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  if not parameter_grid:
    parameter_grid = BAYES_OPT_PARAM_GRID
  
  best_params = optimize_catboost_params(pd_df=pd_df, start_week=start_week,
                                     holdout_duration=holdout_duration,
                                     error_metric_func=error_metric_func,
                                     parameter_grid=parameter_grid,
                                     init_points=init_points,
                                     n_iter=n_iter)['params']
    
  params = {   
      'depth': int(best_params['depth']),
      'learning_rate': best_params['learning_rate'],
      'subsample': best_params['subsample'],
      'l2_leaf_reg': best_params['l2_leaf_reg'],
      'grow_policy': 'Depthwise',
      'iterations': 200
   }
  
  print(params)
  return params


def optimize_catboost_params(pd_df, start_week, holdout_duration, error_metric_func, parameter_grid, init_points, n_iter):
  
  """
  Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  
  Returns
  -------
  - max : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  def evaluate_catboost(depth, learning_rate, subsample, l2_leaf_reg):
    
    params = {
        'depth': int(depth),
        'learning_rate': learning_rate,
        'subsample': subsample,
        'l2_leaf_reg': l2_leaf_reg,
        'grow_policy': 'Depthwise',
        'iterations': 200
    } 
    
    n = holdout_duration
    sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
    holdout_period_list = [sorted_time_list[i * n:(i + 1) * n] for i in range((len(sorted_time_list) + n - 1) // n)]
    
    if len({len(i) for i in holdout_period_list}) != 1:
        print('Set start date and holdout duration so all sublists have the same number of periods!')
        print('Holdout Set List:', holdout_period_list)
        return
    
    error_list = []
    
    for time_period in holdout_period_list:
        
        train_pd = pd_df[pd_df[TIME_VAR] < time_period[0]]
        holdout_pd = pd_df[(pd_df[TIME_VAR] >= time_period[0]) & (pd_df[TIME_VAR] <= time_period[len(time_period)-1])]

# change this
        catboost_model = train_catboost(train_pd, params)
        predictions = predict_catboost(holdout_pd, catboost_model)
        y = holdout_pd[TARGET_VAR].values
      
        error = error_metric_func(y, predictions)
        error_list.append(error)
      
    np_errors = np.asarray(error_list)    
    return -1.0 * np.mean(np_errors)

  catboost_bo = BayesianOptimization(evaluate_catboost, parameter_grid)
  catboost_bo.maximize(init_points, n_iter)
  
  return catboost_bo.max

# COMMAND ----------

########################################################################
## Added by Corey during development of SDK on 4/07/2021 ###############
########################################################################

def bayesian_opt_rforest_params(pd_df, start_week, holdout_duration, parameter_grid=None,\
                                error_metric_func=calc_APE, early_stop_rounds=100, init_points=5, n_iter=10):
  
  """
  Run Bayesian optimization over param grid to find best parameters for Random Forest.
  WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  - init_points: int, Number of time Bayesian Optimization is called - higher the better.
  - n_iter: int, Number of random parameters sets to pick from while maximizing target.
  
  Returns
  -------
  - params : dict, Parameters on grid that maximize chosen error metric of Random Forest model per Bayesian optimization.
  
  """
  
  if not parameter_grid:
    parameter_grid = BAYES_OPT_PARAM_GRID
  
  best_params = optimize_rforest_params(pd_df=pd_df, start_week=start_week, holdout_duration=holdout_duration, error_metric_func=error_metric_func, parameter_grid=parameter_grid, early_stop_rounds=early_stop_rounds, init_points=init_points, n_iter=n_iter)['params']
    
  params = {
      'criterion': 'mse',
      'n_jobs': -1,
      'n_estimators': int(best_params['n_estimators']),
      'max_depth': int(best_params['max_depth']),
      'min_samples_split': best_params['min_samples_split'],
      'max_features': best_params['max_features'],
      'max_samples': best_params['max_samples'],
      }
  
  print(params)
  return params



def optimize_rforest_params(pd_df, start_week, holdout_duration, error_metric_func, parameter_grid,                            early_stop_rounds, init_points, n_iter):
  
  """
  Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for Random Forest model fitting.
  
  Returns
  -------
  - max : dict, Parameters on grid that maximize chosen error metric of Random Forest model per Bayesian optimization.
  
  """
                       
                       
  def evaluate_rforest(n_estimators, max_depth, min_samples_split, max_features, max_samples):
    
    params = {
          'n_estimators': int(n_estimators),
          'max_depth': int(max_depth),
          'min_samples_split': min_samples_split,
          'max_features': max_features,
          'max_samples': max_samples,
          'n_jobs': -1
          } 
    
    n = holdout_duration
    sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
    holdout_period_list = [sorted_time_list[i * n:(i + 1) * n] for i in                            range((len(sorted_time_list) + n - 1) // n)]
    
    if len({len(i) for i in holdout_period_list}) != 1:
        print('Set start date and holdout duration so all sublists have the same number of periods!')
        print('Holdout Set List:', holdout_period_list)
        return
    
    error_list = []
    
    for time_period in holdout_period_list:
        
        train_pd = pd_df[pd_df[TIME_VAR] < time_period[0]]
        holdout_pd = pd_df[(pd_df[TIME_VAR] >= time_period[0]) & (pd_df[TIME_VAR] <= time_period[len(time_period)-1])]
      
        rforest_model = train_random_forest(train_pd, params)
        predictions = predict_random_forest(holdout_pd, rforest_model)
        y = holdout_pd[TARGET_VAR].values
      
        error = error_metric_func(y, predictions)
        error_list.append(error)
      
    np_errors = np.asarray(error_list)    
    return -1.0 * np.mean(np_errors)

  rforest_bo = BayesianOptimization(evaluate_rforest, parameter_grid)
  rforest_bo.maximize(init_points, n_iter)
  
  return rforest_bo.max


# COMMAND ----------

## Alternative: %pip install [lib-name-here] - ie, can run directly using Jupyter special commands
def juypter_library_install(lib_name_text):
    import pip
    pip.main(['install', lib_name_text])


from pandas import ExcelWriter

def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in list_dfs.items():
            df.to_excel(writer, 'cluster%s' % n)
        writer.save()


# In[ ]:


def clean_col_headers(input_pd):
    """
    Removes weird characters and spaces from column names + makes lowercase
    """
    output_pd = input_pd.copy()
    output_pd.columns = output_pd.columns.str.strip()                                         .str.lower()                                         .str.replace(' ', '_')                                         .str.replace('(', '')                                         .str.replace(')', '')                                         .str.replace("'", '')
    return output_pd


# In[ ]:


def split_col_names(input_pd, orig_col, one_col, two_col, separator):
    """
    orig_col: string name of the column we want to split
    one_col: string name of the first new column
    two_col: string name of the second new column
    separator: string - this is the character(s) that will define where to split
    """
    output_pd = input_pd.copy()
    output_pd[[one_col,two_col]] = output_pd[orig_col].str.split(separator, expand=True)
    return output_pd


# In[ ]:


def filter_dataframe_cols(input_pd, filter_string):
    output_pd = input_pd.filter(regex=filter_string)
    return output_pd


# In[ ]:


def calc_time_diff(input_pd, new_col_str, col_one_str, col_two_str, time_ind='months'):
    import datetime as dt
    if time_ind == 'months': input_pd[new_col_str] = (input_pd[col_one_str] - input_pd[col_two_str]).dt.months
    elif time_ind == 'weeks': input_pd[new_col_str] = (input_pd[col_one_str] - input_pd[col_two_str]).dt.weeks
    elif time_ind == 'days': input_pd[new_col_str] = (input_pd[col_one_str] - input_pd[col_two_str]).dt.days
    elif time_ind == 'years': input_pd[new_col_str] = (input_pd[col_one_str] - input_pd[col_two_str]).dt.years
    else: print('Please select an appropriate time_ind = months, weeks, days, years!')
    return input_pd


# In[ ]:


def adjust_date(input_date_num, date_check='month'):
    
    if len(str(input_date_num)) != 6:
        print('Please enter a date using following format: YYYYWW, YYYYMM, YYYYQQ')
    
    else:
        year_end_check = int(str(input_date_num)[-2:])

        if date_check == 'month':
            if year_end_check > 12:
                output_date_num = input_date_num + 100 + (year_end_check - 12) - year_end_check 
            else:
                output_date_num = input_date_num

        if date_check == 'week':
            if year_end_check > 52:
                output_date_num = input_date_num + 100 + (year_end_check - 52) - year_end_check 
            else:
                output_date_num = input_date_num

        if date_check == 'quarter':
            if year_end_check > 4:
                output_date_num = input_date_num + 100 + (year_end_check - 4) - year_end_check 
            else:
                output_date_num = input_date_num

        return output_date_num


# In[ ]:


def weighted_avg(values, weights):
    return (values * weights).sum() / weights.sum()

def grouped_weighted_avg(values, weights, by):
    return (values * weights).groupby(by).sum() / weights.groupby(by).sum()

# COMMAND ----------

########################################################################
## Added by Corey during development of SDK on 4/07/2021 ###############
########################################################################

def bayesian_opt_lgbm_params(pd_df, start_week, holdout_duration, parameter_grid=None, error_metric_func=calc_APE, boost_rounds=500, early_stop_rounds=100, init_points=5, n_iter=10):
  
  """
  Run Bayesian optimization over param grid to find best parameters for LightGBM.
  WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  - init_points: int, Number of time Bayesian Optimization is called - higher the better.
  - n_iter: int, Number of random parameters sets to pick from while maximizing target.
  
  Returns
  -------
  - params : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  if not parameter_grid:
    parameter_grid = BAYES_OPT_PARAM_GRID
  
  best_params = optimize_lgbm_params(pd_df=pd_df,                                     start_week=start_week,                                     holdout_duration=holdout_duration,                                     error_metric_func=error_metric_func,                                     parameter_grid=parameter_grid,                                     boost_rounds=boost_rounds,                                     early_stop_rounds=early_stop_rounds,                                     init_points=init_points,                                     n_iter=n_iter)['params']
    
  params = {
      'metric': 'rmse',
      'max_depth': int(best_params['max_depth']),
      'min_gain_to_split': best_params['min_gain_to_split'],
      'learning_rate': best_params['learning_rate'],
      'min_child_weight': best_params['min_child_weight'],
      'subsample': best_params['subsample'],
      'colsample_bytree': best_params['colsample_bytree'],
      'max_leaves': int(best_params['max_leaves']),         
      'num_boost_round': boost_rounds,
           }
  
  print(params)
  return params


def optimize_lgbm_params(pd_df, start_week, holdout_duration, error_metric_func, parameter_grid,                         boost_rounds, early_stop_rounds, init_points, n_iter):
  
  """
  Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  
  Returns
  -------
  - max : dict, Parameters on grid that maximize chosen error metric of LightGBM model per Bayesian optimization.
  
  """
  
  def evaluate_lgbm(max_depth, min_gain_to_split, learning_rate,                     min_child_weight, subsample, colsample_bytree, max_leaves):
    
    params = {
          'metric': 'rmse',
          'max_depth': int(max_depth),
          'min_gain_to_split': min_gain_to_split,
          'learning_rate': learning_rate,
          'min_child_weight': min_child_weight,
          'subsample': subsample,
          'colsample_bytree': colsample_bytree,
          'max_leaves': int(max_leaves)
          } 
    
    n = holdout_duration
    sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
    holdout_period_list = [sorted_time_list[i * n:(i + 1) * n] for i in                            range((len(sorted_time_list) + n - 1) // n)]
    
    if len({len(i) for i in holdout_period_list}) != 1:
        print('Set start date and holdout duration so all sublists have the same number of periods!')
        print('Holdout Set List:', holdout_period_list)
        return
    
    error_list = []
    
    for time_period in holdout_period_list:
        
        train_pd = pd_df[pd_df[TIME_VAR] < time_period[0]]
        holdout_pd = pd_df[(pd_df[TIME_VAR] >= time_period[0]) & (pd_df[TIME_VAR] <= time_period[len(time_period)-1])]
      
        lgbm_model = train_lightGBM(train_pd, params)
        predictions = predict_lightGBM(holdout_pd, lgbm_model)
        y = holdout_pd[TARGET_VAR].values
      
        error = error_metric_func(y, predictions)
        error_list.append(error)
      
    np_errors = np.asarray(error_list)    
    return -1.0 * np.mean(np_errors)

  lgbm_bo = BayesianOptimization(evaluate_lgbm, parameter_grid)
  lgbm_bo.maximize(init_points, n_iter)
  
  return lgbm_bo.max


# In[ ]:




# COMMAND ----------

########################################################################
## Added by Corey during development of SDK on 4/07/2021 ###############
########################################################################

def bayesian_opt_xgb_params(pd_df, start_week, holdout_duration, parameter_grid=None, error_metric_func=calc_APE, boost_rounds=500, early_stop_rounds=100, init_points=5, n_iter=10):
  
  """
  Run Bayesian optimization over param grid to find best parameters for XGBoost.
  WARNING: this process takes a great deal of time / computing power and shouldn't be done regularly.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  - init_points: int, Number of time Bayesian Optimization is called - higher the better.
  - n_iter: int, Number of random parameters sets to pick from while maximizing target.
  
  Returns
  -------
  - params : dict, Parameters on grid that maximize chosen error metric of XGBoost model per Bayesian optimization.
  
  """
  
  if not parameter_grid:
    parameter_grid = BAYES_OPT_PARAM_GRID
  
  best_params = optimize_xgb_params(pd_df=pd_df,                                    start_week=start_week,                                    holdout_duration=holdout_duration,                                    error_metric_func=error_metric_func,                                    parameter_grid=parameter_grid,                                    boost_rounds=boost_rounds,                                    early_stop_rounds=early_stop_rounds,                                    init_points=init_points,                                    n_iter=n_iter)['params']

    
  params = {'metric': 'l2, l1',
            'objective': 'regression',
            'num_machines': 2,
            'nthread': 32,
            'verbose': 100,
            'max_depth': int(best_params['max_depth']),
            'gamma': best_params['gamma'],
            'learning_rate': best_params['learning_rate'],
            'min_child_weight': best_params['min_child_weight'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'max_leaves': int(best_params['max_leaves']),         
            'num_boost_round': boost_rounds,
           }
  
  print(params)
  return params


def optimize_xgb_params(pd_df, start_week, holdout_duration, error_metric_func, parameter_grid,                        boost_rounds, early_stop_rounds, init_points, n_iter):
  
  """
  Run a Bayesian optimization to test and find the best value for each parameter along a grid of parameters.
  
  Parameters
  ----------
  - pd_df : pandas or koalas DataFrame, Training data for model.
  - start_week: int, Week of year after which rolling training/testing is performed.
  - holdout_duration: int, Duration of holdout used for dynamic testing.  
  - error_metric_function: function, Function used to calculate error to measure hyperparameter/training performance.
  - param_grid : dict, Dictionary of parameter-value pairs.
  - boost_rounds : int (default = 1000), Number of boosting rounds to perform.
  - early_stop_rounds : int (default = 100), Number of early stopping rounds for LightGBM model fitting.
  
  Returns
  -------
  - max : dict, Parameters on grid that maximize chosen error metric of XGBoost model per Bayesian optimization.
  
  """
  
  def evaluate_xgboost(max_depth, gamma, learning_rate, min_child_weight,                       subsample, colsample_bytree, max_leaves):
    
    params = {
        'max_depth': int(max_depth),
        'gamma': gamma,
        'learning_rate': learning_rate,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'max_leaves': int(max_leaves),
        } 
    
    n = holdout_duration
    sorted_time_list = sorted(pd_df.loc[pd_df[TIME_VAR] >= start_week, TIME_VAR].unique())
    holdout_period_list = [sorted_time_list[i * n:(i + 1) * n] for i in                            range((len(sorted_time_list) + n - 1) // n)]
    
    if len({len(i) for i in holdout_period_list}) != 1:
        print('Set start date and holdout duration so all sublists have the same number of periods!')
        print('Holdout Set List:', holdout_period_list)
        return
    
    error_list = []
    
    for time_period in holdout_period_list:
        
        dtrain = pd_df[pd_df[TIME_VAR] < time_period[0]]
        holdout_pd = pd_df[(pd_df[TIME_VAR] >= time_period[0]) & (pd_df[TIME_VAR] <= time_period[len(time_period)-1])]
      
        xgb_model = train_xgboost(dtrain, params)
        predictions = predict_xgboost(holdout_pd, xgb_model)
        y = holdout_pd[TARGET_VAR].values
      
        error = error_metric_func(y, predictions)
        error_list.append(error)
      
    np_errors = np.asarray(error_list)    
    return -1.0 * np.mean(np_errors)

  xgb_bo = BayesianOptimization(evaluate_xgboost, parameter_grid)
  xgb_bo.maximize(init_points, n_iter)
  
  return xgb_bo.max

# COMMAND ----------

def pull_SHAP_dict(input_train_pd, lgbm='Include', rforest='Include'):

    if '_log' not in TARGET_VAR:
        TARGET_VAR = TARGET_VAR + '_log'
        
    ## Remove target variable
    filtered_pd = input_train_pd.drop(TARGET_VAR, axis=1, inplace=False)
    
    ## Pull for Light GBM
    if lgbm == 'Include':
        lgbm_explainer = shap.TreeExplainer(MODEL_DICT['lightGBM_model_stage1'])
        lgbm_shap_values = lgbm_explainer.shap_values(filtered_pd)
        lgbm_shap_sum = np.abs(lgbm_shap_values).mean(axis=0)
        lgbm_importance_pd = pd.DataFrame([filtered_pd.columns.tolist(), lgbm_shap_sum.tolist()]).T
        lgbm_importance_pd.columns = ['column_name', 'lgbm_shap_imp']
        
    ## Pull for Random Forest
    if rforest == 'Include':
        rforest_explainer = shap.TreeExplainer(MODEL_DICT['rforest_model_stage1'])
        rforest_shap_values = rforest_explainer.shap_values(filtered_pd)
        rforest_shap_sum = np.abs(rforest_shap_values).mean(axis=0)
        rforest_importance_pd = pd.DataFrame([filtered_pd.columns.tolist(), rforest_shap_sum.tolist()]).T
        rforest_importance_pd.columns = ['column_name', 'rforest_shap_imp']

    if (lgbm == 'Include' & rforest == 'Include'):
        importance_pd = lgbm_importance_pd.merge(rforest_importance_pd, on='column_name')                                          .sort_values('lgbm_shap_imp', ascending=False)
        return importance_pd
    
    elif lgbm == 'Include':
        return lgbm_importance_pd
    
    elif rforest == 'Include':
        return rforest_importance_pd


# In[ ]:


def quantile_std_columns(pd_df, col_list, quantiles=10, fill_val=0):
    
    from sklearn.preprocessing import QuantileTransformer
    
    if not col_list:
        col_list = COLS_TO_QUANT_STD
        
    col_list = ensure_is_list(col_list)
    new_column_names = [feature + '_std' for feature in col_list]
    
    pd_df[col_list] = pd_df[col_list].replace([np.inf, -np.inf, np.nan], fill_val)
    
    scaler = QuantileTransformer(n_quantiles=quantiles, random_state=None, copy=False)
    qtrans_pd = pd.DataFrame(scaler.fit_transform(pd_df[col_list]))
    qtrans_pd.columns = new_column_names
    
    final_pd = pd_df.join(qtrans_pd)
    final_pd = final_pd.drop(col_list, axis=1)
    
    if TARGET_VAR in col_list:
        update_target_var(final_pd)
        
    return final_pd


########################################################################
## Added by Corey during development of SDK on 4/08/2021 ###############
########################################################################

def generate_residuals_output(input_pd, model_pred_cols=None, target_var_override=None,\
                              time_var_override=None, col_flag='_model_', col_deflag='_calc_'):
    
    ## Libraries to (likely) use for post-residual-generation plotting
    from pandas.plotting import autocorrelation_plot
    from statsmodels.graphics.gofplots import qqplot
    
    if target_var_override: target_var = target_var_override
    else: target_var = TARGET_VAR
        
    if time_var_override: time_var = time_var_override
    else: time_var = TIME_VAR
        
    if model_pred_cols: model_cols = ensure_is_list(model_pred_cols)
    else:
        model_cols = [model_col for model_col in input_pd.columns if                       col_flag in model_col and col_deflag not in model_col]
    
    model_output_cols = [target_var] + [time_var] + model_cols
    residuals_pd = input_pd[model_output_cols]
    
    for each_col in model_cols:
        residuals_pd[each_col + '_resid'] = residuals_pd[TARGET_VAR] - residuals_pd[each_col]  ## actuals less predicted
    
    residuals_pd.drop(columns=model_cols, inplace=True)
    return residuals_pd


########################################################################
## Added by Corey during development of SDK on 4/08/2021 ###############
########################################################################

def calculate_feature_VIF(input_pd, cols_to_drop=[], feat_col='feature', vif_col='vif'):
    
    ## General Documentation: https://www.geeksforgeeks.org/detecting-multicollinearity-with-vif-python/
    ## In VIF method, we pick each feature and regress it against all of the other features
    ## Large VIF on an independent variable indicates a highly collinear relationship to the other variables
    ## Greater VIF denotes greater correlation - a better candidate to strip out of the data
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    
    ## Se-up for subsetting our modeling dataframe
    interim_cols_to_drop = input_pd.select_dtypes(exclude=np.number).columns.tolist()
    final_cols_to_drop = [TARGET_VAR] + [TIME_VAR] + interim_cols_to_drop + cols_to_drop

    X = input_pd.drop(columns=final_cols_to_drop)
    vif_data = pd.DataFrame() 
    vif_data[feat_col] = X.columns 
    
    ## Using function from 'statsmodels' library
    vif_data[vif_col] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] 

    ## Setting very high value to fill in for inf values
    vif_data_clean = vif_data.replace([np.inf, -np.inf], np.nan).fillna(100000)
    
    return vif_data_clean

  
def return_collinear_features_VIF(vif_pd, threshold_level=15,feature_col='feature', vif_col='vif'):    
    vif_features = vif_pd[vif_pd[vif_col] > threshold_level][feature_col].to_list()   
    return vif_features

  
def drop_collinear_features_VIF(vif_pd, threshold_level=15,feature_col='feature', vif_col='vif'):
    
    ## Variance inflation factors range from 1 upwards
    ## What percentage the variance (i.e. the standard error squared) is inflated for each coefficient
    ## For example, a VIF of 1.9 tells you that the variance of a particular coefficient is 90% bigger
    ## than what you would expect if there was no multicollinearity — if there was no correlation with other predictors
    
    vif_features = vif_pd[vif_pd[vif_col] > threshold_level][feature_col].to_list()   
    vif_output_pd = vif_pd[~vif_pd[feature_col].isin(vif_features)]
    
    print('Features with VIF greater than {}:'.format(threshold_level), vif_features)
    return vif_output_pd


# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/05/2021 ###############
########################################################################

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def combine_csv_files(list_of_file_names, list_of_merge_features, merge_type='inner'):
    output_df = pd.read_csv(list_of_file_names[0])
    for each_file in list_of_file_names[1:]:
        temp_df = pd.read_csv(each_file)
        output_df = output_df.merge(temp_df, on=list_of_merge_features, how=merge_type)   
    return output_df


# In[27]:


########################################################################
## Added by Corey during development of SDK on 4/05/2021 ###############
########################################################################

import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from functools import reduce
from itertools import combinations
 
from datetime import datetime, timedelta


def get_google_trends(keyword_in_list, start_date='2018-01-01', end_date=datetime.today(),                      category=0, geo_cntry='US', geo_resolution='STATE', sleep_time=5):
    
    ## Library Documentation: https://pypi.org/project/pytrends/
    ## Category Details: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
    ## Category Info: all = 0 and shopping = 18; our default setting category=0
    from pytrends.request import TrendReq
    from progressbar import ProgressBar
    
    ## Defining date parsing by week
    dates = (
        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
          .apply(lambda x: x['date'] - timedelta(days = (x['date'].weekday() + 1) % 7), axis=1)
          .unique()
    )
    
    weeks = list(dates)
    pytrend = TrendReq()
    df_list = []
    
    ## Setting API loop to pull search terms
    pbar = ProgressBar()
    for week in pbar(weeks):

        import time
        time.sleep(sleep_time)

        weekstart = str(pd.to_datetime(week).date())
        weekend = pd.to_datetime(week) + timedelta(days=6)
        weekend = str(weekend.date())
        timethread = weekstart + ' ' + weekend
               
        pytrend.build_payload(kw_list=keyword_in_list, 
                              geo=geo_cntry, 
                              timeframe=timethread, 
                              cat=category)

        df = pytrend.interest_by_region(resolution=geo_resolution)
        df['WeekOfYear'] = week
        df_list.append(df)

    ## Getting merged dataframe for all words in keyword list
    trends_df = pd.concat(df_list)
    trends_df = trends_df.reset_index()   
    
    return trends_df


def get_related_google_queries(key_list, time_horizon='today 3-m', geo_cntry='US', category=0):
    
    ## Library Documentation: https://pypi.org/project/pytrends/
    ## Sample Implementation: https://towardsdatascience.com/google-trends-api-for-python-a84bc25db88f
    ## Pulls queries related to the search terms entered
    from pytrends.request import TrendReq
    from progressbar import ProgressBar
    
    ## Function set-up
    top_pd_list = []
    rising_pd_list = []
    pytrend = TrendReq()
    
    ## Build API from specifications
    pytrend.build_payload(kw_list=key_list, 
                          geo=geo_cntry, 
                          timeframe=time_horizon, 
                          cat=category)

    queries_dict = pytrend.related_queries()
    
    pbar = ProgressBar()
    for each_key in pbar(key_list):
        
        ## "Top" queries related to this keyword
        temp_top_pd = pd.DataFrame(queries_dict[each_key]['top'])
        temp_top_pd['search_term'] = each_key
        top_pd_list.append(temp_top_pd)
        
        ## "Rising" queries related to this keyword
        temp_rising_pd = pd.DataFrame(queries_dict[each_key]['rising'])
        temp_rising_pd['search_term'] = each_key
        rising_pd_list.append(temp_rising_pd)
        
    top_queries_pd = pd.concat(top_pd_list).reset_index(drop=True)
    top_queries_pd.rename(columns={'search_term':'search_term',                                   'query':'related_query',                                   'value':'index_value'},                          inplace=True)
    top_queries_pd = top_queries_pd[['search_term', 'related_query', 'index_value']]
    top_queries_pd['time_horizon'] = time_horizon
    
    rising_queries_pd = pd.concat(rising_pd_list).reset_index(drop=True)
    rising_queries_pd.rename(columns={'search_term':'search_term',                                      'query':'related_query',                                      'value':'rising_perc_increase'},                             inplace=True)
    rising_queries_pd = rising_queries_pd[['search_term', 'related_query', 'rising_perc_increase']]
    rising_queries_pd['time_horizon'] = time_horizon
    
    return top_queries_pd, rising_queries_pd


# In[33]:


########################################################################
## Added by Corey during development of SDK on 4/05/2021 ###############
########################################################################

def train_svm_regressor(input_pd, svm_params, *args, **kwargs):
    
    ## SVM Regressor Docs: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    ## SVM Implementation Example: https://www.analyticsvidhya.com/blog/2020/03/support-vector-regression-tutorial-for-machine-learning/
    ## PCA Docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html  
    
    ## SCALES POORLY: The fit time complexity is more than quadratic with the number of samples which 
    ## makes it hard to scale to dataset with more than a couple of 10000 samples.
    
    import scipy
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
    from sklearn.svm import SVR
    
    svm_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    svm_std_scale_pd = StandardScaler().fit_transform(svm_pd)
    
    pca = PCA(n_components=svm_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(svm_std_scale_pd)).reset_index(drop=True)    
    
    ## Recommend using PCA - or else will have dimensionality issue
    if svm_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(svm_std_scale_pd).reset_index(drop=True)
    
    svm_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    svm_full_pd = svm_merge_pd.join(temp_merge_pd)
    
    ## Alternative approach using CONCAT() function
    ## svm_full_pd = pd.concat([svm_merge_pd, pca_fit_pd], axis=1, join='outer', ignore_index=True)
    ## svm_full_pd.rename(columns={0:TIME_VAR, 1:TARGET_VAR}, inplace=True)
    
    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    y = svm_full_pd[TARGET_VAR].values
    X = svm_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
    
    ## Note - long runtimes - SVM/SVR scales poorly with datasize
    svm_model = SVR(kernel=svm_params['kernel'],                    gamma=svm_params['gamma'],                    C=svm_params['C'],                    epsilon=svm_params['epsilon']
                   )
    
    svm_model.fit(X, y)
    return svm_model


def predict_svm_regressor(input_pd, svm_model):
    
    svm_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    svm_std_scale_pd = StandardScaler().fit_transform(svm_pd)
    
    pca = PCA(n_components=svm_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(svm_std_scale_pd)).reset_index(drop=True)    
    
    ## Recommend using PCA - or else will have dimensionality issue
    if svm_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(svm_std_scale_pd).reset_index(drop=True)
    
    svm_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    svm_full_pd = svm_merge_pd.join(temp_merge_pd)
    
    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    X = svm_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
    
    predictions = svm_model.predict(X)
    return predictions


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/04/2021 ###############
########################################################################

def gridsearch_params_cv(indexed_pd, holdout_time_start, holdout_duration, stage1_dict, param_grid,                         model_to_gridsearch="lightgbm", time_ind=None, insample_periods_to_report=6, *args, **kwargs):
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
    
    temp_gridsearch_preds = run_dynamic_cross_validation(indexed_pd=indexed_pd, 
                                                         holdout_time_start=holdout_time_start, 
                                                         stage1_dict=stage1_dict,
                                                         holdout_duration=holdout_duration,
                                                         insample_periods_to_report=insample_periods_to_report, 
                                                         *args, **kwargs)

          
    columns_to_keep = [col for col in temp_gridsearch_preds if "_model_" in col] + ["sample"]
    
    results = temp_gridsearch_preds.groupby(['sample'])[columns_to_keep].agg(['mean', 'median'])
    results.columns = list(map('_'.join, list(results.columns)))
    results = results.reset_index()

    results.insert(loc=0, column='feature_set', value=str(param_update_dict))
    
    final_pd = final_pd.append(results, ignore_index=True)
    
  return final_pd


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/04/2021 ###############
########################################################################

def show_matplotlib_colors(plot_chart=True, num_cols=5, size_tuple=(8,5)):
    
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    # Sort colors by hue, saturation, value and name.
    by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                    for name, color in colors.items())
    sorted_names = [name for hsv, name in by_hsv]

    n = len(sorted_names)
    ncols = num_cols
    nrows = n // ncols + 1
    
    if plot_chart:

        fig, ax = plt.subplots(figsize=size_tuple)

        # Get height and width
        X, Y = fig.get_dpi() * fig.get_size_inches()
        h = Y / (nrows + 1)
        w = X / ncols

        for i, name in enumerate(sorted_names):
            col = i % ncols
            row = i // ncols
            y = Y - (row * h) - h

            xi_line = w * (col + 0.05)
            xf_line = w * (col + 0.25)
            xi_text = w * (col + 0.3)

            ax.text(xi_text, y, name, fontsize=(h * 0.8),
                    horizontalalignment='left',
                    verticalalignment='center')

            ax.hlines(y + h * 0.1, xi_line, xf_line,
                      color=colors[name], linewidth=(h * 0.6))

        ax.set_xlim(0, X)
        ax.set_ylim(0, Y)
        ax.set_axis_off()

        fig.subplots_adjust(left=0, right=1,
                            top=1, bottom=0,
                            hspace=0, wspace=0)
        plt.show()
        
    else:
        print(sorted_names)


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/03/2021 ###############
########################################################################
## Note - seems to predict the same thing for every time period
## Corey to check with the Trellis team
## Open Qs - do we need to scale & PCA? (same as KNN)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

def train_lstm_tensorflow_model(input_pd, lstm_params, *args, **kwargs):
    
    ## Sequential Model Docs: https://keras.io/guides/sequential_model/
    ## Implementation Reference Doc: https://towardsdatascience.com/understanding-lstm-and-its-quick-implementation-in-keras-for-sentiment-analysis-af410fd85b47
    ## Implementation Reference Doc: https://machinelearningmastery.com/tune-lstm-hyperparameters-keras-time-series-forecasting/
    ## Loss Functions: https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    ## Dropout Regularization: https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/
    ## PCA Reference Docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    
    lstm_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    lstm_std_scale_pd = StandardScaler().fit_transform(lstm_pd)
    
    pca = PCA(n_components=lstm_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(lstm_std_scale_pd)).reset_index(drop=True)
    
    ## Recommend NOT using PCA - detracts from Deep Learning pattern recognition
    if lstm_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(lstm_std_scale_pd).reset_index(drop=True)
    
    lstm_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    lstm_full_pd = lstm_merge_pd.join(temp_merge_pd)

    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    y = lstm_full_pd[TARGET_VAR].values
    X = lstm_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
    
    val_X = X.iloc[int(X.shape[0] * lstm_params['validation_perc']):, :]
    val_y = y[int(len(y) * lstm_params['validation_perc']):]
    
    # MUST reshape the training data to fit into this shape
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))
    val_X = val_X.values.reshape((val_X.shape[0], 1, val_X.shape[1]))
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(int(lstm_params['units']), input_shape=(X.shape[1], X.shape[2]), dropout=lstm_params['dropout_perc']))
    lstm_model.add(Dense(int(lstm_params['dense']), activation=lstm_params['activation']))
    lstm_model.compile(loss=lstm_params['loss'], optimizer=lstm_params['optimizer'], metrics=lstm_params['metrics'])
    es = EarlyStopping(monitor=lstm_params['loss'], mode='min', verbose=1, patience=lstm_params['patience'])
    
    history = lstm_model.fit(X, y, epochs=int(lstm_params['epochs']), callbacks=[es],                             batch_size=int(lstm_params['batch_size']), validation_data=(val_X, val_y),                             shuffle=False, verbose=2)
    
    ## return history
    return lstm_model


def predict_lstm_tensorflow_model(input_pd, lstm_model):
    
    ## Model Setup: https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/
    ## Prediction Setup: https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html
    
    lstm_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    lstm_std_scale_pd = StandardScaler().fit_transform(lstm_pd)
    
    pca = PCA(n_components=lstm_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(lstm_std_scale_pd)).reset_index(drop=True)
    
    ## Recommend NOT using PCA - detracts from Deep Learning pattern recognition
    if lstm_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(lstm_std_scale_pd).reset_index(drop=True)
    
    lstm_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    lstm_full_pd = lstm_merge_pd.join(temp_merge_pd)

    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    X = lstm_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
    
    ## MUST reshape the training data to fit into this shape
    X = X.values.reshape((X.shape[0], 1, X.shape[1]))
    predictions = lstm_model.predict(X, batch_size=int(lstm_params['batch_size']))
    
    return predictions


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/02/2021 ###############
########################################################################

import gc
import scipy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer

def train_knn_regressor(input_pd, knn_params, *args, **kwargs):
    
    ## KNN Regressor Docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
    ## PCA Docs: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    
    knn_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    knn_std_scale_pd = StandardScaler().fit_transform(knn_pd)
    
    pca = PCA(n_components=knn_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(knn_std_scale_pd)).reset_index(drop=True)
    
    ## Recommend using PCA - or else will have dimensionality issue
    if knn_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(knn_std_scale_pd).reset_index(drop=True)
    
    knn_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    knn_full_pd = knn_merge_pd.join(temp_merge_pd)
    
    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    y = knn_full_pd[TARGET_VAR].values
    X = knn_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
        
    knn_model = KNeighborsRegressor(n_neighbors=knn_params['neighbors'],                                    weights=knn_params['dist_weight'],                                    algorithm=knn_params['algo_choice'])
    knn_model.fit(X, y)    
    return knn_model


def predict_knn_regressor(input_pd, knn_model):
    
    knn_pd = input_pd.drop([TIME_VAR, TARGET_VAR], axis=1)
    knn_std_scale_pd = StandardScaler().fit_transform(knn_pd)
    
    pca = PCA(n_components=knn_params['n_comps'], svd_solver='auto')
    pca_fit_pd = pd.DataFrame(pca.fit_transform(knn_std_scale_pd)).reset_index(drop=True)
    
    ## Recommend using PCA - or else will have dimensionality issue
    if knn_params['pca_trans']:
        temp_merge_pd = pca_fit_pd.copy()
    else:
        temp_merge_pd = pd.DataFrame(knn_std_scale_pd).reset_index(drop=True)
    
    knn_merge_pd = input_pd[[TIME_VAR, TARGET_VAR]].reset_index(drop=True)
    knn_full_pd = knn_merge_pd.join(temp_merge_pd)
    
    ## Drop the TIME_VAR from X, too - will bias results otherwise
    ## Especially if PCA transformation was used for data preparation
    X = knn_full_pd.drop([TIME_VAR, TARGET_VAR], axis=1)                   .replace([np.inf, -np.inf], np.nan)                   .fillna(0)
    
    predictions = knn_model.predict(X)    
    return predictions


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/02/2021 ###############
########################################################################
## Note - builds from 'run_rolling_window_cv' function (in old code)
## Note - Corey updated to add 'walkforward' (for a single time period) on 5/11/2021

def run_dynamic_cross_validation(indexed_pd, holdout_time_start, holdout_duration, stage1_dict, stage2_dict=None,\
                                 train_time_start=None, holdout_time_end=None, time_ind=None,\
                                 walkforward=False, insample_periods_to_report=6, *args, **kwargs):
    if not time_ind:
        time_ind = TIME_VAR
    
    sorted_time_list = sorted(indexed_pd.loc[indexed_pd[time_ind] >= holdout_time_start, time_ind].unique())
    
    if holdout_time_end:
        sorted_time_list = [time for time in sorted_time_list if time <= holdout_time_end]
    
    n = holdout_duration
    
    if walkforward:
      holdout_tuples_list = [[time, time + n - 1] for time in sorted_time_list]
    
    else:
      holdout_tuples_list = [sorted_time_list[i * n:(i + 1) * n] for i in range((len(sorted_time_list) + n - 1) // n)]
        
    ## Useful for cutting down on model training time (if deep historical data)
    if train_time_start:
        indexed_pd = indexed_pd[indexed_pd[time_ind] >= train_time_start]
    
    print('Holdout Tuples List: %s' % (holdout_tuples_list))
    print('Stage1 Models: %s' % (stage1_dict.keys()))
    if stage2_dict:
        print('Stage2 Models: %s' % (stage2_dict.keys()))
    
    final_pd = pd.DataFrame()
    
    ## Corey - net new code
    for holdout_tuple in holdout_tuples_list:        
        temp_train_pd = indexed_pd[indexed_pd[time_ind] < holdout_tuple[0]]
        temp_holdout_pd = indexed_pd[(indexed_pd[time_ind] >= holdout_tuple[0]) & (indexed_pd[time_ind] <= holdout_tuple[len(holdout_tuple)-1])]
        
        print("Running models for %s -- Train Size: %s, Test Size: %s" % (holdout_tuple, temp_train_pd.shape, temp_holdout_pd.shape))
        print("Num periods in train set: %s -- Periods in holdout set: %s" % (temp_train_pd[time_ind].nunique(), temp_holdout_pd[time_ind].unique()))
        print("Train Dataframe Min/Max Time:", temp_train_pd[TIME_VAR].min(), temp_train_pd[TIME_VAR].max())
        print("Holdout Dataframe Min/Max Time:", temp_holdout_pd[TIME_VAR].min(), temp_holdout_pd[TIME_VAR].max())

        temp_pred_pd = run_models(temp_train_pd, temp_holdout_pd, stage1_dict, stage2_dict, *args, **kwargs)
        
        print('Modeling complete for CV set %s' % (holdout_tuple))
        
        ## Filter output size to save memory
        temp_pred_pd = temp_pred_pd[temp_pred_pd[time_ind] >= (holdout_tuple[0] - insample_periods_to_report)]
        
        ## COREY FLAG - NET NEW CODE
        ## Did not work - might need to just 
        tuple_string = str(holdout_tuple)
        temp_pred_pd['oos_periods'] = tuple_string
        
        ## Append iterative dataframes to aggregated "final" dataframe
        final_pd = final_pd.append(temp_pred_pd, ignore_index=True)
        print('Shape of aggregate dataframe:', final_pd.shape)
    
    ## Remove from loop and return final appended dataframe
    return final_pd

  
## BELOW VERSION IS DEPRECATED - DOES NOT INCLUDE THE 'WALKFORWARD' CONCEPT
## KEEPING HERE IN COMMENTED-OUT FORM UNTIL THE NEW VERSION CAN BE RIGOROUSLY TESTED
# def run_dynamic_cross_validation(indexed_pd, holdout_time_start, holdout_duration, stage1_dict, stage2_dict=None,\
#                                  train_time_start=None, holdout_time_end=None, time_ind=None, insample_periods_to_report=6, *args, **kwargs):
#     if not time_ind:
#         time_ind = TIME_VAR
    
#     sorted_time_list = sorted(indexed_pd.loc[indexed_pd[time_ind] >= holdout_time_start, time_ind].unique())
    
#     if holdout_time_end:
#         sorted_time_list = [time for time in sorted_time_list if time <= holdout_time_end]
    
#     n = holdout_duration
#     holdout_tuples_list = [sorted_time_list[i * n:(i + 1) * n] for i in range((len(sorted_time_list) + n - 1) // n)]
        
#     ## Useful for cutting down on model training time (if deep historical data)
#     if train_time_start:
#         indexed_pd = indexed_pd[indexed_pd[time_ind] >= train_time_start]
    
#     print('Holdout Tuples List: %s' % (holdout_tuples_list))
#     print('Stage1 Models: %s' % (stage1_dict.keys()))
#     if stage2_dict:
#         print('Stage2 Models: %s' % (stage2_dict.keys()))
    
#     final_pd = pd.DataFrame()
    
#     for holdout_tuple in holdout_tuples_list:        
#         temp_train_pd = indexed_pd[indexed_pd[time_ind] < holdout_tuple[0]]
#         temp_holdout_pd = indexed_pd[(indexed_pd[time_ind] >= holdout_tuple[0]) & (indexed_pd[time_ind] <= holdout_tuple[len(holdout_tuple)-1])]
        
#         print("Running models for %s -- Train Size: %s, Test Size: %s" % (holdout_tuple, temp_train_pd.shape, temp_holdout_pd.shape))
#         print("Num periods in train set: %s -- Periods in holdout set: %s" % (temp_train_pd[time_ind].nunique(), temp_holdout_pd[time_ind].unique()))

#         temp_pred_pd = run_models(temp_train_pd, temp_holdout_pd, stage1_dict, stage2_dict, *args, **kwargs)
        
#         print('Modeling complete for CV set %s' % (holdout_tuple))
        
#         ## Filter output size to save memory
#         temp_pred_pd = temp_pred_pd[temp_pred_pd[time_ind] >= (holdout_tuple[0] - insample_periods_to_report)]
        
#         ## COREY FLAG - NET NEW CODE
#         ## Did not work - might need to just 
#         tuple_string = str(holdout_tuple)
#         temp_pred_pd['oos_periods'] = tuple_string
        
#         ## Append iterative dataframes to aggregated "final" dataframe
#         final_pd = final_pd.append(temp_pred_pd, ignore_index=True)
#         print('Shape of aggregate dataframe:', final_pd.shape)
    
#     ## Remove from loop and return final appended dataframe
#     return final_pd

########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################

def top_performer_flag(input_pd, grouping_dimension, feature_of_interest, agg_method='sum', cutoff_value_override=None):
    
    output_pd = input_pd.copy()
    new_col_name = 'top_flag_' + str(agg_method) + '_' + str(grouping_dimension)
    
    if cutoff_value_override:
        cutoff_value = cutoff_value_override
        
    else:
        cutoff_value = int(np.round(output_pd[grouping_dimension].nunique() * 0.20, 1))
    
    ## Sorting top to bottom to capture LOWEST performers in dimension of interest
    flag_pd = output_pd.groupby(grouping_dimension).agg({feature_of_interest:agg_method}).reset_index()
    top_values = flag_pd.sort_values(by=feature_of_interest, ascending=False)[grouping_dimension][0:cutoff_value].to_list()
    
    output_pd[new_col_name] = 0
    output_pd.loc[output_pd[grouping_dimension].isin(top_values), new_col_name] = 1

    return output_pd


def bot_performer_flag(input_pd, grouping_dimension, feature_of_interest, agg_method='sum', cutoff_value_override=None):
    
    output_pd = input_pd.copy()
    new_col_name = 'bot_flag_' + str(agg_method) + '_' + str(grouping_dimension)
    
    if cutoff_value_override:
        cutoff_value = cutoff_value_override
        
    else:
        cutoff_value = int(np.round(output_pd[grouping_dimension].nunique() * 0.20, 1))
    
    ## Sorting bottom to top to capture LOWEST performers in dimension of interest
    flag_pd = output_pd.groupby(grouping_dimension).agg({feature_of_interest:agg_method}).reset_index()
    top_values = flag_pd.sort_values(by=feature_of_interest, ascending=True)[grouping_dimension][0:cutoff_value].to_list()
    
    output_pd[new_col_name] = 0
    output_pd.loc[output_pd[grouping_dimension].isin(top_values), new_col_name] = 1

    return output_pd


def time_period_flag(input_pd, time_period_ref_col, time_period_list):
    
    output_pd = input_pd.copy()
    new_col_name = str(time_period_ref_col) + '_addtl_flag'
    time_period_list = ensure_is_list(time_period_list)
    
    output_pd[new_col_name] = 0
    output_pd.loc[output_pd[time_period_ref_col].isin(time_period_list), new_col_name] = 1
    
    return output_pd


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################

def aggregate_predictions(prediction_pd, prod_level_col, bus_level_col, agg_func='sum', ind_cols_list=None, error_func_list=[calc_APA, calc_Bias, calc_SE]):
    """
    Return a dataframe of aggregated error metrics by levels of the product and business hierarchies
    """
    
    if not ind_cols_list:
        ind_cols_list = []
  
    ## Handle the exception where the TARGET_VAR has already been converted to the base_target_var name
    base_target_var = get_base_target_var() 
    if base_target_var in prediction_pd.columns:
        tar_var = base_target_var
    else:
        tar_var = TARGET_VAR
    
    ## Use set to remove duplicates in case the user wants to indicate by a level of the hierarchy
    grouping_cols = list(set([prod_level_col] + [bus_level_col] + ['sample'] + ind_cols_list))
    
    models_dict = {column:np.sum for column in list(prediction_pd.columns) if '_model' in column and '_calc' not in column}
    actuals_dict = {tar_var:np.sum}
    agg_dict = {**models_dict, **actuals_dict}
    
    ## Will automatically drop the alreay-calculated columns
    agg_pd = prediction_pd.groupby(grouping_cols).agg(agg_dict).reset_index()
    
    ## Then calculates the error metrics AFTER aggregating the predictions levels
    final_pd = calc_error_metrics(agg_pd, tar_var, error_func_list=error_func_list)
    
    return final_pd


def aggregate_error_metrics(prediction_pd, prod_level_col, bus_level_col, agg_func, ind_cols_list=None):
    """
    Return a dataframe of aggregated error metrics by levels of the product and business hierarchies
    """
    
    if not ind_cols_list:
        ind_cols_list = []
  
    ## Handle the exception where the TARGET_VAR has already been converted to the base_target_var name
    base_target_var = get_base_target_var() 
    if base_target_var in prediction_pd.columns:
        tar_var = base_target_var
    else:
        tar_var = TARGET_VAR
    
    if agg_func == 'weighted_mean':
        agg_func = get_weighted_average_func(prediction_pd, tar_var)
    
    ## Use set to remove duplicates in case the user wants to indicate by a level of the hierarchy
    grouping_cols = list(set([prod_level_col] + [bus_level_col] + ['sample'] + ind_cols_list))
    
    model_dict = {column:np.sum for column in list(prediction_pd.columns) if '_model' in column}
    error_dict = {column:agg_func for column in list(prediction_pd.columns) if '_calc' in column}
    actuals_dict = {tar_var:np.sum}
    
    new_error_names_dict = {old:old.replace('_calc', '_aggcalc') for old in error_dict.keys()}
    agg_dict = {**model_dict, **error_dict, **actuals_dict}      
    
    agg_pd = prediction_pd.groupby(grouping_cols).agg(agg_dict)
    final_pd = agg_pd.rename(new_error_names_dict, axis=1)
    
    return final_pd.reset_index()


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################
## Note - not working for our example

def train_predict_holt_winters(pd_df, forecast_cols, grouping_cols, agg_method='mean',                               seasonal_periods=52, seasonal='additive', trend='add', steps=8):
    '''
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
    '''
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    
    grouping_cols = ensure_is_list(grouping_cols)
    grouping_cols = list(set([TIME_VAR] + [TARGET_VAR] + grouping_cols))
    forecast_cols = ensure_is_list(forecast_cols)
    
    agg_dict = {col_name:agg_method for col_name in forecast_cols}
    agg_pd = pd_df.groupby(grouping_cols).agg(agg_dict).reset_index()
    
    agg_pd = agg_pd.sort_values(by=TIME_VAR).reset_index(drop=True)
    agg_pd[TARGET_VAR] = agg_pd[TARGET_VAR].astype('float')
    agg_pd[TARGET_VAR] = agg_pd[TARGET_VAR].replace(np.nan, agg_pd[TARGET_VAR].mean())
    
    X = agg_pd.drop(forecast_cols, axis=1)
    y = agg_pd[forecast_cols]
    
    hw_model = HWES(y, seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)
    fitted_hw_model = hw_model.fit(optimized=True, use_brute=True)
    
    predictions = fitted_hw_model.forecast(steps=steps)
    return predictions


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################

def gridsearch_params_stage2(training_pd, testing_pd, stage1_dict, stage2_dict, param_grid,                             model_to_gridsearch="lightgbm", time_ind=None, *args, **kwargs):
  """
  Run a param gridsearch using a single training and holdout dataframe 
  """
  import itertools as it
  
  def get_key_to_gridsearch(model_dict):
    models_to_gridsearch = [key for key in list(model_dict.keys()) if model_to_gridsearch in key.lower()]
    assert len(models_to_gridsearch) == 1, "You have two or more keys in your modeling dictionary containing %s" % model_to_gridsearch

    return models_to_gridsearch[0]

  model_key = get_key_to_gridsearch(stage2_dict)

  sorted_names = sorted(param_grid)
  combinations = list(it.product(*(param_grid[name] for name in sorted_names)))

  starter_params = stage2_dict[model_key][0][1].copy()

  final_pd = pd.DataFrame()
    
  pbar = ProgressBar()
  for param_combination in pbar(combinations):
    # update param dictionary for target model
    param_update_dict = {sorted_names[index]:param_combination[index] for index in range(len(sorted_names))}
    new_params = starter_params.copy()
    new_params.update(param_update_dict)
    stage2_dict[model_key][0][1] = new_params
    
    temp_predictions = run_models(train_pd=training_pd, 
                                  test_pd=testing_pd, 
                                  stage1_dict=stage1_dict, 
                                  stage2_dict=stage2_dict, 
                                  print_errors=False,
                                  *args, **kwargs)
          
    columns_to_keep = [col for col in temp_predictions if "_model_" in col] + ["sample"]

    results = temp_predictions.groupby(['sample'])[columns_to_keep].agg(['mean', 'median', 'std'])
    results.columns = list(map('_'.join, list(results.columns)))
    results = results.reset_index()

    results.insert(loc=0, column='feature_set', value=str(param_update_dict))
    
    final_pd = final_pd.append(results, ignore_index=True)
    
  return final_pd


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################

def underpred_penalty_metric(input_pd, target_var_col, prediction_col, penalty_term=2.0):
    
    """
    Creating 'dummy' metric that assigns more of a penalty for underprediction
    User has the ability to adjust this penalty term based on context
    """
    
    input_pd['baseline'] = input_pd[target_var_col] - input_pd[prediction_col]
    input_pd['interim_metric'] = input_pd['baseline']
    input_pd.loc[input_pd['baseline'] > 0,  'interim_metric'] = np.fabs(input_pd['interim_metric']) * penalty_term
    
    input_pd[prediction_col + '_penalty_metric'] = (1 - (np.fabs(input_pd['interim_metric'])/input_pd[target_var_col]))
    output_pd = input_pd.drop(columns=['baseline', 'interim_metric'], inplace=False)
    
    return output_pd


# In[ ]:


########################################################################
## Added by Corey during development of SDK on 4/01/2021 ###############
########################################################################

def calc_SHAP_values(pd_df, approximate_xgb=True, include_rf=False, tree_limit=None, path_indicator=None):
  """
  Save the SHAP values for all model
  """
  
  if approximate_xgb:
    print("NOTE: Approximating XGB SHAP values.")
  
  tar_var_list = [TARGET_VAR, TARGET_VAR + '_log', TARGET_VAR + '_std']
  tar_var_to_drop = list(set(tar_var_list).intersection(set(train_pd)))

  xgb_models = [model for model in MODEL_DICT if "xgb_model_stage1" in model]
  lgbm_models = [model for model in MODEL_DICT if "lightGBM_model_stage1" in model]
  rf_models = [model for model in MODEL_DICT if "rforest_model_stage1" in model]
  
  filtered_pd = pd_df.drop(TARGET_VAR, axis=1, inplace=False)

  shap_dict = {}
  
  for model in xgb_models:
    print("Calculating SHAP values for %s ..." % model)
    explainer = shap.TreeExplainer(MODEL_DICT[model])
    shap_values = explainer.shap_values(filtered_pd, tree_limit=tree_limit, approximate=approximate_xgb)
    dict_entry = {"Values":shap_values, "Explainer":explainer}
    shap_dict[model] = dict_entry
    
  for model in lgbm_models:
    print("Calculating SHAP values for %s ..." % model)
    explainer = shap.TreeExplainer(MODEL_DICT[model])
    shap_values = explainer.shap_values(filtered_pd, tree_limit=tree_limit)
    dict_entry = {"Values":shap_values, "Explainer":explainer}
    shap_dict[model] = dict_entry
  
  if include_rf:
      for model in rf_models:
        print("Calculating SHAP values for %s ..." % model)
        explainer = shap.TreeExplainer(MODEL_DICT[model])
        shap_values = explainer.shap_values(filtered_pd, tree_limit=tree_limit)
        dict_entry = {"Values":shap_values, "Explainer":explainer}
        shap_dict[model] = dict_entry
  
  return shap_dict


########################################################################
## Added by Corey during development of SDK on 3/29/2021 ###############
########################################################################

def print_missing_row_percentage(pd_df):
    print('Percent of rows with any missing data: %3.3f%%' %
          (pd_df[pd_df.isna().any(axis=1) == True].shape[0] / pd_df.shape[0] * 100))


def create_lag_dictionary(feature_list_to_lag, period_list_of_lags, suffix='_lag_'):
    lagged_columns_dict = {}
    for each_period in period_list_of_lags:
        col_list = []
        for x, y in itertools.product(feature_list_to_lag, [each_period]):
            col_list.append(x + suffix + str(y))
        lagged_columns_dict[each_period] = col_list
    return lagged_columns_dict


def created_lagged_feature_list(input_pd, lag_dictionary, cols_to_lag_list):
    output_pd = input_pd.copy()
    for each_period in lag_dictionary.keys():
        output_pd[lag_dictionary[each_period]] = output_pd[cols_to_lag_list].shift(each_period)                                                    .where(output_pd[TIME_VAR] >= np.min(output_pd[TIME_VAR]) + each_period)
    return output_pd


def plot_box_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10, show_outliers=False, 
                  fig_size=(10,10), x_label=None, y_label=None, **kwargs):
  """
  Plot a boxplot in seaborn
  TODO clean this function up; standardize arguments between plotting functions
  """  
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
    
  plt.figure(figsize=fig_size)
  plot = sns.boxplot(x=x_var, y=y_var, data=pd_df, hue=hue, showfliers=show_outliers, **kwargs)
  plt.title(title, size=title_size)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  
  sns.despine()
  
  return plot


def plot_violin_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10, show_outliers=False,
                   fig_size=(10,10), x_label=None, y_label=None, **kwargs):
  """
  Plot a violin plot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.violinplot(x=x_var, y=y_var, data=pd_df, hue=hue, showfliers=show_outliers, **kwargs)
  plt.title(title, size=title_size)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  if x_label:
    plt.xlabel(x_label)

  if y_label:
    plt.ylabel(y_label)
    
  sns.despine()
  
  return plot


def plot_joint_plot(pd_df, x_var, y_var, kind="scatter", title=None, title_size=10, fig_size=(10,10),
                    y_limit_list=None, x_label=None, y_label=None, **kwargs):
  """
  Plot a joint plot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.jointplot(x=x_var, y=y_var, kind=kind, data=pd_df, **kwargs)
  plt.title(title, size=title_size)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
  
  if y_limit_list:
    plt.ylim(y_limit_list[0], y_limit_list[1])
    
  if x_label:
    plt.xlabel(x_label)

  if y_label:
    plt.ylabel(y_label)
    
  sns.despine()
  
  return plot


def plot_scatter_plot(pd_df, x_var, y_var, hue=None, size=None, stylehue=None, title=None, title_size=10,
                      fig_size=(10,10), x_label=None, y_label=None, **kwargs):
  """
  Plot a boxplot in seaborn
  """  
  x_ticks=get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.scatterplot(x=x_var, y=y_var, data=pd_df, hue=hue, style=stylehue, size=size, **kwargs)
  plt.title(title, size=title_size)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  
  sns.despine()
  
  return plot



def get_google_trends(keyword_in_list, start_date='2018-01-01', end_date=datetime.today(),\
                      category=0, geo_cntry='US', geo_resolution='STATE', sleep_time=2):
    
    ## Defining date parsing by week
    dates = (
        pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['date'])
          .apply(lambda x: x['date'] - timedelta(days = (x['date'].weekday() + 1) % 7), axis=1)
          .unique()
    )
    
    weeks = list(dates)
    pytrend = TrendReq()
    df_list = []
    
    ## Setting API loop to pull search terms
    pbar = ProgressBar()
    for week in pbar(weeks):

        import time
        time.sleep(sleep_time)

        weekstart = str(pd.to_datetime(week).date())
        weekend = pd.to_datetime(week) + timedelta(days=6)
        weekend = str(weekend.date())
        timethread = weekstart + ' ' + weekend
               
        pytrend.build_payload(kw_list=keyword_in_list, 
                              geo=geo_cntry, 
                              timeframe=timethread, 
                              cat=category)

        df = pytrend.interest_by_region(resolution=geo_resolution)
        df['WeekOfYear'] = week
        df_list.append(df)

    ## Getting merged dataframe for all words in keyword list
    trends_df = pd.concat(df_list)
    trends_df = trends_df.reset_index()   
    
    return trends_df


# ### Revisiting GBM Modeling

# In[ ]:


def train_sklearn_gbm(pd_df, alpha_level, n_est=500, max_depth=6):
    from sklearn.ensemble import GradientBoostingRegressor
    
    index_cols = get_hierarchy()
    transformed_index_cols = [col + "_index" for col in index_cols]
    
    y = pd_df[TARGET_VAR].values
    X = pd_df.drop([TARGET_VAR], axis=1, inplace=False)
    
    GBM_Model = GradientBoostingRegressor(loss='quantile', alpha=alpha_level, n_estimators=n_est, max_depth=max_depth)
    GBM_Model.fit(X,y)
    
    return GBM_Model

def predict_sklearn_gbm(pd_df, GBM_Model):
    
    X = pd_df.drop([TARGET_VAR], axis=1, inplace=False)             .replace([np.inf, -np.inf], np.nan)             .fillna(0)             .values
    
    predictions = GBM_Model.predict(X)
    
    return predictions


# ### Future Shell Creation: exponential smoothing

# In[ ]:


def BuildHWSeasonalSmoothing(df, Target_Var, Time_Var, ntest, nfuture, grp_lvl):
    try:
        df = df.sort_values(by = Time_Var).reset_index(drop=True)
        df_mrds_orig = pd.DataFrame(df, columns=[Time_Var,Target_Var])
    #     .set_index(Time_Var)
        df_mrds = df_mrds_orig.set_index(Time_Var)
        df_mrds= df_mrds.sort_values([Time_Var])
        nmonths = df_mrds.shape[0]
        Train = df_mrds[0:(nmonths-nfuture)]
        Test =  df_mrds[((nmonths-nfuture)+1):]
    #     model = ExponentialSmoothing(Train, trend="add", seasonal="add", seasonal_periods=52)
        model = ExponentialSmoothing(Train, trend="add", seasonal=None)
        fit = model.fit()
        futforecasts = fit.forecast(nfuture+1)

        df[Target_Var][-(nfuture+1):] = futforecasts.to_list()
    except Exception as exc:
        print('{}:'.format(exc))
        print("Forcast Column:{}, Group:{}".format(Target_Var, df[grp_lvl].unique()))
        
    return df


# In[ ]:


def BuildHistorical(df, Target_Var, Time_Var, ntest, nfuture):
    df = df.sort_values(by = Time_Var).reset_index(drop=True)
    df_mrds = pd.DataFrame(df, columns=[Time_Var,Target_Var]).set_index(Time_Var)
    df_mrds= df_mrds.sort_values([Time_Var])
    nmonths = df_mrds.shape[0]
    Train = df_mrds[0:(nmonths-nfuture)]
    Test =  df_mrds[((nmonths-nfuture)+1):]
    LaggedActuals = df[Target_Var].shift(12)[-nfuture:]
    df[Target_Var] = Train[Target_Var].append(LaggedActuals).to_list()
    return df


# In[ ]:


## Palaash Version - hardcoded with dates
## TODO - once we confirm the below version, can delete this version
# def forecast_data_monthly(data, time_val, group_level, columns_to_forecast, future_period):
    
#     data[columns_to_forecast] = data[columns_to_forecast].astype(float)

#     max_date = np.max(data[time_val])
#     future_days = [max_date + i if (max_date + i)<=202012 else \
#                    (202100+(max_date+i)-202012 ) for i in range(1, future_period+2)]


#     shell_data = data[data[time_val] == max_date]
#     shell_data[columns_to_forecast] = 0

#     for future_per in future_days:
#         temp_df = shell_data.copy()
#         temp_df[time_val] = future_per
#         data = data.append(temp_df, ignore_index=True).copy(0)
   
#     if len(group_level)>0:
#         for item in columns_to_forecast:
#             data = data.groupby(group_level[0])\
#                        .apply(BuildHWSeasonalSmoothing, Target_Var = item, Time_Var = time_val, ntest= 0, nfuture = future_period, grp_lvl = group_level[0])\
#                        .reset_index(drop=True)
#     else:
#         for item in columns_to_forecast:
#             data['temp'] = 'temp'
#             data = data.groupby('temp')\
#                        .apply(BuildHWSeasonalSmoothing, Target_Var = item, Time_Var = time_val, ntest= 0, nfuture = future_period, grp_lvl = 'temp')\
#                        .reset_index(drop=True)
#             data.drop(columns = 'temp', inplace=True)

#     return data.copy()


# In[ ]:


## Corey Version - update of the above without hardcoded dates

def forecast_data_monthly(data, time_val, group_level, columns_to_forecast,                          future_period, last_period_of_year=202012):
    
    data[columns_to_forecast] = data[columns_to_forecast].astype('float')
    
    max_date = np.max(data[time_val])
    transition_date = last_period_of_year - 12 + 100
    
    future_days = [max_date + i if (max_date + i) <= last_period_of_year else                    (transition_date + (max_date + i) - last_period_of_year) for                    i in range(1, future_period + 2)]

    shell_data = data[data[time_val] == max_date]
    shell_data[columns_to_forecast] = 0

    for future_per in future_days:
        temp_df = shell_data.copy()
        temp_df[time_val] = future_per
        data = data.append(temp_df, ignore_index=True).copy(0)
   
    if len(group_level) > 0:
        for item in columns_to_forecast:
            data = data.groupby(group_level[0])                       .apply(BuildHWSeasonalSmoothing, Target_Var=item, Time_Var=time_val,                              ntest=0, nfuture=future_period, grp_lvl=group_level[0])                       .reset_index(drop=True)
    else:
        for item in columns_to_forecast:
            data['temp'] = 'temp'
            data = data.groupby('temp')                       .apply(BuildHWSeasonalSmoothing, Target_Var=item, Time_Var=time_val,                              ntest=0, nfuture=future_period, grp_lvl='temp')                       .reset_index(drop=True)
            data.drop(columns='temp', inplace=True)

    return data.copy()


# In[ ]:


def forecast_data_weekly(data, time_val, group_level, columns_to_forecast,                         future_period, last_period_of_year=201952):
    
    data[columns_to_forecast] = data[columns_to_forecast].astype('float')
    
    max_date = np.max(data[time_val])
    transition_date = last_period_of_year - 52 + 100
    
    future_days = [max_date + i if (max_date + i) <= last_period_of_year else                    (transition_date + (max_date + i) - last_period_of_year) for                    i in range(1, future_period + 1)]

    shell_data = data[data[time_val] == max_date]
    shell_data[columns_to_forecast] = 0

    for future_per in future_days:
        temp_df = shell_data.copy()
        temp_df[time_val] = future_per
        data = data.append(temp_df, ignore_index=True).copy(0)
   
    if len(group_level) > 0:
        for item in columns_to_forecast:
            data = data.groupby(group_level[0])                       .apply(BuildHWSeasonalSmoothing, Target_Var=item, Time_Var=time_val,                              ntest=0, nfuture=future_period, grp_lvl=group_level[0])                       .reset_index(drop=True)
    else:
        for item in columns_to_forecast:
            data['temp'] = 'temp'
            data = data.groupby('temp')                       .apply(BuildHWSeasonalSmoothing, Target_Var=item, Time_Var=time_val,                              ntest=0, nfuture=future_period, grp_lvl='temp')                       .reset_index(drop=True)
            data.drop(columns='temp', inplace=True)

    return data.copy()


# In[ ]:


def forecast_data(data, time_val, group_level, columns_to_forecast, future_period):
    data[time_val] = pd.to_datetime(data[time_val])
    data[time_val] = data[time_val].dt.normalize()
    data[columns_to_forecast] = data[columns_to_forecast].astype(float)
    
    max_date = np.max(data[time_val])
    future_days = [max_date + timedelta(days=7*i) for i in range(1, future_period+1)]

    shell_data = data[data[time_val] == max_date]
    shell_data[columns_to_forecast] = 0

    for future_per in future_days:
        temp_df = shell_data.copy()
        temp_df[time_val] = future_per
        data = data.append(temp_df, ignore_index=True).copy(0)
   
    if len(group_level)>0:
        for item in columns_to_forecast:
            data = data.groupby(group_level[0])                       .apply(BuildHWSeasonalSmoothing, Target_Var = item, Time_Var = time_val, ntest= 0, nfuture = future_period, grp_lvl = group_level[0])                       .reset_index(drop=True)
    else:
        for item in columns_to_forecast:
            data['temp'] = 'temp'
            data = data.groupby('temp')                       .apply(BuildHWSeasonalSmoothing, Target_Var = item, Time_Var = time_val, ntest= 0, nfuture = future_period, grp_lvl = 'temp')                       .reset_index(drop=True)
            data.drop(columns = 'temp', inplace=True)

    return data.copy()


# In[ ]:


print('Supplement Source Code Imported')


# In[ ]:





# ### CODE IN DEVELOPMENT

# In[ ]:


def train_holt_winters(pd_df, seasonal_periods=12, seasonal='additive', trend='add'):
    '''
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
    '''
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    
    ## Assumes the data is subset at level of granularity
    ## Eg - for Hermes, looping by dept and store (and predicting at month)
    pd_df = pd_df.sort_values(by=TIME_VAR).reset_index(drop=True)
    pd_df[TARGET_VAR] = pd_df[TARGET_VAR].astype('float')
    pd_df[TARGET_VAR] = pd_df[TARGET_VAR].replace(np.nan, pd_df[TARGET_VAR].mean())
    
    X = pd_df.drop([TARGET_VAR], axis=1)
    y = pd_df[TARGET_VAR]
    
    hw_model = HWES(y, seasonal=seasonal, seasonal_periods=seasonal_periods, trend=trend)
    fitted_hw_model = hw_model.fit(optimized=True, use_brute=True)
    
    return fitted_hw_model


# In[ ]:


def predict_holt_winters(fitted_hw_model, steps=8):
    '''
    https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
    https://medium.com/datadriveninvestor/how-to-build-exponential-smoothing-models-using-python-simple-exponential-smoothing-holt-and-da371189e1a1
    '''
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
    predictions = fitted_hw_model.forecast(steps=steps)
    return predictions


# In[ ]:


def refresh_target_var(how='add', suffix='log'):
    '''
    how = ['add', 'remove']
    suffix = ['log', 'std', 'nrm']
    To add addiitional amendments, update this function!
    Note set_target_var() is a function in the original src codebase
    '''
    
    for indicator in ['_log', '_nrm', '_std']:
        if indicator in TARGET_VAR:
            new_target = TARGET_VAR.replace(indicator, '')
        else:
            new_target = TARGET_VAR
            
    if how == 'remove':
        return set_target_var(new_target)
    
    elif how == 'add':
        new_target = new_target + '_' + suffix
        return set_target_var(new_target)
    
    else:
        print('Please use "add" or "remove" to dictate how to refresh!')
        return set_target_var(TARGET_VAR)


# In[ ]:


## COREY added on 1/11/2021
## COREY TO TEST THIS IN PIPELINE
## Not yet tested - not sure if needed based on our current implementation

def lag_by_group_helper(key, value_df, lag_periods=8, date_index_col='date', shift_freq='MS'):
    df = value_df.assign(group=key) ## this pandas method returns a copy of the df, with group columns assigned the key value
    return (df.sort_values(by=[date_index_col], ascending=True)
        .set_index([date_index_col])
        .shift(lag_periods, freq=shift_freq)) # the parenthesis allow you to chain methods and avoid intermediate variable assignment

def lag_features_by_group(input_pd, lagging_cols_list, dt_format='%Y%m'):
    
    output_pd = input_pd.copy()
    
    sort_hierarchy = [TIME_VAR] + get_hierarchy()
    output_pd.sort_values(by=sort_hierarchy, inplace=True)
    
    datetime_col = output_pd[TIME_VAR].apply(lambda x: pd.to_datetime(x, format=dt_format)) 
    output_pd['date'] = datetime_col
    
    grouped_pd = output_pd[lagging_cols_list + ['date']].groupby(get_hierarchy())
    df_pd_list = [lag_by_group_helper(g, grouped_pd.get_group(g)) for g in grouped_pd.groups.keys()]
    
    lagged_pd = pd.concat(df_pd_list, axis=0)                  .reset_index()
    
    return lagged_pd


# COMMAND ----------



# COMMAND ----------

## Developed by Bhavya and ported over to NewSourceCode by Corey on 5/3/2021
## Assuming no code issues, we can over-write the outdated versions on these in /src code
## Holding separately now so we can isolate/correct issues more easily, if needed

def raise_df_value_error():
  """
  Throw a value error if an object is not a pandas or PySpark DataFrame.
  """
  raise ValueError('Object is not a Pandas or PySpark dataframe')
  

def is_pandas_df(obj):
  """
  Check if an object is a pandas DataFrame.
  """
  return isinstance(obj, pd.DataFrame)


def get_target_var(df, search_str_length=8):
  """
  Get name of global variable target_var after transformations to that column in a pandas or koalas DataFrame.
  NOTE: does not change global var

  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
  DataFrame with aggregated values of global variable target_var.
  """  
 
  if is_pandas_df(df):
    column_names = df.columns.tolist()
    
  else:
    raise_df_value_error()
    
  new_name = [feature for feature in column_names if TARGET_VAR[:search_str_length] in feature]
  
  if len(new_name) > 1: 
    raise ValueError('Returns more than one variable')
  
  return new_name[0]


def set_target_var(name):
  """
  Manually update the target var
  """
  global TARGET_VAR
  TARGET_VAR = name

  
def update_target_var(df, search_str_length=8):
  """
  Automatically update the target_var based on columns in a dataframe
  """  
  new_name = get_target_var(df, search_str_length)
  set_target_var(new_name)
  
  
def log_columns(pd_df, col_list):
  """
  Log Transformation - uses the above adjustments - might not need to adjust the below?
  """
  
  new_column_names = [feature + "_log" for feature in col_list]
  
  trans_pd = pd_df[col_list].apply(lambda x: np.log1p(x))
  trans_pd.columns = new_column_names
  
  final_pd = pd_df.join(trans_pd)
  
  final_pd.drop(col_list, axis=1, inplace=True)
  
  # After log trasformation we changed the qty which is global variable to qty_log
  # This code allows us to make qty_log as taget variable on global level instead of qty
  if TARGET_VAR in col_list:
    update_target_var(final_pd)

  return final_pd


# COMMAND ----------

## Adjustments to conform to PEP dimensions for prediction

def get_hierarchy():  
  """
  Return a list comprised of the concatenated product and business hierarchies up to the user-specified levels (as defined in global variables).
  """

  return PRODUCT_HIER[:PRODUCT_LEVEL] + CUSTOMER_HIER[:CUSTOMER_LEVEL] + LOCATION_HIER[:LOCATION_LEVEL]  ## removed OTHER_CATEGORICALS from this

# COMMAND ----------

## Ported over on 5/3/2021 by Corey
## This code was updated in-place in "OrigSourceCode" - moved here for now (for error handling)

def pull_feature_importance_scores(lgbm_importance='gain', importance_type='weight', sort_by_col='xgb_model_stage1'):
    '''
    Workaround for issues with saving the Feature Importance
    Output here is dataframe - does not automatically save
    '''
    
    xgb_models = [model for model in MODEL_DICT if 'xgb' in model]
    lgbm_models = [model for model in MODEL_DICT if 'lightGBM' in model]
    
    result_list = {}
    
    for model in xgb_models:
        feature_importance_dict = MODEL_DICT[model].get_score(importance_type=importance_type)
        result_list[model] = feature_importance_dict
        
    for model in lgbm_models:
        lightGBM_importance_array = MODEL_DICT[model].feature_importance(importance_type=lgbm_importance)
        feature_names = MODEL_DICT[model].feature_name()
        feature_importance_dict = dict(zip(feature_names, lightGBM_importance_array.T))
        result_list[model] = feature_importance_dict
    
    final_pd = pd.DataFrame.from_dict(result_list).reset_index()
    final_pd.sort_values(by=sort_by_col, axis=0, inplace=True, ascending=False, na_position='last')
    
    return final_pd

  
def get_encv_coeffs(model, train_pd, nonzero_only=False):
    '''
    Return coefficients from a cross-validated Elastic Net model.
    Input training dataframe and ENCV model training during /Modeling runs.
    '''   

    tar_var_list = [TARGET_VAR, TARGET_VAR + '_log', TARGET_VAR + '_std']
    tar_var_to_drop = list(set(tar_var_list).intersection(set(train_pd)))
    
    X_pd = train_pd.drop(tar_var_to_drop, axis=1, inplace=False)
    
    coefficients = pd.DataFrame(model.coef_.transpose(), index = X_pd.columns, columns = ['value'])
    non_zero_coefficients = coefficients[coefficients['value'] != 0]
    
    if nonzero_only:
        output_pd = pd.DataFrame(non_zero_coefficients)
    else:
        output_pd = coefficients
    
    output_pd.sort_values(by=['value'], axis=0, inplace=True, ascending=False, na_position='last')
    
    return output_pd

# COMMAND ----------

## Added from DemandBrain 1.5 - these are "utilities" associated with that codebase
## Pulling these over to get saving capabilities (without issue of overlapping function names)

def get_secrets(in_scope, secret_dict):
  """
  Secrets are used to connect to Azure SQL in a confidential manner.  This function
    returns the credentials to get within "secret_dict" that are within the "in_scope" server.

  Parameters
  ----------
  in_scope : String
      Secret scope
  secret_dict : Dictionary
      output_secret_name : actual_secret_key

  Returns
  -------
  out_credentials : Dictionary
      Dictionary containing redacted secret values
  """

  out_credentials = {}
  for i in secret_dict.keys():
    out_credentials[i] = dbutils.secrets.get(scope = in_scope, key = secret_dict.get(i) )
  return(out_credentials)


def convertDFColumnsToDict(df, key_col, value_col):
  """
  Converts PySpark DF to Pandas dictionary. Each row is key and value pair (no duplicate id rows can be fed in).
  """
  dict  = {row[key_col]:row[value_col] for row in df.collect()}
  return dict


def convertDFColumnsToList(df, col):
  "Converts Dataframe column to list and returns"
  return(df.select(col).distinct().rdd.map(lambda r: r[0]).collect())


def convert_df_cols_to_list(df, group_cols, value_cols):
  """
  Aggregates data and stores value_cols into a list stored as a separate column.  Collect_list does not unwind in order of listed output.
    In order to preserve order, sort pyspark dataframe prior to using the function.
      E.g.: If ultimate output will be [MODEL_ID : variable], sort by model_id, variable prior

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      List of columns over which you wish to aggregate data
  value_cols : List
      Input columns that you wish to store as list.  The output column will be in the following format:
        arr[{input col 1: value},
            {input col 2: value}]

  Returns
  -------
  df : PySpark dataframe
      Aggregated dataset with new column "list_out"
  """
  df = df.groupBy(group_cols).agg(collect_list(struct(value_cols)).alias("list_out"))
  return df


def row_sum_DF(df, sum_cols, out_col):
  "Row-wise sum of PySpark dataframe"
  sum_cols = list(set(sum_cols) & set(df.columns))

  #Sum columns
  if len(sum_cols)>0:
    #Impute null to 0
    #df = impute_to_value(df, sum_cols ,0) #This should be done outside the function

    expression = '+'.join(sum_cols)
    df = df.withColumn(out_col, expr(expression))
  else:
    df = df.withColumn(out_col, lit(0))
  return (df)


def melt(df, id_vars, value_vars, var_name: str="variable", value_name: str="value"):
    """
    Melts dataset from wide to long
    """
    _vars_and_vals = array(*(
        struct(lit(c).alias(var_name), col(c).alias(value_name))
        for c in value_vars))
    _tmp = df.withColumn("_vars_and_vals", explode(_vars_and_vals))
    cols = id_vars + [
            col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)

## List utilities
def intersect_two_lists(list1, list2):
  "Returns intersection between two lists"
  return(list(set(list1) & set(list2)))

def subtract_two_lists(list1, list2):
  "Returns missing elements between two lists"
  return(list(set(list1) - set(list2)))

def add_prefix_to_list(in_list, prefix):
  """
  Adds a prefix to all elements in a list
  """
  return([prefix + sub for sub in in_list])

def add_suffix_to_list(in_list, suffix):
  """
  Adds a suffix to all elements in a list
  """
  return([sub + suffix for sub in in_list])


##Delta tables
def save_df_as_delta(df, path, enforce_schema=True):
  """Saves Spark Dataframe as a Delta table"""
  if enforce_schema == False:
    #df.write.format("delta").option("mergeSchema", "true").mode("overwrite").save(path) #allows user to add columns to schema.  If columns are deleted they will remain in schema but be null in delta table.  Columns that previously and currently exist must have identical schemas.
    df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").save(path) #allows for full schema overwrite but requires rewriting all data
  else:
    #Enforces schema exactly
    df.write.format("delta").mode("overwrite").save(path)

def load_delta_info(path):
    """Loads delta table information"""
    delta_info = DeltaTable.forPath(spark, path)
    return delta_info

def set_delta_retention(delta_tbl, period):
    """Sets how long to retain delta table historical data"""
    delta_tbl.deletedFileRetentionDuration = "interval " + period
    delta_tbl.logRetentionDuration = "interval " + period

def load_delta(path, version=None):
    """Loads delta table as Spark Dataframe"""
    if version is None:
      latest_version = spark.sql("SELECT max(version) FROM (DESCRIBE HISTORY delta.`" + path +"`)").collect()
      df = spark.read.format("delta").option("versionAsOf", latest_version[0][0]).load(path)
    else:
      df = spark.read.format("delta").option("versionAsOf", version).load(path)
    return(df)

#SQL tables
def save_df_to_sql(df, in_table, in_jdbcurl, in_user, in_password, in_mode):
  """
  Bulk inserts data to sql
  https://docs.microsoft.com/en-us/sql/connect/spark/connector?view=sql-server-ver15

  Parameters
  ----------
  df : PySpark dataframe
  in_table : String
    Name of output table
  in_jdbcurl : String
    connection jdbcurl
  in_user : String
    connection user
  in_password : String
    connection password
  in_mode : String
    Allowable values include: overwrite, append

  """
  try:
    df.write \
      .format("com.microsoft.sqlserver.jdbc.spark") \
      .mode(in_mode) \
      .option("url", in_jdbcurl) \
      .option("dbtable", in_table) \
      .option("user", in_user) \
      .option("password", in_password) \
      .save()
    print("Data saved to SQL table:" + in_table)
  except ValueError as error :
      print("Connector write failed", error)

#Local csv files
def load_csv(path, sql_context):
  """Reads in local laptop csv file as PySpark dataframe"""
  df = (sql_context.read.format("csv").options(header="true")
      .load(path))
  return df


def convert_str_to_list(in_str, delim):
  """Separates string into list by delimeter"""
  # TODO: Add test for none/empty value for the parameter
  if in_str is None:
    out_list = None
  else:
    in_str = str(in_str)
    out_list = [s.strip() for s in in_str.split(delim)]
  return out_list

def load_parameters(param_table, name_col, val_col):
  """
  Loads parameters in input table to an output dictionary

  Parameters
  ----------
  param_table : Pandas dataframe
      Input config table
  name_col : String
      Name of column that has the config name
  val_col: String
      Name of column that has the config value

  Returns
  -------
  out_params : Python dictionary
      Dictionary with keys equal to name_col and values equal to val_col
  """

  #Check fields exist in data
  if len(intersect_two_lists([name_col],param_table.columns)) == 0:
    return(None)
  if len(intersect_two_lists([val_col],param_table.columns)) == 0:
    return(None)

  #TO-DO: don't we have a convertDFtoDict function? Does that only do 1D tables?
  #Find integer column position of "name" field and "value" field
  val_loc = param_table.columns.get_loc(val_col)
  name_loc = param_table.columns.get_loc(name_col)

  #Load dictionary with table values
  out_params = {}
  for i in range(len(param_table)):
    value = param_table.iloc[i,val_loc]
    name = param_table.iloc[i,name_loc]
    out_params[name] = value
  return(out_params)

# COMMAND ----------

## COREY ADDED on 5/11/2021
## Previous version does not return an actua object (used as a helper function)
## Created new version that returns dataframe using different function name
## Also added 'vars_to_lag' to the vars to keep, as returned in output_pd

def create_lag_variables_pd(pandas_df, vars_to_lag, lag=1, grouping_vars=None):
  """
  Create lagged features on particular variables. By default, lags your target variable (e.g., Sales) by 1 week.
  NOTE: The internals of this function assumes you're using the YYYYWW format for TIME_VAR.
  
  Parameters
  ----------
  pandas_df : pandas or koalas DataFrame
      DataFrame with variables to lag and "datetime" variable.
  
  vars_to_lag : list or str
      Variable(s) of pandas_df for which to create lagged features.
  
  lag : list or int
      Number(s) of units of time for which to create lagged features.

  Returns
  -------
  final_df : pandas or koalas DataFrame
      DataFrame grouped by global variables PRODUCT_LEVEL and BUSINESS_LEVEL and with specified lagged variables.

  TODO: this code could be cleaned up to filter first; could calc grouping_vars only once
  """
  assert "datetime" in pandas_df.columns, "Please create the datetime column using assign_datetime_var() before calling this function"
  
  if type(vars_to_lag) is str:
    vars_to_lag = [vars_to_lag] 
    
  if type(lag) is int:
    lag = [lag] 
  
  if not grouping_vars:
      grouping_vars = get_hierarchy()

  var_list = []
  for shift in lag:
    for var in vars_to_lag:
      var_name = var + "_lag_" + str(shift) 
      var_list += [var_name]
      
      pandas_df[var_name] = pandas_df.groupby(grouping_vars)[var]\
                                     .shift(shift)\
                                     .fillna(0)
      
  vars_to_keep = grouping_vars + ["datetime"] + [TIME_VAR] + vars_to_lag + var_list  
  final_df = pandas_df.loc[:, vars_to_keep]
  
  return final_df

# COMMAND ----------

print('Corey updates confirmed')  ## to delete after validation exercise