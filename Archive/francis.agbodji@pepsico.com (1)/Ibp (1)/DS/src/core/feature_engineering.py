# Databricks notebook source
import numpy as np

# COMMAND ----------

# DBTITLE 1,Pandas Time Features
def assign_datetime_var(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """reduce_df_size
  Add a new "datetime" column containing a date_time copy of TIME_VAR (assuming that TIME_VAR follows YYYYWW by default).
  """
  
  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override
  
  # the YYYYWW we often use will throw an error since it has no interpretable day column
  if datetime_format in ["%Y%U-%w", "%Y%W-%w"]:
    pandas_df["datetime"] = pd.to_datetime(pandas_df[time_var].astype(str) + '-1', format=datetime_format)
  else:
    pandas_df["datetime"] = pd.to_datetime(pandas_df[time_var].astype(str), format=datetime_format)
    
  return pandas_df


def ensure_datetime_in_pd(df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Adds datetime column to dataframe if it's not present already
  """
  if "datetime" not in df.columns:
    df = assign_datetime_var(df, datetime_format, time_var_override)
  
  return df


def get_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Year' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Year = pandas_df['datetime'].dt.year)


def get_day_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Day' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Day = pandas_df['datetime'].dt.day)


def get_day_of_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Day_Of_Year' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Day_Of_Year = pandas_df['datetime'].dt.dayofyear)


def get_day_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'DayYear' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(DayYear = (pandas_df['datetime'].dt.year.astype(str) + pandas_df['datetime'].dt.dayofyear.astype(str)).astype(int))


def get_day_of_week_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Day_Of_Week' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Day_Of_Week = pandas_df['datetime'].dt.dayofweek.astype(np.int8))


def get_weekend_flag_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Weekend_Flag' feature to the dataframe using a datetime column
  """
  assert 'Day_Of_Week' in list(pandas_df.columns), "Please run get_day_of_week_feature first."
  
  return pandas_df.assign(Weekend_Flag = (pandas_df['Day_Of_Week'] >= 5).astype(np.int8))


def get_week_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Week' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Week = pandas_df['datetime'].apply(lambda x: x.strftime("%U")).astype(int))


def get_week_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'WeeKYear' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(WeekYear = (pandas_df['datetime'].dt.year.astype(str) + pandas_df['datetime'].apply(lambda x: x.strftime("%U")).astype(str)).astype(int))


def get_quarter_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Quarter' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Quarter = pandas_df['datetime'].dt.quarter)


def get_quarter_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'QuarterYear' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(QuarterYear = (pandas_df['datetime'].dt.year.astype(str) + pandas_df['datetime'].dt.quarter.astype(str)).astype(int))


def get_week_of_month_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'WeekOfMonth' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(WeekOfMonth = (pandas_df['datetime'].dt.day-1)//7 + 1)


def get_month_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'Month' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(Month = (pandas_df['datetime'].dt.month))
  
  
def get_month_year_feature(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Add a 'MonthYear' feature to the dataframe using a datetime column
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  return pandas_df.assign(MonthYear = (pandas_df['datetime'].dt.year.astype(str) + pandas_df['datetime'].dt.month.astype(str)).astype(int))


def get_linear_trend_features(pandas_df, datetime_format="%Y%U-%w", time_var_override=None):
  """
  Calculate the linear value of the time variable and include the sin and cos to account for cyclicality.
  """
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  pandas_df = pandas_df.assign(LinearTime = pd.to_numeric(pandas_df["datetime"]))
  
  pandas_df = pandas_df.assign(SinLinearTime = np.sin(pandas_df["LinearTime"]),\
                               CosLinearTime = np.cos(pandas_df["LinearTime"]))

  pandas_df.drop(["LinearTime"], axis=1, inplace=True)

  return pandas_df


def get_time_features(pandas_df, datetime_format="%Y%U-%w", time_var_override=None, func_bundle='weekly'):
  """
  Get all time_var related features and drop datetime when finished
  """
  
  pandas_df = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override)
  
  prebuilt_bundles = {
    'weekly' : [
      get_year_feature, get_week_feature, get_quarter_feature, get_month_feature,
      get_month_year_feature, get_quarter_year_feature, get_week_of_month_feature,
      get_linear_trend_features
    ], 
    'daily' : [
      get_day_feature, get_day_of_week_feature, get_day_year_feature, get_day_of_year_feature, get_weekend_flag_feature,
      get_year_feature, get_week_feature, get_quarter_feature, get_month_feature,
      get_month_year_feature, get_quarter_year_feature, get_week_of_month_feature,
      get_linear_trend_features
    ], 
  }
  
  # allow user to pass custom list of function if they want
  try:
    time_functions = prebuilt_bundles[func_bundle]
  except:
    time_functions = func_bundle
  
  for func in time_functions:
    try:
      pandas_df = func(pandas_df)
    except: 
      print("Failed to run %s" % func)
      
  pandas_df.drop('datetime', axis=1, inplace=True)
    
  return reduce_df_size(pandas_df)

# COMMAND ----------

# DBTITLE 1,Pandas Imputation Functions
def fill_missing_rows(df_pd, ffill_columns=None, other_agg_dict=None, groupby_columns=None, num_agg_periods=52):
  """
  Inputs
  ------
  df_pd : pandas DataFrame
  ffill_columns : list (default = None)
      List of columns to forward fill (by group) using last available record
      If ffill_columns is None and other_agg_dict is None,
         then ffill_columns = all columns except TARGET_VAR, TIME_VAR, and groupby_columns
  other_agg_dict : dict (default = None)
      Dictionary of column name to pandas groupby aggregation function
      NOTE that 'ffill' can't be used here since aggregating using ffill returns a different output
         than aggregating by any other function
  groupby_columns : list (default = None)
      Columns to group by for forward filling
      If none provided, then uses columns from get_hierarchy() with appropriate suffix 
  num_agg_periods : int (default = 52)
      Positive number of periods before and including END_WEEK to use for filling with an aggregated value
      This parameter is only used for other_agg_dict aggregations
  
  Outputs
  -------
  df_pd : pandas DataFrame
      DataFrame with filled values
  """
  
   
  if not groupby_columns:
    groupby_columns = correct_suffixes_in_list(df_pd, get_hierarchy())
  
  if not ffill_columns and not other_agg_dict:
    columns_with_nulls = df_pd.columns[df_pd.isna().any()].tolist()
    ffill_columns = [col for col in columns_with_nulls \
                    if col not in [TARGET_VAR, TIME_VAR] + groupby_columns]
  
  if not ffill_columns:
    ffill_columns = []
  
  if not other_agg_dict:
    other_agg_dict = {}

  assert TARGET_VAR not in ffill_columns and TARGET_VAR not in other_agg_dict, \
         '\'{0}\' (TARGET_VAR) in ffill_columns or other_agg_dict. It should not be filled'.format(TARGET_VAR)
  
  assert TIME_VAR not in ffill_columns and TIME_VAR not in other_agg_dict, \
         '\'{0}\' (TIME_VAR) in ffill_columns or other_agg_dict. It should not be filled'.format(TIME_VAR)

  assert 'ffill' not in other_agg_dict.values(), \
         '\'ffill\' should not be used in other_agg_dict. Add the column to ffill_columns instead'  
  
  assert (not ffill_columns) or (ffill_columns and set(groupby_columns) == set(correct_suffixes_in_list(df_pd, get_hierarchy()))), \
         'Forward filling using a subset of the hierarchy columns as your groupby_columns would forward fill arbitrarily (e.g., forward filling by brand would mean forward filling the last SKU arbitrarily)'
  
  for groupby_column in groupby_columns:
    assert groupby_column not in ffill_columns and groupby_column not in other_agg_dict, \
           '\'{0}\' in groupby_columns should not be filled'.format(groupby_column)
  
  assert num_agg_periods > 0, 'num_agg_periods provided is <= 0' 
    
  df_pd = df_pd.sort_values(by=groupby_columns + [TIME_VAR])
  
  if ffill_columns:
    df_pd.update(df_pd.groupby(groupby_columns)[ffill_columns].ffill())

  if other_agg_dict:
    past_row_indexes = df_pd[TIME_VAR] <= END_WEEK
    update_pd = df_pd.loc[past_row_indexes].groupby(groupby_columns).tail(num_agg_periods)\
                                           .groupby(groupby_columns).agg(other_agg_dict).dropna(subset = list(other_agg_dict.keys()))
    
    df_pd = df_pd.set_index(get_hierarchy())
    df_pd.update(update_pd, overwrite = False)
    df_pd = df_pd.reset_index()
    
  return df_pd


def predict_missing_values(df_pd, fill_column, feature_columns, stage1_dict, stage2_dict=None, 
                           final_model='xgb_model', splitting_function=None, error_metric='APA', verbose=False):
  """
  Inputs
  ------
  df_pd : pandas DataFrame
  fill_column : str
      Column with empty values for which to predict/fill
  feature_columns : list
      List of columns in df_pd to use to predict values of fill_column
  stage1_dict : dict
      Stage 1 modeling dictionary in format
      model_name : ((train_function, model_parameters), predict_function format)
  stage2_dict : dict (default = None)
      Stage 2 modeling dictionary
  final_model : str
      Name of model to use for final predictions
      If stage2_dict provided, then final_model should be a model_name in stage2_dict
      Otherwise, final_model should be a model_name in stage1_dict
  splitting_function : func (default = sklearn.model_selection.train_test_split)
      Function to use to split records into training/testing (among those with populated values for fill_column)
  error_metric : str (default = 'APA')
      Error metric to use for visualization
  verbose : bool (default = False)
      If True, prints a summary of the changes made
  
  Outputs
  -------
  df_pd : pandas DataFrame
      DataFrame with filled values
  """
  
  # changing global variable TARGET_VAR temporarily to fill_column since it is the target variable for these predictions
  global TARGET_VAR
  target_var_copy = TARGET_VAR
  TARGET_VAR = fill_column

  assert all(feature_column in df_pd.columns.values for feature_column in feature_columns), \
         'Not all feature_columns are columns in df_pd. Consider using correct_suffixes_in_list() to make sure columns match'
  
  if df_pd[feature_columns].isna().sum().sum() > 0:
    print('Warning: There are null values in feature_columns')     
  
  if stage2_dict:
    assert final_model in stage2_dict, 'final_model is not a key in stage2_dict'
    prediction_col_name = final_model + '_stage2'
  else:
    assert final_model in stage1_dict, 'final_model is not a key in stage1_dict, and stage2_dict was not provided'
    prediction_col_name = final_model + '_stage1'
    
  if not splitting_function:
    from sklearn.model_selection import train_test_split
    splitting_function = train_test_split  
  
  rows_with_missing_data = df_pd[fill_column].isna()

  # Training + testing data = rows where TARGET_VAR is populated
  # Holdout data = rows where TARGET_VAR is NA
  train_pd, test_pd = splitting_function(df_pd.loc[~rows_with_missing_data, [fill_column] + feature_columns])
  missing_values_pd = df_pd.loc[rows_with_missing_data, \
                                [fill_column] + feature_columns]
  
  train_test_predictions_pd = run_models(train_pd, test_pd,
                                         stage1_dict=stage1_dict, stage2_dict=stage2_dict,
                                         descale=False, suppressErrorMetrics=False, print_errors=False)
  
  missing_value_predictions_pd = predict_modeling_pipeline(missing_values_pd, modeling_dict)
  
  df_pd.loc[rows_with_missing_data, fill_column] = missing_value_predictions_pd[prediction_col_name].values

  if verbose:
    count_filled_values = rows_with_missing_data.sum()
    print('Filled {0} missing values in column \'{1}\' with predictions using columns {2} \n'.format(count_filled_values, fill_column, feature_columns))
    
    filled_values_describe = missing_value_predictions_pd[prediction_col_name].describe()
    non_missing_values_describe = df_pd.loc[~rows_with_missing_data, fill_column].describe()
    
    describe_df = pd.concat([non_missing_values_describe, filled_values_describe], axis = 1)
    describe_df.columns = ['Non-missing values', 'Filled / predicted values']
    print(describe_df)
    print('\n')
    
    error_col_name = prediction_col_name + '_calc_' + error_metric
    is_error_describe = train_test_predictions_pd.loc[train_test_predictions_pd['sample'] == 'IS', error_col_name].describe()
    oos_error_describe = train_test_predictions_pd.loc[train_test_predictions_pd['sample'] == 'OOS', error_col_name].describe()
    error_df = pd.concat([is_error_describe, oos_error_describe], axis = 1)
    error_df.columns = [error_col_name + ' (IS)', error_col_name + ' (OOS)']
    print(error_df)
    print('\n')
    
  # returning global variable MODEL_DICT back to initial state after additions from run_models()
  global MODEL_DICT
  MODEL_DICT = {}
  
  # returning global variable TARGET_VAR to value before this function was called
  TARGET_VAR = target_var_copy
  
  return df_pd

# COMMAND ----------

# DBTITLE 1,Pandas Lagged Features
def get_lagged_features(pandas_df, vars_to_lag, LAGGED_PERIODS, time_var_override=None, vars_to_group_by=None,
                        datetime_format="%Y%U-%w", resample_format='W-MON', time_feature_func=get_week_year_feature,
                        drop_resampled_rows=True, sort_dataframe=True):
  """
  Calculate lagged features for a DataFrame given a set of variables to lag, lag times, and pandas date and resample formats.
  """
  
  warnings.warn("get_lagged_features fills missing values with nulls by default; ensure this treatment is appropriate for your treatment variables")
    
  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override
    
  if not vars_to_group_by:
    vars_to_group_by = get_hierarchy()
  
  vars_to_group_by = correct_suffixes_in_list(pandas_df, vars_to_group_by)
    
  full_hierarchy = vars_to_group_by + ensure_is_list(time_var)

  datetime_pd = ensure_datetime_in_pd(pandas_df, datetime_format, time_var_override=time_var)
  datetime_pd = fill_missing_timeseries(datetime_pd, resample_format, vars_to_group_by)
  
  create_lag_variables(datetime_pd, vars_to_lag, LAGGED_PERIODS, vars_to_group_by)  
  
  if drop_resampled_rows:
    datetime_pd = filter_using_indexes(datetime_pd, pandas_df, full_hierarchy)    
    assert len(datetime_pd) == len(pandas_df), print("Your dataframe wasn't filtered correctly, likely because \
                                                      one of your date formats is incorrectly specified. Are you using a bad default?")
  else:
    datetime_pd[time_var] = time_feature_func(datetime_pd, datetime_format=datetime_format)[time_var]
  
  datetime_pd.drop("datetime", axis=1, inplace=True)  
    
  if sort_dataframe:
    datetime_pd.sort_values(full_hierarchy, inplace=True)

  return datetime_pd


def fill_missing_timeseries(pandas_df, date_format='W-MON', grouping_vars=None):
  """
  Fills the missing rows in a timeseries, filling in nulls with 0's by default
  """
  
  assert "datetime" in pandas_df.columns, "Please create the datetime column using assign_datetime_var() before calling this function"
  
  if not grouping_vars:
      grouping_vars = get_hierarchy()

  filled_pd = (pandas_df.set_index('datetime')
               .groupby(grouping_vars)
               .apply(lambda d: d.reindex(
                 pd.date_range(pandas_df.datetime.min(),
                               pandas_df.datetime.max(),
                               freq=date_format)))
               .drop(grouping_vars, axis=1)
               .reset_index())
  
  # fill newly-created columns
  filled_pd[grouping_vars] = filled_pd[grouping_vars].ffill(downcast='infer')
  filled_pd[TARGET_VAR] = filled_pd[TARGET_VAR].fillna(0)
  
  # rename datetime
  datetime_col = [col for col in list(filled_pd.columns) if "level_" in col]
  assert len(datetime_col) == 1
  
  filled_pd = filled_pd.rename({datetime_col[0] : 'datetime'}, axis=1)
  
  return filled_pd


def create_lag_variables(pandas_df, vars_to_lag, lag=1, grouping_vars=None):
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
      
  vars_to_keep = grouping_vars + ["datetime"] + var_list  
  final_df = pandas_df.loc[:, vars_to_keep]

# COMMAND ----------

# DBTITLE 1,Pandas Indexed Features
def get_indexed_features(input_pd, target_vars):
  """
  Generate new features as the indexed values of existing columns.
  
  Parameters
  ----------
  input_pd : pandas or koalas DataFrame
      DataFrame with features to index.
  
  target_vars : list or str
      Variable(s) to index.
  
  Returns
  -------
  final_pd : DataFrame with original features and indexed versions of variable(s) in target_vars.
  """
  
  # allow users to enter a string
  if is_string(target_vars):
    target_vars = [target_vars] 

  new_variable_names = [col + "_indexed" for col in target_vars]
    
  filtered_pd = input_pd.loc[:,target_vars]
    
  mean_of_columns = filtered_pd.mean(axis=0)
  mean_of_columns.replace(0, 1, inplace=True)

  
  indexed_features = filtered_pd.div(mean_of_columns, axis='columns')
    
  indexed_features.columns = new_variable_names
  
  final_pd = pd.concat([input_pd.drop(target_vars, axis=1), indexed_features], axis=1)

  return final_pd


def get_dummy_vars(input_pd, categorical_vars):
  """
  Create dummy variables for a provided list of categorical variables.
  
  Parameters
  ----------
  input_pd : pandas or koalas DataFrame
      DataFrame with categorical variables.
  
  categorical_vars: list or str
      Categorical variable(s) of input_pd for which to create dummy variables.
  
  Returns
  -------
  df_with_dummies : pandas or koalas DataFrame
      DataFrame with original columns for categorical_vars and corresponding dummy variables.
  """
  
  if is_string(categorical_vars):
     categorical_vars = [categorical_vars]
  
  dummy_pd = pd.get_dummies(input_pd[categorical_vars])
  
  return pd.concat([input_pd, dummy_pd],axis=1)

# COMMAND ----------

# DBTITLE 1,Pandas Product Features
def get_competitor_ratio(input_pd, TARGET_VAR, competitor_var):
  """
  Calculate the ratio between two columns (target variable and competitor variable).

  Parameters
  ----------
  input_pd : pandas or koalas DataFrame
      DataFrame with target and competitor variables.
  
  TARGET_VAR : str
      Name of column in input_pd.
  
  competitor_var : str
      Name of column in input_pd.
  
  Returns
  -------
  final_pd : pandas or koalas DataFrame
      input_pd with additional column for row-wise quotient of target variable and competitor variable.
      
  """
  print("Replacing zero values in competitor_var with 1 to avoid missing values")
  input_pd[competitor_var] = input_pd.loc[:, competitor_var].replace(0, 1)
  
  final_pd = input_pd.copy()
  final_pd[TARGET_VAR + "_" + competitor_var + "_ratio"] = final_pd[TARGET_VAR]/final_pd[competitor_var]
  
  return(final_pd)
  
  
def get_product_age_feature(df, product_cols):
  """
  Adds a flag when the column at the lowest level of the global product hierarchy had its first sale
  """
  #product_cols = PRODUCT_HIER[PRODUCT_LEVEL-1]

  filtered_df = df[(df[TARGET_VAR] > 0)]
  time_stats = filtered_df.groupby(product_cols)[TIME_VAR].min()\
                                                          .reset_index()\
                                                          .rename({TIME_VAR:"Product_Age"}, axis=1)

  final_df = pd.merge(df, time_stats, on=[product_cols], how="left")

  final_df["Product_Age"] = (END_WEEK - final_df["Product_Age"]).fillna(0)

  return final_df
  

def get_new_product_flag_feature(df, time_threshold=26):
    """
    Adds a flag when the column at the lowest level of the global product hierarchy didn't have sales more than time_threshold periods ago 
    
    Parameters
    ----------
    df : Pandas DataFrame
    time_threshold : Prior number of weeks from end_date after which sales are to be considered.
    
    Global Parameters
    ----------
    product_cols : The groupby columns to indicate a product. Ex. MaterialId, CustomerId, ..
    TIME_VAR : The column indicating a time series in the dataframe. Ex. WeekId, MonthId, DateId, ..
    TARGET_VAR : To filter on this column such that only valid data points are considered. Ex. Sales, Orders, ..
    END_WEEK : The latest week until which data is present.
    
    Returns
    -------
    df : Pandas DataFrame with an additional column "New_Product_Flag" 0's for old, and 1's for new product.
    """
  
    product_cols = PRODUCT_HIER[PRODUCT_LEVEL-1]
    
    filtered_df = df[(df[TARGET_VAR] > 0)]
    time_stats = filtered_df.groupby(product_cols)[TIME_VAR].min().reset_index()
        
    time_cutoff = update_time(END_WEEK, -1 * time_threshold)
      
    time_stats['New_Product_Flag'] = np.where((time_stats[TIME_VAR] >= int(time_cutoff)), 1, 0)
    time_stats.drop([TIME_VAR], axis=1, inplace=True)
  
    final_df = pd.merge(df, time_stats, how='left', left_on=product_cols, right_on=product_cols)
    final_df['New_Product_Flag'] = final_df['New_Product_Flag'].fillna(0)

    return final_df
  
  
def get_low_velocity_flag_feature(pd_df, hierarchy_cols, target_threshold=0, time_threshold=4):
  """
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or less] sales in the historical data, then LowVelocityFlag = 1.
  """
  
  filt_pd = pd_df.loc[pd_df[TARGET_VAR] <= target_threshold]
  filt_pd.sort_values(by=hierarchy_cols + [TIME_VAR], axis=0, inplace=True)
  
  filt_pd['streak'] = filt_pd.groupby(hierarchy_cols + [filt_pd[TIME_VAR].diff(-1).ne(-1).shift().bfill().cumsum()]).transform('count')[TARGET_VAR]

  consecutive_weeks_no_sales_pd = filt_pd.loc[filt_pd['streak'] >= time_threshold, hierarchy_cols]\
                                         .drop_duplicates()

  consecutive_weeks_no_sales_pd["LowVelocityFlag"] = 1
  
  final_pd = pd_df.merge(consecutive_weeks_no_sales_pd, on=hierarchy_cols, how='left')
  
  final_pd["LowVelocityFlag"] = final_pd["LowVelocityFlag"].fillna(0)
  
  return final_pd


def get_high_velocity_flag_feature(pd_df, hierarchy_cols, target_threshold=1, time_threshold=8):
  """
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or more] sales in the historical data, then HighVelocityFlag = 1.
  """

      
  filt_pd = pd_df[pd_df[TARGET_VAR] >= target_threshold]
    
  filt_pd.sort_values(by=hierarchy_cols + [TIME_VAR], axis=0, inplace=True)
  
  filt_pd['streak'] = filt_pd.groupby(hierarchy_cols + [filt_pd[TIME_VAR].diff(-1).ne(-1).shift().bfill().cumsum()]).transform('count')[TARGET_VAR]
    
  consecutive_weeks_high_sales_pd = filt_pd.loc[filt_pd['streak'] > time_threshold, hierarchy_cols]\
                                           .drop_duplicates()
  
  consecutive_weeks_high_sales_pd["HighVelocityFlag"] = 1
  
  final_pd = pd_df.merge(consecutive_weeks_high_sales_pd, on=hierarchy_cols, how='left')
  
  final_pd["HighVelocityFlag"] = final_pd["HighVelocityFlag"].fillna(0)
  
  return final_pd

# COMMAND ----------

# DBTITLE 1,Pandas Holiday Features
def get_holiday_flag_features(df):
    """
    . Calendar Df Code - nusa-dhpr-dbr-dev(WORKSPACE) - anand.singh/Advacedforecasting/0.2 Data Prep - Dbo, Loc, Prod Hier, Cal, Syn_Pos
    . Dataset Link - https://www.kaggle.com/gsnehaa21/federal-holidays-usa-19662020
    
    Parameters
    ----------
    df : Pandas DataFrame
    
    Global Parameters
    ----------
    START_WEEK : The starting week after which calendar, holidays are to be considered.
    END_WEEK : The ending week before which calendar, holidays are to be considered.
    TIME_VAR : The column name identifying week_id on the input df.

    Returns
    -------
    df : Pandas DataFrame with 2 additional columns.
      'Upcoming_Holiday_Flag' = 1 only if there is a holiday "coming" in the next 2 weeks.
      'Holiday_Flag' = 1 is the current week has a holiday.
    """
    
    # read in us holidays list, filter and keep only required data.
    usholidays = load_spark_df(get_filestore_path() + "usholidays.csv").toPandas()
    usholidays = usholidays[['Date', 'Holiday']]

    # read in a full year date-wise calendar, filter and keep only required data.
    calendar = load_spark_df(get_filestore_path() + "calendar.csv").toPandas()
    calendar = calendar[['Date', 'week_id']]
    calendar = calendar[(calendar['week_id'] >= START_WEEK) & (calendar['week_id'] <= END_WEEK)].sort_values("Date")

    # Merge the 2, to capture holidays for each day in main calendar df in a new column 'Holiday_Flag'
    cal = calendar.merge(usholidays, how='left', left_on=['Date'], right_on=['Date'])
    cal['Holiday_Flag'] = np.where((cal['Holiday'].isnull()), 0, 1)
    cal = cal[['Date', 'week_id', 'Holiday_Flag']]
        
    # Groupby, to create a dict to capture data if the week has a holiday.
    week_hol_dict = cal.groupby(['week_id'])['Holiday_Flag'].max()

    # create a seperate dataframe from this dictionary. And add a new column for upcoming holidays
    week_hol_df = week_hol_dict.reset_index()
    week_hol_df['Upcoming_Holiday_Flag'] = 0
        
    for index, row in week_hol_df.iterrows():
        curr_week = row['week_id']
        next_week = curr_week + 1
        next_to_next_week = curr_week + 2
  
        if curr_week % 100 == 51 or curr_week % 100 == 52:
            week_hol_df.at[index, 'Upcoming_Holiday_Flag'] = 1
            continue
          
        if (next_week % 100 > 52):
            next_week = next_week - (next_week % 100) + 101
        if (next_to_next_week % 100 > 52):
            next_to_next_week = next_to_next_week - (next_to_next_week % 100) + 101
        if (week_hol_dict[next_week] == 1) or (week_hol_dict[next_to_next_week] == 1):
            week_hol_df.at[index, 'Upcoming_Holiday_Flag'] = 1
            
    df = df.merge(week_hol_df, how='left', left_on=TIME_VAR, right_on=['week_id'])
    df.drop(['week_id'], axis=1, inplace=True)
    
    return df

# COMMAND ----------

# DBTITLE 1,Pandas Numeric Aggregation Features
def get_lagged_hierarchical_agg_features(df, columns_to_aggregate, prod_lvl, bus_lvl, time_ind=None, agg_func=np.average, lag_periods=[1], *args, **kwargs):
    """
    Calculate the aggregated value of specified columns based on levels within the product and business hierarchy. Must be lagged.
    """
    
    if not time_ind:
      time_ind = TIME_VAR
    
    groupby_cols = PRODUCT_HIER[:prod_lvl] + BUSINESS_HIER[:bus_lvl] + [time_ind]
    
    target_columns = ensure_is_list(columns_to_aggregate)
    new_columns_dict = {column : (agg_func.__name__ + '_' + PRODUCT_HIER[prod_lvl-1] + "_" + BUSINESS_HIER[bus_lvl-1] + "_" + column) for column in target_columns}
    new_columns_list = list(new_columns_dict.values())
        
    agg_df = df.groupby(groupby_cols)[target_columns].agg(agg_func).reset_index()
    agg_df.rename(columns=new_columns_dict, inplace=True)
    
    join_df = df.merge(agg_df, how='left', on=groupby_cols)
    
    final_pd = get_lagged_features(join_df, new_columns_list, lag_periods, *args, **kwargs)
    final_pd.drop(new_columns_list, inplace=True, axis=1)
    
    return final_pd

  
#TODO use the same prod_lvl and bus_lvl syntax described above; be sure that the aggregation levels make it into the column names (also done above)
def get_quantile_features(df, groupby_cols, columns_to_quantile, quantile=.25, inter='linear'):

    '''
    Assign quantiles to both unit sales and dollar sales for the desired 
    intersection of the customer and product hierarchy. 
  
    Returns
    -------
    df : pandas DataFrame
    Modeling dataframe with two new quantile columns for unit and dollar sales
    
    '''
    
    columns_to_quantile = list(columns_to_quantile)
    
    new_column_names_dict = {col: col + "_" + str(quantile) + "_quantile" for col in columns_to_quantile}
    
    column_list = groupby_cols + columns_to_quantile
    
    grouped_pd = df[column_list].groupby(groupby_cols)\
                   .quantile(q=quantile, interpolation=inter)\
                   .rename(columns=new_column_names_dict)
        
    final_pd = df.join(grouped_pd, on= groupby_cols, how="left")
    
    return final_pd

# COMMAND ----------

# DBTITLE 1,Pandas Statistical Features
def calc_consecutive_periods(pandas_df):
  """
  Populate consecutive weeks of both orders and no orders for sales data
  """
  def fill_consecutive_weeks(pd):
    """
    Calculate both consecutive weeks of an order and consecutive weeks of no order
    """
    new_pd = pd.filter([TARGET_VAR])
    new_pd = new_pd != 0

    num_consecutive_weeks = new_pd.cumsum() - new_pd.cumsum().where(~new_pd).ffill().fillna(0).astype(int)
    
    new_pd = ~new_pd
    num_no_consecutive_weeks = new_pd.cumsum() - new_pd.cumsum().where(~new_pd).ffill().fillna(0).astype(int)
    
    pd['num_consec_weeks'] = num_consecutive_weeks[TARGET_VAR]
    pd['num_consec_no_weeks'] = num_no_consecutive_weeks[TARGET_VAR]
    
    return pd
  
  def fill_last_order_date(pd):
    """
    Return PD with last week of non-zero sales noted for each store
    """
    def return_max(pd):
      """
      Function to return maximum datetime value for a given df column
      """
      recent_date = pd['datetime'].max()
      return recent_date
    
    pd = pd.loc[pd[TARGET_VAR] != 0]
    last_order = pd.groupby(PRODUCT_HIER, as_index=False).apply(return_max)
    return last_order
  
  def shift_pd(df, target):
    '''
    Function to shift sales data to create a 1 week lag 
    '''
    df[TARGET_VAR] = df[TARGET_VAR].shift(-1)
    return df
   
  datetime_pd = assign_datetime_var(pandas_df)
  datetime_pd = fill_missing_timeseries(datetime_pd)
   
  first_group = get_hierarchy() + ['datetime', TARGET_VAR]
  datetime_pd = datetime_pd.filter(first_group).sort_values(first_group)
  
  datetime_pd = datetime_pd.groupby(get_hierarchy()).apply(shift_pd, TARGET_VAR)
  datetime_pd = datetime_pd.groupby(get_hierarchy()).apply(fill_consecutive_weeks)
  return datetime_pd


def calc_rolling_frequency(df, calc_col, rolling_window, period, agg_dict="mean"):
  """
  Calculates rolling aggregated count of orders by rolling window timeframe
  """
  def rolling_calc(df, calc_col, rolling_window, period, agg_dict="mean"):
    column_name = 'rolling_frequency_' + str(period)

    df[column_name] = df[calc_col].rolling(window=rolling_window)\
                                  .agg(agg_dict)
    
    df[column_name] = df[column_name].fillna(0)
    return df
  
  rolling_df = df.groupby(get_hierarchy()).apply(rolling_calc, calc_col, rolling_window, period, agg_dict)
  return rolling_df

# COMMAND ----------

# DBTITLE 1,PySpark Lagged Features
def get_lagged_features_pyspark(input_df, vars_to_lag, LAGGED_PERIODS = [1,2,3,4,7,52], date_format = "yyyy-ww"):
  """
  Create lagged features on particular variables and for particular time periods.
  NOTE: The internals of this function assumes you're using the YYYY-WW format for TIME_VAR.
  """
  
  from pyspark.sql.window import Window
  
  lagged_periods_list = ensure_is_list(LAGGED_PERIODS)
  lagged_vars_list = ensure_is_list(vars_to_lag)
  
  lagged_feature_list = [var + "_lag_" + str(number) for var in lagged_vars_list for number in lagged_periods_list]

  grouping_vars = get_hierarchy()
  
  overWindow = Window.partitionBy(grouping_vars).orderBy(TIME_VAR) 
  
  new_var_list = []
  
  for var in lagged_vars_list:
    for number in lagged_periods_list:
      new_var_name = var + "_lag_" + str(number)
      new_var_list.append(new_var_name)
      input_df = input_df.withColumn(new_var_name, lag(var, number).over(overWindow))
                  
  final_df = input_df.fillna(0, subset=new_var_list)
  
  return final_df


def get_lagged_high_sales_week_flag_feature(pd_df, lag_periods=[52], tar_var=None, *args, **kwargs):
    """
    Flag to identify weeks with higher than average sales in a specified list of lag periods (this variable must be lagged to make sense)
    """
    
    if not tar_var:
      tar_var = TARGET_VAR
    
    hierarchy_cols = get_hierarchy()
      
    avg_name = 'AVG_BY_PROD_' + tar_var.upper()
            
    avg_pd = pd_df.groupby(hierarchy_cols)[tar_var].mean().reset_index()
    avg_pd.rename(columns={tar_var : avg_name}, inplace=True)
    
    agg_pd = pd.merge(pd_df, avg_pd,\
                      how='left',
                      left_on=hierarchy_cols, right_on=hierarchy_cols)
        
    agg_pd['High_Sales_Week_Flag'] = np.where((agg_pd[avg_name] < agg_pd[tar_var]), 1, 0)
    agg_pd.drop(avg_name, axis=1, inplace=True)
            
    final_pd = get_lagged_features(agg_pd, "High_Sales_Week_Flag", lag_periods, *args, **kwargs)
    final_pd.drop("High_Sales_Week_Flag", inplace=True, axis=1)

    return final_pd

# COMMAND ----------

# DBTITLE 1,PySpark Indexed Features
def get_indexed_features_pyspark(input_df, vars_to_index):
  """
  Generate new features as the indexed values of existing columns
  """
  
  vars_list = ensure_is_list(vars_to_index)
  df = ensure_is_pyspark_df(input_df)
  
  filt_df = df.select(vars_list)
  
  filt_col_names = filt_df.schema.names
  new_var_names = [column + "_indexed" for column in vars_list]
  
  col_means = filt_df.groupBy()\
                     .mean()\
                     .toDF(*filt_col_names)
                   
  for var in filt_df.columns:
    mean = col_means.select(var).first()[var]
    filt_df = filt_df.withColumn(var, col(var)/(mean))
    
  filt_df = filt_df.toDF(*new_var_names)
     
  final_indexed_df = columnwise_union_pyspark(df, filt_df)
  
  return final_indexed_df


def get_dummy_vars_pyspark(df, categorical_vars):
  """
  Create dummy variables for a provided list of categorical variables.
  
  Parameters
  ----------
  df : pandas or koalas DataFrame
      DataFrame with categorical variables.
  
  categorical_vars: list or str
      Categorical variable(s) of input_pd for which to create dummy variables.
  
  Returns
  -------
  df : PySpark DataFrame
      DataFrame with original columns for categorical_vars and corresponding dummy variables.
  """
  
  if is_string(categorical_vars):
    categorical_vars = [categorical_vars]
  
  # inpiration: https://stackoverflow.com/questions/46528207/dummy-encoding-using-pyspark/46587741
  # sould just be looping through a handful of categorical variables (not every single row) 
  for categorical_var in categorical_vars:
    categories = df.select(categorical_var).distinct().rdd.flatMap(lambda x:x).collect()
    dummies = [F.when(F.col(categorical_var) == category, 1).otherwise(0).alias(str(category)) for category in categories]
    df = df.select(dummies + df.columns)
  
  return df

# COMMAND ----------

# DBTITLE 1,PySpark Calculated Features
def get_quantile_features_pyspark(df, groupby_cols, column_to_quantile, quantile = 0.25):
  
  """
  Assign quantiles to both unit sales and dollar sales for the desired 
  intersection of the customer and product hierarchy. Used in feature engineering

  Returns
  -------
  final_df : PySpark DataFrame
      Modeling dataframe with two new quantile columns for unit and dollar sales

  """
  
  groupby_cols = list(groupby_cols)

  # Important: PySpark only allows you to get quantiles for one feature at a time 
  quantile_sql_function = F.expr('percentile_approx(' + column_to_quantile + ', ' + str(quantile) + ')')
    
  new_column_names_list = groupby_cols + [column_to_quantile + "_" + str(quantile) + "_quantile"]
  
  grouped_df = df.select(groupby_cols + [column_to_quantile]).groupBy(groupby_cols).agg(quantile_sql_function).toDF(*new_column_names_list)
  
  final_df = df.join(grouped_df, on = groupby_cols, how = 'left')
  
  return final_df 

# COMMAND ----------

def get_competitor_ratio_pyspark(df, TARGET_VAR, competitor_var):
  """
  Calculate the ratio between two columns (target variable and competitor variable).

  Parameters
  ----------
  df : PySpark DataFrame
      DataFrame with target and competitor variables.
  
  TARGET_VAR : str
      Name of column in input_pd.
  
  competitor_var : str
      Name of column in input_pd.
  
  Returns
  -------
  df : PySpark DataFrame
      Input DataFrame with additional column for row-wise quotient of target variable and competitor variable.
  """
   
  df = df.withColumn(competitor_var, when(col(competitor_var) == 0, 1).otherwise(col(competitor_var)))
  
  return df.withColumn(TARGET_VAR + "_" + competitor_var + "_ratio",
                       col(TARGET_VAR) / col(competitor_var))              
                     
  

  
def get_low_velocity_flag_pyspark(df, target_threshold=0, time_threshold=3):
  """
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or less] sales in the historical data, then LowVelocityFlag = 1.
  """
  
  hierarchy_cols = get_hierarchy()
      
  filt_df = df.filter(col(TARGET_VAR) <= target_threshold)
  filt_df = filt_df.orderBy(hierarchy_cols + [TIME_VAR], ascending= True)
        
  # inspiration for this implementation came from: https://stackoverflow.com/questions/54445961/pyspark-calculate-streak-of-consecutive-observations
  window1 = Window.partitionBy([col(x) for x in hierarchy_cols]).orderBy(col(TIME_VAR))
  window2 = Window.partitionBy([col(x) for x in (hierarchy_cols + [TARGET_VAR])]).orderBy(col(TIME_VAR))
  streak_df = filt_df.withColumn('grp',F.row_number().over(window1) - F.row_number().over(window2))
  window3 = Window.partitionBy([col(x) for x in (hierarchy_cols + [TARGET_VAR, 'grp'])]).orderBy(col(TIME_VAR))
  streak_df = streak_df.withColumn('streak', F.when(col(TARGET_VAR) > target_threshold, 0).otherwise(F.row_number().over(window3)))
  streak_max = streak_df.groupBy(hierarchy_cols).max('streak')

  low_velocity_lambda_function = F.udf(lambda x: 1 if x >= time_threshold else 0)
  
  low_velocity_flags = streak_max.withColumn('LowVelocityFlag', low_velocity_lambda_function(col('max(streak)'))).drop('max(streak)')
  
  df = df.join(low_velocity_flags, on = hierarchy_cols, how = 'left')
  df = df.withColumn('LowVelocityFlag', col('LowVelocityFlag').cast('int')).fillna({'LowVelocityFlag':0})
  
  return df


def get_high_velocity_flag_pyspark(df, target_threshold=1, time_threshold=8):
  """
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or more] sales in the historical data, then HighVelocityFlag = 1.
  """
  
  hierarchy_cols = get_hierarchy()
      
  filt_df = df.filter(col(TARGET_VAR) <= target_threshold)
  filt_df = filt_df.orderBy(hierarchy_cols + [TIME_VAR], ascending= True)
    
  # inspiration for this implementation came from: https://stackoverflow.com/questions/54445961/pyspark-calculate-streak-of-consecutive-observations
  window1 = Window.partitionBy([col(x) for x in hierarchy_cols]).orderBy(col(TIME_VAR))
  window2 = Window.partitionBy([col(x) for x in (hierarchy_cols + [TARGET_VAR])]).orderBy(col(TIME_VAR))
  streak_df = filt_df.withColumn('grp',F.row_number().over(window1) - F.row_number().over(window2))
  window3 = Window.partitionBy([col(x) for x in (hierarchy_cols + [TARGET_VAR, 'grp'])]).orderBy(col(TIME_VAR))
  streak_df = streak_df.withColumn('streak', F.when(col(TARGET_VAR) < target_threshold, 0).otherwise(F.row_number().over(window3)))
  streak_max = streak_df.groupBy(hierarchy_cols).max('streak')
  
  high_velocity_lambda_function = F.udf(lambda x: 1 if x >= time_threshold else 0)
  
  high_velocity_flags = streak_max.withColumn('HighVelocityFlag', high_velocity_lambda_function(col('max(streak)'))).drop('max(streak)')
  
  df = df.join(high_velocity_flags, on = hierarchy_cols, how = 'left')
  df = df.withColumn('HighVelocityFlag', col('HighVelocityFlag').cast('int')).fillna({'HighVelocityFlag':0})
  
  return df


def get_holiday_flag_features_pyspark(df):
  """
  . Calendar Df Code - nusa-dhpr-dbr-dev(WORKSPACE) - anand.singh/Advacedforecasting/0.2 Data Prep - Dbo, Loc, Prod Hier, Cal, Syn_Pos
  . Dataset Link - https://www.kaggle.com/gsnehaa21/federal-holidays-usa-19662020

  Parameters
  ----------
  df : PySpark DataFrame

  Global Parameters
  ----------
  START_WEEK : The starting week after which calendar, holidays are to be considered.
  END_WEEK : The ending week before which calendar, holidays are to be considered.
  TIME_VAR : The column name identifying week_id on the input df.

  Returns
  -------
  df : PySpark DataFrame with 2 additional columns.
    'Upcoming_Holiday_Flag' = 1 only if there is a holiday "coming" in the next 2 weeks.
    'Holiday_Flag' = 1 is the current week has a holiday.
  """
  
  def check_upcoming_holidays(curr_week):
    next_week = curr_week + 1
    next_to_next_week = curr_week + 2

    if curr_week % 100 == 51 or curr_week % 100 == 52:
      return 1
    if (next_week % 100 > 52):
      next_week = next_week - (next_week % 100) + 101
    if (next_to_next_week % 100 > 52):
      next_to_next_week = next_to_next_week - (next_to_next_week % 100) + 101
    if (week_hol_dict[next_week] == 1) or (week_hol_dict[next_to_next_week] == 1):
      return 1
    else:
      return 0

  # read in us holidays list, filter and keep only required data.
  usholidays = load_spark_df(get_filestore_path() + "usholidays.csv")
  usholidays = usholidays.select(*['Date', 'Holiday'])

  # read in a full year date-wise calendar, filter and keep only required data.
  calendar = load_spark_df(get_filestore_path() + "calendar.csv")
  calendar = calendar.select(*['Date', 'week_id'])
  calendar = calendar.filter( (col('week_id') >= START_WEEK) & (col('week_id') <= END_WEEK) ).orderBy("Date", ascending = True)

  cal = calendar.join(usholidays, on = 'Date', how = 'left')                          
  cal = cal.withColumn('Holiday_Flag', isNull('Holiday').cast('int')).select('Date', 'week_id', 'Holiday_Flag')      

  # Groupby, to create a dataframe to capture data if the week has a holiday. And add a new column for upcoming holidays
  week_hol_df = cal.groupBy('week_id').max('Holiday_Flag')
  week_hol_df = week_hol_df.withColumn('Upcoming_Holiday_Flag', 0)

  checkIfHoliday_udf = F.udf(check_upcoming_holidays)  
  week_hol_df = week_hol_df.withColumn('Upcoming_Holiday_Flag', checkIfUpcomingHoliday_udf(col('week_id')))
  
  df = df.join(week_hol_df,\
               df.col('TIME_VAR') == week_hol_df.col('week_id'),\
               how='left')
  df = df.drop(['week_id'])

  return df


def get_lagged_hierarchical_agg_features_pyspark(df, columns_to_aggregate, prod_lvl, bus_lvl, time_ind=None, agg_func="avg", lag_periods=[1], *args, **kwargs):
  """
  Calculate the aggregated value of specified columns based on levels within the product and business hierarchy. Must be lagged.
  """

  if not time_ind:
    time_ind = TIME_VAR

  groupby_cols = PRODUCT_HIER[:prod_lvl] + BUSINESS_HIER[:bus_lvl] + [time_ind]

  target_columns = ensure_is_list(columns_to_aggregate)
  
  new_columns_dict = {column:(agg_func + '_' + PRODUCT_HIER[prod_lvl-1] + "_" + BUSINESS_HIER[bus_lvl-1] + "_" + column) for column in target_columns}
  new_columns_list = list(new_columns_dict.values())
  agg_function_dict = {x:agg_func for x in target_columns}

  agg_df = df.select(groupby_cols + target_columns).groupBy(groupby_cols).agg(agg_function_dict).toDF(*(groupby_cols + new_columns_list))

  join_df = df.join(agg_df, on = groupby_cols, how = 'left')
  
  final_df = get_lagged_features_pyspark(join_df, new_columns_list, lag_periods, *args, **kwargs)
  final_df = final_df.drop(*new_columns_list)

  return final_df

# COMMAND ----------

# DBTITLE 1,PySpark Time Features
def get_year_feature_pyspark(df):
  """
  Calculate the year in PySpark given a TIME_VAR column in the format YYYYWW.
  
  Parameters
  ----------
  df : PySpark DataFrame
      PySpark DataFrame with column name equal to global variable TIME_VAR in format YYYYWW (e.g., 201801).
  
  Returns
  -------
  df_with_year : PySpark DataFrame
      PySpark DataFrame with additional column "Year" in the format YYYY (int).
  """
  return df.withColumn("Year", col(TIME_VAR).cast('string').substr(0,4).cast('int'))


def get_week_feature_pyspark(df):
  """
  Calculate the week in PySpark given a TIME_VAR column in the format of YYYYWW.
  
  Parameters
  ----------
  df : PySpark DataFrame
      PySpark DataFrame with column name equal to global variable TIME_VAR in format YYYYWW (e.g., 201801).
  
  Returns
  -------
  df_with_week : PySpark DataFrame
      PySpark DataFrame with additional column "Week" in the format WW (int).
  """
  return df.withColumn("Week", col(TIME_VAR).cast('string').substr(-2, 2).cast('int'))


def get_quarter_feature_pyspark(df):
  """
  Calculate the quarter in PySpark given a "Week" column in the format of WW (int).
  
  Parameters
  ----------
  df : PySpark DataFrame
      PySpark DataFrame with column "Week" in format of WW (e.g., 51).
  
  Returns
  -------
  df_with_quarter : PySpark DataFrame
      PySpark DataFrame with additional column "Quarter" (int).
  """
  from itertools import chain

  assert ("Week" in df.columns), "Please calculate Week variable before trying to build this feature"
  quarter_dict = dict([(n, 1) for n in range(1, 14)] +
                      [(n, 2) for n in range(14, 27)] +
                      [(n, 3) for n in range(27, 40)] +
                      [(n, 4) for n in range(40, 53)])
  
  quarter_map = create_map([lit(x) for x in chain(*quarter_dict.items())])

  return df.withColumn("Quarter", quarter_map.getItem(col("Week").cast('int')))
 
  
def get_quarter_year_feature_pyspark(df):
  """
  Calculate the QuarterYear (in the format of YYYYQQ) given "Quarter" and "Year" columns.
  
  Parameters
  ----------
  df : PySpark DataFrame
      PySpark DataFrame with columns "Quarter" and "Year" (int).
  
  Returns
  -------
  df_with_quarter_year : PySpark DataFrame
      PySpark DataFrame with additional column "QuarterYear" (string).
  """
  assert ("Quarter" in df.columns) & ("Year" in df.columns), "Please calculate both Quarter and Year before trying to build this feature"
                   
  return df.withColumn("QuarterYear", concat(col('Year').cast('String'), col('Quarter').cast('string')))

# (DJMP) Could we nest the functions inside of each other instead of creating df's with successively longer names?
# (NL) we could, but writing it this way makes it easier to debug if one of the functions breaks. might be worth refactoring later.
def get_time_features_pyspark(init_df):
  """
  Calculate the week, year, quarter, and quarterYear given a TIME_VAR column in the format of YYYYWW.
  
  Parameters
  ----------
  init_df : PySpark or pandas/koalas DataFrame
      DataFrame with column name equal to global variable TIME_VAR in format YYYYWW (e.g., 201801).
  
  Returns
  -------
  df_with_qtryear_qtr_week_year : PySpark or pandas/koalas DataFrame, same type as init_df
      DataFrame with additional columns "QuarterYear", "Quarter", "Week", and "Year".
  """
  
  if is_pandas_df(init_df):
    df = spark.createDataFrame(init_df)
  else:
    df = init_df
  
  df_with_year = get_year_feature_pyspark(df)
  df_with_week_year = get_week_feature_pyspark(df_with_year)
  df_with_qtr_week_year = get_quarter_feature_pyspark(df_with_week_year)
  df_with_qtryear_qtr_week_year = get_quarter_year_feature_pyspark(df_with_qtr_week_year)
  
  if is_pandas_df(init_df):
    return df_with_qtryear_qtr_week_year.toPandas()
  else:
    return df_with_qtryear_qtr_week_year
  
  
def get_week_in_month_feature_pyspark(input_df, time_var_override = None):
  """
  Calculate the week number within a given month.
  
  Parameters
  ----------
  input_df : PySpark DataFrame
      DataFrame with column TIME_VAR.
      
  time_var_override : str (default None)
      Variable to use as TIME_VAR (default, None, uses global variable TIME_VAR).
  
  Returns
  -------
  final_df : PySpark DataFrame
      DataFrame with column "WeekInMonth" (integer between 1 and 4).
  """
  
  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override
  
  input_with_datetime_df = assign_datetime_var_pyspark(input_df, time_var_override=time_var)
  
  final_df = input_with_datetime_df.withColumn("WeekInMonth", (F.floor(F.dayofmonth(col("datetime")) / 7) + 1).cast('int'))
  final_df = final_df.drop("datetime")

  return final_df


def get_month_index_feature_pyspark(input_df, time_var_override = None):
  """
  Calculate the month number.
  
  Parameters
  ----------
  input_df : PySpark DataFrame
      DataFrame with column TIME_VAR.
      
  time_var_override : str (default None)
      Variable to use as TIME_VAR for the purpose of calculating month index.

  Returns
  -------
  final_pd : PySpark DataFrame
      DataFrame with column "Month" (int between 1 and 12).
  """
  
  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override

  input_with_datetime_df = assign_datetime_var_pyspark(input_df, time_var_override = time_var)
  
  final_df = input_with_datetime_df.withColumn("Month", F.month(col("datetime")))
  final_df = final_df.drop("datetime")

  return final_df
    
  
def get_month_year_pyspark(df):
  """
  Calculate the MonthYear (in the format of YYYYMM) given "Month" and "Year" columns.
  """
  
  assert ("Month" in df.columns) & ("Year" in df.columns), "Please calculate both Quarter and Year before trying to build this feature"

  #TODO is there a built-in to do this?
  pad_month_string_function = F.udf(lambda x: str(x).rjust(2, ' '))

  return df.withColumn("MonthYear", concat(col("Year").cast('string'),
                                           pad_month_string_function(col("Month"))).cast('int'))


def get_linear_trend_features_pyspark(input_df, time_var_override = None):
  """
  Calculate the linear value of the time variable and include the sin and cos to account for cyclicality.
  
  Parameters
  ----------
  input_pd : PySpark DataFrame
      DataFrame with column TIME_VAR.
      
  time_var_override : str (default None)
      Variable to use as TIME_VAR (default, None, uses global variable TIME_VAR).

  Returns
  -------
  final_pd : pandas or koalas DataFrame
      DataFrame with columns "SinLinearTime" and "CosLinearTime".
  """

  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override
  
  input_with_datetime_df = assign_datetime_var_pyspark(input_df, time_var_override = time_var)

  date_to_str_udf = F.udf(lambda x: x.strftime("%Y-%m-%d"))
  
  final_df = input_with_datetime_df.withColumn("LinearTime", F.unix_timestamp(date_to_str_udf(col("datetime")), 'yyyy-MM-dd'))
    
  final_df = final_df.withColumn("SinLinearTime", F.sin(col("LinearTime")))\
                     .withColumn("CosLinearTime", F.cos(col("LinearTime")))\
                     .drop("datetime","LinearTime")

  return final_df



def assign_datetime_var_pyspark(df, date_format = '%Y%U-%w', time_var_override = None):
  """
  Add a new "datetime" column containing a date_time copy of TIME_VAR (assuming that TIME_VAR follows YYYYWW by default).

  Parameters
  ----------
  df : PySpark DataFrame
      DataFrame with column TIME_VAR.
      
  date_format : str (default = "%Y%U-%w")
      pandas datetime format for new column "datetime".
      
  time_var_override : str (default None)
      Variable to use as TIME_VAR (default, None, uses global variable TIME_VAR).

  Returns
  -------
  df : PySpark DataFrame
      DataFrame with column "datetime" in format date_format.
  """
  
  if not time_var_override:
    time_var = TIME_VAR
  else:
    time_var = time_var_override
  
  format_conversion_lambda_function = F.udf(lambda x: datetime.strptime(x, date_format), DateType())
  
  df = df.withColumn('datetime', format_conversion_lambda_function(concat(col(time_var).cast('string'), F.lit('-1'))))
  
  return df

# COMMAND ----------

# DBTITLE 1,External Data Mining
def pull_weekly_google_trends(search_keyword_list, start_date_dt, end_date_dt=None, col_name='date', resampling='W',\
                            geo_target='US', geo_resolution='STATE', category=0, gprop=None, convert_date_to_dt=True):
  '''
  Google Trends is a website by Google that analyzes the popularity of top search queries in Google Search across various regions and languages
  Idea is to create this dataframe at level of interest (eg, US state) and time of interest (eg, weekly for period of time)
  Then combine with an existing dataframe modelign dataframe as feature or set of features
  
  PYTRENDS API INFORMATION:
  kw_list: string keywords, ie, keywords for which to get data; each kw will be pulled into it's own column
  cat: defines category to narrow search results; cat=0 allows for results from all categories; for full overview/list: https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
  geo: two-letter country abbreviation; original default is 'WORLD'
  timeframe: sets the time for which search term is reviewed
  gprop: Google property to use; default is websearches; can be 'images','news','youtube'
  interest_by_region: levels include - COUNTRY, STATE, CITY, DMA, REGION
  
  Full Documentation for PyTrends: https://pypi.org/project/pytrends/
  '''
  
  import time
  import datetime
  from pytrends.request import TrendReq
  
  
  def create_date_pd(start_date_dt, end_date_dt=None, col_name='date', resampling='W'):
    '''
    Creates a shell dataframe at resampled period of user's choice; start and end dates dictated by the user
    Designed to be used in conjunction with the GoogleTrends function - pull_google_trends()
    '''

    if not end_date_dt:
      dates_pd = pd.DataFrame(pd.date_range(start=start_date_dt, end=datetime.date.today()), columns=[col_name])

    else: 
      dates_pd = pd.DataFrame(pd.date_range(start=start_date_dt, end=end_date_dt), columns=[col_name])

    indexed_pd = dates_pd.set_index(col_name)
    resampled_pd = indexed_pd.resample(resampling)\
                             .sum()\
                             .reset_index()
    return resampled_pd 

   
  pytrend = TrendReq()
  search_pd_list = []
  resampled_date_pd = create_date_pd(start_date_dt, end_date_dt=end_date_dt, col_name=col_name, resampling=resampling)
  
  weeks = list(resampled_date_pd[col_name].unique())
  search_keyword_list = ensure_is_list(search_keyword_list)
  
  for week in weeks:
    time.sleep(1)
    
    weekstart = str(pd.to_datetime(np.datetime64(week - np.timedelta64(6,'D'))).date())
    weekend = str(pd.to_datetime(np.datetime64(week)).date())
    
    timethread = weekstart + ' ' + weekend
    pytrend.build_payload(kw_list=search_keyword_list, geo=geo_target, timeframe=timethread, cat=category)

    search_pd = pytrend.interest_by_region(resolution=geo_resolution, inc_low_vol=True, inc_geo_code=False)
    search_pd[resampling] = week
    search_pd_list.append(search_pd)
    
  
  merged_search_pd = pd.concat(search_pd_list).reset_index()
  
  if convert_date_to_dt:
    merged_search_pd[resampling] = pd.to_datetime(merged_search_pd[resampling]).apply(lambda x: x.date())
  
  return merged_search_pd

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Workbench - M5 Forecasting Features
def calc_descriptive_statistic_features(df, grouping_cols, calc_column, operation_dict=None, text_flag=None):
    """
    Create new summary statistic features based on existing column
    #TODO accept list
    """
    if not operation_dict:
        operation_dict = {
            calc_column + "_min" + text_flag : "min",
            calc_column + "_max" + text_flag : "max",
            calc_column + "_mean" + text_flag : "mean",
            calc_column + "_std" + text_flag : "std",
            calc_column + "_med" + text_flag : "median",
          
        }
        
    # transform doesn't accept a list of functions, so build features iterably    
    for key, value, in operation_dict.items():
        df[key] = df.groupby(grouping_cols)[calc_column].transform(value)
        
    return df


def calc_unique_count(df, grouping_cols, count_col, final_column_name=None):
    """
    Calculates the number of different counting_col for every unique set of grouping_cols
    """
    if not final_column_name:
        final_column_name = count_col + "_count"
    
    df[final_column_name] = df.groupby(grouping_cols)[count_col].transform('nunique')
    
    return df


def calc_ratio_vs_average(df, grouping_cols, ratio_col, final_column_name=None):
    """
    Calculates the ratio of a column divided by its historical average across some grouping (which should include a time feature)
    """
    if not final_column_name:
        final_column_name = ratio_col + "_hist_momentum"
        
    df[final_column_name] = df[ratio_col]/df.groupby(grouping_cols)[ratio_col].transform('mean')
    
    return df


def calc_ratio_vs_prior_period(df, grouping_cols, ratio_col, final_column_name=None):
    """
    Calculates the ratio of a column in a given time period divided by that same column in the previous time period
    NOTE: Assumes you don't need to resample
    """
    if not final_column_name:
        final_column_name = ratio_col + "_momentum_vs_prior"
        
    df[final_column_name] = df[ratio_col]/df.groupby(grouping_cols)[ratio_col].transform(lambda x: x.shift(1))
    
    return df


def standardize_features(df, column_list):
    """
    Standardize features using z-score standardization
    """
    for column in column_list:
        df[column + "_standardized"] = (df[column] - df[column].mean()) / df[column].std()
    
    return df


def normalize_features(df, column_list):
    """
    Normalize features using mean-normalization
    """
    for column in column_list:
        df[column + "_normalized"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df


def calc_date_features(df, date_col='date'):
    df = df.assign(
            day = df[date_col].dt.day.astype(np.int8),
            week = df[date_col].dt.week.astype(np.int8),
            month = df[date_col].dt.month.astype(np.int8),
            year = df[date_col].dt.year.astype(np.int16),
            day_of_week = df[date_col].dt.dayofweek.astype(np.int8))
    
    df = df.assign(weekend_flag = (df['day_of_week']>=5).astype(np.int8))
    
    return df
  

def calc_lagged_features(pandas_df, vars_to_lag, time_var, grouping_cols, lag_periods):
    """
    Create lagged features on particular variables. Assumes your data doesn't need to be resampled
    """

    vars_to_lag = ensure_is_list(vars_to_lag)
    grouping_cols = ensure_is_list(grouping_cols)


    var_list = []
    for shift in lag_periods:
        for var in vars_to_lag:
          var_name = var + "_lag_" + str(shift) 
          var_list += [var_name]

          pandas_df[var_name] = pandas_df.groupby(grouping_cols)[var]\
                                         .shift(shift)
        
    return pandas_df

# COMMAND ----------



# COMMAND ----------

