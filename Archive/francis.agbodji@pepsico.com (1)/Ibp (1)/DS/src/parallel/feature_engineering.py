# Databricks notebook source
def get_dummies(df, columns, delim=None):
  """
  Creates dummy variables for unique values of list of columns

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  columns : List
      List of columns which you want to create dummy variables
  delim : Str
      String containing delimeter that should separate input column values - indicators will be created for all values separated by delim

  Returns
  -------
  df : PySpark dataframe
      Input dataset with appended dummy fields
  """
  #TO-DO: Get rid of loop
  if isinstance(columns, str):
      columns = [columns]

  for c in columns:
    if c in df.columns:
      unique_values = df.select(c).dropDuplicates()
      unique_values = convertDFColumnsToList(unique_values, c)
      unique_values = ['NULL' if x==None else x for x in unique_values]
      unique_values = [str(x) if isinstance(x, str)==False  else x for x in unique_values] #Handle numeric unique types
      if delim is None:
        indicator_dict = {i: [(c+'_'+i).replace(" ","_")] for i in unique_values} #Create dictionary from values
      else:
        indicator_dict = {i: [(c+'_'+j).replace(" ","_") for j in str.split(i,delim)] for i in unique_values} #Sep delim dictionary

      #Initialize to 0
      for ind in list(itertools.chain(*indicator_dict.values())):
        df = df.withColumn(ind, lit(0))

      #Create indicators
      for this_key in indicator_dict.keys():
          for this_indicator in indicator_dict[this_key]:
            if this_key == 'NULL':
                df = df.withColumn(this_indicator, when(col(c).isNull(), lit(1)).otherwise(col(this_indicator)))
            else:
              df = df.withColumn(this_indicator, when(col(c)==this_key, lit(1)).otherwise(col(this_indicator)))

  return(df)



def get_time_vars(df, date_field, funcs = ["YearIndex","MonthIndex","WeekIndex","WeekInMonth"]):
  """Create time fields using a date field"""

  if len(intersect_two_lists(funcs, ["YearIndex"]))>0:
    df = df.withColumn("YearIndex", year(date_field))
    
  if len(intersect_two_lists(funcs, ["MonthIndex"]))>0:
    df = df.withColumn("MonthIndex", month(date_field))
    
  if len(intersect_two_lists(funcs, ["WeekIndex"]))>0:
    df = df.withColumn("WeekIndex", lpad(weekofyear(date_field), 2, '0').astype("int"))
    
  if len(intersect_two_lists(funcs, ["WeekInMonth"]))>0:
    df = df.withColumn("WeekInMonth", ceil(dayofmonth(date_field)/7))
    
  if len(intersect_two_lists(funcs, ["WeekOfYear"]))>0:
    df = df.withColumn("WeekOfYear", concat("YearIndex","WeekIndex").astype("int"))
    
  if len(intersect_two_lists(funcs, ["Quarter"]))>0:
    df = df.withColumn("Quarter", quarter(col(date_field)).astype("int"))
    
  if len(intersect_two_lists(funcs, ["QuarterYear"]))>0:
    df = df.withColumn("QuarterYear", concat(year(date_field),lpad(quarter(col(date_field)), 2, 'Q')))
    
  if len(intersect_two_lists(funcs, ["LinearTrend"]))>0:
    df = df.withColumn("LinearTrend", datediff(df[date_field],lit('1970-01-01')))
    
  if len(intersect_two_lists(funcs, ["CosX"]))>0:
    df = df.withColumn("CosX", cos(col("LinearTrend")))
    
  if len(intersect_two_lists(funcs, ["SinX"]))>0:
    df = df.withColumn("SinX", sin(col("LinearTrend")))
    
  return(df)



def fill_missing_timeseries_pyspark(df, hierarchy, time_variable='Week_Of_Year'):
  """
  Fills the missing rows in a timeseries - ie, resamples for missing time periods
  """
  distinct_hier = df.select(hierarchy).distinct()
  distinct_time = df.select(time_variable).distinct()
  continous_df = distinct_hier.join(distinct_time, how='cross')
  return continous_df


#TO-DO: The refactor to take in lists assumes -number is a lead and +number is a lag (not intuitive)
def do_lags_N(df, order, lagvars, n, partition = None, drop_lagvars=True):
    """ Creates lag and lead columns for as many values as specified by partitioning over a window, and only lags if a list is specified"""
    #If string inputs were given, convert to lists
    if isinstance(lagvars, str):
      lagvars = [lagvars]

    if isinstance(partition, str):
      partition = [partition]

    if partition:
        lag_window = Window.partitionBy(partition).orderBy(order)
    else:
        lag_window = Window.partitionBy().orderBy(order)
        
    def n_iter(df_, col_name):
      
      if isinstance(n, list):
        df_ = reduce(lambda df__, i: df__.withColumn(str(col_name + "_lag" +str(i)), lag(col_name, i).over(lag_window)), n, df_)
      else:
        df_ = reduce(lambda df__, i: df__.withColumn(str(col_name + "_lag" +str(i)),\
                                                   lag(col_name, i).over(lag_window)).withColumn(str(col_name + "_lead" +str(i)),\
                                                                                                 lead(col_name, i).over(lag_window)), range(1,n+1), df_)
      return df_
        
    df = reduce(lambda memo_df, col_name: n_iter(memo_df, col_name), lagvars, df)
    if drop_lagvars:
      df = df.drop(*lagvars)
        
    return df

#Update to use map reduce instead of loop through stats
def get_hierarchical_statistics(df, group_col, input_var, stats):
  """
  Calculates a percentile by group

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_col : List
      List of fields over which to calculate the percentile
  perc_var: String
      Column on which you wish to perform the calculation
  percentiles: List
      List of percentiles you wish to calculate

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing percentile variable
  """
  grp_window = Window.partitionBy(group_col)
  df = (reduce(
    lambda df, this_stat:
      df.withColumn(this_stat.__name__ + "_" + input_var, this_stat(input_var).over(grp_window)),
    stats,
    df
  ))
  return df



def calculate_ratio(df,ratio_dict):
  """
  Calculates ratio of two variables using ratio_dict

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  ratio_dict: Nested Dictionary
       {'Unit_Price':{'top_var':'Dollar_Sales','bottom_var':'Unit_Sales'},
        'Pct_Baseline':{'top_var':'Baseline_Sales','bottom_var':'Total_Sales'}
       }

  Returns
  -------
  df : PySpark dataframe
      Input dataset with new ratio variables appended
  """
  #TO-DO: get rid of loop
  for this_var in ratio_dict.keys():
    this_dict = ratio_dict[this_var]
    if set(list(this_dict.values())).issubset(set(df.columns)):
      df = df.withColumn(this_var, col(this_dict.get("top_var")) /col(this_dict.get("bottom_var")))
  return(df)

#TO-DO: Finish dev
#TO-DO: Create unit test
def calculate_adstock(df,adstock_var,adstock_factor, date_field, partition = None):
  """
  Calculates decayed ad-stock/carry over effect for a particular variable

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  adstock_var : String
      Variable on which to perform adstock (e.g., grp)
  adstock_factor: DoubleType
      Decay rate
  date_field : String
      Name of date field in input dataset

  Returns
  -------
  df : PySpark dataframe
      Input dataset with adstocked variable appended
  """

  hierarchy = [date_field] + partition
  w = Window.partitionBy(hierarchy)
  #At = Xt + adstock rate * At-1.

  df = do_lags_N(df, ["RDATE"], adstock_var, -1, ["MODEL_ID"]) #Lag
  new_adstock_name = adstock_var + "adstock"
  adstock_lag = adstock_var + "lag1"
  df = df.withColumn(new_adstock_name, adstock_var + (adstock_factor*adstock_lag).over(w)) #Calculate carry over effect

  return(df)


def calc_ratio_vs_average_pyspark(df, group_cols, ratio_cols):
  """
  Calculates the ratio of a column divided by its historical average across some grouping (which should include a time feature)

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      Hierarchy over which to calculate the average
  ratio_cols: List
      List of fields over which to perform the ratio calculation

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing the ratio variables
  """
  
  if isinstance(group_cols, str):
    group_cols = [group_cols]
  if isinstance(ratio_cols, str):
    ratio_cols = [ratio_cols]
    
  #Get hierarchical average
  df = (reduce(
    lambda df, this_col:
      get_hierarchical_statistics(df, group_cols, this_col, [avg]),
    ratio_cols,
    df
  ))
  
  #Create ratio and drop average
  df = (reduce(
    lambda df, this_col:
      df.withColumn(this_col + "_hist_momentum", col(this_col)/col("avg_" + this_col)),
    ratio_cols,
    df
  ))
  df = (reduce(
    lambda df, this_col:
      df.drop("avg_" + this_col),
    ratio_cols,
    df
  ))
  return df

def calc_ratio_vs_prior_period_pyspark(df, group_cols, ratio_cols, sort_cols):
  """
  Calculates the ratio of a column in a given time period divided by that same column in the previous time period

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      Hierarchy over which to calculate the average
  ratio_cols: List
      List of fields over which to perform the ratio calculation

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing the ratio variables
  """
  
  if isinstance(group_cols, str):
    group_cols = [group_cols]
  if isinstance(ratio_cols, str):
    ratio_cols = [ratio_cols]
  if isinstance(sort_cols, str):
    sort_cols = [sort_cols]

  #Get prior period
  windowSpec = Window.partitionBy(*group_cols).orderBy(*sort_cols)
  df = (reduce(
    lambda df, this_col:
      df.withColumn(this_col + "_shift_lag", lag(col(this_col), 1).over(windowSpec)),
    ratio_cols,
    df
  ))
  
  #Create ratio and drop prior period
  df = (reduce(
    lambda df, this_col:
      df.withColumn(this_col + "_momentum_vs_prior", col(this_col)/col(this_col + "_shift_lag")),
    ratio_cols,
    df
  ))
  df = (reduce(
    lambda df, this_col:
      df.drop(this_col + "_shift_lag"),
    ratio_cols,
    df
  ))
  return df

def get_velocity_flag_pyspark(df, group_cols, sales_var, time_var, velocity_type="high", target_threshold=1, time_threshold=8):
  """
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or more] sales in the historical data, 
  then HighVelocityFlag = 1.
  If a product has ever had [time_threshold] consecutive weeks of [target_threshold or less] sales in the historical data, 
  then LowVelocityFlag = 1.

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      Hierarchy over which to calculate the velocity (e.g., Customer-SKU)
  sales_var: String
      Variable over which we want to calculate velocity (often the target variable)
  time_var: String
      The date field in the dataset
  velocity_type: String
      "High" or "Low" to indicate whether we wish to calculate a high velocity or low velocity feature
  target_threshold: Integer
      The threshold which sales needs to meet 
  time_threshold: Integer
      The number of weeks where sales needs to meet threshold to be classifed as high or low

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing the velocity variables
  """
  
  if isinstance(group_cols, str):
    group_cols = [group_cols]

  #Determine if sales week is within threshold
  if velocity_type == "high":
    df = df.withColumn("within_bounds", when(col(sales_var) >= target_threshold,1).otherwise(0))
  else:
    df = df.withColumn("within_bounds", when(col(sales_var) < target_threshold,1).otherwise(0))
    
  #Cumulative Sum
  df = df.withColumn("grp", sum((col("within_bounds") == 0).cast("int")).over(Window.partitionBy(group_cols).orderBy(time_var)))
  df = df.withColumn("cum_sum",sum(col("within_bounds")).over(Window.partitionBy(group_cols + ["grp"]).orderBy(time_var)))
  
  #Determine if time threshold was met
  df = get_hierarchical_statistics(df, group_cols, "cum_sum", [max])
  if velocity_type == "high":
    df = df.withColumn("High_Velocity_Flag",when(col("max_cum_sum")>=time_threshold,1).otherwise(0))
  else:
    df = df.withColumn("Low_Velocity_Flag",when(col("max_cum_sum")>=time_threshold,1).otherwise(0))
    
  df = df.drop("within_bounds").drop("grp").drop("cum_sum").drop("max_cum_sum")
  
  return df


def calc_unique_count_pyspark(df, group_cols, count_cols):
  """
  Calculates the unique number of "count_cols" within the unique set of grouping_cols

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      Hierarchy over which to calculate the uniqueness 
  count_cols: List
      Variable over which to calculate the uniqueness
  new_names: List
      List of new variable names

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing the velocity variables
  """
  
  if isinstance(group_cols, str):
    group_cols = [group_cols]
  if isinstance(count_cols, str):
    count_cols = [count_cols]

  df = (reduce(
    lambda df, this_col:
      get_hierarchical_statistics(df, group_cols, this_col, [approx_count_distinct]),
    count_cols,
    df
  )) 
  
  for this_count_col in count_cols:
    df = df.withColumnRenamed("approx_count_distinct_" + this_count_col,"distinct_" + '_'.join(group_cols) + "_"+this_count_col)
  
  return df


def get_performance_flag(df, group_col, hierarchy_col, input_var, stats, top_bottom_flag):
  """
  Gets top / bottom performers of a certain hierarchy (customer, product, location) on a certain variable (e.g., total sales, stddev sales)

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_col : List
      List of fields over which to calculate the percentile (e.g., Customer)
  hierarchy_col: List
      List of fields over which to calculate the hierarchical statistics (e.g., Category, Brand)
  input_var: String
      Variable over which you want to perform calculations (typically demand column)
  stats: List
      Statistics you wish to calculate (e.g., [sum, stddev, max])
  top_bottom_flag: String
      Whether you are identifying top or bottom (can take values "top" or "bottom").  If top, the flag will identify hierarchies that 
      are greater than the 80th percentile.  If low, the flag will identify hierarchies that are less than 20th percentile.

  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended columns containing the top / bottom flags
  """
  
  if isinstance(hierarchy_col, str):
    hierarchy_col = [hierarchy_col]
  if isinstance(group_col, str):
    group_col = [group_col]
  
  #Determine if we are finding "top" or "bottom" performers
  if top_bottom_flag == "top":
    percentile = [0.8] #Top 20% percentile
    var_name_tag = "top"
  else:
    percentile = [0.2] #Bottom 20% percentile
    var_name_tag = "bottom"
    
  #Get hierarchical stats (e.g., total demand by unit)
  df = get_hierarchical_statistics(df,hierarchy_col,input_var,stats)
  
  #Create flag if it's above or below desired percentile
  for i in stats:
    aggregated_var_name =  i.__name__ + "_" + input_var #e.g., sum_sales
    percentile_var_name = aggregated_var_name + str((percentile[0]*10)).replace('.', '') #e.g., sum_sales_80
    new_var_name = var_name_tag + "_performer_"+ i.__name__ + "_" + '_'.join(hierarchy_col) #e.g., top_performer_sum_CAT_BRAND
    
    df = calculate_percentile(df, group_col, aggregated_var_name, percentile)
    if top_bottom_flag == "top":
      df = df.withColumn(new_var_name,when(col(aggregated_var_name) >= col(percentile_var_name),1).otherwise(0))
    else:
      df = df.withColumn(new_var_name,when(col(aggregated_var_name) < col(percentile_var_name),1).otherwise(0))
    df = df.drop(aggregated_var_name)
    df = df.drop(percentile_var_name)
  
  return df


def standardize_columns_pyspark(df, group_col, COLS_TO_STANDARDIZE):
  """
  Standardize columns of a DataFrame distributively using PySpark and store results in global variable standardization_dict to undo transformation.
  
  Parameters
  ----------
  input_df : PySpark DataFrame
      DataFrame with columns to standardize.
  
  COLS_TO_STANDARDIZE : list or str
      Column(s) of input_df to standardize.
  do
  Returns
  -------
  final_df : PySpark DataFrame
      DataFrame with all columns of input_df, except COLS_TO_STANDARDIZE, and standardized versions of COLS_TO_STANDARDIZE.

  TODO: robust scaler that excludes 0's and outliers?
  """
  
  if isinstance(COLS_TO_STANDARDIZE, str):
    COLS_TO_STANDARDIZE = [COLS_TO_STANDARDIZE]

  def z_score(c):
      return (col(c) - mean(c).over(Window.partitionBy(group_col))) / stddev(c).over(Window.partitionBy(group_col))

  df = (reduce(
      lambda df_, this_feature:
        df_.withColumn(this_feature,  z_score(this_feature)),
      COLS_TO_STANDARDIZE,
      df
    ))
  
  return df


def get_product_age_feature_pyspark(df):
  """
  Adds a flag when the column at the lowest level of the global product hierarchy had its first sale
  """
  
  product_cols = PRODUCT_HIER[PRODUCT_LEVEL - 1]
  
  col_names_time_stats = [product_cols, TIME_VAR]
  
  time_stats = df.filter(col(TARGET_VAR) > 0)\
                 .groupBy(product_cols)\
                 .min(TIME_VAR)\
                 .toDF(*col_names_time_stats)
  
  time_stats = time_stats.withColumn("Product_Age", col(TIME_VAR)).drop(TIME_VAR)
  
  final_df = df.join(time_stats, on = [product_cols], how = 'left')
  
  subtract_from_end_week_lambda_function = F.udf(lambda x: END_WEEK - x)
 
  final_df = final_df.fillna({"Product_Age":0}).withColumn("Product_Age", subtract_from_end_week_lambda_function(col("Product_Age")))
#                      .fillna({"Product_Age":0})
  
  final_df = final_df.withColumn("Product_Age", col("Product_Age").cast('int'))
  
  return final_df


def get_new_product_flag_feature_pyspark(df, time_threshold = 26):
  """
  Adds a flag when the column at the lowest level of the global product hierarchy didn't have sales more than time_threshold periods ago 

  Parameters
  ----------
  df : PySpark DataFrame
  time_threshold : Prior number of weeks from end_date after which sales are to be considered.

  Global Parameters
  ----------
  product_cols : The groupby columns to indicate a product. Ex. MaterialId, CustomerId, ..
  TIME_VAR : The column indicating a time series in the dataframe. Ex. WeekId, MonthId, DateId, ..
  TARGET_VAR : To filter on this column such that only valid data points are considered. Ex. Sales, Orders, ..
  END_WEEK : The latest week until which data is present.

  Returns
  -------
  df : PySpark DataFrame with an additional column "New_Product_Flag" 0's for old, and 1's for new product.
  """

  product_cols = PRODUCT_HIER[PRODUCT_LEVEL - 1]
  
  filtered_df = df.filter(col(TARGET_VAR) > 0)
  
  col_names_time_stats = [product_cols, TIME_VAR]

  time_stats = filtered_df.groupBy(product_cols).min(TIME_VAR).toDF(*col_names_time_stats)
  
  time_cutoff = update_time(END_WEEK, -1 * time_threshold)
  
  time_above_cutoff_lambda_function = F.udf(lambda x: 1 if x >= int(time_cutoff) else 0)
  
  time_stats = time_stats.withColumn("New_Product_Flag",
                                     time_above_cutoff_lambda_function(col(TIME_VAR)))
  time_stats = time_stats.drop(TIME_VAR)

  
  final_df = df.join(time_stats, on = product_cols, how = 'left')
  final_df = final_df.fillna({"New_Product_Flag":0})
  final_df = final_df.withColumn("New_Product_Flag", col("New_Product_Flag").cast('int'))
  
  return final_df
  