# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

def get_model_id(df,col_name,hier):
    """
    Returns concatenation of model ID hierarchy as a new variable
    """
    df = df.withColumn(col_name, F.concat(*hier))
    return df

def get_train_ind(df,train_id_field,date_field,train_start_date,train_end_date):
  "Develops standardized TRAIN_IND field which indicates data is in training period"
  df = df.withColumn(train_id_field, when(col(date_field).between(train_start_date, train_end_date), 1).otherwise(0))
  return(df)

def get_date(df, date_var, date_format, field_name):
  """
  Converts date variable into standardized RDATE field (output format: YYYY-MM-DD).  Can take in a string, date or timestamp.

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  date_var : String
      Column name of the unstandardized date field
  date_format: String
      Date format of the unstandardized date field.  Can accomodate typical date/time formats:
      https://spark.apache.org/docs/latest/api/python/pyspark.sql.html
      "yyyy-MM-dd"
  field_name: String
      Name of desired column name for standardized date (e.g., RDATE)

  Returns
  -------
  df : PySpark dataframe
      Input dataset with new date field appended called RDATE in the following format: YYYY-MM-DD
  """
  #Convert to RDATE based on format
  df = df.withColumn(field_name, to_date(col(date_var),date_format))
  return df

def align_week_start_date(df,datecol,day):
  """Aligns a date column to a particular day of week (e.g., Sun, Mon)"""
  return df.withColumn(datecol, date_add(next_day(col(datecol),day),-7))

def find_dupicates(df, id_vars):
  """
  Finds duplicate records of the id variables and outputs a dataframe with the records

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  id_vars : List
      The combination of fields that should be unique in the dataset

  Returns
  -------
  df : PySpark dataframe
      Duplicate records
  """

  df = df.exceptAll(df.drop_duplicates([id_vars]))
  return(df)

def filter_values(df, filter_dict):
  """
  Filters rows based on criterion/columns specified in filter dictionary

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  filter_dict : Dictionary
      {field on which to filter : filter expression}

  Returns
  -------
  df : PySpark dataframe
      Filtered output dataset
  """

  for this_key in filter_dict.keys():
    if this_key in df.columns:
      df = df.filter(filter_dict[this_key])
  return(df)

def bin_categorical_vars(df, variable, bin_dict):
  """
  Re-bins a variable based on a binning dictionary
  #TO-DO: If unique value is missing from bin_dict it replaces it with Null, change so that it retains original value

  Parameters
  ----------
  df : PySpark dataframe
      Input data
  variable : String
      Variable that you want to bin
  bin_dict : Dictionary
      Dictionary containing old value (key) and new value (item)

  Returns
  -------
  df : PySpark dataframe
      Re-binned dataset
  """
  if variable in df.columns:
    mapping_expr = create_map([lit(x) for x in chain(*bin_dict.items())])
    df = df.withColumn(variable, mapping_expr[df[variable]])
  return df

def log_columns_ps(input_df, columns_to_log):
  """
  Standardize columns of a DataFrame distributively in PySpark using log-plus-one method.

  Parameters
  ----------
  input_df : PySpark DataFrame
      DataFrame with columns to standardize with log-plus-one method.

  columns_to_log : list or str
      Column(s) of input_df to standardize with log-plus-one method.

  Returns
  -------
  final_df : PySpark DataFrame
      DataFrame with all columns of input_df, except columns_to_log, and standardized versions of columns_to_log.
  """

  if isinstance(columns_to_log, str):
    columns_to_log = [columns_to_log]

  dbl_cols = [f.name for f in input_df.schema.fields if isinstance(f.dataType, DoubleType)]
  columns_to_log = list(set(dbl_cols) & set(columns_to_log))

  filtered_df = input_df.select(*columns_to_log)

  filtered_df = filtered_df.select(*[log1p(col).alias(col) for col in filtered_df.columns])

  rest_of_input_df = input_df.drop(*columns_to_log)

  final_df = columnwise_union(filtered_df, rest_of_input_df)

  return final_df

def columnwise_union(df1, df2):
  """
  Merge two PySpark DataFrames together columnwise.
  #TO-DO: If data is shuffled on cores, sometimes mono id doesn't link up properly, need to add unit test

  Parameters
  ----------
  df1 : PySpark DataFrame

  df2 : PySpark DataFrame

  Returns
  -------
  final_df : PySpark DataFrame
      DataFrame with the columns of df1 and df2.
  """
  df1 = df1.withColumn('row_index', monotonically_increasing_id())
  df2 = df2.withColumn('row_index', monotonically_increasing_id())

  final_df = df2.join(df1, on=["row_index"]).sort("row_index") \
                .drop("row_index")

  return final_df


def aggregate_data(df, group_cols, cols_to_agg, stats):
  """
  This function performs outputs aggregation summaries for listed columns to aggregate. E.g., sum sales and count distinct weeks.
  Example Call:
    aggregate_data(mrd,["MODEL_ID"],[sales, date],[sum,countDistinct])

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  group_cols : List
      List of group by columns
  cols_to_agg : List
      Variables over which to calculate statistics
  stats : List of aggregate functions - nonstring
      List of aggregate statistics to calculate (e.g., [avg,sum,min,max])
      Refer for list of aggregate functions: https://jaceklaskowski.gitbooks.io/mastering-spark-sql/content/spark-sql-functions.html
      Some options include: avg, sum, min, max, kurtosis, skewness, stddev, countDistinct, etc.

  Returns
  -------
  agg_dat : PySpark dataframe
      Aggregated dataset
  """

  if isinstance(group_cols, str):
      group_cols = [group_cols]
  if isinstance(cols_to_agg, str):
      cols_to_agg = [cols_to_agg]

  #Aggregate
  exprs = [f(F.col(c)) for (f, c) in zip(stats, cols_to_agg)]
  agg_dat = df.groupby(*group_cols).agg(*exprs)

  #Rename
  prefix = [s.__name__ for s in stats]
  new_names = group_cols + [a + "_" + b for (a, b) in zip(prefix, cols_to_agg)]
  agg_dat = agg_dat.toDF(*new_names)

  return agg_dat


def get_cumsum(df, id_hier , cum_hier, sales_var):
  """
  Gets cumulative sum % by group (e.g., top 99% of products)

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  id_hier : List
      Unique hierarchy of data (e.g., Customer-SKU)
  cum_hier : List
      Hierarchy over which to perform cumulative % (e.g., Customer)
  sales_var: String
      Sales variable over which to perform cumulative sum

  Returns
  -------
  df : PySpark dataframe
      Aggregated dataset at the level of id_hier with appended cumulative sum fields
  """
  #Get cumulative sum by group
  agg_sales = aggregate_data(df,id_hier + cum_hier,sales_var,[sum])
  windowval = (Window.partitionBy(cum_hier).orderBy(desc('sum_' + sales_var)).rangeBetween(Window.unboundedPreceding,0))
  agg_sales = agg_sales.withColumn('cum_sum', sum('sum_' + sales_var).over(windowval))

  #Get total sales by group
  windowval = (Window.partitionBy(cum_hier))
  agg_sales = agg_sales.withColumn('total_sum', sum('sum_' + sales_var).over(windowval))

  #Get cum % and output
  agg_sales = agg_sales.withColumn('cum_pct', col('cum_sum') / col('total_sum'))

  return agg_sales

def impute_cols_ts(df, cols ,order_col, fill_type,partition_cols='MODEL_ID'):
  """
  Impute null data using close values (forward fill or backward fill)
  #TO-DO: Add nearest neighbor
  #TO-DO: Get rid of for loop

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  cols : List
      Columns to impute
  fill_type : String
      "bfill" or "ffill" for for backward/forward fill types

  Returns
  -------
  df : PySpark dataframe
      Imputed dataset
  """

  if isinstance(cols, str):
      cols = [cols]

  for c in cols:
    if c in df.columns:
      if (fill_type == "bfill"):
        #Do backward fill
        window = Window.partitionBy(partition_cols).orderBy(order_col).rowsBetween(0,sys.maxsize)
        df = df.withColumn(c, first(df[c], ignorenulls=True).over(window))

      if (fill_type == "ffill"):
        # Do forward fill
        window = Window.partitionBy(partition_cols).orderBy(order_col).rowsBetween(-sys.maxsize, 0)
        df = df.withColumn(c, last(df[c], ignorenulls=True).over(window))

  return df

def impute_to_col(df, impute_cols ,impute_val_cols):
  "Imputes NULL values to different column"
  #TO-DO: Get rid of loop
  i = 0
  for c in impute_cols:
    df = df.withColumn(c, when(col(c).isNull(), col(impute_val_cols[i])).otherwise(col(c)))
    i = i+1
  return (df)

def impute_to_value(df, impute_cols ,value):
  "Imputes NULL values to single value"
  #TO-DO: Get rid of loop
  for c in impute_cols:
    df = df.withColumn(c, when(col(c).isNull(), lit(value)).otherwise(col(c)))
  return (df)

def interpolate_col(df, impute_col, impute_method):
    """
    Impute null data using interpolation.  Should be run at grouped MODEL_ID level in order to avoid
    interpolating across customer / product hierarchies.

    Parameters
    ----------
    df : Pandas dataframe
      Input dataset
    impute_col : String
      Columns to impute
    impute_method : String
      Interpolation type (e.g., 'linear','spline')
      https://drnesr.medium.com/filling-gaps-of-a-time-series-using-python-d4bfddd8c460

    Returns
    -------
    df : Pandas dataframe
      Imputed dataset
    """

    df[impute_col]=df[impute_col].interpolate(method=impute_method)
    return df

###Outlier detection
def minmax_scale_ps(df, vars_to_scale):
  """
  Applies mix max scaling using pandas udf
  """
  scaler = MinMaxScaler(feature_range=(0, 1))
  df[vars_to_scale] = scaler.fit_transform(df[vars_to_scale])
  return(df)

class DetectOutliers:
    """
    Class to detect outliers at a specified level of aggregation
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'outlier_vars',
            'detect_level',
            'time_var',
            'classifiers'
        ]

        self.__dict__.update(kwargs)
        self._check_required_attrs_received()

    def _check_required_attrs_received(self):
        self.missing_attrs = [attr for attr in self._required_attrs if attr not in self.__dict__]
        if self.missing_attrs:
            missing = ', '.join(self.missing_attrs)
            err_msg = f'The following parameters are required but were not provided: {missing}'
            raise TypeError(err_msg)

def predict_outliers(df, model_info,**kwargs):
    """
    Applies pyod outlier detection methods on fed in data
    """

    df_return = df
    X = df[model_info.outlier_vars]
    for i, (clf_name,clf) in enumerate(model_info.classifiers.items()) :

            clf.fit(X) # fit the dataset to the model
            scores_pred = clf.decision_function(X)*-1 # predict raw anomaly score
            y_pred = clf.predict(X) # prediction of a datapoint category outlier or inlier

    df_return['pred']=pd.DataFrame(y_pred, columns=["pred"])
    df_return = df[ ["MODEL_ID"] + model_info.time_var  + ["pred"]]
    return(df_return)
  

def calculate_percentile(df, group_col, perc_var, percentiles):
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
    lambda df, perc:
      df.withColumn(perc_var + str((perc*10)).replace('.', ''), F.expr('percentile_approx('+perc_var+','+ 
                                                 str(perc)
                                                 +')').over(grp_window)),
    percentiles,
    df
  ))
  return df



def index_cols_ps(df, cols_to_index, suffix):
  """
  Converts character columns into numeric indices

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  cols_to_index : List
      List of columns to index
  suffix : String
      String to append to new column names
      
  Returns
  -------
  df : PySpark dataframe
      Original dataframe with appended indexed columns
  """
  cols_to_index_names = [s + suffix for s in cols_to_index]
  indexer = StringIndexer(inputCols=cols_to_index, outputCols=cols_to_index_names)
  df = indexer.setHandleInvalid("skip").fit(df).transform(df)
  return df