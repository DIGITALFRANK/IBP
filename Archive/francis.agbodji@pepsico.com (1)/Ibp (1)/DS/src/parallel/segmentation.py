# Databricks notebook source
class DTWModelInfo:
    """
    Class to contain the required information for segmenting time series using Dynamic Time Warping (DTW).  DTW is a distance metric that
    can be used in clustering algorithms (KNN, Hierarchical, etc.). This metric can compare time series that are not time aligned unlike
    euclidean distances.
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'model_id',
            'time_field',
            'target_field',
            'n_clusters'
        ]

        # Check required attributes exist
        self.__dict__.update(kwargs)
        self._check_required_attrs_received()

    def _check_required_attrs_received(self):
        self.missing_attrs = [attr for attr in self._required_attrs if attr not in self.__dict__]
        if self.missing_attrs:
            missing = ', '.join(self.missing_attrs)
            err_msg = f'The following parameters are required but were not provided: {missing}'
            raise TypeError(err_msg)

    def set_param(self, **kwargs):
      """
      Overrides a parameter
      """
      self.__dict__.update(kwargs)
def calculate_dtw_segment(df, segmentation_cls):
  #Pivot so each column is a week
  df_pivoted = df.pivot(index=segmentation_cls.model_id, 
                        columns=segmentation_cls.time_field, 
                        values=segmentation_cls.target_field)
  df_pivoted = df_pivoted.fillna(0)
  
  #Convert data into 2d numpy array and scale
  ts_dat = df_pivoted.values
  ts_dat = scale(ts_dat, axis=1)

  #Convert to 3d array
  ts_dat = to_time_series_dataset(ts_dat)
  ts_dat.shape
  
  #Cluster
  model = TimeSeriesKMeans(n_clusters=segmentation_cls.n_clusters, metric="dtw", max_iter=10)
  cluster_pred = model.fit_predict(ts_dat)
  
  #Output
  output_clusters = pd.DataFrame(df_pivoted.index)
  output_clusters['STAT_CLUSTER'] = pd.DataFrame(cluster_pred)
  return(output_clusters)

# COMMAND ----------

def calculate_dtw_elbow(df, segment_group):
  #Pivot so each column is a week
  df_pivoted = df.pivot(index=segment_group, columns=TIME_VAR, values='target_var_QTY')
  df_pivoted = df_pivoted.fillna(0)
  
  #Convert data into 2d numpy array and scale
  ts_dat = df_pivoted.values
  ts_dat = scale(ts_dat, axis=1)

  #Convert to 3d array
  ts_dat = to_time_series_dataset(ts_dat)
  ts_dat.shape
  
  iertia = []
  n = []
  for i in (2,5,6,8,10):
      print(i)
      #model = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=10)
      #cluster_pred = model.fit_predict(ts_dat)
      iertia.append(3)
      n.append(i)

  iertia = pd.DataFrame(iertia, columns=["inertia"])
  iertia['n'] = pd.DataFrame(n)
  iertia[segment_group] = df[segment_group].drop_duplicates()
  iertia = iertia[[segment_group,"inertia","n"]]

  return(iertia)

# #Generate automatic schema of scaling output table
# keep_vars = STAT_SEGMENT_GROUP + ["inertia","n"]
# auto_schema = mrd.withColumn("inertia",lit(1.0)).withColumn("n",lit(1))
# #keep_vars = intersect_two_lists(keep_vars, mrd.columns)
# auto_schema = auto_schema.select(keep_vars)
# auto_schema = auto_schema.limit(1)
# auto_schema = auto_schema.schema
# auto_schema
# #Calculate elbow
# schema = StructType([
# StructField('MODEL_ID', StringType()),
# StructField('inertia', DoubleType()),
# StructField('n', DoubleType())])

# @pandas_udf(auto_schema, PandasUDFType.GROUPED_MAP)
# def predict_dtw_segment_udf(data):
#     return calculate_dtw_elbow(data, STAT_SEGMENT_GROUP)

# cluster_elbow = mrd.groupBy(STAT_SEGMENT_GROUP).apply(predict_dtw_segment_udf)
# cluster_elbow.cache()
# cluster_elbow.count()
# display(cluster_elbow)

# COMMAND ----------

def get_cumsum_simple(df, id_hier, sales_var):
  """
  Gets cumulative sum % by group (e.g., top 99% of products)

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  id_hier : List
      Unique hierarchy of data (e.g., Customer-SKU)
  sales_var: String
      Sales variable over which to perform cumulative sum

  Returns
  -------
  df : PySpark dataframe
      Aggregated dataset at the level of id_hier with appended cumulative sum fields
  """
  
  ## Derive total sales for time period of dataframe
  total_sales = df.select(F.sum(sales_var)).collect()[0][0] 
  
  ## Get cumulative sum by group
  agg_sales = aggregate_data(df, id_hier, sales_var, [sum])    
  windowval = (Window.orderBy(desc('sum_' + sales_var)).rangeBetween(Window.unboundedPreceding,0))
  agg_sales = agg_sales.withColumn('cum_sum', sum('sum_' + sales_var).over(windowval))
  
  ## Total sales for cumsum
  agg_sales = agg_sales.withColumn('total_sum', lit(total_sales))

  ## Get cum % and output
  agg_sales = agg_sales.withColumn('cum_pct', col('cum_sum') / col('total_sum'))

  return agg_sales

# COMMAND ----------

def calculate_cv_segmentation(df, cv_var, report_level=None):
    """
    Calculates coefficient of variation of variable at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    cv_var : String
      Variable
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated CV
    """
    
    ## Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    ## Check variables exists in data
    if len(intersect_two_lists([cv_var],df.columns)) == 0:
        return(None)

    ## Perform CV
    df = df.groupBy(report_level).agg((stddev(col(cv_var))/mean(col(cv_var))).alias("CV"),
                                     (stddev(col(cv_var))).alias('STD'),
                                     (mean(col(cv_var)).alias('MEAN')))
    return(df)

# COMMAND ----------

