# Databricks notebook source
# DBTITLE 1,Pandas Preprocessing
def factorize_columns(pandas_df, columns):
  """
  Transforms string or category columns into factors (e.g., replacing all of the strings with an integer encoding, i.e. dummy variable)
  """
    
  def factorize_column(series):
    array, labels = series.factorize()
    return {'array':array, "labels": list(labels)}
  
  columns = ensure_is_list(columns)
  
  label_dict = {}
  
  for column in columns:
    factorize_dict = factorize_column(pandas_df[column])
    pandas_df[column + "_index"] = pd.to_numeric(factorize_dict['array'], downcast='integer')    
    #pandas_df.drop(column, axis=1, inplace=True)
    label_dict.update({column + "_index" : factorize_dict['labels']})
    
  return pandas_df, label_dict


def index_features(df, path_spec=None):
  """
  Index string columns and return a DataFrame that is ready for modeling.
  """

  input_df = ensure_is_pandas_df(df)
  
  string_features = list(df.select_dtypes(include=['object', 'category']).columns)
  assert string_features, "No object, category, or string columns found!"
  
  factorized_df, labels = factorize_columns(df, string_features)  
  new_columns = [col + '_index' for col in string_features]
  mapping_array = (factorized_df[string_features + new_columns]).drop_duplicates()
  

  # save mapping array
  if path_spec:
    save_path = get_temp_path() + "_label_array" + path_spec
  else:
    save_path = get_temp_path() + "_label_array"
    
  save_pickle(mapping_array, path=save_path)

  factorized_df.drop(string_features, inplace=True, axis=1)  
    
  return factorized_df


def deindex_features(df):
  """
  Deindex columns in a dataframe using a pre-saved array
  """
  # get mapping array
  load_path = get_temp_path() + "_label_array"
  mapping_array = load_pickle(load_path)

  indexed_features = [col for col in list(df.columns) if "_index" in col]
  mapped_indexed_features = [col for col in list(mapping_array.columns) if "_index" in col]
  
  missing_from_mapping_array = [col for col in indexed_features if col not in mapped_indexed_features]
  if missing_from_mapping_array:
    print("The following columns are missing in your mapping array and won't be deindexed: %s" % missing_from_mapping_array)
  
  deindexed_df = df.merge(mapping_array, on=mapped_indexed_features, how='inner')
  deindexed_df = deindexed_df.drop(mapped_indexed_features, axis=1)
  
  return deindexed_df


def categorize_column_types(df):
  """
  Create two lists for a DataFrame: one which contains the names of all string columns, and one which contains the name of all other columns.
  
  Parameters
  ----------
  df : pandas or koalas DataFrame

  Returns
  -------
  (str_col_set, num_col_set) : (set, set)
      Two mutually exclusive and collectively exhautive sets of columns of df:
      1. str_col_set: all columns not of type int or float
      2. num_col_set: all columns of type int or float
  """
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

  num_col_set = set(df.select_dtypes(include=numerics).columns.tolist())

  str_col_set = set(df.columns.tolist()) - num_col_set

  return (str_col_set, num_col_set)


def split_by_time(df, threshold, time_var=None):
  """
  Split a time-series dataframe into two based on some threshold 
  """
  
  if not time_var:
    time_var = TIME_VAR
    
  training = df[df[time_var] <= threshold]
  testing = df[df[time_var] > threshold]
  
  return (training, testing)


def find_week_splits(df, k_folds):
  """
  Split sorted dates into k even groups and fold them left in order to create indexes for developing rolling validation windows.
  """
  from itertools import accumulate

  def splitList(seq, num):
      avg = len(seq) / float(num)
      out = []
      last = 0.0

      while last < len(seq):
          out.append(seq[int(last):int(last + avg)])
          last += avg

      return out
    
  if is_pyspark_df(df):
    time_seq = sorted(df.select(TIME_VAR).distinct().collect())  
    time_list = [int(row.asDict()[TIME_VAR]) for row in time_seq]  
    
  elif is_pandas_df(df):
    time_list = sorted(df[TIME_VAR].unique())
    
    
  min_val = time_list[0]
  max_val = time_list[-1]
  week_difference = max_val - min_val

  # split into the desired number of folds
  train_test_splits = splitList(time_list, k_folds)

  training_splits = list(accumulate(date_range for date_range in train_test_splits[:-1]))

  test_splits = train_test_splits[1:]

  return zip(training_splits, test_splits)

# COMMAND ----------

# DBTITLE 1,PySpark Preprocessing
def index_features_pyspark(df):
  """
  Index string columns and return a DataFrame that is ready for modeling.
  """
  
  file_name_ending_in_dot_npy = get_pipeline_name() + "_label_dict.npy" 
  
  if is_pandas_df(df): 
    input_df = spark.createDataFrame(df)
  else:
    input_df = df
    
  filled_df = input_df.fillna("missing")
  
  string_features = get_string_cols_pyspark(filled_df)

  indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(filled_df) for column in string_features]
  
  pipeline = Pipeline(stages=indexers)
  
  indexer_pipeline = pipeline.fit(filled_df)
  final_df = indexer_pipeline.transform(filled_df).drop(*string_features)

  if is_pandas_df(df):
    final_df = final_df.toPandas()
    
  # save labels for later
  label_dict = {feature.getOutputCol(): feature.labels for feature in indexer_pipeline.stages}
  save_path = get_temp_path() + file_name_ending_in_dot_npy
  save_dict(label_dict, path=save_path)

  return final_df


def deindex_features_pyspark(df, specified_columns=None, rename_columns=True):
  """
  Index string columns and return a DataFrame that is ready for modeling.
  """
  
  file_name_ending_in_dot_npy = get_pipeline_name() + "_label_dict.npy" 
  
  if is_pandas_df(df):
    input_df = spark.createDataFrame(df)
  else:
    input_df = df
  
  if specified_columns:
    indexed_features = ensure_is_list(specified_columns)
  else: 
    indexed_features = [col for col in list(input_df.columns) if "_index" in col]
  
  fs_load_path = get_temp_path() + file_name_ending_in_dot_npy
  label_dict = load_dict(fs_load_path)
  
  final_df = input_df
  
  for feature in indexed_features:
    if rename_columns:
      new_name = feature.replace("_index", "")
    else:
      new_name=feature
      
    final_df = IndexToString(inputCol=feature, outputCol=new_name, labels=label_dict[feature]).transform(final_df)
    
  # drop indexed columns
  final_df = final_df.drop(*indexed_features)
      
  if is_pandas_df(df):
    return final_df.toPandas()

  return final_df


def get_string_cols_pyspark(df):
  """
  Return a list of all string columns in a PySpark DataFrame.
  
  Parameters
  ----------
  df : PySpark DataFrame
  
  Returns
  -------
  string_columns : list
      List of all features of df whose dtype starts with 'string'.
  """
  return [feature[0] for feature in df.dtypes if feature[1].startswith('string')]


def assemble_vectors_pyspark(spark_df):
  """
  Assemble features as vectors for modeling.
  
  Paramaters
  ----------
  spark_df : PySpark DataFrame
      DataFrame with features and a column TARGET_VAR.
  
  Returns
  -------
  final_df : PySpark DataFrame
      DataFrame with columns "features" (vector) and TARGET_VAR.
  """
  feature_list = spark_df.columns
  feature_list.remove(TARGET_VAR)

  assembler = VectorAssembler(
      inputCols=feature_list,
      outputCol="features")

  output = assembler.transform(spark_df)
  selected_df = output[["features", TARGET_VAR]]

  final_df = selected_df.toDF("features", "label")
  
  return final_df