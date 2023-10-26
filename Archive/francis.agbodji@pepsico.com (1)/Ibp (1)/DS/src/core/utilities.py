# Databricks notebook source
from typing import Iterable
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

# COMMAND ----------

# DBTITLE 1,Pandas I/O
def save_csv(df, path):
  """
  Save csv to path
  """
  save_path = path + '.csv'
  make_dir(save_path)
  
  df.to_csv(save_path)
 
  print("Saved to %s \n" % save_path)  
  
  
def save_pickle(df, path):
  """
  Save pickle to path
  """
  save_path = path + '.pkl'
  make_dir(save_path)
  
  df.to_pickle(save_path)
   
  print("Saved to %s \n" % save_path)
  
                   
def quick_save(df, indicator="", save_func=save_pickle, temp_override=False):
  """
  Quickly save the dataframe to throwaway folder
  """
  if not temp_override:
    assert OUTBOUND_PATH
    save_path = get_local_path(OUTBOUND_PATH)
  else:
    save_path = get_temp_path() 
        
  save_func(df, save_path + "checkpoints/pickles/checkpoint_" + indicator)

  
def load_csv(path, *args, **kwargs):
  """
  Load a csv into pandas
  """
  load_path = path + '.csv'

  return reduce_df_size(pd.read_csv(load_path, *args, **kwargs))
  
  
def load_pickle(path, *args, **kwargs):
  """
  Load a pkl into pandas
  """
  load_path = path + '.pkl'

  return reduce_df_size(pd.read_pickle(load_path, *args, **kwargs))


def quick_load(indicator="", load_func=load_pickle, temp_override=False):
  """
  Quickly load a file into pandas from a throwaway folder
  """
  if not temp_override:
    assert OUTBOUND_PATH
    save_path = get_local_path(OUTBOUND_PATH)
  else:
    save_path = get_temp_path() 

  return load_func(save_path + "checkpoints/pickles/checkpoint_" + indicator)
  

def save_dict_as_csv(dictionary, path):
  """
  Save a dictionary as a comma-separated values (.csv) file.
  
  Parameters
  ----------
  dictionary : dict
      Dictionary to save.
      
  path : string
      ADLS filepath location to store the dictionary.
  """

  dict_pd = pd.DataFrame(dictionary, index=[0])

  dict_df = spark.createDataFrame(dict_pd)

  dict_df \
  .repartition(1) \
  .write.format('csv') \
  .mode("overwrite") \
  .option("header", True) \
  .save(path)
    
  print("File saved at %s" % path)
  
  
def load_dict_from_csv(path):
  """
  Load a dictionary from a comma-separated values (.csv) file.
  
  Parameters
  ----------
  path : string
      ADLS filepath to the dictionary.
  
  Returns
  -------
  dict_pd : pandas DataFrame
      Dictionary cast as DataFrame (oriented by records). 
  """
  dict_df = spark.read.format("csv") \
  .option("header", "true") \
  .option("inferSchema", "true") \
  .load(path)
  
  dict_pd = dict_df.toPandas().to_dict(orient='records')[0]
  
  return dict_pd


def save_dict(dict_obj, path):
  """
  Save dictionary as numpy file
  """
  import os
  np.save(os.path.expanduser(path), dict_obj)
  
  
def load_dict(path):
  """
  Load dictionary from numpy pickle
  """
  return np.load(path, allow_pickle='TRUE').item()

#TODO test with quick_save
def quick_save_parquet(df, indicator=""):
  """
  Quickly save the dataframe to throwaway folder
  """
  assert OUTBOUND_PATH
  save_path = OUTBOUND_PATH + "checkpoints/parquets/checkpoint_" + indicator
  save_parquet_file(df, save_path, verbose=False)
  
#TODO test with quick_load
def quick_load_parquet(indicator="", pandas=True):
  """
  Quickly load the dataframe from a throwaway folder
  """  
  assert OUTBOUND_PATH
  save_path = OUTBOUND_PATH + "checkpoints/parquets/checkpoint_" + indicator + ".parquet"
  
  #TODO see if get_local_path is no longer needed
  if pandas:
    return read_parquet((save_path))
  else:
    return read_parquet_pyspark(save_path)
  
  
#TODO split into pyspark and pandas versions
def save_parquet_file(df, path, verbose=True, partition=True):
  """
  Write PySpark dataframe to parquet file (no need to include the parquet extension)
  """
  
  #TODO handle
  assert '.parquet' not in path, "No need to include the parquet extension in your path!"

  if is_pandas_df(df):
    
    folder = get_local_path(path)
    file = folder + '.parquet'
  
    df.to_parquet(file)
    
  else:
    # Spark needs a different path than python
    folder = path
    file = folder + '.parquet'
    
    if partition:
      df.write\
        .mode('overwrite')\
        .option('encoding', 'utf-8')\
        .option('header', 'true')\
        .parquet(folder)
    
    else:
      df.repartition(1)\
        .write\
        .mode('overwrite')\
        .option('encoding', 'utf-8')\
        .option('header', 'true')\
        .parquet(folder)
    
      # Save it as just the file, not a folder
      temp_path = dbutils.fs.ls(folder)[-1].path
      dbutils.fs.cp(temp_path, file)
      dbutils.fs.rm(folder, recurse=True)

  if verbose:
    print('Saved to %s' % file)
  
  
def read_parquet_pyspark(path):
  """
  Read parquet file into pyspark dataframe
  """
  #TODO handle
  assert '.parquet' in path, "Missing parquet extension!"
    
  return spark.read.parquet(path)


def read_parquet(path):
  """
  Read parquet file to pandas dataframe
  """
  if '.parquet' not in path:
    path = path + '.parquet'
  
  path = get_local_path(path)
  
  return pd.read_parquet(path)


def get_filestore_path():
  """
  Return root path of Databricks' Filestore, which we use to save parameters and temporary dataframes
  """

  return "/FileStore/tables/"


def get_temp_path(use_outbound=False):
  """
  Create / fetch a path for housing temporary files created in the modeling workflow
  """
  from pathlib import Path

  if use_outbound:
    assert OUTBOUND_PATH, "Missing OUTBOUND_PATH."
    path = get_local_path(OUTBOUND_PATH + get_pipeline_name() + '/temp/') 
  else:
    path = get_local_path(get_filestore_path() + get_pipeline_name() + '/temp/')
    
  Path(path).mkdir(parents=True, exist_ok=True)
  
  return path


def load_all_pickles_from_dir(path):
  """
  Load multiple pickles into a single dataframe
  """
  paths_list = list_files_in_directory(path)
  
  paths_list = [get_local_path(path) for path in paths_list]
  
  first_path = paths_list.pop(0)
  
  df = pd.read_pickle(first_path)
  
  for path in paths_list:
    try:
      temp_df = pd.read_pickle(path)
      df = pd.concat([df, temp_df], axis=0)
    except: 
      warnings.warn("Failed to load" + path, UserWarning)
      
  return reduce_df_size(df)


def load_all_dfs_from_dir_pyspark(path):
  """
  Load all DFs in a directory in a dictionary for querying
  """
  files_in_path = list_files_in_directory(path)

  df_dict = {}
  name_list = [] # used to avoid querying spark when creating pointers
  
  print("Starting bulk upload.. \n.")
  
  for file in files_in_path:
    name = get_table_name_from_path(file)
    print("Loading ", name)
    try: 
      df_dict[name] = load_spark_df(file)
    except: 
      warnings.warn("Failed to load" + name, UserWarning)
      
  print("\n Created Spark DataFrame dictionary containing the following tables: ", df_dict.keys())
    
  return df_dict


def filter_using_indexes(new_df, orig_df, filter_columns):
  """
  Filter one dataframe using a multiindex from another
  """

  new_index = new_df.set_index(filter_columns).index
  original_index = orig_df.set_index(filter_columns).index
  
  return new_df[new_index.isin(original_index)]


def filter_using_dict(df_pd, column_dict):
  """
  Without this function, the standard way of getting the records for a single hierarchy combination from a pandas DataFrame looks like:
     pd_df.loc[pd_df['hier_col_1'] == 'value', pd_df['hier_col_2'] == 'value', ...]
  This function simplifies that with a dictionary of column name to column value
  
  Inputs
  ------
  df_pd : pandas DataFrame
  column_dict : dict
      Dictionary of column names to values to select
    
  Returns
  -------
  select_pd : pandas DataFrame
  """
      
  individual_conditions = tuple(df_pd[column] == column_dict[column] for column in column_dict.keys())
  
  condition_series = pd.DataFrame(individual_conditions).transpose().all(axis = 1)
  
  return df_pd.loc[condition_series]


def list_files_in_directory(path):
  """
  Get a list of files within a given directory
  """
  
  if path[0:5] != "/dbfs":
    path = "/dbfs" + path
        
  files = glob.glob(path + "/*")
  
  filtered_files = [path.replace("/dbfs", "") for path in files]
  
  return filtered_files


def get_local_path(path):
  """
  Python isn't aware of spark context, so you have to add "/dbfs/ to the beginning of paths"
  """
  return "/dbfs" + path


def list_root_files_in_directory(path):
  """
  Print all available files within a folder structure using BFS
  """
  import glob

  def any_exist(avalue, bvalue):
    return any(any(x in y for y in bvalue) for x in avalue)
  
  queue = [path]
  file_paths = []
  
  delimiters = ['txt', 'csv']
  
  while len(queue) > 0:
    next_path = queue[0]
    files_in_path = list_files_in_directory(next_path)
    
    if any_exist(delimiters, files_in_path):
      print("Found root:" + next_path)
      file_paths.append(next_path)
    
    else:
      queue += files_in_path
          
    queue.pop(0)

  return file_paths


def print_files_in_dir(filepath):
  """
  Print files in directory at filepath and corresponding filesize information 
  """
  
  files = dbutils.fs.ls(filepath)
  total_size = np.sum([file.size for file in files])
  print(f'{len(files):d} files in ' + filepath + ' totaling ' + format_bytes(total_size) + ': ')

  for i in range(len(files)):
    print(f'{i + 1:2d}. ', files[i].name, "(" + format_bytes(files[i].size) + ")")

    
def remove_dir(filepath_incl_dir_name):
  """
  Delete folder at filepath_incl_file_or_dir_name
  """
  
  if filepath_incl_dir_name[-1] != '/':
    filepath_incl_dir_name = filepath_incl_dir_name + '/'
  
  filepath_excl_dir_name = filepath_incl_dir_name.rsplit('/',2)[0]
  dir_name = filepath_incl_dir_name.rsplit('/',2)[1]
  
  dbutils.fs.rm(filepath_incl_dir_name, recurse = True)
  print(dir_name + " (folder) removed from directory")

  
def remove_file(filepath):
  """
  Delete file at filepath
  """
  
  filepath_excl_filename, filename = os.path.split(filepath)
  
  files_in_root_dir = dbutils.fs.ls(filepath_excl_filename)
  filenames_in_root_dir = [file.name for file in files_in_root_dir]

  file_index = filenames_in_root_dir.index(filename)
  
  dbutils.fs.rm(filepath)
  print(filename + " (" + format_bytes(files_in_root_dir[file_index].size) + ") removed from directory")
  
  
def remove_files_in_dir(path):
  """
  Delete all files within a directory
  """
  path = get_local_path(path) + '/*'
  
  files = glob.glob(path)
  for f in files:
      os.remove(f)
      

def make_dir(file_path):
  """
  Create a new directory at a given path
  """
  import os
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

# COMMAND ----------

# DBTITLE 1,Memory Management
def print_memory_config():
  """
  Prints important config information that could explain OOM issues
  Docs: https://spark.apache.org/docs/latest/configuration.html#memory-management
  """
  print("Max executor memory: %s" % sc._conf.get('spark.executor.memory'))
  print("Max memory allowed to be moved onto the driver node: %s" % sc._conf.get('spark.driver.maxResultSize'))


def set_memory_configs(maxResultSize='16g', executorMemory='32g'):
  """
  Sets your maxResultSize to be higher than default; may cause your driver to crash if you exceed your driver nodes memory allowance.
  """
  sc._conf.set('spark.driver.maxResultSize', maxResultSize)
  sc._conf.set('spark.executor.memory', executorMemory)
  

def get_memory_usage(pandas_df):
  """
  Returns the number of bytes used by a pandas dataframe
  """
  return pandas_df.memory_usage().sum()


def print_memory_usage(pandas_df):
  """
  Returns the number of bytes used by a pandas dataframe in a formatted string
  """
  return format_bytes(get_memory_usage(pandas_df))


def reduce_df_size(input_pd, recast_floats=True, verbose=False):
  """
  Reduces dataframe size by safely downcasting
  NOTE: recast_floats may chop-off extreme digits; see https://en.wikipedia.org/wiki/Single-precision_floating-point_format for more information
  """
    
  starting_memory = get_memory_usage(input_pd) 
  
  # convert objects to categories
  object_cols = list(input_pd.select_dtypes(include=['object']).columns)

  if object_cols:
    cast_columns(input_pd, object_cols, 'category')
  else:
    if verbose:
      print("No object columns found.")
  
  # Recast numerics
  downcast_int(input_pd)
  
  if recast_floats:
    downcast_float(input_pd)
  
  if verbose:
    # Calculate efficiency improvements
    ending_memory = get_memory_usage(input_pd) 
    percent_reduction = __builtin__.round((starting_memory - ending_memory)/starting_memory, 4)*100
    print("Reduced dataframe size by %s%% \n" % __builtins__.round(percent_reduction, 4))
  
  return input_pd


def profile_runtime(func):
  """
  Decorator function you can use to profile a bunch of nested functions.
  Docs: https://docs.python.org/2/library/profile.html#module-cProfile
  Example:

    @profile_python_code
    def profileRunModels(*args, **kwargs):
      return run_models(*args, **kwargs)
  
  """
  
  def wrap(*args, **kwargs):
    import cProfile
        
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    profiler.print_stats(sort='cumulative')
      
    return result

  return wrap


def profile_memory_usage(func, *args, **kwargs):
  """
  Profile the amount of memory used in a python function
  """
  from memory_profiler import profile
  
  return profile(func(*args, **kwargs))
  

def format_bytes(size):
  """
  Takes a byte size (int) and returns a formatted, human-interpretable string
  """
  # 2**10 = 1024
  power = 2**10
  n = 0
  power_labels = {0 : ' bytes', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
  while size > power:
      size /= power
      n += 1
  return str(__builtins__.round(size, 2)) + power_labels[n]


def print_environment_details():
  """
  Prints available / used RAM and CPU for diagnostic purposes
  """
  import psutil
  print("Percent of CPU used: %s%%" % psutil.cpu_percent())
  
  memory_used = dict(psutil.virtual_memory()._asdict())
  total_bytes = format_bytes(memory_used['total'])
  available_bytes = format_bytes(memory_used['available'])
  
  print("Total quantity of virtual memory on system: %s" % total_bytes)
  print("Available quantity of virtual memory on system: %s" % available_bytes)
  print("Percent of virtual memory used: %s%%" % memory_used['percent'])

# COMMAND ----------

# DBTITLE 1,Dask Utilities
def convert_pandas_to_dask(pandas_df, npartitions=None, distributed=True, partition_size="100MB"):
  """
  Convert a pandas dataframe to a distributed dask dataframe, enabling lazy evaluation that can be prompted using .compute() or .persist()
  """
  from distributed import Client
  
  if npartitions:
    dask_df = dd.from_pandas(pandas_df, npartitions=npartitions)
  else:
    dask_df = dd.from_pandas(pandas_df, npartitions=4)\
                .repartition(partition_size=partition_size)
  
  if distributed:
    if DASK_CLIENT is None: 
      DASK_CLIENT = Client()
    
    dask_df = DASK_CLIENT.persist(dask_df)

  return dask_df


def profile_dask_client():
  """
  Print scheduler statistics
  """
  assert dask_client, "No dask client has been defined globally."
  return dask_client.profile()

# COMMAND ----------

# DBTITLE 1,Pandas UDFs
def get_weighted_average_func(pd_df, tar_var):
  """
  Returns lambda function for calculating weighted average
  """
  mask = pd_df[tar_var] != 0
  return lambda x: np.ma.average(x[mask], weights=pd_df.loc[x[mask].index, tar_var])


# COMMAND ----------

# DBTITLE 1,Pandas / Python Utilities
# TODO requires testing
# def merge_by_concat(big_df, small_df, index_columns=None, how='left'):
#   """
#   Merge two dataframes by concatenation. Avoids trying to join two high-dimensionality dataframes in memory
#   by joining them on an index and then adding in the other columns later.
#   """
  
#   if not index_columns:
#     index_columns = big_df[get_hierarchy() + [TIME_VAR]]
  
#   merged_df = big_df[index_columns].merge(small_df, on=index_columns, how=how)
#   merged_df.drop(index_columns, axis=1, inplace=True)
  
#   return pd.concat([big_df, merged_df], axis=1)


def concatenate_dfs(df1, df2):
  """
  Safely concatenates dataframes, handling for the different treatment between panda and dask implementations
  """
  if is_pandas_df(df1) & is_pandas_df(df2):
    return pd.concat([df1, df2])
  
  elif is_dask_df(df1) & is_dask_df(df2):
    return dd.concat([df1, df2])
    
  else:
    raise InputError("DataFrames of the wrong class or of differnet classes")


def set_target_var(name):
  """
  Manually update the target var
  """  
  TARGET_VAR = name
  
  
def update_target_var(df, search_str_length=10):
  """
  Automatically update the TARGET_VAR based on columns in a dataframe
  """  
  new_name = get_target_var(df, search_str_length)
  set_target_var(new_name)
  
  
def get_base_target_var():
  """
  Get the root feature of the TARGET_VAR
  """
  
  for indicator in ["_log", "_nrm", "_std"]:
    if indicator in TARGET_VAR:
      return TARGET_VAR.replace(indicator, "")
      
  raise ValueError('No transformation indicator found!')
    
    
  
def get_target_var(df, search_str_length=10):
  """
  Get name of global variable TARGET_VAR after transformations to that column in a pandas or koalas DataFrame.
  NOTE: does not change global var
  
  Parameters
  ----------
  pd_df : pandas or koalas DataFrame
      DataFrame with aggregated values of global variable TARGET_VAR.
  """
  
  if is_pyspark_df(df):
    column_names = df.schema.names
    
  elif is_pandas_df(df):
    column_names = df.columns.tolist()
    
  else:
    raise_df_value_error()
    
  new_name = [feature for feature in column_names if TARGET_VAR[:search_str_length] in feature]
  #TARGET_VAR[:search_str_length]

  if len(new_name) > 1: 
    raise ValueError('Returns more than one variable')
  
  return new_name[0]


def get_pipeline_name():
  """
  Return a string that can be appended to various file names
  """
  return PIPELINE_NAME


def run_function_in_parallel(func, t_split):
  """
  Multiprocess a python function
  """
  from multiprocessing import Pool
  import psutil
  
  N_CORES = psutil.cpu_count()
  
  num_cores = np.min([N_CORES,len(t_split)])
  pool = Pool(num_cores)
  df = pd.concat(pool.map(func, t_split), axis=1)
  pool.close()
  pool.join()
  return df


def correct_suffixes_in_list(input_pd, lst, substring='_index'):
  """
  Ensure every element in a list has the right suffix based on the columns in a Pandas dataframe
  """
  dataframe_columns = list(input_pd.columns)
  
  missing_from_dataframe = [item for item in lst if item not in dataframe_columns]
  
  corrected = [item for substring in missing_from_dataframe for item in dataframe_columns if item.startswith(substring)]  
  
  assert len(missing_from_dataframe) == len(corrected), "Not able to correct all missing items in grouping list. \n List 1: % s \n List 2: %s" % (missing_from_dataframe, corrected)
  
  if corrected: 
    lst.extend(corrected)
    lst = [item for item in lst if item not in missing_from_dataframe]
  
  return lst   


def get_hierarchy():  
  """
  Return a list comprised of the concatenated product and business hierarchies up to the user-specified levels (as defined in global variables PRODUCT_LEVEL and BUSINESS_LEVEL).
  """

  return PRODUCT_HIER[:PRODUCT_LEVEL] + BUSINESS_HIER[:BUSINESS_LEVEL] + OTHER_CATEGORICALS


def display_list(lst):
  """
  Conveniently displays a python list as a Spark DF
  """
  display(spark.createDataFrame(lst, StringType()))
  
  
def get_table_name_from_path(path):
  """
  Given a file path, return the table name
  Ex.: If path is /dbfs//mnt/WALMART US/ml_services/sre/outbound/MAPPINGFILES/LISTPRICE_NEW, return LISTPRICE_NEW
  """
  return path[path[:-1].rindex('/')+1:]


def update_time(series_or_int, adjustment, time_unit='weeks', datetime_format="%Y%U-%w"):
  """
  Correctly adds or subtracts units from a time period
  """  
  
  # we may want to treat time periods as a monotonically increasing integer
  # without breaking all of our other functions
  if time_unit == "int":
    return series_or_int + adjustment
  
  casted_series = pd.Series(series_or_int).astype(str)
  
  # get datetime col
  if datetime_format in ["%Y%U-%w", "%Y%W-%w"]:
    datetime_series = pd.to_datetime(casted_series.astype(str) + '-1', format=datetime_format)
  else:
    datetime_series = pd.to_datetime(casted_series.astype(str), format=datetime_format)
    
  # make adjustment
  adjustment_delta = pd.DateOffset(**{time_unit:adjustment})  
  adjusted_series = (datetime_series + adjustment_delta)
  
  # return the series in the original
  final_series = adjusted_series.dt.strftime(datetime_format)
  
  if datetime_format in ["%Y%U-%w", "%Y%W-%w"]:
    final_series = final_series.str.extract(r'(.*)-\d+', expand=False).astype(int)
  else:
    final_series = final_series.astype(int)
    
  if isinstance(series_or_int, (int, np.integer)): 
    assert final_series.shape == (1,)
    return final_series[0]
  
  return final_series


#TODO delete
def update_week(initial_val, adjustment_value):
  """
  Correctly adds or subtracts values from a "WeekOfYear" style indicator. Assumes adjustment-value is an integer
  """
  raise ValueError("update_week is deprecated; use update_time instead")
#   initial_str = str(initial_val)

#   week = initial_str[4:]
#   year = initial_str[:4]
    
#   adj_week = int(week) + int(adjustment_value)
  
#   if adj_week > 52:
#     adj_year = int(year) + int(adj_week/52)
#     adj_week = adj_week % 52e
  
#   elif adj_week < 1:
#     adj_year = int(year) + int(adj_week/52) - 1
#     adj_week = adj_week % 52
#     if adj_week == 0:
#       adj_week = 52
      
#   else:
#     adj_year = year
#     adj_week = adj_week
    
#   return str(adj_year) + str(adj_week).zfill(2)

  
def cast_columns(pd, column_list, primitive):
  """
  Cast multiple pandas columns to a given type
  """
    
  column_list = ensure_is_list(column_list)
  
  pd[column_list] = pd[column_list].astype(primitive)
  
  return pd


def downcast_int(input_pd):
  """
  Downcasts integers to save space (down to int8 at the minimum)
  """
  int_cols = list(input_pd.select_dtypes(include=['integer']).columns)
  if not int_cols:
    return input_pd
  else:
    input_pd[int_cols] = input_pd[int_cols].apply(pd.to_numeric, downcast='integer')
  
  return input_pd


def downcast_float(input_pd):
  """
  Downcasts floats to save space (down to float32 at the minimum)
  """
  float_cols = list(input_pd.select_dtypes(include=['float64']).columns)

  if not float_cols:
    return input_pd
  else:
    input_pd[float_cols] = input_pd[float_cols].apply(pd.to_numeric, downcast='float')
  
  return input_pd


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


def is_dask_df(obj):
  """
  Check if an object is a dask DataFrame.
  """
  return isinstance(obj, dd.DataFrame)

  
def is_string(obj):
  """
  Check if an object is a string.
  """
  return isinstance(obj, str)
  
  
def is_pyspark_df(obj):
  """
  Check if an object is a PySpark DataFrame.
  """
  return isinstance(obj, pyspark.sql.dataframe.DataFrame)


def is_list(obj):
  """
  Check if an object is a list in python 
  """
  return isinstance(obj, list) 
  

def ensure_is_list(obj):
  """
  Return an object in a list if not already wrapped
  """
  if not is_list(obj):
    return [obj]
  else:
    return obj
  
  
def convert_to_pandas_df(df):
  """
  Converts a Spark DF to a Pandas DF 
  """
  
  return df.toPandas()

  
def convert_to_pyspark_df(df):
  """
  Converts a Pandas DF to a PySpark DF 
  """
  
  return spark.createDataFrame(df)


def ensure_is_pyspark_df(input_df):
  """
  Ensures that a dataframe is a PySpark dataframe (not a pandas DF)
  """
  if is_pandas_df(input_df):
    return spark.createDataFrame(input_df)
  else:
    return input_df
  
  
def ensure_is_pandas_df(input_df):
  """
  Ensures that a dataframe is a pandas dataframe (not a PySpark DF) and downcast automatically
  """
  if not is_pandas_df(input_df):
    return reduce_df_size(input_df.toPandas())
  else:
    return reduce_df_size(input_df)


def exp_increase_df_size(df, n):
  """
  Exponentially multiples dataframe size for feasibility testing
  """
  for i in range(n):
    df = pd.concat([df, df], axis=0)
  
  return df

# COMMAND ----------

# DBTITLE 1,PySpark DataFrame Extensions
def pipe(self, f, *args, **kwargs):
  """
  Add a pipe method for use on the pyspark.sql.DataFrame interface
  """
  return f(self, *args, **kwargs)

def cast_columns_pyspark(df, column_list, cast_type):
  """
  Cast multiple PySpark columns to a given type
  """
    
  column_list = ensure_is_list(column_list)
  
  for col_name in column_list:
    df = df.withColumn(col_name, col(col_name).cast(cast_type))
 
  return df

def rename_columns_pyspark(df, column_naming_dict):
  """
  Rename multiple columns in PySpark
  """
  for old, new in column_naming_dict.items():
    df = df.withColumnRenamed(old, new)
    
  return df

# COMMAND ----------

# DBTITLE 1,PySpark I/O
def quick_save_spark(df, indicator=""):
  """
  Quickly save the dataframe to throwaway folder
  """
  assert OUTBOUND_PATH
  save_path = OUTBOUND_PATH + "checkpoints/quick_save" + indicator
  print("Saving to " + save_path)
  save_spark_df(df, save_path)
  
  
def quick_load_spark(indicator=""):
  """
  Quickly load the dataframe from a throwaway folder
  """  
  assert OUTBOUND_PATH

  return load_spark_df(OUTBOUND_PATH + "checkpoints/quick_save" + indicator)


def save_spark_df(df, path):
  """
  Save a Spark DataFrame to ADLS using partitioned files.
  
  Parameters
  ----------
  df : PySpark DataFrame
      DataFrame to save.
  
  path : string
      ADLS filepath location to store the DataFrame.
  """
  
  if is_pandas_df(df):
    df = convert_to_pyspark_df(df)
  
  df.write.format("com.databricks.spark.csv") \
  .option("header", "true")\
  .mode("overwrite") \
  .save(path)
    
  
def load_spark_df(path):
  """
  Load Spark DataFrame from partitioned files.
  
  Parameters
  ----------
  path : string
      ADLS filepath to the DataFrame.
  """
  df = spark.read \
  .option("inferSchema", "true")\
  .option("header", "true")\
  .format("csv")\
  .load(path)
  
  return df


def load_pandas_df_from_spark(path):
  """
  Load and optimize pandas df through Spark interface
  """
  return ensure_is_pandas_df(load_spark_df(path))

# COMMAND ----------

# DBTITLE 1,PySpark Utilities
def compare_pyspark_dfs(df1, df2, cols_to_compare):
  """
  Compare list of unique elements in two PySpark dataframes
  """
  capture_dict = {}
  for feature in cols_to_compare:
    set1 = set(df1.select(feature).distinct().collect())
    set2 = set(df2.select(feature).distinct().collect())
    intersection_count = len(set1 & set2)
    print("%s intersection - Count: %s, Percent of DF1: %s, Percent of DF2: %s" % (feature, intersection_count, intersection_count/len(set1), intersection_count/len(set2)))
    capture_dict[feature]: [set1, set2]
      
  return capture_dict


def remove_aggregate_aliases(string):
  """
  Remove junk strings added by .agg() (e.g., "max()")
  """
  for substring in ["avg(", "sum(", "min(", "max(", ")"]:
    string = string.replace(substring, "")

  return string


def columnwise_union_pyspark(df1, df2):
  """
  Merge two PySpark DataFrames together columnwise.
  
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


def exp_increase_df_size_pyspark(df, n):
  """
  Exponentially multiples dataframe size for feasibility testing
  """
  for i in range(n):
    df = df.union(df)
  
  return df


def save_mllib_model_pyspark(model_object, path=None, name=None):
  """
  Sales MLlib modeling object to Filestore for future reference
  """
  
  if not name:
    name = model_object.__name__
  
  if not path:
    path = get_filestore_path()
  
  model_object.write.overwrite().save(path + name)
  

def melt_df_pyspark(df: DataFrame, id_vars: Iterable[str], value_vars: Iterable[str], var_name: str="variable", value_name: str="value") -> DataFrame:
    """
    Melt pyspark dataframe (similar to pd.melt())
    """
    # Create array<struct<variable: str, value: ...>>
    _vars_and_vals = F.array(*(
        F.struct( F.lit(c).alias(var_name), F.col(c).alias(value_name) )
        for c in value_vars))

    # Add to the DataFrame and explode
    _tmp = df.withColumn("_vars_and_vals", F.explode(_vars_and_vals))

    cols = id_vars + [
            F.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]]
    return _tmp.select(*cols)
  

def save_df_as_csv_spark(df, path):
  """
  Save a Spark DataFrame as a comma-separated values (.csv) file.
  """
    
  if is_pandas_df(df):
    df = convert_to_pyspark_df(df)
  
  df \
  .repartition(1) \
  .write.format('csv') \
  .mode("overwrite") \
  .option("header", True) \
  .save(path)
    
  print("File saved at %s" % path)

# COMMAND ----------

# DBTITLE 1,Workbench - multiple file upload from CSV - Corey
# Utility function for easier file upload + compilation
# Beverage and Nutrition files were compiled from multiple CSV sources
# NOTE: this function pulled from Order Cleansing codebase
#TODO better variable names, list comprehension instead of for loops
def upload_multiple_files(file_path, file_list, file_extension):
  complete_pd = pd.DataFrame()
  for each in range(0, len(file_list)):
    full_file_name = file_path + file_list[each] + file_extension
    temp_pd = load_spark_df(full_file_name).toPandas()
    complete_pd = complete_pd.append(temp_pd, ignore_index=True)
  return complete_pd

# COMMAND ----------

# DBTITLE 1,Workbench - Indexing code that wasn't found to improve groupby speeds
def apply_hierarchical_index(func, *args):
  """
  Decorator function that sets a hierarchical MultiIIndex prior to running a given function.
  Resets the index appropriately before returning the dataframe result.
  """
  
  arguments = list(*args)
  dataframe = arguments.pop(0)
  grouping_vars = arguments.pop(0)
  
  if not is_pandas_df(dataframe):
    raise "Please ensure the first argument in the func call is a pandas dataframe."
  
  
  def wrap(dataframe, grouping_vars, arguments):

    dataframe = ensure_multiindex_exists(dataframe, grouping_vars)
    
    result = func(dataframe, *arguments)

    return result.reset_index()

  return wrap


def check_index_is_set(df, grouping_vars):
  return set(ex_trans_pd.index.names) == set(grouping_vars)

  
def is_default_index(df):
  return df.index.names == [None]


def reset_index_if_not_default(df):
  if not is_default_index(df):
    df.reset_index(inplace=True)
    
  return df


def set_multiindex(df, grouping_vars=None):
  if not grouping_vars:
    grouping_vars = get_hierarchy()
  else:
    grouping_vars = ensure_is_list(grouping_vars)
    
  return df.set_index(grouping_vars)


def ensure_multiindex_exists(df, grouping_vars=None):
  
  if not grouping_vars:
    grouping_vars = get_hierarchy()
  
  if check_index_is_set(df, grouping_vars):
    return df
  else:
    return set_multiindex(df)


def get_index_levels(pandas_df):
  return np.array(pandas_df.index.levels).shape

# COMMAND ----------

# DBTITLE 1,Usability Functions 
def check_or_init_dropdown_widget(name_str,default,choices):
  """widget_out=check_or_init_dropdown_widget(name_str,default,choices).  Checks if widget named "namestr" is present.  If not, it
  creates a dropdown wiget with choices and default
  :param name_str: (str) Will be the name assigned to a widget if a widget fo that name does no exist
  :param default: (str)  The default choice of the widget assigned if widget does not exist  
  :param choices: (list of strs)  The choices that will appear in dropdown  if widget does not exist  
  :output: The current selected choice of the output and some messaging
  """
  try:
    widget_out=dbutils.widgets.get(name_str)
    print(f'The widget {name_str} is already initialized and set to {widget_out}.\nIf another setting is desired, reset before running any more cells')
  except:
    dbutils.widgets.dropdown(name_str,default,choices)
    print(f'The widget {name_str} is activited and set to the default {default}.\nIf another setting is desired, reset before running any more cells.')

def concat_dbfs_path(*args):
  """takes any number of directories(str) with no extension and
  concatanates with / inbetween where first the in list has the form: 'dbfs:/first_level_dir'. Needs to follow format 'dbfs:/path/to/dir' with no file extension
  NOTE: empty strings inserted into the middle of *args will not add a level to the directory structure, the first and last arguments to *args cannot be empty"""
  full_str=''
  for item in args:
    if full_str!='':
      if full_str[-1]!='/':
        full_str+='/'
          
    full_str+=item

    
  if full_str[-1]=='/':
    dbutils.notebook.exit(f"{full_str} does not follow format 'dbfs:/path/to/file',end with '/'--Bad filepath")

        
  if full_str.find('.')!=-1 or full_str[0:6]!='dbfs:/':
    dbutils.notebook.exit(f"{full_str} does not follow format 'dbfs:/path/to/file' with no file extension,--bad filepath")
    
  return full_str    


def copy_table_from_delta(source_path,target_path,version_num='',label='NA'):
  """status_str=copy_table_from_delta(source_path,target_path,version_num='',label='NA')
  copys one version from a delta table to another path.  version_num defaults to most recent version.  
  Adds some info from source old delta table to userMetadata.
  :param source_path: (str) Path of source delta table
  :param target_path: (str) Path of target delta table (or can be path of desired new delta table)
  :param version_num: (int)(optional) Desired version number, defaults to the most recent
  :param label: (str)(optional) Any additional labeling to be added to the userMetaDate
  :output: (str) Sucess or Failure"""
  split='#'*140
  some_delta=DeltaTable.forPath(spark,source_path)
  #get delta history version
  delta_hist_df=some_delta.history().toPandas()
  versions=delta_hist_df['version'].astype(int).tolist()
  
  #check if inputed version exists
  if type(version_num)!=str:
    try:
      version_row=versions.index(version_num)
    except:
      print(f'{version_num} not a valid version num, versions={versions[-1]} to {versions[0]}')
      return 'Failure'
  else:
    version_row=0
    version_num=versions[0]
  #get the delta_info from the desired version
  row=delta_hist_df.iloc[version_row].to_dict()

  #Compile relevant delta_info
  user_name=row.get('userName','Unknown')
  time_stamp=row.get('timestamp','Unknown')
  opt_metrics=row.get('operationMetrics','Unknown')
  if opt_metrics=='Unknown':
    rows='Unknown'
  else:
    rows=opt_metrics.get('numOutputRows','Unknown')
  
  UserMetaStr=f'Version:{version_num}|User:{user_name}|Timestamp:{time_stamp}|rows:{rows}|label:{label}'
  
  print(f'Reading version {version_num} from {source_path}')
  copy_df=spark.read.format("delta").option("versionAsOf", version_num).load(source_path)
  print(split)
  print(f'Adding UserMeta data:\n{UserMetaStr}')
  copy_df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").option('userMetadata',UserMetaStr).save(target_path)
  print(split)
  print(f'Saved in delta table at {target_path}')
  
  return 'Success'



  