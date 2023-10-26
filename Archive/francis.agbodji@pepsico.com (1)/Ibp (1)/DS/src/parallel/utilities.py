# Databricks notebook source
from delta.tables import DeltaTable

# COMMAND ----------

def get_target_path(targetContainer, targetStorageAccount, targetPath):
  """
  Returns bronze / silver / gold target paths for PEP
  """
  tgtPath = "abfss://"+targetContainer+"@"+targetStorageAccount+".dfs.core.windows.net/"+targetPath
  return(tgtPath)

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

def convert_df_pick_to_dict(df, key_col, pick_col, encoding):
    "Converts Dataframe containing a key and value (pickled object) to a key/unpickled object"
    out_dict = {}
    for i in df.toPandas()[key_col].unique():
        this_pick = df.filter(col(key_col)==i)
        this_key = this_pick.select(col(key_col)).collect()[0]
        this_pick = this_pick.select(col(pick_col)).collect()[0]
        this_pick = this_pick[pick_col].encode(encoding)
        this_pick = pickle.loads(this_pick)
        out_dict[i] = this_pick
    return(out_dict)
  
def convert_dict_tuple_to_int(in_dict ,list_to_convert):
  " Converts float tuple elements into an integer for a nested dictionary "
  
  for item in in_dict.keys():
    for key, value in in_dict.get(item).items():
      if key in list_to_convert:
        res = in_dict.get(item).get(key)
        res = [int(tup) for tup in res]
        in_dict[item][key] = res
        
  return(in_dict)

def convert_dict_vals_to_int(in_dict ,list_to_convert):
  " Converts float elements into an integer for a nested dictionary "
  for item in in_dict.keys():
    for key, value in in_dict.get(item).items():
      if key in list_to_convert:
        res = in_dict.get(item).get(key)
        res = int(res)
        in_dict[item][key] = res
        
  return(in_dict)

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

#List utilities
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

#Delta tables
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
def load_csv_to_df(path, sql_context):
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


def run_scope_and_key():
    tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
    client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
    client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
    client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
    storage_account = "cdodevadls2"

    storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
    spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
    spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
                   "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
    spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
    spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
    #spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization","true")
  

def load_data(dir_path,data_format,sep):
  """
  Load Shipment/Product/Customer/Location/Price data 

  Parameters
  ----------
  dir_path : String
    path of the table to be loaded 
  data_format : String
    e.g. csv
  sep : String
    e.g. ";"
  """
  if dir_path:
    df=spark.read.load(path=dir_path,format=data_format, sep=sep, inferSchema="true", header="true")
  else:
    print("Data Load Failed!")
  return df


def dedupe_two_lists(list1, list2):
    "De-dupes two lists"
    return(list(set(list1 + list2)))


# COMMAND ----------

# Function to derive top n correlated hols
def top_n_correlation(corr_dict, n):
  top_n_hols = [k for (k,v) in dict(sorted(corr_dict.items(), key=lambda x: x[1], reverse = True)[:n]).items()]
  hols_to_drop = list(corr_dict.keys() - top_n_hols)
  return top_n_hols, hols_to_drop

def blank_as_null(x):
    return when(col(x) != "", col(x)).otherwise(None)

# COMMAND ----------

# Function to read in files
def read_file(file_location):
  return (spark.read.format("csv").option("inferSchema", "false").option("header", "true").option("sep", ",").load(file_location))

# performs addition of period + time_var, while handling year change
def get_maxforcast(start_date, period, forcast_periods):
  '''
  start_date: Date in YYYYMM/YYYYWW format for month/week
  period: periods to add to start_date
  forcast_periods: list of all TIME_VAR
  '''
  forcast_periods = [x for x in sorted(forcast_periods) if x>=start_date]
  
  if period>len(forcast_periods):
    return forcast_periods[-1]
  else:
    return forcast_periods[period-1]
  
  
# This method gives the next Week_OF_Year or Month_Of_Year bases on no_of_frst. 
# i.e If TIME_VAR is Month_Of_Year, date_var = 202011 and no_of_frst= 2, the function will return 202101.
def next_time_var(time_var,date_var,no_of_frst,calendar_df,CALENDAR_DATEVAR):
  if time_var =='Month_Of_Year':
    date_var = str(date_var)
    date_var = date_var+('01')
    date_time =[]
    date_time.append(date_var)
    df = spark.createDataFrame(date_time, "string").toDF("date")
    df = df.withColumn('date', to_timestamp(df.date, "yyyyMMdd"))
    df = df.withColumn('date', F.add_months(df.date, no_of_frst))
    df = df.withColumn('year',year(col('date')).cast('string'))\
    .withColumn('month',month(col('date')).cast('string'))
    df = df.withColumn('Month_Of_Year',when(F.length(col('month'))==1,F.concat(F.col('year'),lit('0'),F.col('month')))\
                   .otherwise(F.concat(F.col('year'),F.col('month'))).cast("int"))
  else:
    date_var = str(date_var)
    date_time =[]
    date_time.append(date_var)
    df = spark.createDataFrame(date_time, "string").toDF("date")
    df= df.join(calendar_df,df.date==calendar_df.Week_Of_Year,how='inner')
    df = df.withColumn(CALENDAR_DATEVAR, date_add(col(CALENDAR_DATEVAR), no_of_frst*7)).select(col(CALENDAR_DATEVAR))
    df= df.join(calendar_df,df.Week_start_date==calendar_df.Week_start_date,how='inner')
  returnvar = df.select(time_var).collect()[0][0]
  
  return returnvar