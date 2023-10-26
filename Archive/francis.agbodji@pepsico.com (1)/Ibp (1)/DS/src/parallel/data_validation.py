# Databricks notebook source
#Performs data validation checks to ensure data is ready for modeling

#
#Input datasets are at right level of aggregation
#Check schema complies
#Input dataset's aren't missing for certain tests (required ML data)
#Detect significant changes in input data from last run (alert)

def check_schema(df, schema_spec):
    """
    Checks whether input data complies with specified schema

    Parameters
    ----------
    df : PySpark DataFrame
    schema_spec : Schema

    Returns
    -------
    True/False : True if schemas match
    """
    if df.schema != schema_spec:
        return(False)
    else:
        return(True)

def change_schema_in_pyspark(df, string_cols = [], float_cols = [],  double_cols = [], timestamp_cols = [], date_cols = []):
  """
  Changes schema of pyspark dataframe.  Only changes columns that are passed in.
  """
  for col_name in string_cols:
    df = df.withColumn(col_name, col(col_name).cast('string'))
  for col_name in float_cols:
    df = df.withColumn(col_name, col(col_name).cast('float'))
  for col_name in double_cols:
    df = df.withColumn(col_name, col(col_name).cast('double'))
  for col_name in timestamp_cols:
    df = df.withColumn(col_name, col(col_name).cast('timestamp'))
  for col_name in date_cols:
    df = df.withColumn(col_name, col(col_name).cast('date'))
  return df

def validate_pct_missing(df, group_vars, vars_to_summarize):
    nacounts = df.groupBy(group_vars).agg(*((sum(col(c).isNull().cast("int"))/count(col(c))).alias(c) for c in vars_to_summarize))
    return (nacounts)

def validate_pct_positive(df, group_vars, vars_to_summarize):
    pct_rate = df.groupBy(group_vars).agg(*((sum(col(c))/count(col(c))).alias(c) for c in vars_to_summarize))
    return (pct_rate)

def compare_table_keys(table_1, table_2, key_fields):
  """
  Compares key fields between two tables and returns mismatches

  Parameters
  ----------
  table_1 : PySpark dataframe
      Table 1
  table_2 : PySpark dataframe
      Table 2
  key_fields : List
      Key fields between two tables

  Returns
  -------
  df : PySpark dataframe
      Missing keys from both tables
  """

  if isinstance(key_fields, str):
      key_fields = [key_fields]

  #Check if keys exist in tables
  if len(intersect_two_lists(key_fields,table_1.columns)) == 0:
    return(None)

  if len(intersect_two_lists(key_fields,table_2.columns)) == 0:
    return(None)

  #Determine if there are additional or missing key field rows#
  table1_keys = table_1.select(key_fields)
  table2_keys = table_2.select(key_fields)

  #Missing from Table 1
  missing_tbl1_keys = table1_keys.join(table2_keys, on = key_fields, how="left_anti")
  missing_tbl1_keys = missing_tbl1_keys.withColumn("Missing_From",lit("table2"))

  #Missing from in_table
  missing_tbl2_keys = table2_keys.join(table1_keys, on = key_fields, how="left_anti")
  missing_tbl2_keys = missing_tbl2_keys.withColumn("Missing_From",lit("table1"))

  missing_keys = missing_tbl2_keys.union(missing_tbl1_keys)

  return(missing_keys)

def compare_table_fields(table1_pandas, table2_pandas, key_fields, compare_fields=None):
  """
  Compares differences in all fields across two pandas dataframes

  Parameters
  ----------
  table_1 : Pandas dataframe
      Table 1
  table_2 : Pandas dataframe
      Table 2
  key_fields : List
      List of common key fields between the tables

  Returns
  -------
  comparison : Pandas dataframe
      Pandas dataset that contains discrepancies between the two tables
  """

  if isinstance(key_fields, str):
      key_fields = [key_fields]
  if isinstance(compare_fields, str):
      compare_fields = [compare_fields]

  if compare_fields:
    table1_pandas = table1_pandas[[key_fields + compare_fields]]
    table2_pandas = table2_pandas[[key_fields + compare_fields]]

  diff_configs = table1_pandas.compare(table2_pandas, keep_equal = True, align_axis = 1)
  ids = table1_pandas.iloc[list(diff_configs.index)] #Keep ID's for different rows
  ids = ids[key_fields] #Keep ID columns
  comparison = ids.join(diff_configs)

  return(comparison)

def remove_element_from_tuple(in_tuple, substring):
    """
    Removes elements that contain a substring from a tuple

    Parameters
    ----------
    in_tuple : List
      List of tuples. E.g., [('Subband','Gain'),('Mhz','')]
    substring : String
      Substring for which you want to remove all elements in the tuple

    Returns
    -------
    nested_lst_of_tuples : List
      List of tuples. E.g., [('Subband',''),('Mhz','')]
    """

    #Convert tuples to list
    tuple_list = [list(tup) if isinstance(tup, tuple) else tup
             for tup in list(in_tuple)]
    #Remove substrings
    i = 0
    for element in tuple_list:
        if len(element) > 1 and substring in tuple_list[i]:
            tuple_list[i].remove(substring)
        i = i+1

    #Convert back to list of tuples
    nested_lst_of_tuples = [tuple(l) for l in tuple_list]
    return(nested_lst_of_tuples)

def extract_data_to_col(df, old_var, new_var, key_words):
    #Extract data into new column
    df[new_var] = df[old_var]
    #df[new_var] = df[new_var].map('_'.join)
    for i in key_words:
      df[new_var] = pd.np.where(df[new_var].str.contains(i),
                             i,
                             df[new_var])
    return(df)
  
def remove_data_from_col(df, key_words,old_var):
    df[old_var] = df[old_var]   #.map('_'.join) #Note: this is required only for tuples
    for j in key_words:
        df[old_var] = df[old_var].str.replace(r'_'+j, '')
    return(df)