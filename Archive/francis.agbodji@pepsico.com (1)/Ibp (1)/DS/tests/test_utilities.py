# Databricks notebook source
#test_get_secrets is commented out as it requires access to a cloud environment with secrets

def test_row_sum_DF():
  "Tests Row-wise sum of PySpark dataframe"
  #Test 1: Correct summation
  dummy_dat = sqlContext.createDataFrame([("2020-10-03" , 100,      200)],
                                          ["DATE",        "Sales1", "Sales2"])
  expected_output = sqlContext.createDataFrame([("2020-10-03" , 100,      200,      300)],
                                                ["DATE",        "Sales1", "Sales2", "result"])
  actual_output = utilities.row_sum_DF(dummy_dat, ["Sales1","Sales2"], "result")#.toPandas()

  assert_df_equality(expected_output, actual_output)

  #Test 2: Correct handling of null (null handling must occur outside function)
  dummy_dat = sqlContext.createDataFrame([("2020-10-03" , 100,       200),
                                          ("2020-10-03" , None,      200)],
                                           ["DATE",       "Sales1", "Sales2"])
  expected_output = sqlContext.createDataFrame([("2020-10-03" , 100,       200,     300),
                                                ("2020-10-03" , None,         200,     None)],
                                                ["DATE",        "Sales1", "Sales2", "result"])
  actual_output = utilities.row_sum_DF(dummy_dat, ["Sales1","Sales2"], "result")
  assert_df_equality(expected_output, actual_output)


def test_convertDFColumnsToList():
    "Tests function that converts dataframe column into distinct list values"
    dummy_dat = sqlContext.createDataFrame([("Prod1",       "2020-10-19" ,  50,       200),
                                      ("Prod1",       "2020-10-26" ,  60,       200),
                                      ("Prod1",       "2020-10-03" ,  70,       200),
                                      ("Prod2",       "2020-10-03" ,  400,      200),
                                      ("Prod2",       "2020-10-26" ,  450,      200),
                                      ("Prod2",       "2020-10-19" ,  500,      200)],
                                      ["ProductCode", "DATE",         "Sales1", "Sales2"]).sort("ProductCode")
    actual_output = utilities.convertDFColumnsToList(dummy_dat, "ProductCode").sort()
    assert (actual_output == ['Prod1', 'Prod2'])

def test_compare_table_keys():
  "Tests comparing keys of two tables"
  #Test 1: Correctly flags up missing from table 2
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200)],["ID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  schema = StructType([StructField("ID", LongType(), True),StructField("Missing_From", StringType(), False)])
  expected_output = sqlContext.createDataFrame([(3, "table2")],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 2: Correctly flags up missing from table 2
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(3, 300)],["ID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  schema = StructType([StructField("ID", LongType(), True),StructField("Missing_From", StringType(), False)])
  expected_output = sqlContext.createDataFrame([(2, "table1")],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 3: Correctly flags up no missing keys
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  assert actual_output.count() == 0

  #Test 4: Check if returns None if keys are missing from table 1
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["SalesID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  assert actual_output == None

  #Test 4: Check if returns None if keys are missing from table 2
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["ID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["SalesID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  assert actual_output == None

  #Test 4: Check if returns None if keys are missing from both tables
  dummy_dat_tbl1 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["SalesID","Sales"])
  dummy_dat_tbl2 = sqlContext.createDataFrame([(1, 100),(2, 200),(3, 300)],["SalesID","Sales"])
  actual_output = utilities.compare_table_keys(dummy_dat_tbl1, dummy_dat_tbl2, ["ID"])
  assert actual_output == None
  #TO-DO: Update function so that it can handle multiple columns

def test_load_parameters():
  "Tests parameter loading"
  #Test 1: Correctly loads parameters
  dummy_dat = sqlContext.createDataFrame([("Config1", "100"),("Config2", "FALSE")],["ConfigName","ConfigValue"])
  actual_output = utilities.load_parameters(dummy_dat, "ConfigName", "ConfigValue")
  assert actual_output.get('Config1') == '100'
  assert actual_output.get('Config2') == 'FALSE'

  #Test 2: Returns none if columns don't exist in table
  actual_output = utilities.load_parameters(dummy_dat, "not_a_config", "ConfigValue")
  assert actual_output == None
  actual_output = utilities.load_parameters(dummy_dat, "ConfigName", "not_a_config")
  assert actual_output == None
  actual_output = utilities.load_parameters(dummy_dat, "not_a_config", "not_a_config")
  assert actual_output == None

  #Test 3: Fails if fields are not strings
  dummy_dat = sqlContext.createDataFrame([("Config1", 100.0),("Config2", 250.0)],["ConfigName","ConfigValue"])
  actual_output = utilities.load_parameters(dummy_dat, "ConfigName", "ConfigValue")
  assert actual_output == None

  dummy_dat = sqlContext.createDataFrame([(250.0, "100"),(350.0, "FALSE")],["ConfigName","ConfigValue"])
  actual_output = utilities.load_parameters(dummy_dat, "ConfigName", "ConfigValue")
  assert actual_output == None

# def test_get_secrets():
#   "Tests secret loading"
#   #Test 1: Correctly loads DEV-UK
#   market = "UK"
#   creds_to_get = {'user': "tpo-demandbrain-" + market + "-user",
#                 'password' : "tpo-demandbrain-" + market + "-password",
#                 'database' : "tpo-demandbrain-" + market + "-database",
#                 'server' : "tpo-demandbrain-sqlserver" }
#   credentials = get_secrets(in_scope = "bieno-da-d-80173-appk-01", secret_dict=creds_to_get)
#
#   #Set globals
#   user = credentials.get("user")
#   password = credentials.get("password")
#   database = credentials.get("database")
#   server = credentials.get("server")
#   jdbcurl = "jdbc:sqlserver://"+server+";database="+database+";user="+user+";password="+password
#
#   actual_output = spark.read.format("jdbc").option("url", jdbcurl.format(database)).option("dbtable", "DBO_test_sql_connect")\
#               .option("user", user).option("password", password).option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver").load()
#   expected_output = sqlContext.createDataFrame([("DEV-UK", 10.0)],["SQL_ENVIRONMENT", "dummycol"])
#   expected_output = expected_output.drop("dummycol")
#   assert_df_equality(actual_output, expected_output)
#
#   #Test 2: Correctly loads DEV-US
#   market = "US"
#   creds_to_get = {'user': "tpo-demandbrain-" + market + "-user",
#                 'password' : "tpo-demandbrain-" + market + "-password",
#                 'database' : "tpo-demandbrain-" + market + "-database",
#                 'server' : "tpo-demandbrain-sqlserver" }
#   credentials = get_secrets(in_scope = "bieno-da-d-80173-appk-01", secret_dict=creds_to_get)
#
#   #Set globals
#   user = credentials.get("user")
#   password = credentials.get("password")
#   database = credentials.get("database")
#   server = credentials.get("server")
#   jdbcurl = "jdbc:sqlserver://"+server+";database="+database+";user="+user+";password="+password
#
#   actual_output = spark.read.format("jdbc").option("url", jdbcurl.format(database)).option("dbtable", "DBO_test_sql_connect")\
#               .option("user", user).option("password", password).option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver").load()
#   expected_output = sqlContext.createDataFrame([("DEV-US", 10.0)],["SQL_ENVIRONMENT", "dummycol"])
#   expected_output = expected_output.drop("dummycol")
#   assert_df_equality(actual_output, expected_output)
#
#   #Test 2: Correctly loads DEV-ID
#   market = "ID"
#   creds_to_get = {'user': "tpo-demandbrain-" + market + "-user",
#                 'password' : "tpo-demandbrain-" + market + "-password",
#                 'database' : "tpo-demandbrain-" + market + "-database",
#                 'server' : "tpo-demandbrain-sqlserver" }
#   credentials = get_secrets(in_scope = "bieno-da-d-80173-appk-01", secret_dict=creds_to_get)
#
#   #Set globals
#   user = credentials.get("user")
#   password = credentials.get("password")
#   database = credentials.get("database")
#   server = credentials.get("server")
#   jdbcurl = "jdbc:sqlserver://"+server+";database="+database+";user="+user+";password="+password
#
#   actual_output = spark.read.format("jdbc").option("url", jdbcurl.format(database)).option("dbtable", "DBO_test_sql_connect")\
#               .option("user", user).option("password", password).option("driver", "com.microsoft.sqlserver.jdbc.SQLServerDriver").load()
#   expected_output = sqlContext.createDataFrame([("DEV-ID", 10.0)],["SQL_ENVIRONMENT", "dummycol"])
#   expected_output = expected_output.drop("dummycol")
#   assert_df_equality(actual_output, expected_output)

def test_change_schema_in_pyspark():
  #Change double to string
  schema = StructType([StructField("long_col", LongType(), True),StructField("change_col", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0)], schema)
  new_output = utilities.change_schema_in_pyspark(dummy_dat, string_cols=["change_col"])
  assert new_output.dtypes[1][1] == 'string'

  #Change string to timestamp
  schema = StructType([StructField("long_col", LongType(), True),StructField("change_col", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "2020-12-11T23:09:58.392+0000")], schema)
  new_output = utilities.change_schema_in_pyspark(dummy_dat, timestamp_cols=["change_col"])
  assert new_output.dtypes[1][1] == 'timestamp'

  #Change string to float
  schema = StructType([StructField("long_col", LongType(), True),StructField("change_col", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "20")], schema)
  new_output = utilities.change_schema_in_pyspark(dummy_dat, float_cols=["change_col"])
  assert new_output.dtypes[1][1] == 'float'

  #Change string to date
  schema = StructType([StructField("long_col", LongType(), True),StructField("change_col", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "2020-12-11")], schema)
  new_output = utilities.change_schema_in_pyspark(dummy_dat, date_cols=["change_col"])
  assert new_output.dtypes[1][1] == 'date'