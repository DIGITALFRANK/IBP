# Databricks notebook source
def test_impute_to_value():
  "Tests imputation of NULL values in column to a specific value"
  #Test 1: Correct imputation
  dummy_dat = sqlContext.createDataFrame([("Prod1",      "2020-10-19" , 50,       None),
                                          ("Prod2",      "2020-10-26" , None,     200)],
                                          ["ProductCode","DATE",        "Sales1", "Sales2"])
  expected_output = sqlContext.createDataFrame([("Prod1",       "2020-10-19" , 50,       750),
                                                ("Prod2",       "2020-10-26" , 750,      200)],
                                                ["ProductCode", "DATE",        "Sales1", "Sales2"])
  actual_output = data_cleansing.impute_to_value(dummy_dat, ["Sales1","Sales2"], 750)
  assert_df_equality(expected_output, actual_output)

def test_impute_to_col():
  "Tests imputation of NULL values to different columns in a dataset"
  #Test 1: Correct imputation
  dummy_dat = sqlContext.createDataFrame([("Prod1",      "2020-10-19" , 50,       None,      0,          100),
                                          ("Prod2",      "2020-10-26" , None,     200,       0,          300)],
                                          ["ProductCode","DATE",        "Sales1", "Sales2",  "newSales1","newSales2"])
  expected_output = sqlContext.createDataFrame([("Prod1",      "2020-10-19" , 50,       100,       0,          100),
                                                ("Prod2",      "2020-10-26" , 0,        200,       0,          300)],
                                                ["ProductCode","DATE",        "Sales1", "Sales2", "newSales1","newSales2"])
  actual_output = data_cleansing.impute_to_col(dummy_dat, ["Sales1","Sales2"], ["newSales1","newSales2"])
  assert_df_equality(expected_output, actual_output)

def test_filter_values():
  "Tests function that filters based on specified criterion data works properly"
  filter_dict = {'end_of_life_flag': 'end_of_life_flag  == 0',
               'sales': 'sales >= 50'}

  #Test 1: Proper filtering of data
  dummy_dat = sqlContext.createDataFrame([
  (1, 0, 49.9),
  (2, 0, 99.9),
  (3, 1, 99.9),
  (4, 1, 99.9),
  (5, 0, 99.9),
  (6, 0, 99.9)],
  ["ID",  "end_of_life_flag", "sales"])

  actual_output = data_cleansing.filter_values(dummy_dat, filter_dict).sort("ID")
  expected_output = sqlContext.createDataFrame([
    (2, 0, 99.9),
    (5, 0, 99.9),
    (6, 0, 99.9)],
    ["ID",  "end_of_life_flag", "sales"])
  assert_df_equality(expected_output, actual_output)

  #Test 2: Doesn't error out if column doesn't exist
  filter_dict = {'i_dont_exist_in_data': 'i_dont_exist_in_data  == 0'}
  actual_output = data_cleansing.filter_values(dummy_dat, filter_dict).sort("ID")
  assert_df_equality(dummy_dat, actual_output)

def test_get_date():
  "Test various date types get converted to standardized RDATE field"
  #Test 1: dd-MM-yyy format
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "02-02-2018")], ["MODEL_ID",    "DATE"])
  actual_output = data_cleansing.get_date(dummy_dat, "DATE", "dd-MM-yyyy", "RDATE")
  expected_output = sqlContext.createDataFrame([("Walmart10001", "02-02-2018","2018-02-02")], ["MODEL_ID","DATE","RDATE"])
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 2: yyyy-MM-dd format
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2018-02-02")], ["MODEL_ID",    "DATE"])
  actual_output = data_cleansing.get_date(dummy_dat, "DATE", "yyyy-MM-dd", "RDATE")
  expected_output = sqlContext.createDataFrame([("Walmart10001", "2018-02-02","2018-02-02")], ["MODEL_ID","DATE","RDATE"])
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 3: MM/dd/yyy format
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "11/25/1991")], ["MODEL_ID",    "DATE"])
  actual_output = data_cleansing.get_date(dummy_dat, "DATE", "MM/dd/yyy", "RDATE")
  expected_output = sqlContext.createDataFrame([("Walmart10001", "11/25/1991","1991-11-25")], ["MODEL_ID","DATE","RDATE"])
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 4: Test it works if column is already a date
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2018-02-02")], ["MODEL_ID",    "DATE"])
  dummy_dat = dummy_dat.withColumn("DATE", dummy_dat["DATE"].cast(DateType()))
  actual_output = data_cleansing.get_date(dummy_dat, "DATE", "yyyy-MM-dd", "RDATE")
  expected_output = sqlContext.createDataFrame([("Walmart10001", "2018-02-02","2018-02-02")], ["MODEL_ID","DATE","RDATE"])
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  expected_output = expected_output.withColumn("DATE", expected_output["DATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 5: Test it works if column is timestamp
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-08-13 21:39:51.880")], ["MODEL_ID",    "DATE"])
  dummy_dat = dummy_dat.withColumn("DATETS",to_timestamp(col("DATE")))
  actual_output = data_cleansing.get_date(dummy_dat, "DATETS", "yyyy-MM-dd", "RDATE")
  expected_output = sqlContext.createDataFrame([("Walmart10001", "2020-08-13 21:39:51.880","2020-08-13")], ["MODEL_ID","DATE","RDATE"])
  expected_output = expected_output.withColumn("DATETS",to_timestamp(col("DATE")))
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  expected_output = expected_output.select("MODEL_ID","DATE","DATETS","RDATE")
  assert_df_equality(expected_output, actual_output)

  #Test 6: If RDATE already exist's it doesn't error out

def test_bin_categorical_vars():
  #Test 1: Ensure proper binning for one variable
  dummy_dat = sqlContext.createDataFrame([
  (1, "SUPC"),
  (2, "SUPC~SUPC"),
  (3, "BOGO")],
  ["ID",  "Promo_Type"])
  bin_dict = {'SUPC':'SUPC_BIN', 'SUPC~SUPC':'SUPC_BIN', 'BOGO':'BOGO'}

  actual_output = data_cleansing.bin_categorical_vars(dummy_dat, "Promo_Type", bin_dict).sort("ID")
  expected_output = sqlContext.createDataFrame([
    (1, "SUPC_BIN"),
    (2, "SUPC_BIN"),
    (3, "BOGO")],
  ["ID",  "Promo_Type"])
  assert_df_equality(expected_output, actual_output)

  #Test 2: Missing bin type doesn't error out
  bin_dict = {'SUPC':'SUPC_BIN', 'BOGO':'BOGO'}
  actual_output = data_cleansing.bin_categorical_vars(dummy_dat, "Promo_Type", bin_dict).sort("ID")
  expected_output = sqlContext.createDataFrame([
    (1, "SUPC_BIN"),
    (2, None),
    (3, "BOGO")],
  ["ID",  "Promo_Type"])
  assert_df_equality(expected_output, actual_output)

  #Test 3: Missing variable doesn't error out
  actual_output = data_cleansing.bin_categorical_vars(dummy_dat, "Feature_Type", bin_dict).sort("ID")
  assert_df_equality(dummy_dat, actual_output)

  #Test 4: Can handle NULL's

def test_impute_cols_ts():
  #Test 1: backward fill works properly
  schema = StructType([
        StructField("ID", LongType(), True),
        StructField("MODEL_ID", StringType(), True),
        StructField("RDATE", StringType(), True),
        StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,None),
  (2, "Walmart10001", '2020-10-17' ,None),
  (3, "Walmart10001", '2020-10-18' ,500.99),
  (4, "Walmart10002", '2020-10-16' ,650.99),
  (5, "Walmart10002", '2020-10-17' ,None),
  (6, "Walmart10002", '2020-10-18' ,300.6)],schema)
  dummy_dat = dummy_dat.withColumn("RDATE", dummy_dat["RDATE"].cast(DateType()))

  actual_output = data_cleansing.impute_cols_ts(dummy_dat, ["Sales"], "bfill").sort("ID")
  expected_output = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99),
  (2, "Walmart10001", '2020-10-17' ,500.99),
  (3, "Walmart10001", '2020-10-18' ,500.99),
  (4, "Walmart10002", '2020-10-16' ,650.99),
  (5, "Walmart10002", '2020-10-17' ,300.6),
  (6, "Walmart10002", '2020-10-18' ,300.6)],schema)
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))

  assert_df_equality(expected_output, actual_output)

  #Test 2: forward fill works properly
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("MODEL_ID", StringType(), True),
      StructField("RDATE", StringType(), True),
      StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99),
  (2, "Walmart10001", '2020-10-17' ,None),
  (3, "Walmart10001", '2020-10-18' ,600.99),
  (4, "Walmart10002", '2020-10-16' ,650.99),
  (5, "Walmart10002", '2020-10-17' ,700.99),
  (6, "Walmart10002", '2020-10-18' ,None)],schema)
  dummy_dat = dummy_dat.withColumn("RDATE", dummy_dat["RDATE"].cast(DateType()))

  actual_output = data_cleansing.impute_cols_ts(dummy_dat, ["Sales"], "ffill").sort("ID")
  expected_output = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99),
  (2, "Walmart10001", '2020-10-17' ,500.99),
  (3, "Walmart10001", '2020-10-18' ,600.99),
  (4, "Walmart10002", '2020-10-16' ,650.99),
  (5, "Walmart10002", '2020-10-17' ,700.99),
  (6, "Walmart10002", '2020-10-18' ,700.99)],schema)
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))

  assert_df_equality(expected_output, actual_output)

  #Test 3: Multiple variables work at a time
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("MODEL_ID", StringType(), True),
      StructField("RDATE", StringType(), True),
      StructField("Sales", DoubleType(), True),
      StructField("Price", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99, 200.99),
  (2, "Walmart10001", '2020-10-17' ,None, None),
  (3, "Walmart10001", '2020-10-18' ,600.99, None)],schema)
  dummy_dat = dummy_dat.withColumn("RDATE", dummy_dat["RDATE"].cast(DateType()))

  actual_output = data_cleansing.impute_cols_ts(dummy_dat, ["Sales","Price"], "ffill").sort("ID")
  expected_output = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99, 200.99),
  (2, "Walmart10001", '2020-10-17' ,500.99, 200.99),
  (3, "Walmart10001", '2020-10-18' ,600.99, 200.99)],schema)
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))

  assert_df_equality(expected_output, actual_output)

  #Test 4: Doesn't error out if column doesn't exist
  actual_output = data_cleansing.impute_cols_ts(dummy_dat, ["DummyCol"], "ffill").sort("ID")
  expected_output = sqlContext.createDataFrame([
  (1, "Walmart10001", '2020-10-16' ,500.99, 200.99),
  (2, "Walmart10001", '2020-10-17' ,500.99, 200.99),
  (3, "Walmart10001", '2020-10-18' ,600.99, 200.99)],schema)
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))

  assert_df_equality(dummy_dat, actual_output)

def test_log_columns():
  #Test 1: logging works properly
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0)], schema)
  actual_output = data_cleansing.log_columns(dummy_dat, ["Sales"]).sort("ID")
  expected_output = sqlContext.createDataFrame([(1, 5.86)], schema)

  #Test 2: Works with 0's
  dummy_dat = sqlContext.createDataFrame([(1, 0.0)], schema)
  actual_output = data_cleansing.log_columns(dummy_dat, ["Sales"]).sort("ID")
  expected_output = sqlContext.createDataFrame([(1, 0.0)],schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Works with negatives
  dummy_dat = sqlContext.createDataFrame([(1, -1.0)], schema)
  actual_output = data_cleansing.log_columns(dummy_dat, ["Sales"]).sort("ID")
  expected_output = sqlContext.createDataFrame([(1, None)],schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 4: Works with multiple columns
  schema = StructType([
        StructField("ID", LongType(), True),
        StructField("Sales", DoubleType(), True),
        StructField("Price", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0, 5.9)], ["ID", "Sales", "Price"])
  actual_output = data_cleansing.log_columns(dummy_dat, ["Sales","Price"]).sort("ID")
  expected_output = sqlContext.createDataFrame([(1, 5.86, 1.93)],schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 5: Doesn't error out if column doesn't exist
  actual_output = data_cleansing.log_columns(dummy_dat, ["EmptyCol"]).sort("ID")
  assert_approx_df_equality(dummy_dat, actual_output, 0.1)

  #Test 6: Doesn't error out if it's not numeric
  dummy_dat = sqlContext.createDataFrame([(1, "hello")], ["ID", "DummyCol"])
  actual_output = data_cleansing.log_columns(dummy_dat, ["Sales","Price"]).sort("ID")
  assert_approx_df_equality(dummy_dat, actual_output, 0.1)

def test_align_week_start_date():
  #Test 1: Properly aligns to Monday (IRI)
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("DATE", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, '2020-10-07')], schema)
  dummy_dat = dummy_dat.withColumn("DATE", dummy_dat["DATE"].cast(DateType()))
  actual_output = data_cleansing.align_week_start_date(dummy_dat, "DATE","Mon").sort("ID")
  expected_output = sqlContext.createDataFrame([(1, '2020-10-05')], schema)
  expected_output = expected_output.withColumn("DATE", expected_output["DATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 2: Properly aligns to Sunday (Nielsen)
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("DATE", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, '2020-10-07')], schema)
  dummy_dat = dummy_dat.withColumn("DATE", dummy_dat["DATE"].cast(DateType()))
  actual_output = data_cleansing.align_week_start_date(dummy_dat, "DATE","Sun").sort("ID")
  expected_output = sqlContext.createDataFrame([(1, '2020-10-04')], schema)
  expected_output = expected_output.withColumn("DATE", expected_output["DATE"].cast(DateType()))
  assert_df_equality(expected_output, actual_output)

  #Test 3: Properly aligns to same day when the date is already aligned
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("DATE", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, '2020-10-04')], schema)
  dummy_dat = dummy_dat.withColumn("DATE", dummy_dat["DATE"].cast(DateType()))
  actual_output = data_cleansing.align_week_start_date(dummy_dat, "DATE","Sun").sort("ID")
  expected_output = sqlContext.createDataFrame([(1, '2020-10-04')], schema)
  expected_output = expected_output.withColumn("DATE", expected_output["DATE"].cast(DateType()))

def test_aggregate_data():
  #Test 1: Test it properly aggregates data by group
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99),
                                          (2, "Walmart10001",100.99),
                                          (3, "Walmart10002",1.0)], schema)
  actual_output = data_cleansing.aggregate_data(dummy_dat,["MODEL_ID"],["Sales"],[max]).sort("MODEL_ID")
  schema = StructType([
            StructField("MODEL_ID", StringType(), True),
            StructField("max_Sales", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([("Walmart10001",500.99),
                                                ("Walmart10002",1.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Can handle multiple columns
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True),
          StructField("Price", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99, 2.99),
                                          (2, "Walmart10001",100.99, -50.0),
                                          (3, "Walmart10002",1.0, 30.0)], schema)
  actual_output = data_cleansing.aggregate_data(dummy_dat,["MODEL_ID"],["Sales","Price"],[max]).sort("MODEL_ID")
  schema = StructType([
            StructField("MODEL_ID", StringType(), True),
            StructField("max_Sales", DoubleType(), True),
            StructField("max_Price", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([("Walmart10001",500.99, 2.99),
                                                ("Walmart10002",1.0, 30.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Can handle multiple statistics
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True),
          StructField("Price", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99, 2.99),
                                          (2, "Walmart10001",100.99, -50.0),
                                          (3, "Walmart10002",1.0, 30.0)], schema)
  actual_output = data_cleansing.aggregate_data(dummy_dat,["MODEL_ID"],["Sales","Price"],[max,sum]).sort("MODEL_ID")
  schema = StructType([
            StructField("MODEL_ID", StringType(), True),
            StructField("max_Sales", DoubleType(), True),
            StructField("max_Price", DoubleType(), True),
            StructField("sum_Sales", DoubleType(), True),
            StructField("sum_Price", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([("Walmart10001",500.99, 2.99, 601.98, -47.01),
                                                ("Walmart10002",1.0, 30.0, 1.0, 30.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_get_cumsum():
  #Test 1: Test it properly performs cum sum
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Customer", StringType(), True),
          StructField("Cat", StringType(), True),
          StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001","Walmart","IceCream",500.99),
                                          (2, "Walmart10001","Walmart","IceCream",100.99),
                                          (3, "Walmart10002","Walmart","Condiments",200.0),
                                          (4, "CVS10001","CVS","Condiments",300.99),
                                          (5, "CVS10001","CVS","Condiments",150.99),
                                          (6, "CVS10002","CVS","Condiments",50.0)], schema)
  actual_output = data_cleansing.get_cumsum(dummy_dat,["MODEL_ID"],["Customer"],"Sales").sort("MODEL_ID")
  actual_output = actual_output.select(["MODEL_ID","cum_pct"])
  schema = StructType([
            StructField("MODEL_ID", StringType(), True),
    StructField("cum_pct", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([("CVS10001",0.90),
                                                ("CVS10002",1.0),
                                               ("Walmart10001",0.75),
                                                ("Walmart10002",1.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #TO-DO: Add more robust tests