# Databricks notebook source
def test_do_lags_N():
  dummy_dat = sqlContext.createDataFrame([("Prod1",       "2020-10-19" ,  50,       200),
                                          ("Prod1",       "2020-10-26" ,  60,       200),
                                          ("Prod1",       "2020-10-03" ,  70,       200),
                                          ("Prod2",       "2020-10-03" ,  400,      200),
                                          ("Prod2",       "2020-10-26" ,  450,      200),
                                          ("Prod2",       "2020-10-19" ,  500,      200)],
                                          ["ProductCode", "DATE",         "Sales1", "Sales2"])

  #Test 1: Correct lead
  actual_output = feature_engineering.do_lags_N(dummy_dat,["DATE"],["Sales1"],1,["ProductCode"]).sort("ProductCode")
  schema = StructType([
            StructField("ProductCode", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("Sales1", LongType(), True),
            StructField("Sales2", LongType(), True),
            StructField("Sales1_lead1", LongType(), True)])
  expected_output = sqlContext.createDataFrame([("Prod1",       "2020-10-03" , 70,        200,     None),
                                                ("Prod1",       "2020-10-19" , 50,        200,     70),
                                                ("Prod1",       "2020-10-26" , 60,        200,     50),
                                                ("Prod2",       "2020-10-03" , 400,       200,     None),
                                                ("Prod2",       "2020-10-19" , 500,       200,     400),
                                                ("Prod2",       "2020-10-26" , 450,       200,     500)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 2: Correct lag
  actual_output = feature_engineering.do_lags_N(dummy_dat,["DATE"],["Sales1"],-2,["ProductCode"]).sort("ProductCode")
  schema = StructType([
            StructField("ProductCode", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("Sales1", LongType(), True),
            StructField("Sales2", LongType(), True),
            StructField("Sales1_lag2", LongType(), True)])
  expected_output = sqlContext.createDataFrame([("Prod1",      "2020-10-03" , 70,       200,     None),
                                              ("Prod1",      "2020-10-19" , 50,       200,     None),
                                              ("Prod1",      "2020-10-26" , 60,       200,     70),
                                              ("Prod2",      "2020-10-03" , 400,      200,     None),
                                              ("Prod2",      "2020-10-19" , 500,      200,     None),
                                              ("Prod2",      "2020-10-26" , 450,      200,     400)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 3: Correctly transforms string inputs into list
  actual_output = feature_engineering.do_lags_N(dummy_dat,"DATE",["Sales1"],-2,"ProductCode").sort("ProductCode")
  schema = StructType([
            StructField("ProductCode", StringType(), True),
            StructField("DATE", StringType(), True),
            StructField("Sales1", LongType(), True),
            StructField("Sales2", LongType(), True),
            StructField("Sales1_lag2", LongType(), True)])
  expected_output = sqlContext.createDataFrame([("Prod1",      "2020-10-03" , 70,       200,     None),
                                              ("Prod1",      "2020-10-19" , 50,       200,     None),
                                              ("Prod1",      "2020-10-26" , 60,       200,     70),
                                              ("Prod2",      "2020-10-03" , 400,      200,     None),
                                              ("Prod2",      "2020-10-19" , 500,      200,     None),
                                              ("Prod2",      "2020-10-26" , 450,      200,     400)],schema)
  assert_df_equality(expected_output, actual_output)

def test_get_dummies():
  "Tests dummy variable creation"

  #Test 1: Can handle multiple columns
  dummy_dat = sqlContext.createDataFrame([
  (1, "SUPC", "Feature"),
  (2, "SUPC", "Feature"),
  (3, "SUPC", "Feature"),
  (4, "2FAP", "Feature"),
  (5, "2FAP", "Feature"),
  (6, "2FAP", "NoFeature")],
  ["ID",  "Promo_Type", "Feature_Type"])

  actual_output = feature_engineering.get_dummies(dummy_dat, ["Promo_Type","Feature_Type"]).sort("ID")
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("Promo_Type", StringType(), True),
      StructField("Feature_Type", StringType(), True),
      StructField("Promo_Type_SUPC", IntegerType(), False),
      StructField("Promo_Type_2FAP", IntegerType(), False),
      StructField("Feature_Type_NoFeature", IntegerType(), False),
      StructField("Feature_Type_Feature", IntegerType(), False)])
  expected_output = sqlContext.createDataFrame([
    (1, "SUPC", "Feature",1,0,0,1),
    (2, "SUPC", "Feature",1,0,0,1),
    (3, "SUPC", "Feature",1,0,0,1),
    (4, "2FAP", "Feature",0,1,0,1),
    (5, "2FAP", "Feature",0,1,0,1),
    (6, "2FAP", "NoFeature",0,1,1,0)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 2: Can handle spaces in string
  dummy_dat = sqlContext.createDataFrame([(1, "SUPC No")], ["ID",  "Promo_Type"])
  actual_output = feature_engineering.get_dummies(dummy_dat, ["Promo_Type"]).sort("ID")
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("Promo_Type", StringType(), True),
      StructField("Promo_Type_SUPC_No", IntegerType(), False)])
  expected_output = sqlContext.createDataFrame([(1, "SUPC No",1)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 3: Can handle None/null values
  dummy_dat = sqlContext.createDataFrame([(1, "SUPC"),(2, None)], ["ID",  "Promo_Type"])
  actual_output = feature_engineering.get_dummies(dummy_dat, ["Promo_Type"]).sort("ID")
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("Promo_Type", StringType(), True),
      StructField("Promo_Type_SUPC", IntegerType(), False),
      StructField("Promo_Type_NULL", IntegerType(), False)])
  expected_output = sqlContext.createDataFrame([(1, "SUPC",1,0),(2, None,0,1)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 4: Can handle non-string values
  dummy_dat = sqlContext.createDataFrame([(1, 3),(2, 5)], ["ID",  "WeekFactor"])
  actual_output = feature_engineering.get_dummies(dummy_dat, ["WeekFactor"]).sort("ID")
  schema = StructType([
      StructField("ID", LongType(), True),
      StructField("WeekFactor", LongType(), True),
      StructField("WeekFactor_5", IntegerType(), False),
      StructField("WeekFactor_3", IntegerType(), False)])
  expected_output = sqlContext.createDataFrame([(1, 3, 0, 1),(2, 5, 1, 0)],schema)
  assert_df_equality(expected_output, actual_output)

  #Test 5: Can handle missing columns
  dummy_dat = sqlContext.createDataFrame([(1, "SUPC")], ["ID",  "Promo_Type"])
  actual_output = feature_engineering.get_dummies(dummy_dat, ["Display_Type"]).sort("ID")
  assert_df_equality(dummy_dat, actual_output)

  #Test 6: Can handle delimeter
  dummy_dat = sqlContext.createDataFrame([(1, "SUPC~SUPC"),(2,'BOGO'), (3,'BOGO~SUPC')], ["ID",  "Promo_Type"])
  actual_output = feature_engineering.get_dummies(dummy_dat, ["Promo_Type"], delim='~').sort("ID")
  schema = StructType([
        StructField("ID", LongType(), True),
        StructField("Promo_Type", StringType(), True),
        StructField("Promo_Type_BOGO", IntegerType(), False),
        StructField("Promo_Type_SUPC", IntegerType(), False)])
  expected_output = sqlContext.createDataFrame([(1, "SUPC~SUPC",0,1),(2,'BOGO',1,0), (3,'BOGO~SUPC',1,1)], schema)
  assert_df_equality(expected_output, actual_output)

def test_get_time_vars():
  "Tests time series variable creation"
  #Test 1: Works properly when RDATE is a date
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("RDATE", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, '2020-10-16')], schema)
  dummy_dat = dummy_dat.withColumn("RDATE", dummy_dat["RDATE"].cast(DateType()))

  actual_output = feature_engineering.get_time_vars(dummy_dat).sort("ID")
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("RDATE", StringType(), True),
          StructField("YearIndex", IntegerType(), True),
          StructField("MonthIndex", IntegerType(), True),
          StructField("WeekIndex", IntegerType(), True),
          StructField("WeekInMonth", LongType(), True),
          StructField("WeekOfYear", IntegerType(), True),
          StructField("Quarter", IntegerType(), True),
          StructField("QuarterYear", StringType(), True),
          StructField("LinearTrend", IntegerType(), True),
          StructField("CosX", DoubleType(), True),
          StructField("SinX", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, '2020-10-16',2020,10,42,3,202042,4,"2020Q4",18551,-.994,0.1044)], schema)
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(DateType()))
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Works properly when RDATE is a string
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("RDATE", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, '2020-10-16')], schema)
  actual_output = feature_engineering.get_time_vars(dummy_dat).sort("ID")
  expected_output = expected_output.withColumn("RDATE", expected_output["RDATE"].cast(StringType()))
  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_get_hierarchical_statistics():
  #Test 1: Calculates statistics properly
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99),
                                          (2, "Walmart10001",100.99),
                                          (3, "Walmart10002",1.0)], schema)
  actual_output = feature_engineering.get_hierarchical_statistics(dummy_dat,["MODEL_ID"],"Sales",[max]).sort(["ID"])
  schema = StructType([
            StructField("ID", LongType(), True),
            StructField("MODEL_ID", StringType(), True),
            StructField("Sales", DoubleType(), True),
            StructField("max_Sales", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, "Walmart10001",500.99, 500.99),
                                                (2, "Walmart10001",100.99, 500.99),
                                                (3, "Walmart10002",1.0, 1.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Doesn't error with non-numeric variables
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", StringType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001","not a number"),
                                          (2, "Walmart10001","not a number 2"),
                                          (3, "Walmart10002","not a number 3")], schema)
  actual_output = feature_engineering.get_hierarchical_statistics(dummy_dat,["MODEL_ID"],"Sales",[kurtosis]).sort(["ID"])
  schema = StructType([
            StructField("ID", LongType(), True),
            StructField("MODEL_ID", StringType(), True),
            StructField("Sales", StringType(), True),
            StructField("kurtosis_Sales", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, "Walmart10001","not a number", None),
                                                (2, "Walmart10001","not a number 2", None),
                                                (3, "Walmart10002","not a number 3", None)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Doesn't error if multiple variables are passed
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True),
          StructField("Price", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99, 0.99),
                                          (2, "Walmart10001",100.99, 6.99),
                                          (3, "Walmart10002",1.0, 3.99)], schema)
  actual_output = feature_engineering.get_hierarchical_statistics(dummy_dat,["MODEL_ID"],["Sales","Price"],[max]).sort(["ID"])
  assert_df_equality(dummy_dat, actual_output)

  #Test 4: Can handle list of multiple hierarchies
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("MODEL_ID", StringType(), True),
          StructField("Sales", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, "Walmart10001",500.99),
                                          (2, "Walmart10001",100.99),
                                          (3, "Walmart10002",1.0)], schema)
  actual_output = feature_engineering.get_hierarchical_statistics(dummy_dat,["MODEL_ID","ID"],"Sales",[max]).sort(["ID"])
  schema = StructType([
            StructField("ID", LongType(), True),
            StructField("MODEL_ID", StringType(), True),
            StructField("Sales", DoubleType(), True),
            StructField("max_Sales", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, "Walmart10001",500.99, 500.99),
                                                (2, "Walmart10001",100.99,100.99),
                                                (3, "Walmart10002",1.0, 1.0)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_calculate_ratio():
  #Test 1: Properly calculates ratio
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("Units", DoubleType(), True),
          StructField("Dollars", DoubleType(), True),
          StructField("Baseline", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0,500.0, 200.0)], schema)
  ratio_dict = {'Pct_Baseline':{'top_var':'Baseline', 'bottom_var': 'Dollars'}}
  actual_output = feature_engineering.calculate_ratio(dummy_dat,ratio_dict)
  schema = StructType([
            StructField("ID", LongType(), True),
            StructField("Units", DoubleType(), True),
            StructField("Dollars", DoubleType(), True),
            StructField("Baseline", DoubleType(), True),
             StructField("Pct_Baseline", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, 350.0,500.0, 200.0, 0.4)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Can handle multiple variables
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("Units", DoubleType(), True),
          StructField("Dollars", DoubleType(), True),
          StructField("Baseline", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0,500.0, 200.0)], schema)
  ratio_dict = {'Pct_Baseline':{'top_var':'Baseline', 'bottom_var': 'Dollars'},
                 'Price':{'top_var':'Dollars', 'bottom_var': 'Units'}}
  actual_output = feature_engineering.calculate_ratio(dummy_dat,ratio_dict)
  schema = StructType([
            StructField("ID", LongType(), True),
            StructField("Units", DoubleType(), True),
            StructField("Dollars", DoubleType(), True),
            StructField("Baseline", DoubleType(), True),
            StructField("Pct_Baseline", DoubleType(), True),
             StructField("Price", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([(1, 350.0,500.0, 200.0, 0.4 ,1.428)], schema)
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Doesn't error if top or bottom variable is missing
  schema = StructType([
          StructField("ID", LongType(), True),
          StructField("Units", DoubleType(), True),
          StructField("Dollars", DoubleType(), True)])
  dummy_dat = sqlContext.createDataFrame([(1, 350.0,500.0)], schema)
  ratio_dict = {'Pct_Baseline':{'top_var':'Baseline', 'bottom_var': 'Dollars'}}
  actual_output = feature_engineering.calculate_ratio(dummy_dat,ratio_dict)
  assert_df_equality(dummy_dat, actual_output)