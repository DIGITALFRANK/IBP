# Databricks notebook source
def test_calculate_wtd_mape():
  #Test 1: Correct wtd. mape calculation
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 200.0,  300.0),
            ("Walmart10001", "2020-10-02",300.0, 400.0),
            ("Walmart10002", "2020-10-01", 300.0,  300.0),
            ("Walmart10002", "2020-10-02",600.0, 400.0)],
            ["MODEL_ID",  "RDATE", "actual", "prediction"])

  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "actual", "prediction", ["MODEL_ID"]).sort("MODEL_ID")

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", 0.4),
              ("Walmart10002", 0.2)],
              ["MODEL_ID",  "WTD_MAPE"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Check string report level still works
  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "actual", "prediction", "MODEL_ID").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Check empty report level still works
  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "actual", "prediction").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 4: Check missing report level column returns None
  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "actual", "prediction", "notacol")
  assert actual_output == None

  #Test 5: Check missing actual column returns None
  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "notacol", "prediction", "MODEL_ID")
  assert actual_output == None

  #Test 6: Check missing predicted column returns None
  actual_output = model_fit.calculate_wtd_mape(dummy_dat, "actual", "notacol", "MODEL_ID")
  assert actual_output == None

def test_calculate_wtd_accuracy():
  #Test 1: Correct wtd. accuracy calculation
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 200.0,  300.0),
            ("Walmart10001", "2020-10-02",300.0, 400.0),
            ("Walmart10002", "2020-10-01", 300.0,  300.0),
            ("Walmart10002", "2020-10-02",600.0, 400.0)],
            ["MODEL_ID",  "RDATE", "actual", "prediction"])

  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "actual", "prediction", ["MODEL_ID"]).sort("MODEL_ID")

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", 0.6),
              ("Walmart10002", 0.78)],
              ["MODEL_ID",  "WTD_ACC"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: Check string report level still works
  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "actual", "prediction", "MODEL_ID").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: Check empty report level still works
  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "actual", "prediction").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 4: Check missing report level column returns None
  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "actual", "prediction", "notacol")
  assert actual_output == None

  #Test 5: Check missing actual column returns None
  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "notacol", "prediction", "MODEL_ID")
  assert actual_output == None

  #Test 6: Check missing predicted column returns None
  actual_output = model_fit.calculate_wtd_accuracy(dummy_dat, "actual", "notacol", "MODEL_ID")
  assert actual_output == None

def test_calculate_corr():
  #Test 1: Correct correlation
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 155.0,  300.0),
            ("Walmart10001", "2020-10-02",300.0, 0.0),
            ("Walmart10001", "2020-10-03",20.0, 75.0),
            ("Walmart10001", "2020-10-04",19.0, 32.0),
            ("Walmart10002", "2020-10-01", 300.0,  300.0),
            ("Walmart10002", "2020-10-02",600.0, 400.0)],
            ["MODEL_ID",  "RDATE", "actual", "prediction"])

  actual_output = model_fit.calculate_corr(dummy_dat, "actual", "prediction", ["MODEL_ID"]).sort("MODEL_ID")

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", -0.03),
              ("Walmart10002", 1.00)],
              ["MODEL_ID",  "CORR"])
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 2: Check string report level still works
  actual_output = model_fit.calculate_corr(dummy_dat, "actual", "prediction", "MODEL_ID").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 3: Check empty report level still works
  actual_output = model_fit.calculate_corr(dummy_dat, "actual", "prediction").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 4: Check missing report level column returns None
  actual_output = model_fit.calculate_corr(dummy_dat, "actual", "prediction", "notacol")
  assert actual_output == None

  #Test 5: Check missing actual column returns None
  actual_output = model_fit.calculate_corr(dummy_dat, "notacol", "prediction", "MODEL_ID")
  assert actual_output == None

  #Test 6: Check missing predicted column returns None
  actual_output = model_fit.calculate_corr(dummy_dat, "actual", "notacol", "MODEL_ID")
  assert actual_output == None

  #TO-DO: Check weird correlation returns (e.g., infinity, etc.)

def test_calculate_cv():
  #Test 1: Correct CV
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 300.0),
            ("Walmart10001", "2020-10-02", 0.0),
            ("Walmart10001", "2020-10-03", 75.0),
            ("Walmart10001", "2020-10-04", 32.0),
            ("Walmart10002", "2020-10-01", 300.0),
            ("Walmart10002", "2020-10-02", 400.0)],
            ["MODEL_ID",  "RDATE", "prediction"])

  actual_output = model_fit.calculate_cv(dummy_dat, "prediction", ["MODEL_ID"]).sort("MODEL_ID")

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", 1.33),
              ("Walmart10002", 0.20)],
              ["MODEL_ID",  "CV"])
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 2: Check string report level still works
  actual_output = model_fit.calculate_cv(dummy_dat, "prediction", "MODEL_ID").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 3: Check empty report level still works
  actual_output = model_fit.calculate_cv(dummy_dat, "prediction").sort("MODEL_ID")
  assert_approx_df_equality(expected_output, actual_output, 0.01)

  #Test 4: Check missing report level column returns None
  actual_output = model_fit.calculate_cv(dummy_dat, "prediction", "notacol")
  assert actual_output == None

  #Test 5: Check missing variable returns None
  actual_output = model_fit.calculate_corr(dummy_dat, "notacol", "MODEL_ID")
  assert actual_output == None

def test_calculate_gof():
  #Test 1: Correct error function call
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 200.0,  300.0),
            ("Walmart10001", "2020-10-02",300.0, 400.0),
            ("Walmart10002", "2020-10-01", 300.0,  300.0),
            ("Walmart10002", "2020-10-02",600.0, 400.0)],
            ["MODEL_ID",  "RDATE", "actual", "prediction"])
  metric_dict = {'OOS_Acc':
                    {'error_func' : model_fit.calculate_wtd_mape,
                     'parameters': [dummy_dat,"actual","prediction",["MODEL_ID"]]}
                   }
  actual_output = model_fit.calculate_gof(dummy_dat, ["MODEL_ID"], metric_dict).sort(["MODEL_ID"])

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", 0.4),
              ("Walmart10002", 0.2)],
              ["MODEL_ID",  "OOS_Acc"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test it properly filters data based on dictionary params
  dummy_dat = sqlContext.createDataFrame([
            ("Walmart10001", "2020-10-01", 0, 200.0,  300.0),
            ("Walmart10001", "2020-10-02", 1, 300.0, 400.0),
            ("Walmart10002", "2020-10-01", 0, 300.0,  300.0),
            ("Walmart10002", "2020-10-02", 1, 600.0, 400.0)],
            ["MODEL_ID",  "RDATE", "TRAIN_IND","actual", "prediction"])
  test_dat = dummy_dat.filter("TRAIN_IND==0")
  train_dat = dummy_dat.filter("TRAIN_IND==1")
  metric_dict = {'OOS_Acc':
                    {'error_func' : model_fit.calculate_wtd_mape,
                     'parameters': [test_dat,"actual","prediction",["MODEL_ID"]]},
                 'IS_Acc':
                    {'error_func' : model_fit.calculate_wtd_mape,
                     'parameters': [train_dat,"actual","prediction",["MODEL_ID"]]}
                   }
  actual_output = model_fit.calculate_gof(dummy_dat, ["MODEL_ID"], metric_dict).sort(["MODEL_ID"])

  expected_output = sqlContext.createDataFrame([
              ("Walmart10001", 0.5, 0.3),
              ("Walmart10002", 0.0, 0.3)],
              ["MODEL_ID",  "OOS_Acc", "IS_Acc"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_binarize_data():
  #Test 1: Correct > Binarizer
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-10-01", 0.8)], ["MODEL_ID",  "RDATE", "mape"])
  actual_output = model_fit.binarize_data(dummy_dat, "mape", 0.75 ,"binarized_mape",True).sort(["MODEL_ID"])
  actual_output = utilities.convertDFColumnsToList(actual_output.select("binarized_mape"),"binarized_mape")
  assert actual_output == [1.0]

  #Test 2: Correct < Binarizer
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-10-01", 0.7)], ["MODEL_ID",  "RDATE", "mape"])
  actual_output = model_fit.binarize_data(dummy_dat, "mape", 0.75 ,"binarized_mape",False).sort(["MODEL_ID"])
  actual_output = utilities.convertDFColumnsToList(actual_output.select("binarized_mape"),"binarized_mape")
  assert actual_output == [1.0]

  #Test 3: Correct <= Binarizer
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-10-01", 0.75)], ["MODEL_ID",  "RDATE", "mape"])
  actual_output = model_fit.binarize_data(dummy_dat, "mape", 0.75 ,"binarized_mape",False).sort(["MODEL_ID"])
  actual_output = utilities.convertDFColumnsToList(actual_output.select("binarized_mape"),"binarized_mape")
  assert actual_output == [1.0]

  #Test 4: Correct empty column
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-10-01", 0.75)], ["MODEL_ID",  "RDATE", "mape"])
  actual_output = model_fit.binarize_data(dummy_dat, "accuracy", 0.75 ,"binarized_mape",True)
  assert actual_output == None

  #Test 5: Can handle multiple columns
  dummy_dat = sqlContext.createDataFrame([("Walmart10001", "2020-10-01", 0.8, 0.8)], ["MODEL_ID",  "RDATE", "mape", "acc"])
  actual_output = model_fit.binarize_data(dummy_dat, [ "mape", "acc"], [0.75, 0.9] ,["binarized_mape","binarized_acc"],True).sort(["MODEL_ID"])
  assert_output = utilities.convertDFColumnsToList(actual_output,["binarized_mape"])
  assert assert_output == [1.0]
  assert_output = utilities.convertDFColumnsToList(actual_output,["binarized_acc"])
  assert assert_output == [0.0]