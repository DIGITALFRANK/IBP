# Databricks notebook source
def test_score_models_using_coefficients():
  #Test 1: Correct scoring
  model_data = sqlContext.createDataFrame([
    ("2020-10-03" , 100, "Walmart", "Ice Cream", "PPG1", 1, .5, 6.5),
    ("2020-10-04" , 200, "Walmart", "Ice Cream", "PPG1", 1, .8, 4.5),
    ("2020-10-05" , 300, "Walmart", "Ice Cream", "PPG1", 1, .99, 3.5),
    ("2020-10-03" , 600, "CVS", "Ice Cream", "PPG1", 1, .5, 6.5),
    ("2020-10-04" , 250, "CVS", "Ice Cream", "PPG1", 1, .8, 4.5),
    ("2020-10-05" , 890, "CVS", "Ice Cream", "PPG1", 1, .99, 3.5)],
  ["Date", "Sales", "Customer", "Category", "PPG", "Intercept", "Distribution", "Price"])

  coef_list = [{'Customer':'Walmart','Intercept':1,'Distribution':1,'Price':1},
             {'Customer':'CVS','Intercept':0,'Price':1}]
  scoring_info_dict = dict(
      model_id     = ["Customer"],
      score_data   = model_data,
      coef_dict    = coef_list,
      predict_key_fields = ['Customer','Date']
  )

  scoring_info_cls = scoring.ScoringInfo(**scoring_info_dict)
  scoring_info_cls.score_models_using_coefficients()
  actual_output = scoring_info_cls.score_data
  actual_output = actual_output.select(["Customer","Date","pred"]).sort(["Customer","DATE"])

  schema = StructType([
      StructField("Customer", StringType(), True),
      StructField("Date", StringType(), True),
      StructField("pred", DoubleType(), True)])
  expected_output = sqlContext.createDataFrame([
      ("CVS", "2020-10-03" , 6.5),
      ("CVS", "2020-10-04" , 4.5),
      ("CVS", "2020-10-05" , 3.5),
      ("Walmart", "2020-10-03" , 8.0),
      ("Walmart", "2020-10-04" , 6.3),
      ("Walmart", "2020-10-05" , 5.49)],schema)
  assert_df_equality(expected_output, actual_output)

def test_do_waterfall():
  #Test 1: Correct scoring
  #Create dummy data
  model_data = sqlContext.createDataFrame([
      ("2020-10-03" , 100, "Walmart", "Ice Cream", "PPG1", 1, .5, 6.5),
      ("2020-10-04" , 200, "Walmart", "Ice Cream", "PPG1", 1, .8, 4.5),
      ("2020-10-05" , 300, "Walmart", "Ice Cream", "PPG1", 1, .99, 3.5),
      ("2020-10-03" , 600, "CVS", "Ice Cream", "PPG1", 1, .5, 6.5),
      ("2020-10-04" , 250, "CVS", "Ice Cream", "PPG1", 1, .8, 4.5),
      ("2020-10-05" , 890, "CVS", "Ice Cream", "PPG1", 1, .99, 3.5)],
    ["DATE", "Sales", "Customer", "Category", "PPG", "Intercept", "Distribution", "Price"])

  coef_list = [{'Customer':'Walmart','Intercept':1,'Distribution':1,'Price':1},
             {'Customer':'CVS','Intercept':0,'Price':1}
            ]
  levels = {'Distrib_Effect':['Distribution'],'Price_Effect':['Price']}
  #Setup scoring information
  scoring_info_dict = dict(
      model_id     = ["Customer"],
      score_data   = model_data,
      coef_dict    = coef_list,
      predict_key_fields = ['Customer','DATE'],
      date_field = ['DATE'],
      waterfall_levels = levels
  )
  scoring_info_cls = scoring.ScoringInfo(**scoring_info_dict)
  actual_output = scoring_info_cls.score_models_using_coefficients()

  expected_output = sqlContext.createDataFrame([
      ("CVS", "2020-10-03" , 6.5),
      ("CVS", "2020-10-04" , 4.5),
      ("CVS", "2020-10-05" , 3.5),
      ("Walmart", "2020-10-03" , 8),
      ("Walmart", "2020-10-04" , 6.3),
      ("Walmart", "2020-10-05" , 5.49)],
    ["Customer","DATE", "pred"])
  assert_df_equality(expected_output, actual_output.select(["Customer","Date","pred"]).sort(["Customer","DATE"]))