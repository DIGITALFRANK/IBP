# Databricks notebook source
def test_get_disallocation_prop_tdfp():
  lower_forecast = sqlContext.createDataFrame([
              (20201003 , 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 100, "Walmart", "Ice Cream", "PPG2")],
            ["DATE", "FCST", "Customer", "Category", "PPG"])

  #Test function outputs correct disaggregation
  actual_output = get_disallocation_prop_tdfp(lower_forecast, ["Customer", "Category"], ["Customer", "Category", "PPG"], "FCST", "DATE")

  expected_output = sqlContext.createDataFrame([
                ("PPG1" , "Ice Cream", "Walmart", 20201003,.04),
                ("PPG2" , "Ice Cream", "Walmart", 20201003,.96),
                ("PPG1" , "Ice Cream", "Walmart",  20201004,.23),
                ("PPG2" , "Ice Cream", "Walmart", 20201004,.77),
                ("PPG1" , "Ice Cream", "Walmart",  20201005,.91),
                ("PPG2" , "Ice Cream", "Walmart", 20201005,.09)],
              ["PPG", "Category", "Customer", "DATE","prop"])


  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_get_disallocation_prop_tdgsa():
  disagg_data = sqlContext.createDataFrame([
              (20201003 , 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 100, "Walmart", "Ice Cream", "PPG2"),
              (20201003 , 600, "CVS", "Ice Cream", "PPG1"),
              (20201004 , 500, "CVS", "Ice Cream", "PPG1"),
              (20201005 , 400, "CVS", "Ice Cream", "PPG1"),
              (20201003 , 300, "CVS", "Ice Cream", "PPG2"),
              (20201004 , 200, "CVS", "Ice Cream", "PPG2"),
              (20201005 , 100, "CVS", "Ice Cream", "PPG2")],
            ["DATE", "Sales", "Customer", "Category", "PPG"])

  #Test outputs are as expected
  actual_output = get_disallocation_prop_tdgsa(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 3)
  expected_output = sqlContext.createDataFrame([
                  ("PPG1" , "Ice Cream", "Walmart", .39),
                  ("PPG2" , "Ice Cream", "Walmart", .61),
                  ("PPG1" , "Ice Cream", "CVS", .73),
                  ("PPG2" , "Ice Cream", "CVS", .27)],
                ["PPG", "Category", "Customer", "prop"]).select(["PPG","Category","Customer","prop"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test function filters to correct dates
  actual_output = get_disallocation_prop_tdgsa(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 2)
  expected_output = sqlContext.createDataFrame([
                ("PPG1" , "Ice Cream", "Walmart", .57),
                ("PPG2" , "Ice Cream", "Walmart", .43),
                ("PPG1" , "Ice Cream", "CVS", .76),
                ("PPG2" , "Ice Cream", "CVS", .25)],
              ["PPG", "Category", "Customer", "prop"]).select(["PPG","Category","Customer","prop"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

def test_get_disallocation_prop_tdgsf():
  disagg_data = sqlContext.createDataFrame([
              (20201003 , 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 100, "Walmart", "Ice Cream", "PPG2"),
              (20201003 , 600, "CVS", "Ice Cream", "PPG1"),
              (20201004 , 500, "CVS", "Ice Cream", "PPG1"),
              (20201005 , 400, "CVS", "Ice Cream", "PPG1"),
              (20201003 , 300, "CVS", "Ice Cream", "PPG2"),
              (20201004 , 200, "CVS", "Ice Cream", "PPG2"),
              (20201005 , 100, "CVS", "Ice Cream", "PPG2")],
            ["DATE", "Sales", "Customer", "Category", "PPG"])

  #Test function outputs correct disaggregation
  actual_output = get_disallocation_prop_tdgsf(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 3)
  expected_output = sqlContext.createDataFrame([
                ("PPG1" , "Ice Cream", "Walmart", .56),
                ("PPG2" , "Ice Cream", "Walmart", .43),
                ("PPG1" , "Ice Cream", "CVS", .71),
                ("PPG2" , "Ice Cream", "CVS", .28)],
              ["PPG", "Category", "Customer", "prop"]).select(["PPG","Category","Customer","prop"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test function filters to correct dates
  actual_output = get_disallocation_prop_tdgsf(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 2)
  expected_output = sqlContext.createDataFrame([
                ("PPG1" , "Ice Cream", "Walmart", .75),
                ("PPG2" , "Ice Cream", "Walmart", .25),
                ("PPG1" , "Ice Cream", "CVS", .75),
                ("PPG2" , "Ice Cream", "CVS", .25)],
              ["PPG", "Category", "Customer", "prop"]).select(["PPG","Category","Customer","prop"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)


def test_disaggregate_forecast():
  #Test 1: forecast proportions
  lower_forecast = sqlContext.createDataFrame([
              (20201003 , 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 100, "Walmart", "Ice Cream", "PPG2")],
            ["DATE", "FCST", "Customer", "Category", "PPG"])

  upper_forecast = sqlContext.createDataFrame([
                (20201003 , 400, "Walmart", "Ice Cream"),
                (20201004 , 300, "Walmart", "Ice Cream"),
                (20201005 , 1000, "Walmart", "Ice Cream")],
              ["DATE", "FCST", "Customer", "Category"])

  #Test outputs are as expected
  props = get_disallocation_prop_tdfp(lower_forecast, ["Customer", "Category"], ["Customer", "Category", "PPG"], "FCST", "DATE")
  actual_output = disaggregate_forecast(lower_forecast, upper_forecast, props ,
                                 ["Customer", "Category"], ["Customer", "Category","PPG"],
                                 "FCST").orderBy(["DATE","Customer", "Category","PPG"])\
                                  .filter(col("DATE")==20201003)
  expected_output = sqlContext.createDataFrame([
                (20201003 , 20, "Walmart", "Ice Cream", "PPG1",15.4),
                (20201003 , 500, "Walmart", "Ice Cream", "PPG2",384.6)],
              ["DATE", "FCST", "Customer", "Category", "PPG","ALLOC_DEMAND"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 2: tdgsa
  lower_forecast = sqlContext.createDataFrame([
              (20201003 , 25, 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 80, 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1005,1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 550, 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 300, 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 200, 100, "Walmart", "Ice Cream", "PPG2")],
            ["DATE", "Sales","FCST", "Customer", "Category", "PPG"])

  upper_forecast = sqlContext.createDataFrame([
                (20201003 , 400, "Walmart", "Ice Cream"),
                (20201004 , 300, "Walmart", "Ice Cream"),
                (20201005 , 1000, "Walmart", "Ice Cream")],
              ["DATE", "FCST", "Customer", "Category"])

  #Test outputs are as expected
  props = get_disallocation_prop_tdgsa(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 3)
  actual_output = disaggregate_forecast(lower_forecast, upper_forecast, props ,
                                 ["Customer", "Category"], ["Customer", "Category","PPG"],
                                 "FCST").orderBy(["DATE","Customer", "Category","PPG"])\
                                  .filter(col("DATE")==20201003)
  expected_output = sqlContext.createDataFrame([
                (20201003 , 25, 20, "Walmart", "Ice Cream", "PPG1",157.1),
                (20201003 , 550, 500, "Walmart", "Ice Cream", "PPG2",242.9)],
              ["DATE", "Sales","FCST", "Customer", "Category", "PPG","ALLOC_DEMAND"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)

  #Test 3: tdgsf
  lower_forecast = sqlContext.createDataFrame([
              (20201003 , 25, 20, "Walmart", "Ice Cream", "PPG1"),
              (20201004 , 80, 75, "Walmart", "Ice Cream", "PPG1"),
              (20201005 , 1005,1000, "Walmart", "Ice Cream", "PPG1"),
              (20201003 , 550, 500, "Walmart", "Ice Cream", "PPG2"),
              (20201004 , 300, 250, "Walmart", "Ice Cream", "PPG2"),
              (20201005 , 200, 100, "Walmart", "Ice Cream", "PPG2")],
            ["DATE", "Sales","FCST", "Customer", "Category", "PPG"])

  upper_forecast = sqlContext.createDataFrame([
                (20201003 , 400, "Walmart", "Ice Cream"),
                (20201004 , 300, "Walmart", "Ice Cream"),
                (20201005 , 1000, "Walmart", "Ice Cream")],
              ["DATE", "FCST", "Customer", "Category"])

  #Test outputs are as expected
  props = get_disallocation_prop_tdgsf(disagg_data, ["Customer"], ["Customer", "Category", "PPG"], "Sales", "DATE", 3)
  actual_output = disaggregate_forecast(lower_forecast, upper_forecast, props ,
                                 ["Customer", "Category"], ["Customer", "Category","PPG"],
                                 "FCST").orderBy(["DATE","Customer", "Category","PPG"])\
                                  .filter(col("DATE")==20201003)
  expected_output = sqlContext.createDataFrame([
                (20201003 , 25, 20, "Walmart", "Ice Cream", "PPG1",225.2),
                (20201003 , 550, 500, "Walmart", "Ice Cream", "PPG2",174.8)],
              ["DATE", "Sales","FCST", "Customer", "Category", "PPG","ALLOC_DEMAND"])
  assert_approx_df_equality(expected_output, actual_output, 0.1)