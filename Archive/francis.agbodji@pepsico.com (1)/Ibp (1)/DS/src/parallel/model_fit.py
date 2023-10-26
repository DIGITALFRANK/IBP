# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

#Model goodness-of-fit (GOF).  Contains function around model accuracy, baseline/incremental fit (phantom spikes, etc.)

def calculate_gof(df, report_level, metric_dict):
  """
  Wrapper function to calculate goodness of fit metrics for pricing/promo models

  Parameters
  ----------
  metric_dict : Nested dictionary
      Outer Dictionary contains metric name (key)
      Inner Dictionary contains function to call, and parameters
      Example:
      gof_nested_dict = {'OOS_Mape':
                          {'error_func' : calculate_wtd_mape,
                           'parameters': [test_dat,"actual","prediction",["MODEL_ID"]]},
                         'IS_Mape':
                          {'error_func' : calculate_wtd_mape,
                           'parameters': [train_dat,"actual","prediction",["MODEL_ID"]]}
                        }

  Returns
  -------
  metric_out : PySpark dataframe
      Goodness-of-fit report at specified report level
  """
  metric_out = df.select(report_level).distinct()
  for metric_name in metric_dict.keys():
      #Get parameters
      params = metric_dict[metric_name].get("parameters")
      error_func = metric_dict[metric_name].get("error_func")

      #Run error metric and rename
      metric = error_func(*params)
      idx = len(metric.columns)-1
      oldColumns = metric.schema.names
      metric = metric.withColumnRenamed(oldColumns[idx], metric_name)
      metric_out = metric_out.join(metric, on=report_level, how="left")
  return(metric_out)

def calculate_line_item_bias(df, actual_var, pred_var, report_level = None):
    """
    Calculates bias at the line item level (for faster processing than aggregating)

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated line item bias
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check actual/pred exists in data
    if len(intersect_two_lists([actual_var],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([pred_var],df.columns)) == 0:
        return(None)

    #Perform wtd. bias calc
    df = df.withColumn(("BIAS"), (col(pred_var)-col(actual_var))/col(actual_var))
    df = df.select(report_level + ["BIAS"])
    return calculate_wtd_mapedf
  
def calculate_line_item_error(df, actual_var, pred_var, report_level = None):
    """
    Calculates error at the line item level (for faster processing than aggregating)

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated line item error
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check actual/pred exists in data
    if len(intersect_two_lists([actual_var],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([pred_var],df.columns)) == 0:
        return(None)

    #Perform wtd. bias calc
    df = df.withColumn(("error"), (col(pred_var)-col(actual_var)))
    df = df.select(report_level + ["error"])
    return(df)
  
def calculate_line_item_accuracy(df, actual_var, pred_var, report_level = None):
    """
    Calculates accuracy at the line item level (for faster processing than aggregating)

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated line item Accuracies's
    """
    df = df.withColumn(("MAPE"), abs((col(actual_var)-col(pred_var))/col(actual_var)))
    df = df.withColumn(("ACC"), 1 - abs(col("MAPE")))
    df = df.drop(col("MAPE"))
    df = df.select(report_level + ["ACC"])
    return(df)
  
def calculate_line_item_abs_error(df, actual_var, pred_var, report_level = None):
    """
    Calculates absolute error at the line item level (for faster processing than aggregating)

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated line item absolute error
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check actual/pred exists in data
    if len(intersect_two_lists([actual_var],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([pred_var],df.columns)) == 0:
        return(None)

    #Perform wtd. bias calc
    df = df.withColumn(("abs_error"), abs(col(pred_var)-col(actual_var)))
    df = df.select(report_level + ["abs_error"])
    return(df)

  
def calculate_wtd_bias(df, actual_var, pred_var, report_level = None):
    """
    Calculates weighted bias at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated weighted bias
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check actual/pred exists in data
    if len(intersect_two_lists([actual_var],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([pred_var],df.columns)) == 0:
        return(None)

    #Perform wtd. bias calc
    df = df.withColumn(("BIAS"), (col(pred_var)-col(actual_var))/col(actual_var))
    df = df.groupBy(report_level).agg((sum(col("BIAS") * col(actual_var))/sum(col(actual_var))).alias("WTD_BIAS"))
    return(df)
  
def calculate_wtd_mape(df, actual_var, pred_var, report_level = None):
    """
    Calculates weighted mape at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated weighted MAPE's
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check actual/pred exists in data
    if len(intersect_two_lists([actual_var],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([pred_var],df.columns)) == 0:
        return(None)

    #Perform wtd. mape calc
    df = df.withColumn(("MAPE"), F.abs( ( F.col(actual_var) - F.col(pred_var) ) / F.col(actual_var) ))
    df = df.groupBy(report_level).agg(( F.sum( F.col("MAPE") * F.col(actual_var)) / F.sum( F.col(actual_var) ) ).alias("WTD_MAPE"))
    return df

def calculate_wtd_accuracy(df, actual_var, pred_var, report_level = None):
    """
    Calculates weighted accuracy at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual sales value
    pred_var : String
      Predicted sales value
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated weighted Accuracies's
    """
    df = calculate_wtd_mape(df, actual_var, pred_var, report_level)
    if df == None:
        return(None)
    df = df.withColumn(("WTD_ACC"), 1 - F.abs( F.col("WTD_MAPE")) )
    df = df.drop(F.col("WTD_MAPE"))
    return df

def calculate_corr(df, var1, var2, report_level = None):
    """
    Calculates correlation of two variables at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    var1 : String
      Variable 1
    var2 : String
      Variable 2
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated correlation
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check variables exists in data
    if len(intersect_two_lists([var1],df.columns)) == 0:
        return(None)
    if len(intersect_two_lists([var2],df.columns)) == 0:
        return(None)

    #Perform correlation
    df = df.groupBy(report_level).agg((F.corr(col(var1), col(var2))).alias("CORR"))
    return(df)

def calculate_cv(df, cv_var, report_level = None):
    """
    Calculates coefficient of variation of variable at specified level of aggregation

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    cv_var : String
      Variable
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated CV
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Check variables exists in data
    if len(intersect_two_lists([cv_var],df.columns)) == 0:
        return(None)

    #Perform CV
    df = df.groupBy(report_level).agg((stddev(col(cv_var))/mean(col(cv_var))).alias("CV"))
    return(df)

def calculate_confusion_matrix(df, actual_var, pred_var, report_level = None):
    """
    Calculates confusion matrix for two binary variables

    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    actual_var : String
      Actual outcome
    pred_var : String
      Predicted outcome
    report_level : List
      Level of aggregation to generate output

    Returns
    -------
    df : PySpark dataframe
      Dataset at report_level with associated CM
    """
    #Check report level
    if isinstance(report_level, str):
        report_level = [report_level]
    if report_level == None:
        report_level = ["MODEL_ID"]
    if len(intersect_two_lists(report_level,df.columns)) == 0:
        return(None)

    #Calculate outcomes in raw data
    df = df.withColumn("TP", when((col(actual_var)==1)&
                                  (col(pred_var)==1),1).otherwise(0))
    df = df.withColumn("FP", when((col(actual_var)==0)&
                                  (col(pred_var)==1),1).otherwise(0))
    df = df.withColumn("TN", when((col(actual_var)==0)&
                                  (col(pred_var)==0),1).otherwise(0))
    df = df.withColumn("FN", when((col(actual_var)==1)&
                                  (col(pred_var)==0),1).otherwise(0))


    #Perform CV
    df = aggregate_data(df, report_level, ["TP","FP","TN","FN"], [sum,sum,sum,sum])
    new_names = report_level + ["TP","FP","TN","FN"]
    df = df.toDF(*new_names)
    return(df)

def binarize_data(df, input_cols, thresholds, output_cols, gt = True):
    """
    Converts a numeric column into a binary indicator based on threshold.  If gt = True, values greater than threshold will equal 1,
    else values <= threshold will equal 1.  Can handle multiple input columns/thresholds if they are passed in as lists.

    #TO-DO: pyspark.ml.feature 1.4 allows multiple columns for binarizer eliminating need for for loop
    Parameters
    ----------
    df : PySpark dataframe
      Input dataset
    input_col : List
      Non-binarized variable
    threshold : List
      Threshold on which to binarize data
    output_cols : List
      Output column names
    gt : True/False
      True indicates gt should equal 1, False indicates le should equal 1

    Returns
    -------
    df : PySpark dataframe
      Input dataset with appended binarized column
    """

    if isinstance(input_cols, str):
        input_cols = [input_cols]
    if isinstance(output_cols, str):
        output_cols = [output_cols]
    if isinstance(thresholds, float):
        thresholds = [thresholds]

    #Check if variables exists in data
    if len(intersect_two_lists(input_cols,df.columns)) == 0:
        return(None)

    #Binarize data
    i = 0
    for this_var in input_cols:
        binarizer = Binarizer(threshold=thresholds[i], inputCol=this_var, outputCol=output_cols[i])
        df = binarizer.transform(df)
        #If <=, reverse binarizer
        if gt != True:
            df = df.withColumn(output_cols[i], when(col(output_cols[i])==0,1).otherwise(0))
        i = i + 1

    return(df)