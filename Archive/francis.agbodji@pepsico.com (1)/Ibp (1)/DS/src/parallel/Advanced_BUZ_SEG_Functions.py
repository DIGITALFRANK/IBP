# Databricks notebook source
def calc_desc_agg(df, cv_var, report_level = None):
    """
    Calculates Mean of variable at specified level of aggregation

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
      Dataset at report_level with associated mean
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

    #Perform Mean
   
    df_mean = df.groupBy(report_level).agg((mean(col(cv_var))).alias("MEAN")).select(report_level+["MEAN"])
    df_std = df.groupBy(report_level).agg((stddev(col(cv_var))).alias("STD")).select(report_level+["STD"])
    df_min = df.groupBy(report_level).agg((min(col(cv_var))).alias("MIN")).select(report_level+["MIN"])
    df_max = df.groupBy(report_level).agg((max(col(cv_var))).alias("MAX")).select(report_level+["MAX"])
    
    df_agg=df_mean.join(df_std, on=report_level, how="left")\
                  .join(df_min, on=report_level, how="left")\
                  .join(df_max, on=report_level, how="left")

    
    return(df_agg)


# COMMAND ----------

# def calc_STD(df, cv_var, report_level = None):
#     """
#     Calculates STD of variable at specified level of aggregation

#     Parameters
#     ----------
#     df : PySpark dataframe
#       Input dataset
#     cv_var : String
#       Variable
#     report_level : List
#       Level of aggregation to generate output

#     Returns
#     -------
#     df : PySpark dataframe
#       Dataset at report_level with associated mean
#     """
#     #Check report level
#     if isinstance(report_level, str):
#         report_level = [report_level]
#     if report_level == None:
#         report_level = ["MODEL_ID"]
#     if len(intersect_two_lists(report_level,df.columns)) == 0:
#         return(None)

#     #Check variables exists in data
#     if len(intersect_two_lists([cv_var],df.columns)) == 0:
#         return(None)

#     #Perform STD
   
#     #df = df.groupBy(report_level).agg((mean(col(cv_var))).alias("MEAN"))
#     df = df.groupBy(report_level).agg((stddev(col(cv_var))).alias("STD"))
#     #df = df.groupBy(report_level).agg((min(col(cv_var))).alias("MIN"))
#     #df = df.groupBy(report_level).agg((max(col(cv_var))).alias("MAX"))

    
#     return(df)


# COMMAND ----------

# def calc_MIN(df, cv_var, report_level = None):
#     """
#     Calculates MIN of variable at specified level of aggregation

#     Parameters
#     ----------
#     df : PySpark dataframe
#       Input dataset
#     cv_var : String
#       Variable
#     report_level : List
#       Level of aggregation to generate output

#     Returns
#     -------
#     df : PySpark dataframe
#       Dataset at report_level with associated mean
#     """
#     #Check report level
#     if isinstance(report_level, str):
#         report_level = [report_level]
#     if report_level == None:
#         report_level = ["MODEL_ID"]
#     if len(intersect_two_lists(report_level,df.columns)) == 0:
#         return(None)

#     #Check variables exists in data
#     if len(intersect_two_lists([cv_var],df.columns)) == 0:
#         return(None)

#     #Perform MIN
   
#     #df = df.groupBy(report_level).agg((mean(col(cv_var))).alias("MEAN"))
#     #df = df.groupBy(report_level).agg((stddev(col(cv_var))).alias("STD"))
#     df = df.groupBy(report_level).agg((min(col(cv_var))).alias("MIN"))
#     #df = df.groupBy(report_level).agg((max(col(cv_var))).alias("MAX"))

    
#     return(df)


# COMMAND ----------

# def calc_MAX(df, cv_var, report_level = None):
#     """
#     Calculates MAX of variable at specified level of aggregation

#     Parameters
#     ----------
#     df : PySpark dataframe
#       Input dataset
#     cv_var : String
#       Variable
#     report_level : List
#       Level of aggregation to generate output

#     Returns
#     -------
#     df : PySpark dataframe
#       Dataset at report_level with associated mean
#     """
#     #Check report level
#     if isinstance(report_level, str):
#         report_level = [report_level]
#     if report_level == None:
#         report_level = ["MODEL_ID"]
#     if len(intersect_two_lists(report_level,df.columns)) == 0:
#         return(None)

#     #Check variables exists in data
#     if len(intersect_two_lists([cv_var],df.columns)) == 0:
#         return(None)

#     #Perform MAX
   
#     #df = df.groupBy(report_level).agg((mean(col(cv_var))).alias("MEAN"))
#     #df = df.groupBy(report_level).agg((stddev(col(cv_var))).alias("STD"))
#     #df = df.groupBy(report_level).agg((min(col(cv_var))).alias("MIN"))
#     df = df.groupBy(report_level).agg((max(col(cv_var))).alias("MAX"))

    
#     return(df)


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

def compute_quantiles(df, col, quantiles):
  quantiles = sorted(quantiles)

  # 1. compute percentile
  df = df.withColumn("percentile", F.percent_rank().over(Window.orderBy(col)))

  # 2. categorize quantile based on the desired quantile and compute errors
  df = df.withColumn("percentile_cat1", F.lit(-1.0))
  df = df.withColumn("percentile_err1", F.lit(-1.0))
  df = df.withColumn("percentile_cat2", F.lit(-1.0))
  df = df.withColumn("percentile_err2", F.lit(-1.0))

  # check percentile with the lower boundaries
  for idx in range(0, len(quantiles)-1):
    q = quantiles[idx]
    df = df.withColumn("percentile_cat1", F\
                       .when( (F.col("percentile_cat1") == -1.0) & 
                             (F.col("percentile") <= q), q)\
                       .otherwise(F.col("percentile_cat1")))
    df = df.withColumn("percentile_err1", F\
                       .when( (F.col("percentile_err1") == -1.0) & 
                             (F.col("percentile") <= q), 
                             F.pow(F.col("percentile") - q, 2))\
                       .otherwise(F.col("percentile_err1")))

  # assign the remaining -1 values in the error to the largest squared error of 1
  df = df.withColumn("percentile_err1", F\
                     .when(F.col("percentile_err1") == -1.0, 1)\
                     .otherwise(F.col("percentile_err1")))

  # check percentile with the upper boundaries
  for idx in range(1, len(quantiles)):
    q = quantiles[idx]
    df = df.withColumn("percentile_cat2", F\
                       .when((F.col("percentile_cat2") == -1.0) & 
                             (F.col("percentile") <= q), q)\
                       .otherwise(F.col("percentile_cat2")))
    df = df.withColumn("percentile_err2",F\
                       .when((F.col("percentile_err2") == -1.0) & 
                             (F.col("percentile") <= q), 
                             F.pow(F.col("percentile") - q, 2))\
                       .otherwise(F.col("percentile_err2")))

  # assign the remaining -1 values in the error to the largest squared error of 1
  df = df.withColumn("percentile_err2", F\
                     .when(F.col("percentile_err2") == -1.0, 1)\
                     .otherwise(F.col("percentile_err2")))

  # select the nearest quantile to the percentile
  df = df.withColumn("percentile_cat", F\
                     .when(F.col("percentile_err1") < F.col("percentile_err2"), 
                           F.col("percentile_cat1"))\
                     .otherwise(F.col("percentile_cat2")))
  df = df.withColumn("percentile_err", F\
                     .when(F.col("percentile_err1") < F.col("percentile_err2"), 
                           F.col("percentile_err1"))\
                     .otherwise(F.col("percentile_err2")))

  # 3. approximate quantile values by choosing the value with the lowest error at each percentile category
  df = df.withColumn("approx_quantile", F\
                     .first(col).over(Window\
                                      .partitionBy("percentile_cat")\
                                      .orderBy(F.asc("percentile_err"))))

  return df

def extract_quantiles(df):
  df_quantiles = df.select("percentile_cat", "approx_quantile").distinct()
  rows = df_quantiles.collect()
  quantile_values = [ row.approx_quantile for row in rows ]

  return quantile_values