# Databricks notebook source
#Contains functions that disaggregate an aggregated forecast
def get_disallocation_prop_tdgsf(df, upper_hier, lower_hier, demand_var, time_var, time_horizon):
  """
  Determines proportions to output a higher level forecast into a lower level forecast using the
  “top-down Gross-Sohl method F” (tdgsf) method (https://otexts.com/fpp2/top-down.html)

  Parameters
  ----------
  df : PySpark dataframe
      Dataset containing both lower / upper hiarchies, time and demand
  upper_hier : List
      List containing the hierarchy of the aggregated level
  lower_hier : List
      List containing the hierarchy of the lower level
  demand_var: String
      Variable over which to perform the calculation
  time_var: String
      Variable that indicates time period in the dataset
  time_horizon: Integer
      Number of periods to perform the calculation

  Returns
  -------
  df : PySpark dataframe
      Dataset containing upper hierarchy, lower hierarchy and proportion to disallocate
  """
  if isinstance(upper_hier, str):
    upper_hier = [upper_hier]
  if isinstance(lower_hier, str):
    lower_hier = [lower_hier]

  #Filter to required dates
  max_date = aggregate_data(df, upper_hier , [time_var],[max])
  df = df.join(max_date, on=upper_hier, how="left")
  df = df.filter(col(time_var)> col("max_" + time_var) - time_horizon)

  #Calculate aggregate proportion sum
  agg_df = aggregate_data(df, upper_hier + [time_var], [demand_var],[sum])
  agg_df = agg_df.withColumnRenamed("sum_"+demand_var,"agg_"+demand_var)
  agg_df = agg_df.withColumn("agg_"+demand_var,col("agg_"+demand_var)/time_horizon)
  agg_df = aggregate_data(agg_df, upper_hier, ["agg_"+demand_var],[sum])

  #Calculate lower proportion sum
  lower_df = df.withColumn(demand_var,col(demand_var)/time_horizon)
  lower_df = aggregate_data(lower_df, dedupe_two_lists(lower_hier, upper_hier), [demand_var],[sum])

  #Proportions
  proportions = lower_df.join(agg_df, on=upper_hier, how="left")
  proportions = proportions.withColumn("prop",
                                       col("sum_"+demand_var) /
                                       col("sum_agg_"+demand_var))
  proportions = proportions.select(dedupe_two_lists(lower_hier, upper_hier) + ["prop"])

  return (proportions)

def get_disallocation_prop_tdgsa(df, upper_hier, lower_hier, demand_var, time_var, time_horizon):
  """
  Determines proportions to output a higher level forecast into a lower level forecast using the
  “top-down Gross-Sohl method A” (tdgsa) method (https://otexts.com/fpp2/top-down.html)

  Parameters
  ----------
  df : PySpark dataframe
      Dataset containing both lower / upper hiarchies, time and demand
  upper_hier : List
      List containing the hierarchy of the aggregated level
  lower_hier : List
      List containing the hierarchy of the lower level
  demand_var: String
      Variable over which to perform the calculation
  time_var: String
      Variable that indicates time period in the dataset
  time_horizon: Integer
      Number of periods to perform the calculation

  Returns
  -------
  df : PySpark dataframe
      Dataset containing upper hierarchy, lower hierarchy and proportion to disallocate
  """
  if isinstance(upper_hier, str):
    upper_hier = [upper_hier]
  if isinstance(lower_hier, str):
    lower_hier = [lower_hier]

  #Filter to required dates
  max_date = aggregate_data(df, upper_hier , [time_var],[max])
  df = df.join(max_date, on=upper_hier, how="left")
  df = df.filter(col(time_var)> col("max_" + time_var) - time_horizon)

  #Aggregate sales, merge to lower and calculate proportion
  agg_df = aggregate_data(df, upper_hier + [time_var], [demand_var],[sum])
  merged_df = df.join(agg_df, on=upper_hier + [time_var], how="left")
  merged_df = merged_df.withColumn("prop",col(demand_var)/col("sum_"+demand_var))

  #Aggregate proportions
  proportions_df = aggregate_data(merged_df, dedupe_two_lists(lower_hier, upper_hier), ["prop"],[sum])
  proportions_df = proportions_df.withColumnRenamed("sum_prop","prop")
  proportions_df = proportions_df.withColumn("prop",col("prop")/time_horizon)
  proportions_df = proportions_df.select(dedupe_two_lists(lower_hier, upper_hier) + ["prop"])

  return (proportions_df)


def get_disallocation_prop_tdfp(df_lower, upper_hier, lower_hier, demand_var, time_var):
  """
  Determines proportions to output a higher level forecast into a lower level forecast using the
  “top-down forecast proportions” (tdfp) method (https://otexts.com/fpp2/top-down.html)

  Parameters
  ----------
  df_lower : PySpark dataframe
      Dataset containing both lower / upper hiarchies, time and demand
  upper_hier : List
      List containing the hierarchy of the aggregated level
  lower_hier : List
      List containing the hierarchy of the lower level
  demand_var: String
      Variable over which to perform the calculation
  time_var: String
      Variable that indicates time period in the dataset
  time_horizon: Integer
      Number of periods to perform the calculation

  Returns
  -------
  df : PySpark dataframe
      Dataset containing upper hierarchy, lower hierarchy, time and proportion to disallocate
  """
  if isinstance(upper_hier, str):
    upper_hier = [upper_hier]
  if isinstance(lower_hier, str):
    lower_hier = [lower_hier]

  #Determine proportion of forecast based on lower level distribution
  proportions_df = get_hierarchical_statistics(df_lower,upper_hier + [time_var],demand_var,[sum])
  proportions_df = proportions_df.withColumn("prop",col(demand_var)/col("sum_"+demand_var))
  proportions_df = proportions_df.select(dedupe_two_lists(lower_hier, upper_hier) + [time_var,"prop"])


  return (proportions_df)

def disaggregate_forecast(df_lower, df_upper, df_proportions , upper_hier, lower_hier, fcst_var, time_var=None):
  """
  Disaggregates a higher level forecast into a lower level forecast using proportions fed in by "df_proportions"

  Parameters
  ----------
  df_lower : PySpark dataframe
      Dataset containing both lower / upper hiarchies, time and demand
  df_upper : PySpark dataframe
      Dataset containing the upper hiarchies, time and demand
  df_proportions : PySpark dataframe
      Dataset containing the proportions to allocate upper into lower
  upper_hier : List
      List containing the hierarchy of the aggregated level
  lower_hier : List
      List containing the hierarchy of the lower level
  fcst_var: String
      Variable name containing the forecasted demand (should be similarly named in both upper and lower forecast)
  time_var: String
      Variable that indicates time period in the dataset

  Returns
  -------
  df : PySpark dataframe
      Dataset containing upper hierarchy, lower hierarchy, time and disallocated demand
  """
  if isinstance(upper_hier, str):
    upper_hier = [upper_hier]
  if isinstance(lower_hier, str):
    lower_hier = [lower_hier]

  #Disaggregate demand
  df_upper = df_upper.withColumnRenamed(fcst_var,"HIGHER_DEMAND")
  upper_merge = intersect_two_lists(upper_higher + [time_var], df_upper.columns)
  df_out = df_lower.join(df_upper, on=upper_merge, how="left")
  lower_merge = intersect_two_lists(lower_higher + [time_var], df_proportions.columns)
  df_out = df_out.join(df_proportions, on=lower_merge, how="left")
  df_out = df_out.withColumn("ALLOC_DEMAND",col("HIGHER_DEMAND")*col("prop"))
  df_out = df_out.select(df_lower.columns + ["ALLOC_DEMAND"])

  return (df_out)