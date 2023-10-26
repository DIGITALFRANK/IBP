# Databricks notebook source
# DBTITLE 1,PySpark Transformations
# MAGIC %python
# MAGIC 
# MAGIC 
# MAGIC def log_columns_pyspark(input_df, COLS_TO_LOG):
# MAGIC   """
# MAGIC   Standardize columns of a DataFrame distributively in PySpark using log-plus-one method.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame with columns to standardize with log-plus-one method.
# MAGIC   
# MAGIC   COLS_TO_LOG : list or str
# MAGIC       Column(s) of input_df to standardize with log-plus-one method.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : PySpark DataFrame
# MAGIC       DataFrame with all columns of input_df, except COLS_TO_LOG, and standardized versions of COLS_TO_LOG.
# MAGIC   """
# MAGIC   
# MAGIC   if isinstance(COLS_TO_LOG, str):
# MAGIC     COLS_TO_LOG = [COLS_TO_LOG]
# MAGIC 
# MAGIC   filtered_df = input_df.select(*COLS_TO_LOG)
# MAGIC   
# MAGIC   filtered_df = filtered_df.select(*[log1p(col).alias(col + "_log") for col in filtered_df.columns])
# MAGIC   
# MAGIC   rest_of_input_df = input_df.drop(*COLS_TO_LOG)
# MAGIC 
# MAGIC   final_df = columnwise_union_pyspark(filtered_df, rest_of_input_df)
# MAGIC   
# MAGIC   if TARGET_VAR in COLS_TO_LOG:
# MAGIC     update_target_var(final_df)
# MAGIC   
# MAGIC   return final_df
# MAGIC 
# MAGIC 
# MAGIC def undo_scaling_operations_pyspark(df):
# MAGIC   """
# MAGIC   Wrapper to undo scaling operations on prediction df
# MAGIC   """
# MAGIC   if "_log" in TARGET_VAR:
# MAGIC     final_df = delog_df_pyspark(df)
# MAGIC   
# MAGIC   if "_nrm" in TARGET_VAR:
# MAGIC     #TODO
# MAGIC     raise NotImplementedError
# MAGIC     
# MAGIC   if "_std" in TARGET_VAR:
# MAGIC     #TODO
# MAGIC     raise NotImplementedError
# MAGIC           
# MAGIC   return final_df
# MAGIC   
# MAGIC     
# MAGIC def delog_df_pyspark(df):
# MAGIC   """
# MAGIC   De-logp1 a PySpark DF
# MAGIC   """
# MAGIC 
# MAGIC   delogged_df = df.select(*[expm1(col).alias(col) for col in df.columns])\
# MAGIC                   .withColumnRenamed(TARGET_VAR, TARGET_VAR.replace("_log", ""))
# MAGIC   
# MAGIC   return delogged_df
# MAGIC 
# MAGIC 
# MAGIC def get_column_means_pyspark(input_df):
# MAGIC   """
# MAGIC   Calculate the column-wise means for all columns in a PySpark DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame for which column-wise means are desired.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   means : PySpark DataFrame
# MAGIC       DataFrame with means of each column of input_df. 
# MAGIC   """
# MAGIC   return input_df.select(*[mean(col).alias(col + "_mean") for col in input_df.columns])
# MAGIC   
# MAGIC   
# MAGIC def get_column_stds_pyspark(input_df):
# MAGIC   """
# MAGIC   Calculate the column-wise standard deviations for all columns in a PySpark DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame for which column-wise standard deviations are desired.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   std_devs : PySpark DataFrame
# MAGIC       DataFrame with standard deviations of each column of input_df. 
# MAGIC   """
# MAGIC   return input_df.select(*[stddev(col).alias(col + "_stdev") for col in input_df.columns])
# MAGIC 
# MAGIC 
# MAGIC # def standardize_columns_pyspark(input_df, COLS_TO_STANDARDIZE):
# MAGIC #   """
# MAGIC #   Standardize columns of a DataFrame distributively using PySpark and store results in global variable standardization_dict to undo transformation.
# MAGIC   
# MAGIC #   Parameters
# MAGIC #   ----------
# MAGIC #   input_df : PySpark DataFrame
# MAGIC #       DataFrame with columns to standardize.
# MAGIC   
# MAGIC #   COLS_TO_STANDARDIZE : list or str
# MAGIC #       Column(s) of input_df to standardize.
# MAGIC   
# MAGIC #   Returns
# MAGIC #   -------
# MAGIC #   final_df : PySpark DataFrame
# MAGIC #       DataFrame with all columns of input_df, except COLS_TO_STANDARDIZE, and standardized versions of COLS_TO_STANDARDIZE.
# MAGIC 
# MAGIC #   TODO: robust scaler that excludes 0's and outliers?
# MAGIC #   """
# MAGIC #   global standardization_dict
# MAGIC 
# MAGIC #   if isinstance(COLS_TO_STANDARDIZE, str):
# MAGIC #     COLS_TO_STANDARDIZE = [COLS_TO_STANDARDIZE]
# MAGIC 
# MAGIC #   filtered_df = input_df.select(*COLS_TO_STANDARDIZE)
# MAGIC 
# MAGIC #   mean_of_columns = get_column_means_pyspark(filtered_df)
# MAGIC   
# MAGIC #   std_of_columns = get_column_stds_pyspark(filtered_df)
# MAGIC 
# MAGIC #   joined_df = filtered_df.crossJoin(mean_of_columns) \
# MAGIC #                          .crossJoin(std_of_columns)
# MAGIC 
# MAGIC #   # TODO try to implement this step using arrays instead of a loop (don't believe it's possible in PySpark)
# MAGIC #   for feature in filtered_df.columns:
# MAGIC #     joined_df = joined_df.withColumn((feature + "_std"), (col(feature) - col(feature + "_mean"))/col(feature + "_stdev"))\
# MAGIC #                          .drop(*[feature, feature + "_mean", feature + "_stdev"])
# MAGIC 
# MAGIC #   final_df = columnwise_union_pyspark(joined_df, input_df.drop(*COLS_TO_STANDARDIZE))
# MAGIC 
# MAGIC #   # store results in dictionary to undo transformation
# MAGIC #   standardization_dict = {"mean" : mean_of_columns.toPandas(), "stdev" : std_of_columns.toPandas()}
# MAGIC 
# MAGIC #   return final_df
# MAGIC 
# MAGIC 
# MAGIC def get_column_maxes_pyspark(input_df):
# MAGIC   """  
# MAGIC   Calculate the column-wise maximum values for all columns in a PySpark DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame for which column-wise maximums are desired.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   maxs : PySpark DataFrame
# MAGIC       DataFrame with maximums of each column of input_df. 
# MAGIC   """
# MAGIC   return input_df.select(*[max(col).alias(col + "_max") for col in input_df.columns])
# MAGIC   
# MAGIC   
# MAGIC def get_column_mins_pyspark(input_df):
# MAGIC   """  
# MAGIC   Calculate the column-wise minimnum values for all columns in a PySpark DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame for which column-wise minimums are desired.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   mins : PySpark DataFrame
# MAGIC       DataFrame with minimums of each column of input_df. 
# MAGIC   """
# MAGIC   return input_df.select(*[min(col).alias(col + "_min") for col in input_df.columns])
# MAGIC 
# MAGIC 
# MAGIC def normalize_columns_pyspark(input_df, columns_to_normalize):
# MAGIC   """
# MAGIC   Normalize columns of a DataFrame distributively using PySpark and store results in global variable normalization_dict to undo transformation.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_df : PySpark DataFrame
# MAGIC       DataFrame with columns to normalize.
# MAGIC   
# MAGIC   columns_to_nomralize : list or str
# MAGIC       Column(s) of input_df to normalize.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : PySpark DataFrame
# MAGIC       DataFrame with all columns of input_df, except columns_to_normalize, and normalized versions of columns_to_normalize.
# MAGIC   """
# MAGIC   global normalization_dict
# MAGIC 
# MAGIC   if isinstance(columns_to_normalize, str):
# MAGIC     columns_to_normalize = [columns_to_normalize]
# MAGIC 
# MAGIC   filtered_df = input_df.select(*columns_to_normalize)
# MAGIC 
# MAGIC   max_of_columns = get_column_maxes_pyspark(filtered_df)
# MAGIC   min_of_columns = get_column_mins_pyspark(filtered_df)
# MAGIC 
# MAGIC   joined_df = filtered_df.crossJoin(max_of_columns) \
# MAGIC                          .crossJoin(min_of_columns)
# MAGIC 
# MAGIC   # TODO try to implement this step using arrays instead of a loop (don't believe it's possible in PySpark)
# MAGIC   for feature in filtered_df.columns:
# MAGIC     joined_df = joined_df.withColumn((feature + "_nrm"), (col(feature) - col(feature + "_min"))/(col(feature + "_max") - col(feature + "_min")))\
# MAGIC                          .drop(*[feature, feature + "_max", feature + "_min"])
# MAGIC 
# MAGIC   final_df = columnwise_union_pyspark(joined_df, input_df.drop(*COLS_TO_STANDARDIZE))
# MAGIC 
# MAGIC   # store results in dictionary to undo transformation
# MAGIC   normalization_dict = {"max" : max_of_columns.toPandas(), "min" : min_of_columns.toPandas()}
# MAGIC 
# MAGIC   # Update TARGET_VAR global
# MAGIC   if TARGET_VAR in columns_to_normalize:
# MAGIC     update_target_var(final_df)
# MAGIC   
# MAGIC   return final_df
# MAGIC 
# MAGIC 
# MAGIC def filter_dates_pyspark(df):
# MAGIC   """
# MAGIC   Filter a PySpark DataFrame on the column with name equal to global variable TIME_VAR using global variable week indicators START_WEEK and END_WEEK (in the format YYYYMM).
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : PySpark DataFrame
# MAGIC       DataFrame with column name equal to global variable TIME_VAR in the format YYYYMM (e.g., 201801).
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   df_filtered: PySpark DataFrame
# MAGIC       DataFrame filtered by week indicators (in format YYYYMM) stored in global variables START_WEEK and END_WEEK.
# MAGIC   """
# MAGIC   return df.filter((col(TIME_VAR) >= START_WEEK) & (col(TIME_VAR) <= END_WEEK))
# MAGIC   
# MAGIC 
# MAGIC def cast_column_to_numeric_pyspark(df):
# MAGIC   """
# MAGIC   Cast columns to numeric in PySpark using global variable vars_treatment_dict.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : PySpark DataFrame
# MAGIC       DataFrame with columns in keys of global variable AGG_DICT.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   df : PySpark DataFrame
# MAGIC       DataFrame with columns in global variable AGG_DICT cast to 'float' as specified.
# MAGIC   """
# MAGIC   cols_to_make_numeric = list(AGG_DICT.keys())
# MAGIC   float_dict = {key:'float' for key in cols_to_make_numeric}
# MAGIC 
# MAGIC   for feature in cols_to_make_numeric:
# MAGIC     df = df.withColumn(feature, col(feature).cast(float_dict[feature]).alias(feature)) 
# MAGIC   
# MAGIC   return df
# MAGIC 
# MAGIC 
# MAGIC def aggregate_hierarchy_pyspark(df):
# MAGIC   """
# MAGIC   Aggregate PySpark DataFrame according to globally-specified business and product hierarchies.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : PySpark DataFrame
# MAGIC       Ungrouped DataFrame with business and product hierarchy levels and global variable TIME_VAR as columns.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : PySpark DataFrame
# MAGIC       DataFrame according to global variables BUSINESS_HIER/BUSINESS_LEVEL, PRODUCT_HIER/PRODUCT_LEVEL, TIME_VAR, and OTHER_CATEGORICALS. 
# MAGIC   """
# MAGIC 
# MAGIC   agg_hierarchy = get_hierarchy() + [TIME_VAR]
# MAGIC 
# MAGIC   agg_df = df.groupby(agg_hierarchy) \
# MAGIC                .agg(AGG_DICT)
# MAGIC   
# MAGIC   column_names = [remove_aggregate_aliases(col) for col in agg_df.columns]
# MAGIC   
# MAGIC   final_df = agg_df.toDF(*column_names)
# MAGIC   
# MAGIC   return final_df 
# MAGIC 
# MAGIC 
# MAGIC def aggregate_df_pyspark(df):
# MAGIC   """
# MAGIC   Cast and aggregate columns in PySpark.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : PySpark DataFrame
# MAGIC       DataFrame with columns to cast and aggregate.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   aggregated_df : PySpark DataFrame
# MAGIC       DataFrame with casted columns and aggregated according to global variables BUSINESS_HIER/BUSINESS_LEVEL, PRODUCT_HIER/PRODUCT_LEVEL, TIME_VAR, and OTHER_CATEGORICALS.
# MAGIC   """
# MAGIC   casted_df = cast_column_to_numeric_pyspark(df)
# MAGIC   aggregated_df = aggregate_hierarchy_pyspark(casted_df)
# MAGIC   aggregated_df = aggregated_df.withColumn(TIME_VAR, col(TIME_VAR).cast("int").alias(TIME_VAR))
# MAGIC   
# MAGIC   return aggregated_df

# COMMAND ----------

# DBTITLE 1,Pandas / Koalas Transformations
# MAGIC %python
# MAGIC 
# MAGIC # TODO deprecated due to frequent insertion issues
# MAGIC def resample_dataframe(pandas_df, grouping_cols, time_var, resample_format):
# MAGIC   raise ValueError("resample_dataframe has been deprecated; use fill_missing_timeseries instead")
# MAGIC   
# MAGIC   #     """
# MAGIC #     Resample data according to specified format
# MAGIC #     Resample formats found here: https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4
# MAGIC #     """
# MAGIC 
# MAGIC #     pandas_df.sort_values(time_var, inplace=True)
# MAGIC 
# MAGIC #     final_df = (pandas_df.set_index(time_var)
# MAGIC #               .groupby(grouping_cols)
# MAGIC #               .resample(resample_format)
# MAGIC #               .asfreq(fill_value=0)
# MAGIC #               .reset_index()
# MAGIC #              )
# MAGIC 
# MAGIC #     return final_df
# MAGIC 
# MAGIC 
# MAGIC def aggregate_df(df):
# MAGIC   """
# MAGIC   Apply all transformations to a pandas or koalas DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : pandas or koalas DataFrame
# MAGIC       DataFrame to transform.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   aggregated_df : pandas or koalas DataFrame
# MAGIC       Aggregated DataFrame with column TIME_VAR as type int. 
# MAGIC   """
# MAGIC     
# MAGIC   aggregated_df = aggregate_hierarchy(df)
# MAGIC     
# MAGIC   aggregated_df[TIME_VAR] = aggregated_df[TIME_VAR].astype(int)  
# MAGIC   
# MAGIC   return aggregated_df
# MAGIC 
# MAGIC 
# MAGIC ## Corey updated this to be flagged as "old" - made slight adjustments to function below
# MAGIC ## Keeping this version for posterity - may need to be referenced in the future
# MAGIC 
# MAGIC def aggregate_hierarchy_old(df):
# MAGIC   """
# MAGIC   Aggregate pandas or koalas DataFrame using globally defined business and product hierarchies and levels.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : pandas or koalas DataFrame
# MAGIC       DataFrame to aggregate.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       Aggregated DataFrame.
# MAGIC   """
# MAGIC 
# MAGIC   agg_hierarchy = get_hierarchy() + [TIME_VAR]
# MAGIC   cols_to_make_numeric = list(AGG_DICT.keys())
# MAGIC   float_dict = {key:'float' for key in cols_to_make_numeric}
# MAGIC 
# MAGIC   print("Aggregating dataframe using the following hierarchy: %s \n" % agg_hierarchy)
# MAGIC   
# MAGIC   df = df.astype(float_dict)
# MAGIC   
# MAGIC   final_df = df.groupby(agg_hierarchy).agg(AGG_DICT).reset_index()
# MAGIC 
# MAGIC   return final_df
# MAGIC 
# MAGIC 
# MAGIC def aggregate_hierarchy(df):
# MAGIC   """
# MAGIC   Aggregate pandas or koalas DataFrame using globally defined business and product hierarchies and levels.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   df : pandas or koalas DataFrame
# MAGIC       DataFrame to aggregate.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       Aggregated DataFrame.
# MAGIC   """
# MAGIC 
# MAGIC   agg_hierarchy = get_hierarchy() + [TIME_VAR]
# MAGIC   cols_to_make_numeric = list(AGG_DICT.keys())
# MAGIC   ## float_dict = {key:'float' for key in cols_to_make_numeric}  ## corey removed
# MAGIC 
# MAGIC   print("Aggregating dataframe using the following hierarchy: %s \n" % agg_hierarchy)
# MAGIC   
# MAGIC   ## df = df.astype(float_dict)  ## corey removed
# MAGIC   
# MAGIC   final_df = df.groupby(agg_hierarchy).agg(AGG_DICT).reset_index()
# MAGIC 
# MAGIC   return final_df
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC def do_standardize_normalize(pd_df, col_list, operation):
# MAGIC   """
# MAGIC   Standardize or normalize a set of columns of a pandas or koalas DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to standardize or normalize.
# MAGIC       
# MAGIC   col_list : list
# MAGIC       Names of columns of pd_df to standardize or normalize.
# MAGIC       
# MAGIC   operation: str
# MAGIC       String that indicates whether columns in col_list should be standardized ("_std") or normalized ("_nrm").
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_pd : pandas or koalas DataFrame
# MAGIC       DataFrame with standardized or normalized columns and without original columns in col_list.
# MAGIC   """
# MAGIC   from sklearn.preprocessing import StandardScaler, MinMaxScaler
# MAGIC 
# MAGIC   global trans_dict
# MAGIC 
# MAGIC   operation_dict = {"_std": StandardScaler(), "_nrm": MinMaxScaler()}
# MAGIC   
# MAGIC   assert (operation in operation_dict)
# MAGIC   scaler_func = operation_dict[operation]
# MAGIC   
# MAGIC   # create global dict to empower us to denormalize in the feature
# MAGIC   if "trans_dict" not in globals():
# MAGIC     trans_dict = {}
# MAGIC     
# MAGIC   cols_for_transformation = pd_df[col_list]
# MAGIC   
# MAGIC   transformer = scaler_func
# MAGIC   transformer.fit(cols_for_transformation)
# MAGIC   
# MAGIC   # add transformer to global dict for future inverse transformers
# MAGIC   trans_dict.update({operation: transformer})
# MAGIC   
# MAGIC   # transform dataframe
# MAGIC   new_col_names = [feature + operation for feature in col_list]
# MAGIC   transformed_cols = pd.DataFrame(transformer.transform(cols_for_transformation), columns = new_col_names)
# MAGIC   
# MAGIC   # add back original columns
# MAGIC   final_pd = pd_df.join(transformed_cols)
# MAGIC   final_pd = final_pd.drop(col_list, axis=1)
# MAGIC   
# MAGIC   # update TARGET_VAR global if necessary
# MAGIC   if TARGET_VAR in col_list:
# MAGIC     update_target_var(final_pd)
# MAGIC   
# MAGIC   return final_pd
# MAGIC 
# MAGIC 
# MAGIC def do_destandardize_denormalize(pd_df, operation):
# MAGIC   """
# MAGIC   Untransform (either denormalize or destandardize) all previously transformed columns of a given operation type.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to untransform (either denormalize or destandardize).
# MAGIC   
# MAGIC   operation: string
# MAGIC       String that indicates whether columns should be denormalized ("_nrm") or destandardized ("_std").
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df: pandas or koalas DataFrame
# MAGIC       Denormalized or destandardized DataFrame.
# MAGIC   """
# MAGIC   operation_lst = ["_std", "_nrm"]
# MAGIC   
# MAGIC   assert (operation in operation_lst)
# MAGIC   
# MAGIC   transformer = trans_dict[operation]
# MAGIC   
# MAGIC   affected_cols = [feature for feature in pd_df.columns.tolist() if operation in feature and "_lag" not in feature]
# MAGIC   
# MAGIC   new_col_names = [feature.replace(operation, '') for feature in affected_cols]
# MAGIC   
# MAGIC   inversed_pd = pd.DataFrame(transformer.inverse_transform(pd_df[affected_cols]), columns = new_col_names)
# MAGIC   
# MAGIC   final_df = pd_df.join(inversed_pd)
# MAGIC   
# MAGIC   final_df = final_df.drop(affected_cols, axis=1)
# MAGIC   
# MAGIC   return final_df
# MAGIC 
# MAGIC 
# MAGIC def normalize_columns(pd_df, col_list):
# MAGIC   """
# MAGIC   Scale a list of columns of a pandas or koalas DataFrame between 0 and 1.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to normalize.
# MAGIC   
# MAGIC   col_list : list
# MAGIC       Columns of pd_df to normalize.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       DataFrame with normalized columns and without original columns in col_list.
# MAGIC   """
# MAGIC   return do_standardize_normalize(pd_df, col_list, "_nrm")
# MAGIC     
# MAGIC   
# MAGIC def denormalize_df(pd_df):
# MAGIC   """
# MAGIC   Denormalize all columns that were previously normalized.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to denormalize.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       DataFrame with denormalized columns.
# MAGIC   """
# MAGIC   return do_destandardize_denormalize(pd_df, "_nrm")
# MAGIC 
# MAGIC 
# MAGIC def standardize_columns(pd_df, col_list):
# MAGIC   """
# MAGIC   Scale a list of columns of a pandas or koalas DataFrame to conform to normal distribution.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to standardize.
# MAGIC   
# MAGIC   col_list : list
# MAGIC       Columns of pd_df to standardize.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       DataFrame with standardized columns and without original columns in col_list.
# MAGIC   """
# MAGIC   return do_standardize_normalize(pd_df, col_list, "_std")
# MAGIC 
# MAGIC 
# MAGIC def destandardize_columns(pd_df):
# MAGIC   """
# MAGIC   Destandardize all columns that were previously standardized.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to destandardize.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_df : pandas or koalas DataFrame
# MAGIC       DataFrame with destandardized columns.
# MAGIC   """
# MAGIC   return do_destandardize_denormalize(pd_df, "_std")
# MAGIC 
# MAGIC 
# MAGIC def log_columns(pd_df, col_list=None):
# MAGIC   """
# MAGIC   Re-scale a list of columns in a pandas or koalas DataFrame by taking the log(col+1).
# MAGIC   Adding 1 within the log avoids errors caused by 0s and has the neat side-effect of 0s on the original scale remaining 0s on the transformed scale.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to scale logarithmically.
# MAGIC   
# MAGIC   col_list : list
# MAGIC       Names of columns of pd_df to scale logarithmically.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_pd : pandas or koalas DataFrame
# MAGIC       DataFrame with logarithmically scaled columns and without the original columns in col_list.
# MAGIC   """
# MAGIC   if not col_list:
# MAGIC     col_list = COLS_TO_LOG
# MAGIC   
# MAGIC   col_list = ensure_is_list(col_list)
# MAGIC   
# MAGIC   new_column_names = [feature + "_log" for feature in col_list]
# MAGIC   
# MAGIC   trans_pd = pd_df[col_list].apply(lambda x: np.log1p(x))
# MAGIC   trans_pd.columns = new_column_names
# MAGIC    
# MAGIC   final_pd = pd_df.join(trans_pd)
# MAGIC   
# MAGIC   final_pd = final_pd.drop(col_list, axis=1)
# MAGIC   
# MAGIC   if TARGET_VAR in col_list:
# MAGIC     update_target_var(final_pd)
# MAGIC 
# MAGIC   return final_pd
# MAGIC 
# MAGIC 
# MAGIC def undo_scaling_operations(pd):
# MAGIC   """
# MAGIC   Wrapper to undo scaling operations on prediction df in pandas
# MAGIC   """
# MAGIC   if "_log" in TARGET_VAR:
# MAGIC     final_pd = delog_df(pd)
# MAGIC   
# MAGIC   if "_nrm" in TARGET_VAR:
# MAGIC     #TODO
# MAGIC     raise NotImplementedError
# MAGIC     
# MAGIC   if "_std" in TARGET_VAR:
# MAGIC     #TODO
# MAGIC     raise NotImplementedError
# MAGIC           
# MAGIC   return final_pd
# MAGIC 
# MAGIC 
# MAGIC def delog_df(pd_df):
# MAGIC   """
# MAGIC   Undo the logarithmic scale transformation for an entire pandas or koalas DataFrame.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame to untransform.
# MAGIC   
# MAGIC   Returns
# MAGIC   -------
# MAGIC   pd_df : pandas or koalas DataFrame
# MAGIC       DataFrame with untransformed columns.
# MAGIC   """
# MAGIC   hierarchy = get_hierarchy() + [TIME_VAR] + ["sample"]
# MAGIC   column_mask = [col for col in list(pd_df.columns.values) if col not in hierarchy]
# MAGIC   
# MAGIC   pd_df[column_mask] = np.expm1(pd_df[column_mask])
# MAGIC   
# MAGIC   pd_df = pd_df.rename(columns={TARGET_VAR : get_base_target_var()})
# MAGIC   
# MAGIC   return pd_df
# MAGIC 
# MAGIC 
# MAGIC # (DJMP) Did not add docString since function not in use
# MAGIC def transform_columns(pd_df, standardized_cols = [], normalized_cols = []):
# MAGIC   """
# MAGIC   Normalize, standardize, and one-hot encode columns using sklearn
# MAGIC   NOTE: This version can't be inversed; shouldn't be used until sklearn catches up
# MAGIC   """
# MAGIC   from sklearn.compose import ColumnTransformer
# MAGIC   
# MAGIC   # get names for final columns
# MAGIC   orig_features = [standardized_cols + normalized_cols]
# MAGIC   std_names = [col + "_std" for col in standardized_cols]
# MAGIC   nrm_names = [col + "_nrm" for col in normalized_cols]
# MAGIC   trans_features = std_names + nrm_names 
# MAGIC     
# MAGIC   if not trans_features:
# MAGIC     raise Exception('Please enter parameters before running.')
# MAGIC 
# MAGIC   pipeline = ()
# MAGIC   
# MAGIC   if standardized_cols:
# MAGIC     from sklearn.preprocessing import StandardScaler
# MAGIC     scaler = (('std', StandardScaler(), standardized_cols),)
# MAGIC     pipeline += scaler
# MAGIC     
# MAGIC   if normalized_cols:
# MAGIC     from sklearn.preprocessing import MinMaxScaler
# MAGIC     pipeline += (('nrm', MinMaxScaler(), normalized_cols),)
# MAGIC     
# MAGIC   pipeline_lst = list(pipeline)  
# MAGIC           
# MAGIC   preprocessor = ColumnTransformer(
# MAGIC     remainder='drop',
# MAGIC     transformers=pipeline_lst)
# MAGIC 
# MAGIC   transformed_pd = pd.DataFrame(preprocessor.fit_transform(pd_df), columns=trans_features)
# MAGIC     
# MAGIC   final_df = pd_df.join(transformed_pd) 
# MAGIC   final_df.drop(orig_features, axis=1, inplace=True)
# MAGIC   
# MAGIC   return [final_df, preprocessor]
# MAGIC 
# MAGIC 
# MAGIC def replace_features_using_dict(input_pd, replacement_dict):
# MAGIC   """
# MAGIC   Replace values from one column of a pandas or koalas DataFrame with the values of keys from a dictionary.
# MAGIC   
# MAGIC   Parameters
# MAGIC   ----------
# MAGIC   input_pd : pandas or koalas DataFrame
# MAGIC       DataFrame with values to replace.
# MAGIC   
# MAGIC   replacement_dict : dict
# MAGIC       Dictionary of column names-values pairs to use to replace values in DataFrame.
# MAGIC       
# MAGIC   Returns
# MAGIC   -------
# MAGIC   final_pd : pandas or koalas DataFrame
# MAGIC       DataFrame with replaced values.
# MAGIC   """
# MAGIC   final_pd = input_pd.replace(replacement_dict, inplace=False)
# MAGIC   return(final_pd)
# MAGIC 
# MAGIC 
# MAGIC def add_future_rows(pandas_df, number_of_periods_forward, index_columns=None, all_combinations=False, restrain_permutations_using_hierarchies=True, time_unit='weeks', datetime_format="%Y%U-%w"):
# MAGIC     """ 
# MAGIC     Add prediction dates to dataframe. Using set_index().reindex(mux) reads more eloquently, but doesn't scale well due to OOM issues.
# MAGIC     """      
# MAGIC 
# MAGIC     if not index_columns:
# MAGIC       index_columns = get_hierarchy()
# MAGIC 
# MAGIC     add_grid = pd.DataFrame()
# MAGIC     last_available_date = pandas_df[TIME_VAR].max()
# MAGIC     
# MAGIC     if all_combinations:
# MAGIC       def get_all_column_combinations(input_df, columns, restrain_permutations_using_hierarchies):
# MAGIC         df = input_df.set_index(columns)
# MAGIC         multiindex = pd.MultiIndex.from_product(df.index.levels, names=df.index.names)\
# MAGIC                                   .drop_duplicates()\
# MAGIC                                   .to_frame(index=False)
# MAGIC 
# MAGIC         # enforce the pre-set hierarchy (e.g., a particular store can only reside in one state)
# MAGIC         # note that it's wayy faster to filter like this vs. messing with itertools permutations
# MAGIC         if (restrain_permutations_using_hierarchies) & (not index_columns):
# MAGIC 
# MAGIC           for column_set in [PRODUCT_HIER[:PRODUCT_LEVEL], BUSINESS_HIER[:BUSINESS_LEVEL]]:
# MAGIC             multiindex = filter_using_indexes(multiindex, input_df, column_set)
# MAGIC 
# MAGIC         return multiindex
# MAGIC       
# MAGIC       index_dataframe = get_all_column_combinations(pandas_df, index_columns, restrain_permutations_using_hierarchies)
# MAGIC   
# MAGIC     else:
# MAGIC       index_dataframe = pandas_df[index_columns].drop_duplicates()    
# MAGIC     
# MAGIC     
# MAGIC     for i in range(1, number_of_periods_forward + 1):
# MAGIC       temp_df = index_dataframe.copy()
# MAGIC       temp_df[TIME_VAR] = update_time(last_available_date, i, time_unit=time_unit, datetime_format=datetime_format)
# MAGIC       temp_df[TARGET_VAR] = np.nan
# MAGIC       add_grid = pd.concat([add_grid, temp_df])
# MAGIC         
# MAGIC     pandas_df = pd.concat([pandas_df, add_grid]).reset_index(drop=True)
# MAGIC     
# MAGIC     # backfill other hierarchy columns if you only used a subset of get_hierarchy()
# MAGIC     if (len(index_columns) < len(get_hierarchy())) & (set(index_columns).issubset(get_hierarchy())):
# MAGIC       def backfill_hierarchy_columns(pandas_df, index_cols):
# MAGIC         """
# MAGIC         backfill any hierarchy columns that were dropped when using index_columns
# MAGIC         """
# MAGIC                 
# MAGIC         business_cols_to_fill = {
# MAGIC           "joining_cols" : [col for col in BUSINESS_HIER[:BUSINESS_LEVEL] if col in index_cols], 
# MAGIC           "fill_cols" : [col for col in BUSINESS_HIER[:BUSINESS_LEVEL] if col not in index_cols]
# MAGIC         }
# MAGIC 
# MAGIC         product_cols_to_fill = {
# MAGIC           "joining_cols" : [col for col in PRODUCT_HIER[:PRODUCT_LEVEL] if col in index_cols], 
# MAGIC           "fill_cols" : [col for col in PRODUCT_HIER[:PRODUCT_LEVEL] if col not in index_cols]
# MAGIC         }
# MAGIC 
# MAGIC         for column_dict in [business_cols_to_fill, product_cols_to_fill]:
# MAGIC           join_columns = column_dict['joining_cols']
# MAGIC           fill_columns = column_dict['fill_cols']
# MAGIC           if (not join_columns) | (not fill_columns):
# MAGIC             continue
# MAGIC         
# MAGIC           hierarchy_table = (pandas_df[join_columns + fill_columns]
# MAGIC                              .drop_duplicates()
# MAGIC                              .dropna()) # assumes that hierarchy columns can't have nulls
# MAGIC 
# MAGIC           pandas_df = (pandas_df
# MAGIC                        .drop(fill_columns, axis=1)
# MAGIC                        .drop_duplicates()
# MAGIC                        .merge(hierarchy_table, on=join_columns, how='left'))
# MAGIC 
# MAGIC         return pandas_df
# MAGIC       
# MAGIC       pandas_df = backfill_hierarchy_columns(pandas_df, index_columns)
# MAGIC       
# MAGIC     # for some reason, TIME_VAR changed to float by operations above
# MAGIC     pandas_df[TIME_VAR] = pandas_df[TIME_VAR].astype('int')
# MAGIC       
# MAGIC     return pandas_df