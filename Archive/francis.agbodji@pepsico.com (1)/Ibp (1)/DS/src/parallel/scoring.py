# Databricks notebook source
#Score models using coefficient table.  This module also performs "waterfall" decomposition in order to
# decompose non-linear models (e.g., LOG-LOG).
# TO-DO: Update test functions to use this method

from pyspark import SparkContext
from pyspark.sql import SQLContext

# # Spark Context initialisation
# spark_context = SparkContext()
# sqlContext = SQLContext(spark_context)

class ScoringInfo:
    """
    Class to contain the required information for building a scoring a model using coefficients, along with waterfall effect.  
    Scoring using coefficients is used over scoring using model objects since for larger scale implementations, dataframe 
    calculations perform faster than multiple model scoring.
    
    Note(s):
      * Score data should be logged prior to using this class
    """
 
    def __init__(self, **kwargs):
 
        # These arguments are required by the scoring/waterfall functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'model_id',
            'score_data',
            'coef_dict'
        ]
 
        self.__dict__.update(kwargs)
        self._check_required_attrs_received()       
        self.set_coef_data()
        
 
    def _check_required_attrs_received(self):
        self.missing_attrs = [attr for attr in self._required_attrs if attr not in self.__dict__]
        if self.missing_attrs:
            missing = ', '.join(self.missing_attrs)
            err_msg = f'The following parameters are required but were not provided: {missing}'
            raise TypeError(err_msg)
            
    def set_param(self, **kwargs):
      """
      Overrides a parameter
      """
      self.__dict__.update(kwargs)
 
    def set_coef_data(self):
        """
        Merges coefficient dictionary list into the pyspark dataframe as similarly named columns.
        """ 
        self.coef_data = spark.createDataFrame(self.coef_dict)
        self.x_vars = self.coef_data.columns
        self.x_vars.remove(self.model_id[0])
        self.coef_data = self.coef_data.select((*(col(x).alias(x + '_c') for x in self.coef_data.columns)))
        self.coef_data = self.coef_data.withColumnRenamed(self.model_id[0]+"_c",self.model_id[0])    
        self.coef_data = self.coef_data.fillna(0)
        self.score_data = self.score_data.join(self.coef_data, on=self.model_id[0], how="left")
        
    def score_models_using_coefficients(self):
        """
        Scores models with merged in coefficients
        """ 
        self.score_data = (reduce(
        lambda df, col_name: df.withColumn(col_name, col(col_name) * col(col_name + "_c")),
        self.x_vars,
        self.score_data
        ))
        self.score_data = row_sum_DF(self.score_data, self.x_vars, "pred")
        #self.score_data = self.score_data.select(self.predict_key_fields +  ["pred"] )
        
    def do_waterfall(self):
        """
        Performs waterfall (variable attribution changes compared to Prior Period variable values)
        """
        #Get Last period values
        old_score_data = self.score_data #Store unaltered score data
        self.score_data = self.score_data.drop("pred")
        lag_vars = list(set(self.x_vars) - set(["Intercept"]))
        self.score_data = do_lags_N(self.score_data,self.date_field,lag_vars,-1,self.model_id[0])
        
        #Assume this week is last week, store actual this week in "this_week columns"
        for i in lag_vars:
          self.score_data = self.score_data.withColumn(i + "_this_week",col(i))
          self.score_data = self.score_data.withColumn(i,col(i + "_lag1"))
        
        #Loop through waterfall levels
        cnt = 0
        for i in self.waterfall_levels:
          #Increment Last period values for this level
          replace_these_vars = self.waterfall_levels.get(i)
          for this_var in replace_these_vars:
            self.score_data = self.score_data.withColumn(this_var,col(this_var + "_this_week" ))
          
          #Score demand 
          self.score_models_using_coefficients()
          self.score_data = self.score_data.withColumn("waterfall_level", lit(i))
 
          #Append output
          cnt = cnt + 1
          if (cnt==1):
            self.waterfall_data = self.score_data
          else:
            self.waterfall_data = self.waterfall_data.union(self.score_data) 
            
        self.waterfall_data = self.waterfall_data.select(self.predict_key_fields + ["waterfall_level","pred"])
        self.score_data = old_score_data #Store unaltered score data

def select_best_model(df, models, actual_var, select_hier = None):
  """
  Selects the best model for some level of the hierarchy

  Parameters
  ----------
  df : PySpark Dataframe
     Scored predictions
  models : List
      List of column names that contain the model predictions
  actual_var : String
      Column name of actual demand
  select_hier : List
      Level at which to select best model (e.g., country, MODEL_ID).  If left blank, one model will be selected for entire
      population

  Returns
  -------
  gof_report : PySpark Dataframe
      Weighted accuracies at select_hier level and best model selection
  """
  
  if select_hier is None:
    df = df.withColumn("dummy_seg",lit("dummy_seg"))
    select_hier = ["dummy_seg"]
    
  #Calculate accuracies at specified level
  gof_nested_dict = {}
  for i in models:
    this_acc_dict = {'error_func' : calculate_wtd_accuracy,
                     'parameters': [df, actual_var, i, select_hier]
                    }
    gof_nested_dict[i + "_ACC"] = this_acc_dict

  gof_report = calculate_gof(df, select_hier, gof_nested_dict)
  
  #Find maximum accuracy at specified level
  cols = add_suffix_to_list(models, "_ACC")
  cond = "F.when" + ".when".join(["(F.col('" + c + "') == F.col('max_accuracy'), F.lit('" + c + "'))" for c in gof_report.columns])
  gof_report = gof_report.withColumn("max_accuracy", greatest(*cols))\
      .withColumn("best_model", eval(cond))
  gof_report = gof_report.withColumn('best_model', regexp_replace('best_model', '_ACC', ''))
  
  return (gof_report)

#TO-DO: Decomp functions below are too complex after refactoring for PEP.  Modularize and simplify if secondary regression is used
def get_baseline_demand(scoring_class, turn_off_variables, switch_variables, impute_level_dict = None):
  """
  Scores baseline demand by "turning off" promotion variables (e.g., promotions) and switching variables to non promoted values 
  (e.g., avg price to non promo price)

  Parameters
  ----------
  scoring_class : PySpark Dataframe
     Scoring class
  turn_off_variables : List
      List of incremental variables to "turn off" and impute to 0 (e.g., promo acv)
  switch_variables : Dictionary
      List of incremental variables to switch to another value (e.g., Avg Price switch to Non Promo Price)

  Returns
  -------
  baseline_dat : PySpark Dataframe
      Input dataset scored with baseline demand
  """
  
  #Get total demand predictions from class
  orig_data = scoring_class.score_data
  scoring_class.score_models_using_coefficients()
  total_demand_preds = scoring_class.score_data
  total_demand_preds = total_demand_preds.withColumn("total_demand_pred",col("pred"))
  df = orig_data
  
  #Turn off variables
  df = (reduce(
    lambda df, turn_off:
      df.withColumn(turn_off, lit(0)),
    turn_off_variables,
    df
  ))
  
  #Switch variables
  df = (reduce(
    lambda df, switch:
      df.withColumn(switch, col(switch_variables.get(switch))),
    list(set(switch_variables.keys())),
    df
  ))
  
  #Impute variables
  for i in impute_level_dict.keys():
      if len(intersect_two_lists([i], df.columns)) > 0:
        #Get imputation strategy for this variable
        level_at_which_to_impute = impute_level_dict.get(i)
        df = get_hierarchical_statistics(df, level_at_which_to_impute, i, [avg])
        df = df.withColumn(i, col("avg_" + i))
  
  #Get baseline dat
  scoring_class.set_param(**{'score_data' : df})
  scoring_class.score_models_using_coefficients() #Example score 
  
  baseline_preds = scoring_class.score_data
  baseline_preds = baseline_preds.withColumnRenamed("pred","baseline_pred")
  baseline_preds = baseline_preds.join(total_demand_preds.select( scoring_class.predict_key_fields + ["total_demand_pred"]),
                                      on=scoring_class.predict_key_fields, how="left")
  return(baseline_preds)

def turn_off_incrementals(df, scoring_class, turn_off_variables, switch_variables, impute_level_dict = None):
  #Turn off variables
  df = (reduce(
    lambda df, turn_off:
      df.withColumn(turn_off, lit(0)),
    turn_off_variables,
    df
  ))
  
  #Switch variables
  df = (reduce(
    lambda df, switch:
      df.withColumn(switch, col(switch_variables.get(switch))),
    list(set(switch_variables.keys())),
    df
  ))
  
  #Impute variables
  for i in impute_level_dict.keys():
      if len(intersect_two_lists([i], df.columns)) > 0:
        #Get imputation strategy for this variable
        level_at_which_to_impute = impute_level_dict.get(i)
        df = get_hierarchical_statistics(df, level_at_which_to_impute, i, [avg])
        df = df.withColumn(i, col("avg_" + i))
  
  return(df)

def do_detailed_decomposition_using_coefficients(scoring_class, turn_off_variables, switch_variables, impute_level_dict, base_variables):
  """
  Performs decomposition by "levels" variable groups.  This function first turns off relevant variables and then increments
    variables to the actual value and rescores demand.

  Parameters
  ----------
  scoring_class : PySpark Dataframe
     Scoring class
  turn_off_variables : List
      List of incremental variables to "turn off" and impute to 0 (e.g., promo acv)
  switch_variables : Dictionary
      List of incremental variables to switch to another value (e.g., Avg Price switch to Non Promo Price)
  levels : List
      List of variables to perform decomp ({variable group}:[variables that consist of this group]) 
      E.g., {'Own Cannibalization': ['own_comp_1','own_comp_2']}

  Returns
  -------
  baseline_dat : PySpark Dataframe
      Input dataset scored with baseline demand and each level group attributed demand
  """

  orig_dat = scoring_class.score_data
  waterfall_levels = scoring_class.waterfall_levels
  baseline_pred = get_baseline_demand(scoring_class, turn_off_variables, switch_variables, impute_level_dict)  

  #Retain "true" variable values
  coef_ls = list(set(val for dic in scoring_class.coef_dict for val in dic.keys()))
  coef_ls = subtract_two_lists(coef_ls,["MODEL_ID"])
  orig_dat = (reduce(
    lambda df, coef:
      df.withColumn(coef + "_true", col(coef)),
    coef_ls,
    orig_dat
    ))

  orig_dat = turn_off_incrementals(orig_dat, scoring_class, turn_off_variables, switch_variables, impute_level_dict)

  cnt = 0
  for i in waterfall_levels.keys():
    print(i)
    if cnt == 0:
      increment_these_vars = waterfall_levels.get(i) + ["Intecept"] + base_variables
      running_increment_vars = increment_these_vars
      first_pred = i + "_pred"
    else:
      increment_these_vars = waterfall_levels.get(i) 
      running_increment_vars = running_increment_vars + increment_these_vars
    cnt = cnt + 1
    #print("turning on")
    #print(running_increment_vars)

    #Increment variables for this level
    orig_dat = (reduce(
    lambda df, coef:
      df.withColumn(coef, col(coef + "_true")),
    running_increment_vars,
    orig_dat
    ))
  
    #Score demand 
    scoring_class.set_param(**{'score_data' : orig_dat})
    scoring_class.score_models_using_coefficients()
    orig_dat = scoring_class.score_data
    orig_dat = orig_dat.withColumn("pred",exp(col("pred"))-1)
    orig_dat = orig_dat.withColumnRenamed("pred", i + "_pred")
    
  df_out = orig_dat.join(baseline_pred.select(scoring_class.predict_key_fields + ["baseline_pred","total_demand_pred"] ), 
                         on=scoring_class.predict_key_fields, how="left") 
  df_out = df_out.withColumn("baseline_pred",exp(col("baseline_pred"))-1)
  df_out = df_out.withColumn("total_demand_pred",exp(col("total_demand_pred"))-1)
  
  #Do decomp
  j = 1
  levels_list = list(waterfall_levels)
  for i in levels_list:
    if j == 1:
      df_out = df_out.withColumn(i + "_decomp", col(i + "_pred") - col("baseline_pred"))
    else:
      last_pred =  levels_list[j-2] + "_pred"
      df_out = df_out.withColumn(i + "_decomp", col(i + "_pred") - col(last_pred))
    j = j + 1
  
  return (df_out)

#Scale to actuals
def scale_decomps_to_actual(df_to_scale, df_decomps, decomp_vars, scoring_class, actuals_var, levels_to_treat):
  """
  This function scales decomposition factors (% attribution compared to total demand) into actuals.  "Actuals"
  can be either the best forecast or the actual historicals.

  Parameters
  ----------
  scoring_class : PySpark Dataframe
     Scoring class
  turn_off_variables : List
      List of incremental variables to "turn off" and impute to 0 (e.g., promo acv)
  switch_variables : Dictionary
      List of incremental variables to switch to another value (e.g., Avg Price switch to Non Promo Price)

  Returns
  -------
  baseline_dat : PySpark Dataframe
      Input dataset scored with baseline demand
  """
  
  #Get % decomp attribution / total demand
  df_decomps = (reduce(
    lambda df, this_decomp:
      df.withColumn(this_decomp + "_pct", col(this_decomp)/col("total_demand_pred")),
    decomp_vars,
    df_decomps
  ))

  #Apply percentage to actual predictions
  group_pct_list = add_suffix_to_list(decomp_vars, "_pct")
  df_to_scale = df_to_scale.join(df_decomps.select(scoring_class.predict_key_fields + decomp_vars + group_pct_list), 
                                   on=scoring_class.predict_key_fields,
                                   how="left")
  #Treat missing percentages for when there were not enough rows to model
  for i in levels_to_treat:  
    df_to_scale = (reduce(
      lambda df, this_decomp:
        get_hierarchical_statistics(df, i + [TIME_VAR], this_decomp, [avg]),
      group_pct_list,
      df_to_scale
    ))
    df_to_scale = (reduce(
      lambda df, this_decomp:
        df.withColumn(this_decomp, when(col(this_decomp).isNull(),
                                                col("avg_" + this_decomp) ).otherwise(col(this_decomp))),
      group_pct_list,
      df_to_scale
    ))

  #Apply percentage to "actuals" (either actual forecast or actual historicals)
  df_to_scale = (reduce(
    lambda df, this_decomp:
      df.withColumn(this_decomp , col(this_decomp + "_pct")*col(actuals_var)),
    decomp_vars,
    df_to_scale
  ))

  df_to_scale = row_sum_DF(df_to_scale, decomp_vars, "total_demand")
  #df_to_scale = df_to_scale.select(scoring_class.predict_key_fields + [actuals_var])
  return(df_to_scale)