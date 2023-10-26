# Databricks notebook source
from typing import Dict
import pandas as pd

# COMMAND ----------

class StageModelingInfoDict:
    """
    Class to contain the required information for running staged models
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'target',
            'train_id_field', #Field name in data containing a train indicator (for static models)
            'group_level', #Field name in data containing the "model name" that you want to run 
            'train_func_map', #Dictionary of "model name" and the associated training function
            'pred_func_map', #Dictionary of "model name" and the associated prediction function
            'hyperparams', #Dictionary of "model name" and the associated hyperparameters
            'pickle_encoding' #Dictionary of "model name" and the associated hyperparameters
          
        ]

        self.__dict__.update(kwargs)
        self._check_required_attrs_received()

        if self.pickle_encoding is None:
          self.pickle_encoding = 'latin-1'

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

# Palaash Note: modified to handle hyperparameter update
def parallelize_core_models(df:pd.DataFrame, model_info:Dict, hyperparams_df:pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Wrapper function for running stage1 algorithms using PySpark UDF's

    Parameters
    ----------
    df : Pandas dataframe
      Modeling data
    model_info : Dictionary containing required stage 1 modeling info (StageModelingInfoDict class ensures all required fields are present)
      Dictionary
    hyperparams_df: contains the hyperparameter tuning results
    Returns
    -------
    df_to_return : Pandas dataframe
      Dataframe for pickled modeling objects  
    """
    import gc
    import pickle
    
    #Grab parameters from class
    target = model_info.target
    train_id_field = model_info.train_id_field[0]
    group_level = model_info.group_level
    train_func_map = model_info.train_func_map
    pickle_encoding = model_info.pickle_encoding

    # check/get param mapping
    try:
      param_required = [x for x in model_info.group_level if x not in ['FCST_START_DATE']]
      param_available = [x for x in hyperparams_df.columns if x not in ['params', 'one_model_dummy_seg']] #'params'

      if sorted(param_required)==sorted(param_available):
        #get tuned params
        hyperparams = df[param_required].drop_duplicates().merge(hyperparams_df, on=param_required, how='left')
        hyperparams = {hyperparams['train_func'].iloc[0]: eval(hyperparams['params'].iloc[0])}
      else:
        #using default params if the tuning level and prediction level do not match
        hyperparams = model_info.hyperparams
    except:
      #falling back to default params in case of error
      hyperparams = model_info.hyperparams
    
    #Lookup the training and prediction functions associated with "training_name_field" present in this set of data
    this_algo_str = df[model_info.group_level[0]].iloc[0]
    
    this_algo = train_func_map.get(this_algo_str)
    this_hyper = hyperparams.get(this_algo_str)
    
    #TODO: findout why this column was not int
    this_hyper = {x[0]:int(x[1]) if x[0]=='max_leaves' else x[1] for x in this_hyper.items()}
        
    #Run model
    df_new = df.select_dtypes(include='number') #Only keep numeric fields
#     train_pd = df_new[df_new[model_info.train_id_field[0]] == 1]
#     holdout_pd = df_new[df_new[model_info.train_id_field[0]] == 0]    
    model = this_algo(df_new, this_hyper)
    
    #Model pickle
    model_pick = pickle.dumps(model).decode(pickle_encoding)
    
    #Return outputs
    df_to_return = df[group_level].drop_duplicates()  
    df_to_return['model_pick'] = model_pick

    del target
    del train_id_field
    del group_level
    del train_func_map
    del pickle_encoding
    del this_algo_str
    del this_algo
    del this_hyper
    del df_new
    del model
    del model_pick
    gc.collect()
    
    return df_to_return
  
def score_forecast(df:pd.DataFrame, model_info:Dict, OBJECTS_DICT:Dict, **kwargs) -> pd.DataFrame:
    """
    Scores pyspark dataframe with multiple models contained in a dictionary of model objects

    Args:
        df {pandas.DataFrame}: Data passed to the function via pyspark groupBy.apply
        model_info {dict}: StagedModelingInfoDict class containing required info for core models
        pred_pickles {dict}: Model Name + Model Object

    Returns:
        df_to_return {pandas.DataFrame}: Dataframe with scored predictions as new appended column
    """
    import pickle
    import gc

    #Lookup the training and prediction functions associated with "training_name_field" present in this set of data
    this_lookup_str = df["score_lookup"].iloc[0] #TO-DO: Remove hardcoding
    this_algo_str = df[model_info.group_level[0]].iloc[0]

    #Get information from model class
    group_level = model_info.group_level
    pred_func_map = model_info.pred_func_map
    pred_func = pred_func_map.get(this_algo_str)
    id_fields = model_info.id_fields
    encoding = model_info.pickle_encoding
    #    train_id_field = model_info.train_id_field

    #Convert string pickle into model object
    model_object = OBJECTS_DICT.get(this_lookup_str)[this_lookup_str]
    model_object = model_object.encode(encoding)
    model_object = pickle.loads(model_object)

    #Predict
    df_new = df.select_dtypes(include='number')
    preds = pred_func(df_new, model_object)

    #Output
    df_to_return = df[id_fields + group_level]
    df_to_return["pred"] = pd.DataFrame(preds)

    del this_lookup_str
    del this_algo_str
    del group_level
    del pred_func_map
    del pred_func
    del id_fields
    del encoding
    del model_object
    del df_new
    del preds
    gc.collect()
    
    return df_to_return

# COMMAND ----------

def score_forecast_old(df, model_info, model_objects, date_field=None, **kwargs):
    """
    Scores pyspark dataframe with multiple models contained in a dictionary of model objects

    Args:
        df {pandas.DataFrame}: Data passed to the function via pyspark groupBy.apply
        model_info {dict}: StagedModelingInfoDict class containing required info for core models
        pred_pickles {dict}: Model Name + Model Object

    Returns:
        df_to_return {pandas.DataFrame}: Dataframe with scored predictions as new appended column
    """

    #Lookup the training and prediction functions associated with "training_name_field" present in this set of data
    this_algo_str = df[model_info.group_level[0]].iloc[0]

    #Get information from model class
    group_level = model_info.group_level
    pred_func_map = model_info.pred_func_map
    pred_func = pred_func_map.get(this_algo_str)
    id_fields = model_info.id_fields
    encoding = model_info.pickle_encoding
    train_id_field = model_info.train_id_field[0]
    
    #Convert string pickle into model object
    model_object = model_objects.get(this_algo_str)
    #Check if dict is 2D (and thus rolling), need to get inner dictionary
#     if any(isinstance(i,dict) for i in model_objects.values()) == True:
    if date_field:
      this_date_int = int(df[[date_field]].iloc[0])
      model_object = model_object.get(this_date_int)

    model_object = model_object.encode(encoding)
    model_object = pickle.loads(model_object)

    #Predict
    df_new = df.select_dtypes(include='number')
    preds = pred_func(df_new, model_object)

    #Output
    df_to_return = df[id_fields + group_level]
    df_to_return["pred"] = pd.DataFrame(preds)
    #df_to_return = pd.DataFrame(df[], columns=["pred"])

    return (df_to_return)