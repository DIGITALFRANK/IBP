# Databricks notebook source
class ParallelBayesHyperparam:
  """
  Class to contain the required information for running hyper parameter tuning parallely in PySpark
  """
  def __init__(self, **kwargs):

      # These arguments are required by the modelling functions.  Their presence is
      # checked by check_required_attrs_received, with an error being raised if
      # they aren't set.
      self._required_attrs = [
          'start_week',
          'holdout_duration',
          'parameter_grid',
          'error_metric_func'
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
      
def parallel_stage1_hyperparam(df, param_info_cls, **kwargs):
    """
    Runs hyperparameter tuning separately by segment.  Each segment is split onto worker cores.
    """
    
    #Setup data for this udf data
    group_level = param_info_cls.group_level[0]
    group_id = df[group_level].iloc[0]

    df_new = df.select_dtypes(include='number') #Only keep numeric fields
    
    #Run hyperparameter tuning
    bayesian_dict = param_info_cls.algo_func(df_new, param_info_cls.start_week, param_info_cls.holdout_duration,\
                                               parameter_grid=param_info_cls.parameter_grid,\
                                               error_metric_func=param_info_cls.error_metric_func,\
                                               **kwargs)


    #Hyperparam dict
    hyper_pick = pickle.dumps(bayesian_dict).decode(param_info_cls.pickle_encoding)

    #Return outputs
    df_to_return = pd.DataFrame([group_id], columns=[group_level])
    df_to_return['hyper_pick'] = hyper_pick

    return df_to_return
  
def parallel_stage2_hyperparam(df, param_info_cls, **kwargs):
    """
    Runs stage 2 hyperparameter tuning separately by segment.  Each segment is split onto worker cores.
    """
    
    #Setup data for this udf data
    group_level = param_info_cls.group_level[0]
    group_id = df[group_level].iloc[0]

    #Run hyperparameter tuning
    bayesian_dict = param_info_cls.algo_func(df, param_info_cls.start_week, \
                                                 param_info_cls.holdout_duration,\
                                                 param_info_cls.parameter_grid,\
                                                 **kwargs)

    #Hyperparam dict
    hyper_pick = pickle.dumps(bayesian_dict).decode(param_info_cls.pickle_encoding)

    #Return outputs
    df_to_return = pd.DataFrame([group_id], columns=[group_level])
    df_to_return['hyper_pick'] = hyper_pick

    return df_to_return