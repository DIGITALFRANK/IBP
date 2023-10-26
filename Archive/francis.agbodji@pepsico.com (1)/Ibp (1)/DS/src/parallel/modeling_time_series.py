# Databricks notebook source
#Time series modeling.  This module allows the user to parallelize time series models across cores.
def run_time_series(df, model_info, **kwargs):
    """
    Wrapper function for generic statsmodels time series algorithms

    Parameters
    ----------
    df : Pandas dataframe
      Time series data (date field, target field)
    algo_func : Function name
      E.g., ARIMA, Holt
    idx_field : String
      Name of date/id field
    freq : String
      Frequency of data (H, M, W)
    n_ahead : Integer
      Number of future time periods we wish to predict


    Returns
    -------
    vals : Pandas dataframe
      Time series predictions
    """
    #Grab parameters from class
    algo_func = model_info.algo_func
    time_field = model_info.time_field
    target_field = model_info.target_field
    freq = model_info.freq
    n_ahead = model_info.n_ahead

    #model_id = df[model_info.model_id[0]].iloc[0]
    model_id = [df[x].iloc[0] for x in model_info.model_id]
    
    fcst_start_date = df[model_info.fcst_start_field].iloc[0]

    ts_df = df.copy()
    ts_df.sort_values(by=[time_field], inplace=True, ascending=True)
    ts_df = ts_df[[time_field,target_field]]
    ts_df.set_index(time_field, inplace=True)
    #Manual offset adjustment for PEP sunday alignment - make configurable
    new_date_range = pd.date_range(ts_df.index[-1],periods=n_ahead,freq=freq) + pd.DateOffset(1) 
    #Note: commenting out for PEP as it's erroring out (doesn't seem to effect ouputs?)
    #ts_df = ts_df.to_period(freq) 

    #Run desired time series algorithms
    model = algo_func(ts_df, **kwargs)
    model_fit = model.fit()

    #Get predictions
    output = pd.DataFrame(model_fit.forecast(n_ahead))
    output.columns = ['pred']
    output[time_field] = new_date_range
    output = output[[time_field, "pred"]]

    #Shape output
    for x in model_info.model_id:
      output[x] = df[x].iloc[0]
      
    output[time_field] = new_date_range
    output[model_info.fcst_start_field] = fcst_start_date
    #output[time_field] = output[time_field].astype(str)
    
    keep_vars = model_info.model_id +  [model_info.fcst_start_field] +[time_field,"pred"]
    output = output[keep_vars]
    
    output[model_info.fcst_start_field] = output[model_info.fcst_start_field].astype(int)
    output[time_field] = output[time_field].astype(str)   

    return(output)

class TimeSeriesModelInfo:
    """
    Class to contain the required information for building a time series model and
    methods for checking that information.
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'algo_func',
            'time_field',
            'target_field',
            'freq',
            'n_ahead',
            'model_id'
        ]

        # Check required attributes exist
        self.__dict__.update(kwargs)
        self._check_required_attrs_received()

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

def run_auto_arima(df, model_info, **kwargs):
    """
    Wrapper function for auto arima pmdarima

    Parameters
    ----------
    df : Pandas dataframe
      Time series data (date field, target field)
    algo_func : Function name
      E.g., ARIMA, Holt
    idx_field : String
      Name of date/id field
    freq : String
      Frequency of data (H, M, W)
    n_ahead : Integer
      Number of future time periods we wish to predict


    Returns
    -------
    output : Pandas dataframe
      Time series predictions
    """
    #Grab parameters from class
    algo_func = model_info.algo_func
    time_field = model_info.time_field
    target_field = model_info.target_field
    freq = model_info.freq
    n_ahead = model_info.n_ahead

    model_id = df[model_info.model_id[0]].iloc[0]
    fcst_start_date = df[model_info.fcst_start_field].iloc[0]

    ts_df = df.copy()
    ts_df.sort_values(by=[time_field], inplace=True, ascending=True)
    ts_df = ts_df[[time_field,target_field]]
    ts_df.set_index(time_field, inplace=True)
    new_date_range = pd.date_range(ts_df.index[-1],periods=n_ahead,freq=freq) + pd.DateOffset(1) 
    #ts_df = ts_df.to_period(freq)

    #Run desired time series algorithms
    model = algo_func(ts_df, **kwargs)
    output = pd.DataFrame(model.predict(n_ahead))
    output.columns = ['pred']

    #Shape output
    for x in model_info.model_id:
      output[x] = df[x].iloc[0]
      
    output[time_field] = new_date_range
    output[model_info.fcst_start_field] = fcst_start_date
    #output[time_field] = output[time_field].astype(str)
    
    keep_vars = model_info.model_id +  [model_info.fcst_start_field] +[time_field,"pred"]
    output = output[keep_vars]
    
    output[model_info.fcst_start_field] = output[model_info.fcst_start_field].astype(int)
    output[time_field] = output[time_field].astype(str)   

    return(output)