# Databricks notebook source
#Parallelizes scklearn models across cores

from sklearn.linear_model import LogisticRegression
from sklearn import tree
import pickle
import pandas as pd

class SckLearnModelInfo:
    """
    Class to contain the required information for building a scklearn model
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'target',
            'features',
            'model_id',
            'algo_func'
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

    def set_train_data(self, train_data, model_id):
        """
        The train data and features cannot be provided at instantiation as it is provided
        via the UDF wrapper, so this method is provided to handle provision of train data
        within the UDF.
        """
        #Filter for data fed into udf
        self.data = train_data[train_data[self.model_id[0]].isin([model_id])]
        self.X = self.data[self.features]
        self.y = self.data[self.target]

    def set_algo_func(self, new_algo_func):
        """
        Updates the algo function parameter
        """
        self.algo_func = new_algo_func

def train_scklearn(df, model_info,**kwargs):
    """
    Trains a scklearn model and outputs pickles of model objects and predictions
    """

    #Setup data for this udf data subset
    model_id = df[model_info.model_id[0]].iloc[0]
    model_info.set_train_data(df, model_id)

    #Train models
    transformer = model_info.algo_func(**kwargs)
    transformer.fit(model_info.X, model_info.y)

    #Model pickle
    model_pick = pickle.dumps(transformer).decode(model_info.pickle_encoding)

    #Predictions pickle
    predictions = transformer.predict(model_info.X)
    df['pred']=pd.DataFrame(predictions, columns=["pred"])
    keep_vars = [model_info.model_id[0],'pred'] + model_info.predict_key_fields
    pred_out = df[keep_vars]
    preds_pick = pickle.dumps(pred_out).decode(model_info.pickle_encoding)

    #Return outputs
    df_to_return = pd.DataFrame([model_id], columns=[model_info.model_id[0]])
    df_to_return['model_pick'] = model_pick
    df_to_return['preds_pick'] = [preds_pick]

    return(df_to_return)

def explode_predictions(df, model_info, pred_pickles):
    """
    Explodes predictions stored as pickled objects into a flat file (data frame)

    Args:
        df {pandas.DataFrame}: Data passed to the function via pyspark groupBy.apply
        model_info {dict}: At a minimum must contain:
            target {str}: Target variable to predict.
            features {dict[str: str]}: The key and variable fields. E.g., Key : [Variables]
            model_id {str}: Name of the model id field
            predict_key_fields {list}: List of fields to retain in prediction output

        pred_pickles {dict}: Model id + pickled object

    Returns:
        df_to_return {pandas.DataFrame}: Dataframe for all pickled objects
    """

    #Decode pickle
    model_id = df[model_info.model_id[0]].iloc[0]
    these_preds = pred_pickles.get(model_id)
    preds = these_preds.encode(model_info.pickle_encoding)
    preds = pickle.loads(preds)

    return (preds)