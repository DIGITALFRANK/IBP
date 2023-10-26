# Databricks notebook source
#Constrained GLM modeling.  This module allows the user to parallelize GLM models across cores along with
# passing in beta upper/lower bounds.

import glmnet_python


class GlmModelInfo:
    """
    Class to contain the required information for building a constrained elastic net model
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'target',
            'features',
            'model_id',
            'predict_key_fields'
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

        #Filter variables and beta bounds
        self.x_vars = self.features.get(model_id)
        if "lower_bounds" in self.__dict__:
          self.lower = self.lower_bounds.get(model_id)
        if "upper_bounds" in self.__dict__:
          self.upper = self.upper_bounds.get(model_id)
          self.cl = scipy.array([self.lower,self.upper], dtype = scipy.float64)

        #Setup modeling data
        self.X = self.data[self.x_vars].to_numpy(dtype="float64")
        self.y = self.data[self.target].to_numpy(dtype="float64")

    def set_predict_data(self, predict_data, model_id):
        """
        The predict data and features cannot be provided at instantiation as it is provided
        via the UDF wrapper, so this method is provided to handle provision of train data
        within the UDF.
        """
        #Filter for data fed into udf
        self.data = predict_data[predict_data[self.model_id[0]].isin([model_id])]
        self.x_vars = self.features.get(model_id)

        #Setup predict data
        self.X = self.data[self.x_vars].to_numpy(dtype="float64")

def train_glm(df, model_info, **kwargs):
    """
    Trains glmnet using "glmnet_py" - package that allows for upper/lower constraints

    Args:
        df {pandas.DataFrame}: Train data passed to the function via pyspark groupBy.apply
        model_info {dict}: At a minimum must contain:
            target {str}: Target variable to predict.
            features {dict[str: str]}: The key and variable fields. E.g., Key : [Variables]
            model_id {str}: Name of the model id field
            predict_key_fields {list}: List of fields to retain in prediction output

        kwargs {dict}: Keyword arguments to be passed on to glmnet (e.g., lower/upper bounds, alpha)

    Returns:
        out_models {pandas.DataFrame}: Model Id + model object pickle + coefficient pandas dataframe pickle
    """

    #Note: temporary until we remove server conflict preventing this pkg installation on workers
    import glmnet_python
    from glmnet import glmnet
    from glmnetCoef import glmnetCoef;

    #Setup data for this udf data
    model_id = df[model_info.model_id[0]].iloc[0]
    model_info.set_train_data(df, model_id)

    #Run models
    try:
      mod_fit =glmnet(x = model_info.X, y = model_info.y, cl = model_info.cl, **kwargs)
    except:
      mod_fit =glmnet(x = model_info.X, y = model_info.y, **kwargs)


    #Model pickle
    model_pick = pickle.dumps(mod_fit).decode(model_info.pickle_encoding)

    #Coefficient pickle - obtain coefficients associated with minimum lambda
    min_lambda = mod_fit['lambdau'].min()
    c = glmnetCoef(mod_fit, s = scipy.float64([min_lambda]))
    coefs = pd.concat([pd.DataFrame(["Intecept"]+model_info.x_vars, columns = ['variable']),
                       pd.DataFrame(c, columns = ['coefficient'])],axis=1)
    coef_pick = pickle.dumps(coefs).decode(model_info.pickle_encoding)

    #Return outputs
    df_to_return = pd.DataFrame([model_id], columns=[model_info.model_id[0]])
    df_to_return['model_pick'] = model_pick
    df_to_return['coef_pick'] = [coef_pick]

    return df_to_return

def score_glm(df, model_info, model_pickles):
    """
    Trains glmnet using "glmnet_py" - package that allows for upper/lower constraints

    Args:
        df {pandas.DataFrame}: Score data passed to the function via pyspark groupBy.apply
        model_info {dict}: At a minimum must contain:
            target {str}: Target variable to predict.
            features {dict[str: str]}: The key and variable fields. E.g., Key : [Variables]
            model_id {str}: Name of the model id field
            predict_key_fields {list}: List of fields to retain in prediction output

        model_pickles {dict}: Model id + glmnet pickle objects

    Returns:
        df_to_return {pandas.DataFrame}: Scored data using minimum lambda
    """

    #Note: temporary until we remove server conflict preventing this pkg installation on workers
    import glmnet_python
    from glmnet import glmnet
    from glmnetCoef import glmnetCoef
    from glmnetPredict import glmnetPredict

    #Decode pickle
    model_id = df[model_info.model_id[0]].iloc[0]
    this_model = model_pickles.get(model_id)
    mod_fit = this_model.encode(model_info.pickle_encoding)
    mod_fit = pickle.loads(mod_fit)
    model_info.set_predict_data(df, model_id)

    #Predict
    min_lambda = mod_fit['lambdau'].min()
    preds = glmnetPredict(mod_fit, model_info.X, s = scipy.float64([min_lambda]))

    #Output
    df_to_return = pd.concat([df[model_info.predict_key_fields],
                           pd.DataFrame(preds, columns = ['pred'])],axis=1)

    return df_to_return

#Explode coefficients from pickled objects into flat dataframe
def explode_coefficients(df, model_info, coef_pickles):
    """
    Explodes coefficients stored as pickled objects in dictionary into a flat file (data frame)

    Args:
        predict_data {pandas.DataFrame}: Data passed to the function via pyspark groupBy.apply
        model_info {dict}: At a minimum must contain:
            target {str}: Target variable to predict.
            features {dict[str: str]}: The key and variable fields. E.g., Key : [Variables]
            model_id {str}: Name of the model id field
            predict_key_fields {list}: List of fields to retain in prediction output

        model_pickles {dict}: Model id + glmnet pickle objects

    Returns:
        df_to_return {pandas.DataFrame}: Coefficient dataframe for all pickled objects
    """

    #Decode pickle
    model_id = df[model_info.model_id[0]].iloc[0]
    these_coefs = coef_pickles.get(model_id)
    coefs = these_coefs.encode(model_info.pickle_encoding)
    coefs = pickle.loads(coefs)

    #Output
    df_to_return = coefs
    df_to_return[model_info.model_id[0]] = model_id
    cols = model_info.model_id + ['variable','coefficient']
    df_to_return = df_to_return[cols]

    return df_to_return