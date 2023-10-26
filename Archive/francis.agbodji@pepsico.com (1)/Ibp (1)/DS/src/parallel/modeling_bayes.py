# Databricks notebook source
#Bayesian modeling
#Note: UDF passes in predict_data as different pandas format, requiring squeezing

import numpy as np
import pandas as pd
import pymc3 as pm

class BayesModelInfo:
    """
    Class to contain the required information for building a hierarchical regression model and
    methods for checking and sorting that information.
    """

    def __init__(self, **kwargs):

        # These arguments are required by the modelling functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'target',
            'features',
            'train_data',
            'model_id',
            'model_index',
            'predict_key_fields',
        ]

        # Set sigmas to None here, so that we can check if we need to provide defaults
        # once the features have been provided.
        self.sigmas = None
        self.__dict__.update(kwargs)
        self._check_required_attrs_received()

        # We also need the number of rows in the training data and the
        # model indices for each dataset.
        #self.train_index = self.train_data['model_index'].values
        #self.n_index = len(self.train_data['model_index'].unique())
        self.train_index = self.train_data[self.model_index].values
        self.n_index = len(np.unique(self.train_data[self.model_index]))

        # Set the default hyperprior standard deviations to 100 (weak prior), if not provided explicitly.
        if self.sigmas is None:
            self.sigmas = [100 for _ in self.features]

        # Set data and idx attributes.  The default is for training.  This can be updated with the
        # set_train_or_pred method.
        self.data = self.train_data
        self.idx = self.train_index
        self.n = self.n_index

    def _check_required_attrs_received(self):
        self.missing_attrs = [attr for attr in self._required_attrs if attr not in self.__dict__]
        if self.missing_attrs:
            missing = ', '.join(self.missing_attrs)
            err_msg = f'The following parameters are required but were not provided: {missing}'
            raise TypeError(err_msg)

    def set_predict_data(self, predict_data):
        """
        The predict data cannot be provided at instantiation as it is provided via the
        UDF wrapper, so this method is provided to handle provision of predict data
        within the UDF.
        """
        self.n = len(predict_data[self.model_index[0]].unique())
        self.predict_data = predict_data
        self.predict_index = self.predict_data[self.model_index].T.squeeze().values
        unique_ids = self.predict_data[self.model_id].T.squeeze().unique().tolist()
        self.train_data = self.train_data[self.train_data[self.model_id[0]].isin(unique_ids)]
        self.train_index = self.train_data[self.model_index].T.squeeze().values
        #self.n = len(np.unique(self.predict_data[self.model_index]))


    def set_train_or_pred(self, train_or_pred):
        """
        Set whether the instance has training or prediction data for the data and idx attributes.
        """

        if train_or_pred.lower() == 'train':
            self.data = self.train_data
            self.idx = self.train_index
            #self.n = len(np.unique(self.train_data[self.model_index]))
        elif train_or_pred.lower() in ['pred', 'predict']:
            self.data = self.predict_data
            self.idx = self.predict_index
            #self.n = len(np.unique(self.predict_data[self.model_index]))
        else:
            err_msg = f"Expected `train` or `pred[ict]`.  Got {train_or_pred}."
            raise TypeError(err_msg)

def model_factory(model_info, n_skus):
    """
    Create a PyMC3 linear model with two IVs for a given subset of input data.

    Arguments:
        model_info {ModelInfo}: data and parameters required to make the model.

    Returns:
        model {pymc3.Model}: model ready for sampling.
    """

    with pm.Model() as model:
        # Keeping 0/100 so that we have a "weak prior"
        # Hyperpriors for group nodes
        mu_a = pm.Normal('mu_a', mu=0., sigma=100)
        sigma_a = pm.HalfNormal('sigma_a', 5.)
        mu_b = pm.Normal('mu_b', mu=0., sigma=model_info.sigmas[0])
        sigma_b = pm.HalfNormal('sigma_b', 5.)
        mu_c = pm.Normal('mu_c', mu=0., sigma=model_info.sigmas[1])
        sigma_c = pm.HalfNormal('sigma_c', 5.)

        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_skus)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_skus)
        c = pm.Normal('c', mu=mu_c, sigma=sigma_c, shape=n_skus)

        # Model error
        eps = pm.HalfCauchy('eps', 5.)

         # Model the posterior mean
        intercept = a[model_info.idx]
        first_term = b[model_info.idx] * model_info.data[model_info.features[0]].values
        second_term = c[model_info.idx] * model_info.data[model_info.features[1]].values
        sales_est = intercept + first_term + second_term

        # Data likelihood
        y = pm.Normal('y', sales_est, sigma=eps, observed=model_info.data[model_info.target].squeeze('columns'))

        return model

def take_exponent_of_sales_percentiles(sales, percentiles):
    """
    Simple wrapper around numpy's ex1mp and percentiles functions.

    Wraps the two together into a testable form.

    Args:
        sales {array-like}: logged sales distributions.
        percentiles {array-like}: percentiles to compute from the sales distributions.

    Returns:
        vals {array-like}: transposed sales percentiles.
    """

    vals = np.percentile(sales, percentiles, axis=0)
    vals = np.expm1(vals)
    vals = np.transpose(vals)

    return vals

def train_and_predict_hierarchical_bayes(predict_data, model_info, **kwargs):
    """
    Trains a hierarchical Bayesian linear regression model of sales.

    Assumes that all columns other than MODEL_ID, model_index and the target are explanatory
    variables.

    Args:
        predict_data {pandas.DataFrame}: Data passed to the function via pyspark groupBy.apply
        model_info {dict}: At a minimum must contain:
            target {str}: Target variable to predict.
            fields {dict[str: str]}: The key and index fields.  E.g. MODEL_ID, ID (option ID)
                                     and model_index.
            training_data {pandas.DataFrame}: Historical data, including a training data
                                              model ID, model_index (integer ID corresponding
                                              the the model IDs), the target variable and one
                                              explanatory variable, used to generate a posterior
                                              distribution with which out of sample predictions
                                              can be made.
            percentiles {list[float]}: Percentiles of the posterior distribution of sales to
                                       be computed.
            sigmas {dict[str: float]}: Hyperprior standard deviations for the coefficient prior
                                       means (including intercept).

        kwargs {dict}: Keyword arguments to be passed on to pymc3.sample().

    Returns:
        out_preds {pandas.DataFrame}: Sales predictions at each of the provided percentiles.
    """

    # Add the predict data to the model_info object, as it is passed 'live' by the UDF.
    model_info.set_predict_data(predict_data)

    # Ensure the training data is set for training
    model_info.set_train_or_pred('train')
    n_skus = len(predict_data.MODEL_ID.unique())

    #Run hierarchical regression
    with model_factory(model_info, n_skus):
      hierarchical_trace = pm.sample(**kwargs)


    model_info.set_train_or_pred('predict')
    with model_factory(model_info, n_skus):
        ppc = pm.sample_posterior_predictive(hierarchical_trace)

    # Shape data for output
    ids = model_info.data[model_info.predict_key_fields]
    vals = take_exponent_of_sales_percentiles(ppc["y"], model_info.percentiles)

    colnames = ["bayes_pred{}".format(i) for i in model_info.percentiles]
    vals = pd.DataFrame(data=vals, columns=colnames)
    out_preds = pd.concat(
        [ids.reset_index(drop=True), vals.reset_index(drop=True)], axis=1
    )

    return out_preds