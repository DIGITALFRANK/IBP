# Databricks notebook source
# DBTITLE 1,Parameter Estimation
def estimate_gamma_params(arr):
  """
  Estimate params using empirical Bayes, where (shape, scale) => (alpha, beta)
  """
  mu = arr.mean()
  stdev = arr.std()

  scale = mu/(stdev**2)
  shape = scale * mu

  return {'scale':scale, 'shape':shape}


def estimate_normal_params(arr):
  """
  Estimate params using empirical Bayes, where (mu, sigma) => (mu mean, sigma std)
  """ 
  mu = arr.mean()
  sigma = arr.std()

  return {'mu':mu, 'sigma':sigma}


def estimate_cauchy_params(arr):
  """
  Estimate params using empirical Bayes, where (location, scale) => (alpha, beta)
  """ 
  iqr_lower = np.percentile(arr, 25) 
  iqr_upper = np.percentile(arr, 75)
  
  scale = (iqr_upper - iqr_lower) / 2
  
  if scale == 0:
    location = np.mean(arr)
  else:
    location = ( 1 / (scale * 3.14159265) )

  return {'location':location, 'scale':scale}


# COMMAND ----------

# DBTITLE 1,Bayesian Models
#TODO marked for deletion; improper log-likelihood function
# def trainParetoNBD(input_pd, time_switchpoint=None, estimate_switchpoint=True, draws=10000, chains=3, tune=1000, **kwargs):
#   """
#   Finds Pareto-NBD (Poisson mixture of gammas) using MCMC. Assumes:
#   - Customers 'die' (stop purchasing) according to some Pareto distribution
#   - While alive, they make purchases according to some Poisson distribution
#   - Their purchase rate (lambda for their Poisson distribution) is distributed according to the Gamma distribution
#   """
  
#   observations = input_pd[target_var]
  
#   model = pm.Model()

#   with model:
#     if time_switchpoint:
#       time = input_pd[time_var]
      
#       if estimate_switchpoint:
#         switchpoint = pm.DiscreteUniform('switchpoint', lower=time.min(), upper=time.max(), testval=time_switchpoint)
      
#       else:
#         switchpoint = time_switchpoint
      
#       # poisson priors
#       early_params = estimate_gamma_params(observations[time < time_switchpoint])
#       late_params = estimate_gamma_params(observations[time >= time_switchpoint])

#       early_lambda = pm.Gamma('early_lambda', alpha=early_params["shape"], beta=early_params["scale"])
#       late_lambda = pm.Gamma('late_lambda', alpha=late_params["shape"], beta=late_params["scale"])
      
#       # pareto priors
#       early_shape = pm.Poisson('early_shape', mu=early_lambda)
#       late_shape = pm.Poisson('late_shape', mu=late_lambda)
#       _shape = pm.math.switch(switchpoint >= time, early_shape, late_shape)
      
#       _scale = pm.Uniform('pareto_scale', lower=0, upper=1)

#     else:
#       # poisson priors
#       params = estimate_gamma_params(observations)
#       _lambda = pm.Gamma('lambda', alpha=params["shape"], beta=params["scale"])
      
#       # pareto priors
#       _shape = pm.Poisson('_shape', mu=_lambda)
#       _scale = pm.Uniform('pareto_scale', lower=0, upper=1)
      
#     posterior = pm.Pareto('y_obs', alpha=_shape, m=_scale, observed=observations)
      
#     map_estimate = pm.find_MAP()
    
#     step_method = pm.Metropolis() # default would mix NUTS and Metro samplers, which has been shown to cause issues: https://docs.pymc.io/notebooks/sampling_compound_step.html
#     trace = pm.sample(draws=draws, chains=chains, tune=tune, **kwargs)
    
#   return (trace, model)


def train_NBD(input_pd, time_switchpoint=None, estimate_switchpoint=True, show_progress=True, draws=10000, chains=3, tune=1000, **kwargs):
  """
  Finds negative binomial distribution (Poisson mixture of gammas) using MCMC. Assumes:
  - Customers make purchases according to some Poisson distribution
  - Their rate of purchase is distributed according to the Gamma distribution
  """
  
  observations = input_pd[target_var]
  
  model = pm.Model()

  with model:
    if time_switchpoint:
      time = input_pd[time_var]
      
      if estimate_switchpoint:
        switchpoint = pm.DiscreteUniform('switchpoint', lower=time.min(), upper=time.max(), testval=time_switchpoint)
      else:
        switchpoint = time_switchpoint
      
      early_params = estimate_gamma_params(observations[time < time_switchpoint])
      late_params = estimate_gamma_params(observations[time >= time_switchpoint])

      early_lambda = pm.Gamma('early_lambda', alpha=early_params["shape"], beta=early_params["scale"])
      late_lambda = pm.Gamma('late_lambda', alpha=late_params["shape"], beta=late_params["scale"])
      
      _lambda = pm.math.switch(switchpoint >= time, early_lambda, late_lambda)

    else:
      params = estimate_gamma_params(observations)
      _lambda = pm.Gamma('lambda', alpha=params["shape"], beta=params["scale"])
      
    posterior = pm.Poisson('y_obs', mu=_lambda, observed=observations)
      
    map_estimate = pm.find_MAP()
    
    trace = pm.sample(draws=draws, chains=chains, tune=tune, progressbar=show_progress, **kwargs)
    
  return (trace, model)


# COMMAND ----------

# DBTITLE 1,Inference
def predict_bayes_model(trace, model, num_samples=1, agg_func=np.mean):
  """
  Generates predictions using a PyMC3 trace and model. The higher your num_samples, the more stable your forecasts will be, but the less realistic they might be. You can use plot_posterior_samples to visualize this relationship.
  """
  predictions = pm.sample_posterior_predictive(trace, samples=num_samples, model=model)['y_obs']
  
  if agg_func:
    predictions = agg_func(predictions, axis=0)
    
  return predictions

# COMMAND ----------

# DBTITLE 1,Visualizations
def plot_trace_samples(actuals_pd, trace, bayes_model, incl_switchpoint=True, trace_obs='y_obs', adj_plotsize=True, use_subplots=True):
  '''
  Uses globally-defined: time_var, target_var, switchpoint_week
  Documentation: https://docs.pymc.io/notebooks/posterior_predictive.html
  
  INPUTS:
  - input Pandas dataframe - must be Pandas, should be a subset of full pd - should represent UNIQUE customer/product combination
  - trace - training TRACE output associated with that specific customer/product combination
  - bayes model - training MODEL output associated with that specific customer/product combination
  - other inputs can be adjusted as needed by user
  
  RETURNS:
  - side-by-side plots of actuals and re-drawn distribution plots
  '''
  
  before_switch_pd = actuals_pd[actuals_pd[time_var] < switchpoint_week]
  after_switch_pd = actuals_pd[actuals_pd[time_var] >= switchpoint_week]
    
  if adj_plotsize: plt.figure(figsize=(12,6))

  ## Plotting actual data - serves as visual reference for re-drawn data
  if use_subplots: plt.subplot(1,2,1)
  if incl_switchpoint:
    sns.distplot(before_switch_pd[target_var], color='cornflowerblue', hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True})
    sns.distplot(after_switch_pd[target_var], color='darkblue', hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True})
    plt.legend(['Before Switchpoint', 'After Switchpoint'], loc='best')
    plt.grid(b=False)
  else: 
    sns.distplot(actuals_pd[target_var], color='cornflowerblue', hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True})
    plt.legend(['Actuals'], loc='best')
    plt.grid(b=False)

  ## Plotting replicated (re-drawn) data using Bayesian model associated with trace
  if use_subplots: plt.subplot(1,2,2)
  ppc = pm.sample_posterior_predictive(trace, samples=500, model=bayes_model)
  sns.distplot([n.mean() for n in ppc[trace_obs]], color='seagreen', hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True})
  plt.legend(['Replicated Data'], loc='best')
  plt.grid(b=False)

  display()



def plot_trace_histograms(trace, params_to_plot_list=['late_lambda', 'early_lambda'],\
                              num_bins=20, graph_alpha=0.35, title=None, xlabel=None, ylabel=None):
  '''
  Plots histogram of target variable trace output (distribution parameter) from function
  Allows plotting in same space/axis to more directly compare posterior distributions before and after a certain specified 'switchpoint'
  '''
    
  for param in params_to_plot_list: plt.hist(trace[param], histtype='stepfilled', bins=num_bins, alpha=graph_alpha, label=param)
  plt.legend(loc='best'); plt.grid(b=False)
  plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
  
  display()


  
def plot_trace_density(trace, params_to_plot_list=['late_lambda', 'early_lambda'],\
                            title=None, xlabel=None, ylabel=None):
  '''
  Plots density plot of target variable trace output (distribution parameter) from function
  Allows plotting in same space/axis to more directly compare posterior distributions before and after a certain specified 'switchpoint'
  '''
    
  for param in params_to_plot_list: sns.distplot(trace[param], hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True}, label=param)
  plt.legend(loc='best'); plt.grid(b=False)
  plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
  
  display()
  
  

def plot_trace_histograms_regions(trace, param_to_plot='late_lambda', interval_setting=0.50, num_bins=20, graph_alpha=0.75,\
                               col1='cornflowerblue', col2='maroon', title=None, xlabel=None, ylabel=None):
  '''
  Plot histogram distribution of output parameter and user-defined interval
  User-defined interval captures some percentage of the distribution 
  '''
  
  min_boundary = pm.stats.hpd(trace[param_to_plot], credible_interval=interval_setting)[0]
  max_boundary = pm.stats.hpd(trace[param_to_plot], credible_interval=interval_setting)[1]
  
  plt.hist(trace[param_to_plot], histtype='stepfilled', bins=num_bins, alpha=graph_alpha, color=col1, label=param_to_plot) 
  plt.axvspan(min_boundary, max_boundary, alpha=0.15, color=col2)
    
  plt.legend(loc='best'); plt.grid(b=False)
  plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
  
  display()
  
  
  
def plot_trace_density_regions(trace, param_to_plot='late_lambda', interval_setting=0.50, graph_alpha=0.75,\
                                 col1='cornflowerblue', col2='maroon', title=None, xlabel=None, ylabel=None):
  '''
  Plot density distribution of output parameter and user-defined interval
  User-defined interval captures some percentage of the distribution 
  '''
  
  min_boundary = pm.stats.hpd(trace[param_to_plot], credible_interval=interval_setting)[0]
  max_boundary = pm.stats.hpd(trace[param_to_plot], credible_interval=interval_setting)[1]
  
  sns.distplot(trace[param_to_plot], hist=False, kde=True, kde_kws={'linewidth':2, 'shade':True}, color=col1, label=param_to_plot)
  plt.axvspan(min_boundary, max_boundary, alpha=0.15, color=col2)
    
  plt.legend(loc='best'); plt.grid(b=False)
  plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
  
  display()


# COMMAND ----------

# DBTITLE 1,Diagnostics
def plot_marginal_posteriors(trace):
  """
  Plot the posterior distribution for all parameters, with separate curves for each chain. We're looking for similar-looking chain posteriors to know that our models have converged
    Left-hand graph: kernel density estimation of marginal posterior of each random variable
    Right-hand graph: shows which values the MCMC algo. tried in sequential order
  """
  pm.traceplot(trace)
  display()    
    

def plot_posterior(trace):
  """
  Plot posterior distribution of an MCMC trace
  """
  pm.plot_posterior(trace)
  display()
    
    
def print_trace_summary(trace, n_digits=2):
  """
  Print summary metrics for your MCMC trace
  Docs: https://docs.pymc.io/api/stats.html
  
  ess: effective sample size used for each MCMC iteration
  hpd: highest posterior density of an array for a given credible_interval
  rhat: Gelman-Rubin measure of convergence; values greater than one indicate convergence issues.
  """
  return pm.summary(trace).round(n_digits)


def plot_forest_plot(trace):
  """
  Compare credible intervals of parameters for each trace
  """
  pm.forestplot(trace)
  display()

  
def compare_bayes_models(dataset_dict):
  """
  TODO: Compare Bayesian models using WAIC
  Docs: https://docs.pymc.io/api/stats.html
  """
  pass


def plot_chain_autocorrelation(trace):
  """
  Compare autocorrelations between chains
  TODO not sure if this is useful
  """
  pm.autocorrplot(trace)[0]
  display()
    
    
def plot_energy_plot(trace):
  """
  Plot energy plots for MCMC sampler. Ideally will look similar between both energy curves. Long tails may indicate deteriorated sampler efficiency
  Docs: https://eigenfoo.xyz/bayesian-modelling-cookbook/
  """
  pm.energyplot(trace)
  display()
  

def print_convergence_summary(trace):
  diverging = trace['diverging']
  print('Number of Divergent Chains: {}'.format(diverging.nonzero()[0].size))
  diverging_pct = diverging.nonzero()[0].size / len(trace) * 100
  print('Percentage of Divergent Chains: {:.1f}'.format(diverging_pct))
  
  
def plot_pair_plot(trace):
  """
  Create a pairplot to visualize divergences (will appear in orange)
  """
  pm.pairplot(trace, divergences=True)
  display()
  
  
def plot_posterior_samples(trace, model, num_samples=1, actuals=None):
  """
  Samples from the posterior that you built and plots those samples using seaborn
  Docs: https://docs.pymc.io/notebooks/posterior_predictive.html
  """
  ppc = predict_bayes_model(trace, model, num_samples)
  
  sns.distplot(ppc, label='Posterior Samples', color='dodgerblue')

  if actuals is not None:
    sns.distplot(actuals, label='Actuals', color='k')
    
  plt.ticklabel_format(style='plain')
  plt.legend()  
    
  display()


# COMMAND ----------

# DBTITLE 1,Workbench - BGNBD
## COREY - i was doing some reading on this, too - i can see if we can progress on this after getting the dictionary stuff worked out

# def trainBGNBD(input_pd, time_switchpoint=None, estimate_switchpoint=True, draws=10000, chains=3, tune=1000, **kwargs):
#   """
#   Finds BG-NBD using MCMC.
#   Docs: https://discourse.pymc.io/t/bg-nbd-beta-geometric-negative-binomial-model-for-customer-lifetime-value-clv/1215/11 // https://sidravi1.github.io/blog/2018/07/08/fader-hardie-clv
#   """
  
#   import theano.tensor as tt
  
#   def getRFM(pd):
#     """
#     Get the recency, frequency, and time since first purchase of a dataset
#     """
#     #TODO
#     pass
  
#   observations = input_pd[target_var]
#   x = input_pd['frequency'] # count of observations in the data
#   t_x = input_pd['recency'] # duraction between a customer's first and latest purchase
#   T = input_pd['time_since_first_purchase']
#   x_zero = 
  

#   with pm.Model() as model:

#     # Hypers for Gamma params    
#     a = pm.HalfCauchy('a',4)
#     b = pm.HalfCauchy('b',4)

#     # Hypers for Beta params  
#     alpha = pm.HalfCauchy('alpha',4)
#     [id]: url "title" = pm.HalfCauchy('beta',4)

#     lam = pm.Gamma('lam', alpha, r, shape=n_vals)
#     p = pm.Beta('p', a, b, shape=n_vals)

#     def logp(x, t_x, T, x_zero):

#         log_termA = x * tt.log(1-p) + x * tt.log(lam) \
#                                           - t_x * lam

#         termB_1 = -lam * (T - t_x)
#         termB_2 = tt.log(p) - tt.log(1-p)

#         log_termB = pm.math.switch(x_zero,
#                                   pm.math.logaddexp(termB_1, termB_2), termB_1)

#         return tt.sum(log_termA) + tt.sum(log_termB)

#     like = pm.DensityDist('like', logp, observed = {'x':x, 't_x':t_x, 'T':T, 'x_zero':x_zero})
      
#     map_estimate = pm.find_MAP()
    
#     trace = pm.sample(draws=draws, chains=chains, tune=tune, **kwargs)
    
#   return trace

# COMMAND ----------

# NEW DOCUMENTATION: https://discourse.pymc.io/t/how-to-predict-new-values-on-hold-out-data/2568/2
# https://discourse.pymc.io/t/sample-ppc-shape/1530/3

# COMMAND ----------

# DBTITLE 1,"Hierarchical" Bayesian Regression - COREY TODO


# COMMAND ----------

# DBTITLE 1,Bayesian Fitting by "Group"
#TODO deprecated; use predict_bayes_model instead
# def predictHierBayesian(trace_name, num_samples=100, model_name, how=np.mean, obs='y_obs', describe_pred=True):
#   '''
#   Returns predictions by observation using (trace, model) from trainHierBayesian output
#   COREY TO UPDATE
#   '''
  
#   # sample_posterior_predictive(trace, samples, model, vars, var_names, size, keep_size, random_seed, progressbar)
#   ppc_temp = pm.sample_posterior_predictive(trace_name, samples=num_samples, model=model_name)
#   predictions = how(ppc_temp[obs], axis=0)
    
#   if describe_pred:
#     print(stats.describe(predictions))
#     print(predictions.shape)
  
#   return list(predictions)


## COREY TODO - how does this fit with the runModelingPipeline? - what is best way to integrate?
def map_dict_to_df(input_pd, input_dict, grouping_cols_list, grouped_col='grouping_index', prediction_col='Bayes_predictions'):
  '''
  COREY TO UPDATE!! - mapping to join on our modeling df... may be a better way to handle this
  Integration with runModels and the existing pipeline infrastructure?
  '''
  
  grouped_bayes_pd = index_by_group(input_pd, grouping_cols_list=grouping_cols_list, new_col_name=grouped_col)
  
  grouped_bayes_pd[grouped_col] = grouped_bayes_pd[grouped_col].astype('int')
  grouped_bayes_pd[prediction_col] = grouped_bayes_pd[grouped_col].map(input_dict)
  
  return grouped_bayes_pd


def predict_confidence_interval(trace_iterable, pred_function, num_samples=500, lower_bound_percentile=25, upper_bound_percentile=75):
  '''
  Function-based way to set prediction intervals based on the indexed grouping and user-set upper and lower bounds
  Output is dictionary, with keys refering to the indexed grouping number and each 'value' is a list of the upper and lower bounds
  '''
  
  pred_dict_low = pred_function(trace_iterable, aggregate=True, aggregate_func=np.percentile, percentile=lower_bound_percentile, num_samples=num_samples)
  pred_dict_high = pred_function(trace_iterable, aggregate=True, aggregate_func=np.percentile, percentile=upper_bound_percentile, num_samples=num_samples)

  dict_combined = dict(pred_dict_low)

  for k, v in pred_dict_high.items():
    dict_combined[k] = [dict_combined[k], v] if k in dict_combined else v
    
  return dict_combined


def predict_NBD_values(trace_iterable, aggregate=True, aggregate_func=np.mean, percentile=50, num_samples=500, verbose_print=False): 
  ''' 
  Draws from Poisson distribution - using numpy library - to provide single 'prediction' value for each indexed grouping 
  INPUT: trace 'iterable' - a specific parameter from a trace output - trace_name[param_name] (ie, output from the below training functions)
  OUTPUT: dictionary of grouping index (keys) and posterior distribution value (ie, sinlge aggregated 'prediction' as drawn from distribution driven by underlying parameter) 
  '''
  
  ## Dictionary of the actual poisson draws (array of 'predictions') for each grouping
  poisson_distributions_dict = predict_NBD_distributions(trace_iterable, aggregate=True, aggregate_func=aggregate_func,\
                                                       percentile=percentile, num_samples=num_samples, verbose_print=verbose_print)  
  poisson_values_list = []
   
  for k, v in poisson_distributions_dict.items():
    temp_pred = np.mean(v)  ## not giving option to use anything but 'mean' (or else you get integer vals)
    poisson_values_list.append(temp_pred)
  
  poisson_pred_values_dict = dict(zip(poisson_distributions_dict.keys(), poisson_values_list))
  
  return poisson_pred_values_dict


def predict_NBD_distributions(trace_iterable, aggregate=True, aggregate_func=np.mean, percentile=50, num_samples=500, verbose_print=False):
  '''
  Draws from Poisson distribution - using numpy library - to provide array of 'prediction' values for each indexed grouping  
  INPUT: trace 'iterable' - a specific parameter from a trace output - trace_name[param_name] (ie, output from the below training functions)
  OUTPUT: dictionary of grouping index (keys) and posterior distribution array (ie, 'predictions' as drawn from distribution driven by underlying parameter)
  '''
  
  ## Dictionary of the parameter value - lambda - for each grouping
  agg_dict = create_trace_dict_mapping(trace_iterable, aggregate=True, aggregate_func=aggregate_func, percentile=percentile)  
  
  poisson_distributions_list = []
   
  for k, v in agg_dict.items():
    temp_poisson_pred = np.random.poisson(lam=v, size=num_samples)
    poisson_distributions_list.append(temp_poisson_pred)
    if verbose_print:
      print('Summary Stats of Resulting Distribution:', stats.describe(temp_poisson_pred))
  
  poisson_predictions_dict = dict(zip(agg_dict.keys(), poisson_distributions_list))
  
  return poisson_predictions_dict


def create_trace_dict_mapping(trace_iterable, aggregate=False, aggregate_func=np.mean, percentile=50):
  '''
  When Bayesian 'hierarchical' trains using multiple groups/partitions, posterior distributions are fit to each different grouping
  The posterior distribution parameter is referenced based on the index created using the function index_by_group (which is called within the Bayesian training functions)
  This utility function allows that indexing to map against the trace output (an array of parameters) for purposes of predictions that are grouping-specific
  
  INPUT: trace 'iterable' - a specific parameter from a trace output - trace_name[param_name] (ie, output from the below training functions)
  OUTPUT: dictionary of grouping index (keys) and posterior distribution parameters array or parameter array 'roll-up' (values)
  '''
 
  trace_transposed = np.transpose(trace_iterable) ## transpose to get parameter arrays by grouping index
  trace_dict = dict(enumerate(trace_transposed, start=0))  ## set keys of dictionary to the indexed number
  
  if aggregate and aggregate_func == np.percentile:
    output_dict = { k: aggregate_func(v, percentile) for k, v in trace_dict.items() }  ## to bias predictions up or down
  
  elif aggregate:
    output_dict = { k: aggregate_func(v) for k, v in trace_dict.items() }  ## use 'np.mean' or 'np.median'
  
  else:
    output_dict = trace_dict
  
  return output_dict


def train_hier_nbd(input_pd, grouping_cols_list, training_start_date=None, param_name='lambda', 
                         run_models=False, num_samples=3000, num_tune=1000, num_chains=3, **kwargs):
  '''
  Train a unique NBD posterior for every unique combination of the columns in grouping_cols_list
  
  Docs:
    - https://juanitorduz.github.io/intro_pymc3/
    - https://docs.pymc.io/notebooks/multilevel_modeling.html
    - https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Normal
  '''
  
  grouped_bayes_pd = index_by_group(input_pd, grouping_cols_list) 
  
  if training_start_date:
    input_pd = input_pd[input_pd[time_var] >= training_start_date]
  
  with pm.Model() as hier_bayes_model:

    observations = grouped_bayes_pd[target_var] 
    set_shape = grouped_bayes_pd['bayesian_grouping_index'].nunique()
    set_values = grouped_bayes_pd['bayesian_grouping_index'].values.astype('int')

    gamma_params = estimate_gamma_params(observations)
    _lambda = pm.Gamma(param_name, alpha=gamma_params['shape'], beta=gamma_params['scale'], shape=set_shape)
    y_hat = _lambda[set_values]
    y_obs = pm.Poisson('y_obs', mu=y_hat, observed=observations)
    hier_trace = pm.sample(num_samples, tune=num_tune, chains=num_chains, **kwargs)
    
    if run_models:
      return hier_trace[param_name]
    else:
      return (hier_trace, hier_bayes_model)
    
  
def train_hier_normal_poisson(input_pd, grouping_cols_list, training_start_date=None, param_name='lambda', 
                                   run_models=False, num_samples=3000, num_tune=1000, num_chains=3, **kwargs):
  '''
  Train a unique Poisson posterior (normally-distro lambda) for every unique combination of the columns in grouping_cols_list
  
  Docs:
    - https://juanitorduz.github.io/intro_pymc3/
    - https://docs.pymc.io/notebooks/multilevel_modeling.html
    - https://docs.pymc.io/api/distributions/continuous.html#pymc3.distributions.continuous.Normal
  '''
  
  
  
  grouped_bayes_pd = index_by_group(input_pd, grouping_cols_list) 
    
  if training_start_date:
    input_pd = input_pd[input_pd[time_var] >= training_start_date]
  
  with pm.Model() as hier_bayes_model:

    observations = grouped_bayes_pd[target_var] 
    set_shape = grouped_bayes_pd['bayesian_grouping_index'].nunique()
    set_values = grouped_bayes_pd['bayesian_grouping_index'].values.astype('int')
    
    normal_params = estimate_normal_params(observations)
    _lambda = pm.Normal(param_name, mu=normal_params['mu'], sigma=normal_params['sigma'], shape=set_shape)
    y_hat = _lambda[set_values]
    y_obs = pm.Poisson('y_obs', mu=y_hat, observed=observations)
    hier_trace = pm.sample(num_samples, tune=num_tune, chains=num_chains, **kwargs)
    
    if run_models:
      return hier_trace[param_name]
    else:
      return (hier_trace, hier_bayes_model)
  
  
def index_by_group(input_pd, grouping_cols_list, indexed_col_name='bayesian_grouping_index'):
  """
  Return grouped index column for Bayesian groupings
  Doc (for temporal considerations): https://www.quora.com/Does-the-order-of-observations-matter-when-making-Bayesian-inference
  """
  
  grouping_cols_list = ensureIsList(grouping_cols_list)
  
  ## TODO - update correctSuffixesInList - not working
#   grouping_cols_list = correctSuffixesInList(input_pd, grouping_cols_list)
  
  output_pd = input_pd.copy()
  output_pd[indexed_col_name] = output_pd.groupby(grouping_cols_list).ngroup()
  
  return output_pd


# COMMAND ----------

# DBTITLE 1,LIKELY DEPRECATED - NBD Prediction Functions (for looped versions)
# #TODO how is this casted across categories?
# def predictBayesian(trace_list, model_list, how='mean', ppc_obs='y_obs', num_samples=1000):
#   '''
#   Uses built-in PyMC3 functionality to provide set of 'prediction' values
#   Takes in trace + model and re-draws using the posterior distribution, based on user-dictated number of samples
#   '''
  
#   assert((how == 'mean') | (how == 'median')), 'Please enter "mean" or "median" as selected central tendency measure!'
  
#   #TODO refactor so you don't append to list. use dictionary instead with the names of the new columns as a key
#   trace_list = ensureIsList(trace_list)
#   model_list = ensureIsList(model_list)
#   predictions_list = []
  
#   #TODO index,_ in enumerage(trace_list)
#   for each in range(len(trace_list)):
#     temp_ppc = pm.sample_posterior_predictive(trace_list[each], model=model_list[each], samples=num_samples)
#     temp_mean = np.mean(temp_ppc[ppc_obs])
#     temp_median = np.median(temp_ppc[ppc_obs])
    
#     if how == 'mean': 
#       predictions_list.append(temp_mean)
    
#     elif how == 'median': 
#       predictions_list.append(temp_median)
    
#   #TODO don't need the if statement below
#   # NOTE: use newlines in block statements
#   if len(predictions_list) == 1: 
#     return predictions_list[0]
#   else: 
#     return predictions_list  

# COMMAND ----------

# DBTITLE 1,Generalized Data Preparation & Training - looping ... 

## Update list of accepted training functions as additional ones get built
## Or there might be a better way to error handlefor user inputs?

def run_bayesian_models(input_pd, bayes_train_function, switchpoint_week=False, min_sales_week=False, min_week_count=5, num_draws=10000, num_chains=3, num_tune=1000):
  '''
  Ingests dataframe and runs user-specified Bayesian modeling training function on each unique product/customer dataframe
  Runs prep_bayesian_models at outset to establish list of dataframes on which to run
  If the intent is to run this on a subsample of the full modeling pd, then trim the input pd as desired before running
  Uses globally-defined: time_var, target_var, switchpoint_week
  
  INPUTS:
  - input Pandas dataframe - must be Pandas, must include hierarchy features + time var + target var
  - bayes train function - function to be run on each unique product/customer dataframe (currently supports - trainParetoNBD, train_NBD)
  - other inputs can be adjusted as needed by user
  
  RETURNS:
  - feature combination list - list of cust/prod unique combination (if user wants to zip with trace/model outputs)
  - trace output list - trace output to be used for plotting, output review, etc.
  - model output list - model associated with training run (needed for predictions)
  - rhat output list - convergence checks for each - allows for convergence confirmations
  '''
  
  #TODO would a user be able to hurt themselves accidently by passing in the wrong bayes_train_function?
  #assert((bayes_train_function == train_NBD) | (bayes_train_function == trainParetoNBD)), 'Bayesian training function must exist!'
  
  #TODO you're reporting the same code idea three times 
  if all(item in input_pd.columns for item in [business_hier[business_level-1], prod_hier[prod_level-1]]):
    #TODO getHierarchy()?
    features_to_partition_list = [business_hier[business_level-1], prod_hier[prod_level-1]]
  
  else: 
    features_to_partition_list = [business_hier[business_level-1] + '_index', prod_hier[prod_level-1] + '_index']
  
  feature_combination_list = []
  trace_output_list = []
  model_output_list = []
  rhat_output_list = []
   
  list_of_train_pds = prep_bayesian_models(input_pd, switchpoint_week=switchpoint_week, min_sales_week=min_sales_week, min_week_count=min_week_count)
  
  
  for each_pd in list_of_train_pds:  
    #TODO the user should assume that the product / retailer nodes are set by globals 
    feature1 = each_pd[features_to_partition_list[0]].unique()
    feature2 = each_pd[features_to_partition_list[1]].unique()
    print('Running Dataframe Features:', feature1, feature2)
    
    if switchpoint_week: 
      temp_trace, temp_model = bayes_train_function(each_pd, time_switchpoint=switchpoint_week, estimate_switchpoint=True, draws=num_draws, chains=num_chains, tune=num_tune)
    else: 
      temp_trace, temp_model = bayes_train_function(each_pd, time_switchpoint=False, estimate_switchpoint=False, draws=num_draws, chains=num_chains, tune=num_tune)
    
    feature_combination_list.append([*feature1, *feature2])
    trace_output_list.append(temp_trace)
    model_output_list.append(temp_model)
    rhat_output_list.append(pm.summary(temp_trace)['r_hat'].mean())
  
  return (feature_combination_list, trace_output_list, model_output_list, rhat_output_list)


def prep_bayesian_models(input_pd, switchpoint_week=False, min_sales_week=False, min_week_count=5):
  '''
  Ingests dataframe and partitions based on global modeling levels, ie reflects a product category and customer category (eg, ['PL3_Cat', 'State'])
  Includes resampling based on time var and cleansing to remove items with insufficient data for Bayesian training
  Uses globally-defined: time_var, target_var, switchpoint_week
  
  INPUTS:
  - input Pandas dataframe - must be Pandas, must include hierarchy features + time var + target var
  - min sales week - will remove product/customer pd combinations that do not have target var (sales) listed at or more recently than this threshold
  - min week count - will remove product/customer pd combinations that do not have time var (week) beyond this threshold
  
  RETURNS:
  flattened list of dataframes - each dataframe is prepped for Bayesian model training
  '''
   
  if all(item in input_pd.columns for item in [business_hier[business_level-1], prod_hier[prod_level-1]]):
    features_to_partition_list = [business_hier[business_level-1], prod_hier[prod_level-1]]
  else: features_to_partition_list = [business_hier[business_level-1] + '_index', prod_hier[prod_level-1] + '_index']
  
  print('Running at the following level:', features_to_partition_list)
  
  first_pd_list = partition_pandas_df(input_pd, features_to_partition_list[0])
  
  second_pd_list = []
  for each_pd in first_pd_list:
    each_pd = each_pd.pipe(assignDateTimeVar, date_format='%Y%W-%w', time_var_override=None)\
                     .pipe(resamplePandasDataFrame, 'W-MON', grouping_vars=features_to_partition_list)\
                     .pipe(dateTimeExtraction, 'datetime')
    each_pd[time_var] = each_pd[time_var].astype('int')
    second_pd_list.append(each_pd)
  
  third_pd_list = []
  for each_pd in second_pd_list:
    sub_pd_list = partition_pandas_df(each_pd, features_to_partition_list[1])
    third_pd_list.append(sub_pd_list)
  
  flattened_third_pd_list = [val for sublist in third_pd_list for val in sublist]
   
  fourth_pd_list = []
  for each_pd in flattened_third_pd_list:
    
    if switchpoint_week:
      ## Before Switchpoint - lack of sales before/after switchpoint week will cause training loop to break
      filt_by_sales_before = each_pd[each_pd[time_var] < switchpoint_week].groupby(by=features_to_partition_list[1])\
                                                                          .agg({target_var:'sum'})\
                                                                          .reset_index()
      filt_by_sales_before = filt_by_sales_before[filt_by_sales_before[target_var] == 0]
      before_feature_filter_list = list(filt_by_sales_before[features_to_partition_list[1]].unique())

      ## After Switchpoint - lack of sales before/after switchpoint week will cause training loop to break
      filt_by_sales_after = each_pd[each_pd[time_var] >= switchpoint_week].groupby(by=features_to_partition_list[1])\
                                                                          .agg({target_var:'sum'})\
                                                                          .reset_index()
      filt_by_sales_after = filt_by_sales_after[filt_by_sales_after[target_var] == 0]
      after_feature_filter_list = list(filt_by_sales_after[features_to_partition_list[1]].unique())

      ## Create full sales-based list for filtering purposes
      full_feature_filter_list = [*before_feature_filter_list, *after_feature_filter_list]
      
    else:
      filt_by_sales = each_pd.groupby(by=features_to_partition_list[1])\
                             .agg({target_var:'sum'})\
                             .reset_index()
      filt_by_sales = filt_by_sales[filt_by_sales[target_var] == 0]
      full_feature_filter_list = list(filt_by_sales[features_to_partition_list[1]].unique())

    ## Collect those that were 'discontinued' - not recent-enough sales info to include
    ## Helps in conjunction with resampling effort (resampling won't necessarily add to tails)
    for each_item in each_pd[features_to_partition_list[1]].unique():
      if min_sales_week:
        if each_pd[each_pd[features_to_partition_list[1]] == each_item][time_var].max() < min_sales_week: full_feature_filter_list.append(each_item)
      if min_week_count:
        if each_pd[each_pd[features_to_partition_list[1]] == each_item][time_var].nunique() < min_week_count: full_feature_filter_list.append(each_item)

    ## Filter out all features categories in the above list
    each_pd = each_pd[ ~each_pd[features_to_partition_list[1]].isin(full_feature_filter_list) ]
    fourth_pd_list.append(each_pd)
    
  final_pd_list = [pd_item for pd_item in fourth_pd_list if pd_item.empty != True]

  return final_pd_list


def partition_pandas_df(input_pd, feature_to_partition):
  '''
  Ingests a Pandas dataframe and splits it into a separate dataframe based on the unique values for a specified feature
  Each partitioned dataframe is stored in a list - the returned value is a list of dataframes
  '''
  
  pd_list = []
  unique_feature_list = list(sorted(input_pd[feature_to_partition].unique()))
  
  for each_unique in unique_feature_list:
      temp_pd = input_pd[input_pd[feature_to_partition] == each_unique]
      pd_list.append(temp_pd)
  
  return pd_list


# COMMAND ----------

# DBTITLE 1,Bayesian Plots


# COMMAND ----------

# DBTITLE 1,Bayesian Utilities
def compare_trace_params(trace_list, params_to_compare_list=['late_lambda', 'early_lambda']):
  
  trace_list = ensureIsList(trace_list)
  
  assert(all(item in trace_list[0].varnames for item in params_to_compare_list)), 'Please adjust list of names of parameters to compare!'
  
  demand_shift = []
  
  for each_trace in trace_list:
    trace_param_1 = each_trace[params_to_compare_list[0]]
    trace_param_2 = each_trace[params_to_compare_list[1]]
    comparison_output = [x-y for (x, y) in zip(trace_param_1, trace_param_2)]
    demand_numerator = np.sum([1 for i in comparison_output if i > 0])
    demand_denominator = len(comparison_output)
    prob_demand_increase = demand_numerator/demand_denominator
    demand_shift.append(prob_demand_increase)
  
  if len(demand_shift) == 1: return demand_shift[0]
  else: return demand_shift

  
# def traceDistributionPercentiles(trace, parameter_name='late_lambda', list_of_percentiles=[10,50,90]):
#   '''
#   Finds various percentiles for a parameter of a trace
#   Example: distributionPercentiles(pizza_trace, 'late_lambda', [10, 25, 50, 75, 90])
#   '''
#   return calc_distribution_percentiles(trace[parameter_name], list_of_percentiles)


def calc_distribution_percentiles(array_name, list_of_percentiles=[10,50,90]):
  '''
  Finds various percentiles of an array/list
  Different percentiles of interest should be entered as a list and represent percent * 100 -- (eg, [10, 50, 90])
  '''
  return list(np.percentile(array_name, list_of_percentiles))


#TODO hard to read and probably unnecessary
# def createBayesianOutputDF(input_pd, features, predictions, list_of_column_names=False):
#   '''
#   Uses list-based inouts/outputs from Bayesian modeling processing to create a dataframe
#   Might be useful for merging/joining with other prediction values (during modeling runs)
#   '''
  
#   ## NOTE: assumes each item in the features list has product + customer feature
#   feature1 = [each[0] for each in features]
#   feature2 = [each[1] for each in features]

#   if list_of_column_names:
#     list_of_column_names = ensureIsList(list_of_column_names)
#     pred_pd = pd.DataFrame(zip(feature1, feature2, predictions), columns=list_of_column_names)    
    
#   else: 
#     if all(item in input_pd.columns for item in [business_hier[business_level-1], prod_hier[prod_level-1]]):
#       pred_pd = pd.DataFrame(zip(feature1, feature2, predictions), columns=[business_hier[business_level-1], prod_hier[prod_level-1], 'predictions'])
#     else: pred_pd = pd.DataFrame(zip(feature1, feature2, predictions), columns=[business_hier[business_level-1] + '_index', prod_hier[prod_level-1] + '_index', 'predictions'])   
  
#   return pred_pd
  

# ## Note - to actualy be useful, need to run after transforming preds and actuals
# def scoreModelRMSE(trace_name, actuals, model_name, num_samples=1000):
#   from sklearn import metrics
#   ppc = pm.sample_ppc(trace_name, model=model_name, samples=num_samples)
#   pred = ppc['y_obs'].mean(axis=0)
#   return np.sqrt(metrics.mean_squared_error(actuals, pred))



# COMMAND ----------

# DBTITLE 1,Resources

# Youtube videos:
#   Explaination of contractual vs. non-contractual, discrete vs. continuous with discussion of Pareto/NBD: https://www.youtube.com/watch?v=gx6oHqpRgpY&list=PLGVZCDnMOq0rxoq9Nx0B4tqtr891vaCn7&index=45
#   Notebook from their talk: https://github.com/jrgauthier01/pydata-seattle-2017/blob/master/lifetime-value/pareto-nbd.ipynb
  
# Posts:
#   Dealing with convergence issues: https://eigenfoo.xyz/bayesian-modelling-cookbook/