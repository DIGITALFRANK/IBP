# Databricks notebook source
# DBTITLE 1,Import & Instantiate
# MAGIC %run ../src/libraries

# COMMAND ----------

# MAGIC %run ../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ../src/load_src

# COMMAND ----------

# MAGIC %run ../src/config

# COMMAND ----------

# DBTITLE 1,Review Pickle Files
DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT = 'dbfs:/mnt/adls/Tables/DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT'
check = load_delta_info(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
display(check.history())

# COMMAND ----------

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT, 89)  ## select alternative versions if needed - ref above table
pickles_pd = pickles.toPandas()
pickles_pd.lag_model.unique()  ## only retained lag 8 - fine for our revoew purposes here

# COMMAND ----------

# DBTITLE 1,Pull for LightGBM
FEAT_IMP_LAG = 8
FEAT_IMP_MODEL = 'lightGBM_model'

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
pickles_pd = pickles.toPandas()

pickles_pd = pickles_pd[(pickles_pd['lag_model'] == FEAT_IMP_LAG) & (pickles_pd['train_func'] == FEAT_IMP_MODEL)]
pickles_pd = pickles_pd[['train_func', 'model_pick']]
pickles_dict = dict(zip(pickles_pd.train_func, pickles_pd.model_pick))

importance_list = {}

for model in [FEAT_IMP_MODEL]:
  obj = pickles_dict[model].encode('latin-1')
  obj = pickle.loads(obj)
  feature_importance_dict = dict(zip(obj.feature_name(), obj.feature_importance(importance_type='gain').T))
  importance_list[model] = feature_importance_dict
  
importance_pd = pd.DataFrame.from_dict(importance_list)\
                            .reset_index()\
                            .sort_values(by=FEAT_IMP_MODEL, ascending=False)

importance_pd.columns = ['feature_name', FEAT_IMP_MODEL]

importance_pd_lgbm = importance_pd.copy()
importance_pd.head(10)

# COMMAND ----------

# DBTITLE 1,Pull for Quantile GBM Models (x3)
FEAT_IMP_LAG = 8
FEAT_IMP_MODEL = 'gbm_quantile_model'

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
pickles_pd = pickles.toPandas()

pickles_pd = pickles_pd[(pickles_pd['lag_model'] == FEAT_IMP_LAG) & (pickles_pd['train_func'] == FEAT_IMP_MODEL)]
pickles_pd = pickles_pd[['train_func', 'model_pick']]
pickles_dict = dict(zip(pickles_pd.train_func, pickles_pd.model_pick))

importance_list = {}

for model in [FEAT_IMP_MODEL]:
  obj = pickles_dict[model].encode('latin-1')
  obj = pickle.loads(obj)
  feature_importance_dict = dict(zip(obj.feature_name(), obj.feature_importance(importance_type='gain').T))
  importance_list[model] = feature_importance_dict
  
importance_pd = pd.DataFrame.from_dict(importance_list)\
                            .reset_index()\
                            .sort_values(by=FEAT_IMP_MODEL, ascending=False)

importance_pd.columns = ['feature_name', FEAT_IMP_MODEL]

importance_pd_qgbm1 = importance_pd.copy()
importance_pd.head(10)

# COMMAND ----------

FEAT_IMP_LAG = 8
FEAT_IMP_MODEL = 'gbm_quantile_model2'

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
pickles_pd = pickles.toPandas()

pickles_pd = pickles_pd[(pickles_pd['lag_model'] == FEAT_IMP_LAG) & (pickles_pd['train_func'] == FEAT_IMP_MODEL)]
pickles_pd = pickles_pd[['train_func', 'model_pick']]
pickles_dict = dict(zip(pickles_pd.train_func, pickles_pd.model_pick))

importance_list = {}

for model in [FEAT_IMP_MODEL]:
  obj = pickles_dict[model].encode('latin-1')
  obj = pickle.loads(obj)
  feature_importance_dict = dict(zip(obj.feature_name(), obj.feature_importance(importance_type='gain').T))
  importance_list[model] = feature_importance_dict
  
importance_pd = pd.DataFrame.from_dict(importance_list)\
                            .reset_index()\
                            .sort_values(by=FEAT_IMP_MODEL, ascending=False)

importance_pd.columns = ['feature_name', FEAT_IMP_MODEL]

importance_pd_qgbm2 = importance_pd.copy()
importance_pd.head(10)

# COMMAND ----------

FEAT_IMP_LAG = 8
FEAT_IMP_MODEL = 'gbm_quantile_model3'

pickles = load_delta(DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT)
pickles_pd = pickles.toPandas()

pickles_pd = pickles_pd[(pickles_pd['lag_model'] == FEAT_IMP_LAG) & (pickles_pd['train_func'] == FEAT_IMP_MODEL)]
pickles_pd = pickles_pd[['train_func', 'model_pick']]
pickles_dict = dict(zip(pickles_pd.train_func, pickles_pd.model_pick))

importance_list = {}

for model in [FEAT_IMP_MODEL]:
  obj = pickles_dict[model].encode('latin-1')
  obj = pickle.loads(obj)
  feature_importance_dict = dict(zip(obj.feature_name(), obj.feature_importance(importance_type='gain').T))
  importance_list[model] = feature_importance_dict
  
importance_pd = pd.DataFrame.from_dict(importance_list)\
                            .reset_index()\
                            .sort_values(by=FEAT_IMP_MODEL, ascending=False)

importance_pd.columns = ['feature_name', FEAT_IMP_MODEL]

importance_pd_qgbm3 = importance_pd.copy()
importance_pd.head(10)

# COMMAND ----------

## Below are methods for pulling for our other tree-based models
## Note - there is not a simple way to map these back to a feature name itself
## So, for time being, using LGBM-based models to make this mapping easier

# obj.feature_importances_      ## would use for Random Forest
# obj.get_feature_importance()  ## would use for CatBoost

# COMMAND ----------

# DBTITLE 1,Combine All & Review Output
combined_pd = importance_pd_lgbm.merge(importance_pd_qgbm1, on='feature_name', how='outer')\
                                .merge(importance_pd_qgbm2, on='feature_name', how='outer')\
                                .merge(importance_pd_qgbm3, on='feature_name', how='outer')

# COMMAND ----------

model_cols = [col_name for col_name in combined_pd.columns if 'model' in col_name]
combined_pd['imp_sum'] = combined_pd[model_cols].sum(axis=1)
combined_pd['imp_mean'] = combined_pd[model_cols].mean(axis=1)

lgbm_total = combined_pd['lightGBM_model'].sum()
imp_sum_total = combined_pd['imp_sum'].sum()
imp_mean_total = combined_pd['imp_mean'].sum()

# COMMAND ----------

## Sort and use 'lgbm' to dictate importance
combined_pd.sort_values(by='lightGBM_model', ascending=False, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

combined_pd['lgbm_cumsum'] = combined_pd['lightGBM_model'].cumsum()
combined_pd['lgbm_cumsum_perc'] = combined_pd['lgbm_cumsum'] / lgbm_total

combined_pd.head(10)

# COMMAND ----------

## Sort and use 'mean' to dictate import
combined_pd.sort_values(by='imp_mean', ascending=False, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

combined_pd['imp_mean_cumsum'] = combined_pd['imp_mean'].cumsum()
combined_pd['imp_mean_cumsum_perc'] = combined_pd['imp_mean_cumsum'] / imp_mean_total

combined_pd.head(10)

# COMMAND ----------

combined_pd.sort_values(by='lgbm_cumsum_perc', ascending=True, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

sns.lineplot(x=combined_pd.index, y=combined_pd.lgbm_cumsum_perc, color='rebeccapurple')
plt.title('Feature Importance: "LGBM" CumSum Across Features')
plt.xlabel('Num of Features')
plt.show()

# COMMAND ----------

combined_pd.sort_values(by='imp_mean_cumsum_perc', ascending=True, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

sns.lineplot(x=combined_pd.index, y=combined_pd.imp_mean_cumsum_perc, color='goldenrod')
plt.title('Feature Importance: "Multiple Model" CumSum Across Features')
plt.xlabel('Num of Features')
plt.show()

# COMMAND ----------

# DBTITLE 1,Determine Proposed Cols To Drop
FEAT_CUMSUM_THRESH = 0.98

combined_pd.sort_values(by='lgbm_cumsum', ascending=True, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

trimmed_pd = combined_pd[combined_pd.lgbm_cumsum > FEAT_CUMSUM_THRESH]
cols_to_drop_lgbm_cumsum = trimmed_pd.feature_name.tolist()

print(trimmed_pd.shape, trimmed_pd.shape[0] == len(cols_to_drop_lgbm_cumsum))

# COMMAND ----------

combined_pd.sort_values(by='imp_mean_cumsum', ascending=True, inplace=True)
combined_pd.reset_index(inplace=True, drop=True)

trimmed_pd = combined_pd[combined_pd.imp_mean_cumsum_perc > FEAT_CUMSUM_THRESH]
cols_to_drop_mean_cumsum = trimmed_pd.feature_name.tolist()

print(trimmed_pd.shape, trimmed_pd.shape[0] == len(cols_to_drop_mean_cumsum))

# COMMAND ----------

## Find intersection across both lists, and return a list
## This ensures some level of 'robustness' in what we propose to drop

cols_to_drop_final = list(set(cols_to_drop_lgbm_cumsum) & set(cols_to_drop_mean_cumsum))

print(len(cols_to_drop_final))
sorted(cols_to_drop_final)

# COMMAND ----------

# DBTITLE 1,"Manual" Adjustments To List (If Desired)
## For removing certain things that we know we want to include
## Eg - always retain indexed columns since we know we want to retain those

cols_to_drop_final = [feat_name for feat_name in cols_to_drop_final if 'index' not in feat_name]
len(cols_to_drop_final)  # retains ~12 items from above

# COMMAND ----------



# COMMAND ----------

