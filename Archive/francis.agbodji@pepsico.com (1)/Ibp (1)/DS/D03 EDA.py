# Databricks notebook source
# MAGIC %md
# MAGIC ##03 - EDA
# MAGIC 
# MAGIC Performs Exploratory data analysis on the data.  Artifacts are saved to mlflow to guide thresholds for data cleansing / transformations.

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

#initiate widget if needed, otherwise does nothing        
check_or_init_dropdown_widget("TIME_VAR_LOCAL","Week_Of_Year",["Week_Of_Year","Month_Of_Year"])  

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

#Check configurations exist for this script
try:
  required_configs = [
    TARGET_VAR,
    TIME_VAR,
    EDA_SAMPLE_RATE,

    DBA_MRD_EXPLORATORY 
    ]
  print(json.dumps(required_configs, indent=4))
except:
  dbutils.notebook.exit("Missing required configs")

# Optional Configurations for this script listed below
# DBI_ORDERS

# COMMAND ----------

# DBTITLE 1,Setup Mlflow
# For EDA tracking

output_path_mlflow_eda = f'{OUTPUT_PATH_MLFLOW_EXPERIMENTS}/PEP EDA'
mlflow.set_experiment(output_path_mlflow_eda)
mlflow.start_run()
experiment = mlflow.get_experiment_by_name(output_path_mlflow_eda)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

# DBTITLE 1,Load Data
## Corey Note - we are re-loading here - why?
## Can we delete one of the data loads? (likely can delete the one above)

try:
  dataframe_df = load_delta(DBA_MRD_EXPLORATORY)  
except:
  dbutils.notebook.exit("DBA_MRD_EXPLORATORY load failed, Exiting notebok")

# COMMAND ----------

# Adding year column in the data set
dataframe_df = dataframe_df.withColumn('year', (col(TIME_VAR)/100).cast(IntegerType()))

# COMMAND ----------

## Downsample data if needed for EDA
eda_df = dataframe_df.sample(False, EDA_SAMPLE_RATE, seed=10)
eda_pd = eda_df.select("*").toPandas()
print(dataframe_df.count(), eda_df.count(), eda_pd.shape)

# COMMAND ----------

# Getting Max and Min Shipment date
print('Max date ={} Min date={}'.format(eda_pd[TIME_VAR].max(),eda_pd[TIME_VAR].min())) 

# COMMAND ----------

# Profiling for the data set 
eda_pd.describe()

# COMMAND ----------

# Counts based on country and category lavel
df1 = eda_df.groupby('PLANG_LOC_CTRY_ISO_CDV','SRC_CTGY_1_NM').agg(count('PLANG_LOC_CTRY_ISO_CDV').alias('count'))
display(df1)

# COMMAND ----------

# Aggregation of QTY based on year,country and channel
df2 = eda_df.groupby('PLANG_LOC_CTRY_ISO_CDV','year','UDC_CHANNEL').agg(sum('QTY').alias('shipment_QTY'))
display(df2)

# COMMAND ----------

# Distribution of data based on Brand Name
qty_brnd_nm = eda_df.groupby('BRND_NM').agg(count('BRND_NM').alias('count'))
qty_brnd_nm = qty_brnd_nm.orderBy('count', ascending=False)
qty_brnd_nm = qty_brnd_nm.toPandas()

print(qty_brnd_nm.shape)
y_ax = qty_brnd_nm['count'].tolist()
x_ax = qty_brnd_nm['BRND_NM'].tolist()
plt.figure(figsize=(10,4))
sns.barplot(x=x_ax, y=y_ax)
plt.title('Counts by Brand Name', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.show()


# COMMAND ----------

qty_brnd_nm.head(100)

# COMMAND ----------

# Distribution of data based on PCK_CNTNR_SHRT_NM
count_pck_cntnr_nm = eda_df.groupby('PCK_CNTNR_SHRT_NM').agg(count('PCK_CNTNR_SHRT_NM').alias('count'))
count_pck_cntnr_nm = count_pck_cntnr_nm.orderBy('count', ascending=False)
count_pck_cntnr_nm = count_pck_cntnr_nm.toPandas()

print(count_pck_cntnr_nm.shape)
y_ax = count_pck_cntnr_nm['count'].tolist()
x_ax = count_pck_cntnr_nm['PCK_CNTNR_SHRT_NM'].tolist()
plt.figure(figsize=(10,4))
sns.barplot(x=x_ax, y=y_ax)
plt.title('Counts by PCK_CNTNR_SHRT_NM', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.show()

# COMMAND ----------

count_pck_cntnr_nm.head(20)

# COMMAND ----------

# Aggregation of QTy by Location

count_loc = eda_df.groupby('LOC').agg(sum('QTY').alias('sum_QTY'))
count_loc = count_loc.orderBy('sum_QTY', ascending=False)
count_loc = count_loc.toPandas()

print(count_loc.shape)
y_ax = count_loc['sum_QTY'].tolist()
x_ax = count_loc['LOC'].tolist()
plt.figure(figsize=(10,4))
sns.barplot(x=x_ax, y=y_ax)
plt.title('Aggregation of QTY by LOC', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.show()

# COMMAND ----------

# Null value analysis in dataframe. This will give the percentage of null values for each column 
#missing_df = eda_df.select([(count(when(isnan(c) | col(c).isNull(), c))/count(lit(1))).alias(c) for c in eda_df.columns])
#display(missing_df)

# COMMAND ----------

# Null value analysis in dataframe. This will give the count of null values for each column 
#missing_df = eda_df.select([(count(when(isnan(c) | col(c).isNull(), c)).alias(c)) for c in eda_df.columns])
#display(missing_df)

# COMMAND ----------

# DBTITLE 1,Sweet Viz Report
## Subset variables relevant for EDA
keep_vars = intersect_two_lists(EDA_VARS + [TARGET_VAR], eda_pd.columns)
viz_pd = eda_pd[keep_vars]

## Run sweet viz
report = sv.analyze(viz_pd, target_feat=TARGET_VAR)

#Log outputs to mlflow as html file
eda_notebook = report.show_html()
mlflow.log_artifact("SWEETVIZ_REPORT.html")

# COMMAND ----------

# DBTITLE 1, Data Exploration & Review
unique_plot = eda_df.agg(*(countDistinct(col(c)).alias(c) for c in eda_df.columns))
unique_plot =unique_plot.toPandas()
unique_plot = unique_plot.T
unique_plot = unique_plot.rename(columns={0:'Count_Unique'})
unique_plot = unique_plot.sort_values(by=['Count_Unique'],ascending=False)
unique_plot = unique_plot[unique_plot['Count_Unique'] < 500]  ## subset our view

plt.figure(figsize=(12,4))
sns.barplot(x=unique_plot.index, y=unique_plot.Count_Unique)
plt.title('Unique Values by Feature', fontsize=12)
plt.xticks(rotation=90, fontsize=10)
plt.show()

# COMMAND ----------

## Review of target variable over time
## Looking across weeks and months
plt.figure(figsize=(14,4))
# plt.rcParams["figure.figsize"] = (14, 4)

qty_over_time = eda_df.groupby(TIME_VAR).agg(sum(TARGET_VAR).alias(TARGET_VAR))
qty_over_time = qty_over_time.withColumn(TIME_VAR, col(TIME_VAR).cast(StringType()))
qty_by_month = eda_df.groupby('Month_Of_Year').agg(sum(TARGET_VAR).alias(TARGET_VAR))

qty_over_time = qty_over_time.toPandas()
qty_by_month = qty_by_month.toPandas()

plt.subplot(1, 2, 1)
ax = sns.lineplot(x=TIME_VAR, y=TARGET_VAR, data=qty_over_time)
n = 4; [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
plt.title('{} Over Time'.format(TARGET_VAR), fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

plt.subplot(1, 2, 2)
ax = sns.barplot(x='Month_Of_Year', y=TARGET_VAR, data=qty_by_month, color='seagreen')
n = 2; [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
plt.title('{} Aggregated by Month'.format(TARGET_VAR), fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

plt.show()

# COMMAND ----------

# DBTITLE 1,Distribution plots
## Review of overall distribution of our target variable
plt.figure(figsize=(20,4))

plt.subplot(1, 3, 1)
sns.kdeplot(eda_pd[TARGET_VAR], color='dodgerblue', shade=True)
plt.title('{} Dist Plot - Unconstrained'.format(TARGET_VAR), fontsize=12)
plt.grid(b=False)

plt.subplot(1, 3, 2)
constaint1 = 50000
sns.kdeplot(eda_pd[TARGET_VAR], color='seagreen', shade=True)
plt.title('{} Dist Plot - Constrained at {}'.format(TARGET_VAR, constaint1), fontsize=12)
plt.xlim(0, constaint1)
plt.grid(b=False)

plt.subplot(1, 3, 3)
constaint2 = 10000
sns.kdeplot(eda_pd[TARGET_VAR], color='indianred', shade=True)
plt.title('{} Dist Plot - Constrained at {}'.format(TARGET_VAR, constaint2), fontsize=12)
plt.xlim(0, constaint2)
plt.grid(b=False)

plt.show()

# COMMAND ----------

CUST_MERGE_FIELD = intersect_two_lists(MODEL_ID_HIER, CUSTOMER_HIER)
CUST_MERGE_FIELD = CUST_MERGE_FIELD[0]

# COMMAND ----------

## TO-DO: EDA merges if we no longer have CUST_MERGE_FIELD in the aggregated data 

## Since our data is predominantly categorical, building out sample countplots
plt.figure(figsize=(20,4))

plt.subplot(1, 2, 1)
sns.countplot(x=LOC_MERGE_FIELD, data=eda_pd, order=eda_pd[LOC_MERGE_FIELD].value_counts().index)
plt.title('Count of Data by Location Field', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

plt.subplot(1, 2, 2)
sns.countplot(x=CUST_MERGE_FIELD, data=eda_pd, order=eda_pd[CUST_MERGE_FIELD].value_counts().index)
plt.title('Count of Data by Customer Field', fontsize=12)
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

plt.show()

# COMMAND ----------

plot_box_plot(eda_pd, CUST_MERGE_FIELD, TARGET_VAR, title='{} by Customer'.format(TARGET_VAR),\
              title_size=12, show_outliers=False, fig_size=(8,4),\
              order=eda_pd.groupby(CUST_MERGE_FIELD)[TARGET_VAR]\
              .sum().reset_index().sort_values(by=TARGET_VAR, ascending=False)[CUST_MERGE_FIELD])

plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)
plt.show()

# COMMAND ----------

#barplot_data = eda_pd.groupby(CUST_MERGE_FIELD)[TARGET_VAR].sum().reset_index().sort_values(by=TARGET_VAR, ascending=False)
barplot_data = eda_df.groupby(CUST_MERGE_FIELD).agg(sum(TARGET_VAR).alias(TARGET_VAR)).sort(col(TARGET_VAR).desc()).toPandas()
sns.barplot(x=CUST_MERGE_FIELD, y=TARGET_VAR, data=barplot_data, order=barplot_data[CUST_MERGE_FIELD])
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

# COMMAND ----------

# DBTITLE 1,Logged TARGET_VAR Views
# ## Illustration of impact of certain data transformations
np.seterr(all='warn')
log_dataframe = eda_df
log_dataframe = log_dataframe.withColumn("Log_Comparison", F.log1p(F.col(TARGET_VAR))) 
log_dataframe = log_dataframe.fillna(0).toPandas()
plot_violin_plot(log_dataframe, LOC_MERGE_FIELD, 'Log_Comparison', 
                 title='{} by Location - Logged'.format(TARGET_VAR), title_size=12, show_outliers=False, fig_size=(16,4))
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)

# COMMAND ----------

plot_violin_plot(log_dataframe, CATEGORY_FIELD, 'Log_Comparison', 
                 title='{} by Category - Logged'.format(TARGET_VAR), title_size=12, show_outliers=False, fig_size=(16,4))
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)
plt.show()

# COMMAND ----------

plot_violin_plot(log_dataframe, BRAND_FIELD, 'Log_Comparison', 
                 title='{} by Brand - Logged'.format(TARGET_VAR), title_size=12, show_outliers=False, fig_size=(30,4))
plt.xticks(rotation=90, fontsize=8)
plt.grid(b=False)
plt.show()

# COMMAND ----------

# DBTITLE 1,Check Missing Values 
## To review baseline dataframe for missing values
print_missing_row_percentage(eda_pd)
print_missing_percentage(eda_pd)

# COMMAND ----------

# DBTITLE 1,Multicollinearity Review of Baseline Features
## Remove binary fields from numeric correlation

binary_cols_list = []
for each_col in eda_pd.columns:
  temp_val_list = eda_pd[each_col].value_counts().index
  
  if len(temp_val_list) == 2:
    binary_cols_list.append(each_col)
    
corr_cols = eda_pd.select_dtypes(include=np.number).columns.tolist()
corr_cols = list(set(corr_cols) - set(binary_cols_list))
#print(corr_cols)
#print(binary_cols_list)
#Washington's_Birthday
#eda_pd["Washington's_Birthday"].value_counts().index

# COMMAND ----------

## First check for multicollinearity
plot_corr_heatmap(eda_pd, corr_cols, colormap='viridis')
plt.show()

# COMMAND ----------

eda_df.count()

# COMMAND ----------

eda_pd.fillna(value=0, inplace=True)

# COMMAND ----------

## In VIF method, we pick each feature and regress it against all of the other features
## Greater VIF denotes greater correlation - a better candidate to strip out of the data
## Note that this is valuable when working with much larger dataframe
vif_pd = calculate_feature_VIF(eda_pd[corr_cols], cols_to_drop=[], feat_col='feature', vif_col='vif')
vif_pd.sort_values(by='vif', ascending=False)[0:20]

# COMMAND ----------

# ## Remove high VIF fields
# vif_cols_to_drop = return_collinear_features_VIF(vif_pd, threshold_level=VIF_THRESHOLD, feature_col='feature', vif_col='vif')
# dataframe.drop(columns=vif_cols_to_drop, inplace=True)

# ## Review the above changes
# print(vif_cols_to_drop, dataframe.shape)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

