# Databricks notebook source
# MAGIC %md 
# MAGIC ##02 - Outlier Detection
# MAGIC 
# MAGIC This script flags outliers.  The user can specify what algorithm they want to use to detect outliers (LOF, KNN, etc.) through the configuration layer, threshlolds for the detection algorithm and what variables they want to detect (e.g., target variable, price, distribution, etc.).  The data is stored in a delta table for pickup within downstream activities.

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
    MODEL_ID_HIER,  
    CATEGORY_FIELD,
    MIN_ROWS_THRESHOLD,
    OUTLIER_METHOD,
    OUTLIER_DETECT_LEVEL,
    OUTLIER_THRESHOLD,
    OUTLIER_VARS,
    DBA_MRD_EXPLORATORY ,
    DBO_OUTLIERS
    ]
  print(json.dumps(required_configs, indent=4))
except:
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

# DBTITLE 1,Setup Mlflow
#For experiment tracking
output_path_mlflow_outliers = f'{OUTPUT_PATH_MLFLOW_EXPERIMENTS}/PEP Outliers'
mlflow.set_experiment(output_path_mlflow_outliers)
mlflow.start_run()
experiment = mlflow.get_experiment_by_name(output_path_mlflow_outliers)
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))

# COMMAND ----------

#Evaluate and log classification method

#Log parameters to mlflow
mlflow.log_param('TARGET_VAR', TARGET_VAR)
mlflow.log_param('MODEL_ID_HIER', MODEL_ID_HIER)
mlflow.log_param('OUTLIER_METHOD', OUTLIER_METHOD)
mlflow.log_param('OUTLIER_DETECT_LEVEL', OUTLIER_DETECT_LEVEL)
mlflow.log_param('OUTLIER_THRESHOLD', OUTLIER_THRESHOLD)
mlflow.log_param('OUTLIER_VARS', OUTLIER_VARS)

mlflow.log_dict(classifiers, "classifiers.yml")
classifiers

# COMMAND ----------

# DBTITLE 1,Load Data
## Loading simple mrd
# mrd_orig = load_delta(DBA_MRD_EXPLORATORY, 38) #Weekly
# if TIME_VAR=="Month_Of_Year":
#   mrd_orig = load_delta(DBA_MRD_EXPLORATORY, 39) #Monthly

mrd_orig = load_delta(DBA_MRD_EXPLORATORY)
  
#Remove zero sales weeks
mrd = mrd_orig.filter(col(TARGET_VAR)>0)

#Generate automatic schema of scaling output table
keep_vars = OUTLIER_DETECT_LEVEL + ["MODEL_ID"] + [TIME_VAR] + OUTLIER_VARS + [TARGET_VAR]
keep_vars = intersect_two_lists(keep_vars, mrd.columns)
mrd = mrd.select(keep_vars)
auto_schema = mrd.limit(1)
auto_schema = auto_schema.schema
auto_schema

# COMMAND ----------

#Note: PySpark MinMax Scaler only creates vectors, thus manually scaling
scaled_data = mrd
for i in OUTLIER_VARS: 
  min_val = scaled_data.select(i).rdd.min()[0]
  max_val = scaled_data.select(i).rdd.max()[0]
  scaled_data = scaled_data.withColumn(i, (col(i)-min_val) / (max_val - min_val))

#Filter products that have at least record_count rows
num_rows = aggregate_data(scaled_data,["MODEL_ID"] , [TARGET_VAR], [count])
scaled_data = scaled_data.join(num_rows, on=["MODEL_ID"], how="left")
scaled_data = scaled_data.filter(col("count_" + TARGET_VAR)>MIN_ROWS_THRESHOLD)

# COMMAND ----------

# DBTITLE 1,Detect Outliers
#Setup information for detection
model_info_dict = dict(
    outlier_vars = OUTLIER_VARS,
    classifiers  = classifiers,
    detect_level = OUTLIER_DETECT_LEVEL,
    time_var     = [TIME_VAR]
)

outlier_cls = DetectOutliers(**model_info_dict)
outlier_cls.detect_level

# COMMAND ----------

keep_vars = ['MODEL_ID',TIME_VAR,"pred"]
auto_schema = scaled_data.withColumn("pred",lit(1.0))
auto_schema = auto_schema.select(keep_vars)
auto_schema = auto_schema.schema

@pandas_udf(auto_schema, PandasUDFType.GROUPED_MAP)
def predict_outliers_udf(data):
    return predict_outliers(data, outlier_cls)

detected_outliers = scaled_data.groupBy(OUTLIER_DETECT_LEVEL).apply(predict_outliers_udf)
detected_outliers = detected_outliers.withColumnRenamed("pred","OUTLIER_IND")
detected_outliers.cache()

# COMMAND ----------

# DBTITLE 1,Explore Outliers
explore_outliers = detected_outliers.join(mrd_orig.select(OUTLIER_DETECT_LEVEL + 
                                                          [TIME_VAR, CATEGORY_FIELD , TARGET_VAR]), 
                                           on=["MODEL_ID"] + [TIME_VAR], 
                                           how="left")

# COMMAND ----------

#Flag outliers as peak / low sales week
explore_outliers.cache()
avg_vals = aggregate_data(explore_outliers,["MODEL_ID"],[TARGET_VAR],[avg])
explore_outliers = explore_outliers.join(avg_vals, on = ["MODEL_ID"], how="left")

explore_outliers = explore_outliers.withColumn("OUTLIER_TYPE",lit("None"))
cond1 = col("OUTLIER_IND") == 1
cond2 = col(TARGET_VAR) >= col("avg_"+TARGET_VAR)
explore_outliers = explore_outliers.withColumn("OUTLIER_TYPE", when((cond1 & cond2),"PeakWeek").otherwise(col("OUTLIER_TYPE")))

cond1 = col("OUTLIER_IND") == 1
cond2 = col(TARGET_VAR) < col("avg_"+TARGET_VAR)
explore_outliers = explore_outliers.withColumn("OUTLIER_TYPE", when((cond1 & cond2),"LowSales").otherwise(col("OUTLIER_TYPE")))

explore_outliers.cache()
explore_outliers.count()

# COMMAND ----------

#% outliers by Category
num_outliers = aggregate_data(explore_outliers, CATEGORY_FIELD , ["OUTLIER_IND","OUTLIER_IND"], [sum, count])
num_outliers = num_outliers.sort("sum_OUTLIER_IND",ascending = False)
num_outliers = num_outliers.withColumn("Pct_Outlier", col("sum_OUTLIER_IND")/col("count_OUTLIER_IND"))
print(num_outliers.count())
display(num_outliers)

# COMMAND ----------

#% outliers by MODEL_ID
num_outliers = aggregate_data(explore_outliers, "MODEL_ID" , ["OUTLIER_IND","OUTLIER_IND"], [sum, count])
num_outliers = num_outliers.sort("sum_OUTLIER_IND",ascending = False)
num_outliers = num_outliers.withColumn("Pct_Outlier", col("sum_OUTLIER_IND")/col("count_OUTLIER_IND"))
print(num_outliers.count())
display(num_outliers)

# COMMAND ----------

#Plot top 10 largest modeled hierarchies
#Filter top 10 modeled hierarchies
qc_modelids = aggregate_data(explore_outliers,["MODEL_ID"], [TARGET_VAR], [sum])
qc_modelids = qc_modelids.sort("sum_" + TARGET_VAR,ascending = False)
qc_modelids = qc_modelids.limit(10)
top_10_ids =convertDFColumnsToList(qc_modelids, "MODEL_ID")

#Top 10 data
top_10_df = explore_outliers.filter(col("MODEL_ID").isin(top_10_ids))
top_10_df = top_10_df.withColumn('model_char', substring('MODEL_ID', 1,4))
top_10_pd = top_10_df.toPandas()
top_10_pd

#Plot
g = sns.FacetGrid(top_10_pd, col='model_char', hue="OUTLIER_TYPE", col_wrap=3)
g = g.map(sns.scatterplot, TIME_VAR, TARGET_VAR).add_legend()

# COMMAND ----------

#Log image to mlflow
g.savefig("Largest_SKUS.png")
mlflow.log_artifact("Largest_SKUS.png")

# COMMAND ----------

#Bottom top 10 largest modeled hierarchies
#Filter top 10 modeled hierarchies
qc_modelids = aggregate_data(explore_outliers,["MODEL_ID"], [TARGET_VAR], [sum])
qc_modelids = qc_modelids.sort("sum_" + TARGET_VAR)
qc_modelids = qc_modelids.limit(10)
top_10_ids =convertDFColumnsToList(qc_modelids, "MODEL_ID")

#Top 10 data
top_10_df = explore_outliers.filter(col("MODEL_ID").isin(top_10_ids))
top_10_df = top_10_df.withColumn('model_char', substring('MODEL_ID', 1,4))
top_10_pd = top_10_df.toPandas()
top_10_pd

#Plot
g = sns.FacetGrid(top_10_pd, col='model_char', hue="OUTLIER_IND", col_wrap=3)
g = g.map(sns.scatterplot, TIME_VAR, TARGET_VAR).add_legend()

# COMMAND ----------

#Log image to mlflow
g.savefig("Smallest_SKUS.png")
mlflow.log_artifact("Smallest_SKUS.png")

# COMMAND ----------

# DBTITLE 1,Output
#Write as delta table to dbfs
explore_outliers = explore_outliers.select(["MODEL_ID",TIME_VAR,"OUTLIER_IND","OUTLIER_TYPE"])
save_df_as_delta(explore_outliers, DBO_OUTLIERS, enforce_schema=False)

delta_info = load_delta_info(DBO_OUTLIERS)
set_delta_retention(delta_info, "90 days")
display(delta_info.history())

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

