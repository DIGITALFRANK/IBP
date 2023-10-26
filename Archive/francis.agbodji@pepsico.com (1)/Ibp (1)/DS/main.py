# Databricks notebook source
# MAGIC %md 
# MAGIC # Main
# MAGIC 
# MAGIC This script runs the final POC pipeline code in sequence.  It is meant as a reference to walk through how the code links together.  It will also be used to test code enhancements on sample data prior to being approved to Master during the MVP.
# MAGIC 
# MAGIC Note(s):
# MAGIC 
# MAGIC > **Naming Conventions**: 
# MAGIC >> 1.  M = Master, D = DEV.  Master scripts are the production code.  Dev code is development code that is yet to be tested in order to be approved into master
# MAGIC >> 2.  0X = Sequence of scripts.  If scripts have same number they can be run in parallel within Azure Data Factory.
# MAGIC >> 3.  MRD = model ready dataset
# MAGIC >> 4.  DBI = DemandBrain input table
# MAGIC >> 5.  DBA = DemandBrain interim table
# MAGIC >> 6.  DBO = DemandBrain output table
# MAGIC 
# MAGIC > **Table & Configuration Documentation **: Please refer to src/configs for documentation on the input / output tables and configurations
# MAGIC 
# MAGIC > **Experiments**: All code references files in the inherited path as main.  However, artifacts are logged to PEP_Master_Pipeline/Experiments to track experiments against each other.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 - Data Integration
# MAGIC This script maps relevant data sources into a "base" modeling dataset.  Downstream processes will perform an EDA, feature engineering and cleanse the data.  If demand drivers are present for a market, they are mapped to the data in this script.  If they are not present for his market, the will not be included in the final output.
# MAGIC 
# MAGIC * Inputs
# MAGIC   * DBI_ORDERS
# MAGIC   * DBI_SHIPMENTS (shipments is used if orders not present)
# MAGIC   * DBI_PRODUCTS
# MAGIC   * DBI_CUSTOMER
# MAGIC   * DBI_LOC
# MAGIC   * DBI_CALENDAR
# MAGIC   * DBI_MEDIA
# MAGIC   * DBI_PRICING_WEEKLY / DBI_PRICING_MONTHLY
# MAGIC   * DBI_HOLIDAYS_MONTHLY / DBI_HOLIDAYS_WEEKLY
# MAGIC   * DBI_EXTERNAL_VARIABLES_WEEKLY / DBI_EXTERNAL_VARIABLES_MONTHLY
# MAGIC   * DBI_PROMO_ES_WEEKLY / DBI_PROMO_ES_MONTHLY / DBI_PROMO_PT_WEEKLY / DBI_PROMO_PT_MONTHLY
# MAGIC 
# MAGIC * Outputs
# MAGIC   * DBA_MRD_EXPLORATORY

# COMMAND ----------

# # Uncomment below command if widget is not working in cloned branch 
# dbutils.widgets.dropdown("TIME_VAR_LOCAL", "Week_Of_Year", [str(x) for x in ["Week_Of_Year", "Month_Of_Year"]]) 

# COMMAND ----------

# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

dbutils.notebook.run("D01 Data Integration", 6000)

# COMMAND ----------

# MAGIC %md 
# MAGIC ##02 - Competitor Identification
# MAGIC 
# MAGIC This modules creates competitor variables to be used in modeling. There are two possible ways to obtain competitor variables:
# MAGIC 
# MAGIC > **Automatic Generation**: This method automatically finds the most likely competitor set of products given a set of hierarchy configurations and statistical thresholds.
# MAGIC >> 1.  Products are filtered to a set that are within provided hierarchy configurations (e.g., competitors should be same category, pack size)
# MAGIC >> 2.  Products are filtered using statistical thresholds. E.g.
# MAGIC   * Filter Low sales competitors (competitors with <10% of sales compared to own product)
# MAGIC   * Filter Low history (<20 rows of history where both own and competitive products were sold)
# MAGIC   * Keep highly correlated pairs (competitor price vs. own demand)
# MAGIC   * Filter to top X competitors (based on competitor sales or correlation)
# MAGIC >> 3.  The data for the top competitors is then shaped into variables so that it can be merged into mrd
# MAGIC 
# MAGIC > **Client provided list**: This method directly uses competitor pairs provided by the client and only prepares the data into variables for mrd
# MAGIC 
# MAGIC * Inputs
# MAGIC   * DBA_MRD_EXPLORATORY
# MAGIC   
# MAGIC * Outputs
# MAGIC   * DBO_COMPETITOR_VARS
# MAGIC   * DBO_IDENTIFIED_COMPETITORS

# COMMAND ----------

dbutils.notebook.run("D02 Driver Competitor Identification", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##02 - Outlier Detection
# MAGIC 
# MAGIC This script flags outliers.  The user can specify what algorithm they want to use to detect outliers (LOF, KNN, etc.) through the configuration layer, threshlolds for the detection algorithm and what variables they want to detect (e.g., target variable, price, distribution, etc.).  The data is stored in a delta table for pickup within downstream activities.
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD_EXPLORATORY
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBO_OUTLIERS

# COMMAND ----------

dbutils.notebook.run("D02 Outlier Detection", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##02 - Segmentation
# MAGIC 
# MAGIC This script creates business and statistical segmentation for reporting and features into the model
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD_EXPLORATORY
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBO_SEGMENTS

# COMMAND ----------

dbutils.notebook.run("D02 Segmentation", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 03 - EDA
# MAGIC 
# MAGIC Performs Exploratory data analysis on the data.  Artifacts are saved to mlflow to guide thresholds for data cleansing / transformations.
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD_EXPLORATORY
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * SweetViz html report (mlflow)

# COMMAND ----------

dbutils.notebook.run("D03 EDA", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 04 - Cleansing
# MAGIC 
# MAGIC This script cleanses the data prior to automated feature engineering.  Including:
# MAGIC * Filtering
# MAGIC * Imputations
# MAGIC * Outlier handling
# MAGIC * Transformations
# MAGIC * Indexing
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD_EXPLORATORY
# MAGIC   * DBO_SEGMENTS
# MAGIC   * DBO_OUTLIERS
# MAGIC   * DBO_COMPETITOR_VARS
# MAGIC   * DBI_CALENDAR
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBA_MRD_CLEAN
# MAGIC   * DBA_MODELIDS

# COMMAND ----------

dbutils.notebook.run("D04 Cleansing", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 05 - Driver Forecasting
# MAGIC 
# MAGIC This script predicts a list of features using traditional time series (arima, holt, etc.), lagging last year values or historical averaging. 
# MAGIC * It can be used as a benchmark for the target variable to compare against the ML models
# MAGIC * It can also be used to predict future variables in time (e.g., weather, distribution, etc.)
# MAGIC * It runs the predictions using a rolling holdout method and outputs the prediction table in a similar format to Stage 1 & 2 models
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD_CLEAN 
# MAGIC   * DBI_CALENDAR
# MAGIC   * DBI_HOLIDAYS_MONTHLY / DBI_HOLIDAYS_WEEKLY
# MAGIC   * DBI_PROMO_ES_MONTHLY / DBI_PROMO_PT_MONTHLY / DBI_PROMO_ES_WEEKLY / DBI_PROMO_PT_WEEKLY
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBA_DRIVER_FORECASTS

# COMMAND ----------

dbutils.notebook.run("D05_Driver_Forecasting", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 06 - Feature Engineering
# MAGIC 
# MAGIC This script develops additional features for modeling.  The output from this script is the final model ready dataset (mrd).
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_DRIVER_FORECASTS 
# MAGIC   * DBI_CALENDAR
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBA_MRD

# COMMAND ----------

dbutils.notebook.run("D06 Feature Engineering", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 07 - Hyperparameter Tuning
# MAGIC 
# MAGIC This script uses Bayesian Optimization (and Gridsearch) to tune hyperparameters of certain algorithms for Stage1 and Stage2 models.  
# MAGIC * The search is performed individually by segment (HYPERPARAM_LEVEL) and leverages pyspark to parallelize segment search across workers.  If no segment is given, a dummy segment is developed to run one search for the entire dataset.  
# MAGIC * The optimal parameters are stored as dictionary yaml files to mlflow and picked up in the modeling script.
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD 
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBO_HYPERPARAMATER

# COMMAND ----------

dbutils.notebook.run("D07 Hyperparameter Tuning", 12000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 08 - Modeling
# MAGIC 
# MAGIC This script develops the forecast on the mrd (model ready dataset). This runs through a train/test period, uses the results for model selection/feature importance, and then runs on forward-looking period.
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBA_MRD 
# MAGIC   * DBA_MODELIDS
# MAGIC   * DBO_HYPERPARAMATER
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT
# MAGIC   * DBO_PICKLE_STAGE1_ROLLING_BACKTEST
# MAGIC   * DBO_PICKLE_STAGE1_FUTURE_PERIOD
# MAGIC   * DBO_FORECAST_TRAIN_TEST_SPLIT
# MAGIC   * DBO_FORECAST_FUTURE_PERIOD

# COMMAND ----------

dbutils.notebook.run("D08 Production Modeling - Dynamic", 60000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 09 - Report Out Accuracy
# MAGIC 
# MAGIC This script outputs the DemandBrain accuracy compared to a reference point (e.g., PEP, adjusted forecast, final DP submission, etc.).
# MAGIC * It calculates DemandBrain accuracy for all present lags and compares to whatever lags are in the PEP forecast
# MAGIC * It reports accuracy at the global accuracy_report_level config
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBO_FORECAST_ROLLING
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * N/A

# COMMAND ----------

dbutils.notebook.run("D08 Report Out Accuracy", 6000)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 09 - Wireframe Output
# MAGIC 
# MAGIC Outputs tables to the silver layer for wireframe pickup
# MAGIC 
# MAGIC * Inputs: 
# MAGIC   * DBO_FORECAST_FUTURE_PERIOD (dbfs)
# MAGIC 
# MAGIC * Outputs: 
# MAGIC   * DBO_FORECAST_FUTURE_PERIOD (silver layer)

# COMMAND ----------

dbutils.notebook.run("D09 Wireframe Output", 6000)