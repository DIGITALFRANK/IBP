# Databricks notebook source
# DBTITLE 1,Imports
import warnings
import pandas as pd
from bayes_opt import BayesianOptimization
import lightgbm
from catboost import CatBoostRegressor, Pool
from pyod.models.knn import KNN
from collections import namedtuple

# COMMAND ----------

# DBTITLE 1,Spark Config

spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
tenant_id       = "42cc3295-cd0e-449c-b98e-5ce5b560c1d3"
client_id       = "e396ff57-614e-4f3b-8c68-319591f9ebd3"
client_secret   = dbutils.secrets.get(scope="cdo-ibp-dev-kvinst-scope",key="cdo-dev-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
storage_account = "cdodevadls2"

storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
spark.catalog.clearCache()
spark.conf.set("spark.worker.cleanup.enabled", "true")

# COMMAND ----------

# DBTITLE 1,MLFlow
MLFlowConfig = namedtuple('MLFlowConfig', ['experiment_name', 'experiment_path'])
MLFLOW_CONFIGS = dict()

# this one has to be an *absolute* path in the Databricks workspace (could be either in the /Shared or the /User folder)
MLFLOW_EXPERIMENT_NAME_BASE = '/Shared/mlflow-tracking'

# this one is the path where artifact files will be saved (a path in DBFS)
MLFLOW_EXPERIMENT_PATH_BASE = 'dbfs:/mnt/adls/mlflow-tracking'

# for each step in the pipeline
steps = ['modelling']
for s in steps:
  MLFLOW_CONFIGS[s] = MLFlowConfig(experiment_name=f'{MLFLOW_EXPERIMENT_NAME_BASE}/{s}', experiment_path=f'{MLFLOW_EXPERIMENT_PATH_BASE}/{s}')

# COMMAND ----------

# DBTITLE 1,Load Configuration Files
## Inputs (Parameter Files)
#TO-DO: Once we are on PEP, these will be Decisio SQL tables (helper functions exist to read in parallel codebase utilities)
DBP_GLOBALCONFIGS = "DBP_GLOBALCONFIGS"  

#Data Configs
DBP_DATA_PATHS = "/dbfs/FileStore/tables/DBP_DATA_PATHS-1.csv" #Input / Output data location paths
DBP_DATA_CONFIGS = "/dbfs/FileStore/tables/DBP_DATA_CONFIGS.csv" #Configs used to cleanse data
DBP_OUTLIERS = "/dbfs/FileStore/tables/DBP_OUTLIERS-1.csv" #Configs used to detect outliers
DBP_IMPUTATIONS = "/dbfs/FileStore/tables/DBP_IMPUTATION-1.csv" #Cofigs used to determine how to impute missing variables
DBP_TREATMENT = "/dbfs/FileStore/tables/DBP_FEATURE_TREATMENT-3.csv" #Feature treatment (log, lag, etc.)
DBP_HIER_PATHS = "/dbfs/FileStore/tables/DBP_DATA_HIER-6.csv" #Get Product, Customer and Location Hierarchies
DBP_DATA_MERGE_REPORT="/dbfs/FileStore/tables/DBP_DATA_MERGE_REPORT-2.csv" #Get columns for merging core dataframes and reporting feature names
DBP_SEGMENT="/dbfs/FileStore/tables/DBP_SEGMENT_CONFIGS-1.csv" #Get configs to create segments
#Modeling Configs
DBP_HYPERPARAM = "/dbfs/FileStore/tables/DBP_HYPERPARAM_5.csv" #Hyperparameter tuning threshold configs
DBP_HYPERPARAM_DEFAULT="/dbfs/FileStore/tables/DBP_HYPERPARAMETER_DEFAULT.csv" #Default parameters for models
#Report out Configs
DBP_ACCURACY = "/dbfs/FileStore/tables/DBP_ACCURACY-2.csv" #Parameters to guide accuracy report out
DBP_DECOMP = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/configs/DBP_DECOMP"

PROLINK_DATA_PATH="abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/onetimefull-load/ibp-bronze/IBP/EDW/Prolink/"
SHIPMENT_PATH=PROLINK_DATA_PATH+'Shipment Actuals/'
CUSTOMER_PATH=PROLINK_DATA_PATH+'Customer Master/'
PRODUCT_PATH=PROLINK_DATA_PATH+'Product Master/'
LOCATION_PATH=PROLINK_DATA_PATH+'Distribution Master/'

# COMMAND ----------

path_configs = pd.read_csv(DBP_DATA_PATHS)
path_configs = load_parameters(path_configs,"Config","Value")

data_configs = pd.read_csv(DBP_DATA_CONFIGS)
data_configs = load_parameters(data_configs,"Config","Value")

impute_configs = pd.read_csv(DBP_IMPUTATIONS)

treatment_configs = pd.read_csv(DBP_TREATMENT)

outlier_configs = pd.read_csv(DBP_OUTLIERS)
outlier_configs = load_parameters(outlier_configs,"Config","Value")

accuracy_configs = pd.read_csv(DBP_ACCURACY)
accuracy_configs = load_parameters(accuracy_configs,"Config","Value")

hyper_configs = pd.read_csv(DBP_HYPERPARAM)
hyper_def_configs = pd.read_csv(DBP_HYPERPARAM_DEFAULT)

hier_configs=pd.read_csv(DBP_HIER_PATHS)
merge_configs=pd.read_csv(DBP_DATA_MERGE_REPORT)
merge_configs = load_parameters(merge_configs,"Config","Value")

seg_configs=pd.read_csv(DBP_SEGMENT)
seg_configs = load_parameters(seg_configs,"Config","Value")

# COMMAND ----------

# DBTITLE 1,Formatting / Display Options
## Turn-off specific warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# ## Set Pandas display options - decimals, max cols, max rows
pd.set_option('display.float_format', lambda x: '%0.3f' %x)
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 300)

## Handling for "Setting-as-Copy"
pd.options.mode.chained_assignment = None

# COMMAND ----------

# DBTITLE 1,Input / Output Connections
#TO-DO: Once we are setup on PEP environment, hook up to Decisio SQL tables
#Get sql connection details
#Determine if run is ADF or ADB
try:
  #Global configs coming from ADF
  TIME_VAR = dbutils.widgets.get("TIME_VAR_ADF")
  print('ADF Run')
except:
  try:
    #Global configs coming from main
    TIME_VAR = dbutils.widgets.get("TIME_VAR_MAIN")
    print('Main Run')
  except:
    try:
      #Global configs coming from local
      TIME_VAR = dbutils.widgets.get("TIME_VAR_LOCAL")      
    except:
      TIME_VAR = "Week_Of_Year"
    print('Local Run')
  
#Get user connection parameters
# if sql_environment == "DEV":
#   scope = "bieno-da-d-80173-appk-01"
# elif sql_environment == "QA":
#   scope = "bieno-da-q-80174-appk-01"
# elif sql_environment == "PP":
#   scope = "bieno-da-u-80175-appk-01"
# elif sql_environment == "PROD":
#   scope = "bieno-da-p-80176-appk-01"
# else:
#   scope = "bieno-da-d-80173-appk-01"

# creds_to_get = { 'user':      "tpo-demandbrain-" + market + "-user",
#                  'password' : "tpo-demandbrain-" + market + "-password",
#                  'database' : "tpo-demandbrain-" + market + "-database",
#                  'server' :   "tpo-demandbrain-sqlserver" }
# credentials = get_secrets(in_scope = scope, secret_dict=creds_to_get)
# user = credentials.get("user")
# password = credentials.get("password")
# database = credentials.get("database")
# server = credentials.get("server")
# run_location = "Databricks"
# jdbcurl = "jdbc:sqlserver://"+server+";database="+database+";user="+user+";password="+password


# COMMAND ----------

# DBTITLE 1,Input/Output Tables
# Naming convention of tables:
# DBI – Curated datasets from Data Engineering
# DBP – Configuration tables
# DBO – DemandBrain output tables for the front-end
# DBA – Intermediate tables used to pass informaion between scripts

#Define Current DBA/DBO tables paths #################################################################################################
TABLES_ROOT='dbfs:/mnt/adls/Tables/'     #root path, should be static unless major env change
OPTIONAL_EXP_PATH=''               #an empty sting will result in default path dbfs:/mnt/adls/Tables/ 
#using the OPTIONAL_EXP_PATH will result in dbfs:/mnt/adls/Tables/OPTIONAL_EXP_PATH
######################################################################################################################################


### Raw Input Tables #################################################################################################################
#TO-DO: Data Engineering will have to clean our input tables in a similar fashion
# SHIPMENT_PATH_RAW = path_configs.get('SHIPMENT_PATH_RAW') 
# PROD_MASTER_PATH_RAW = path_configs.get('PROD_MASTER_PATH_RAW')
# CUST_MASTER_PATH_RAW = path_configs.get('CUST_MASTER_PATH_RAW') 
# LOC_MASTER_PATH_RAW = path_configs.get('LOC_MASTER_PATH_RAW') 

### Curated Input Tables ###
DBI_PRODUCTS = path_configs.get('DBI_PRODUCTS') 
DBI_CUSTOMER = path_configs.get('DBI_CUSTOMER') 
DBI_LOC = path_configs.get('DBI_LOC') 
DBI_SHIPMENTS = path_configs.get('DBI_SHIPMENTS') 
# DBI_HOLIDAYS = path_configs.get('DBI_HOLIDAYS') 
DBI_CALENDAR = 'dbfs:/mnt/adls/Tables/DBI_CALENDAR'
DBI_EXTERNAL_VARIABLES_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_EXTERNAL_VARIABLES_WEEKLY'
DBI_EXTERNAL_VARIABLES_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_EXTERNAL_VARIABLES_MONTHLY'
DBI_WEATHER_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_WEATHER_WEEKLY'
DBI_WEATHER_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_WEATHER_MONTHLY'
DBI_WEATHER_WEEKLY_HISTORICALS = 'dbfs:/mnt/adls/Tables/DBI_WEATHER_WEEKLY_HISTORICALS'
DBI_WEATHER_MONTHLY_HISTORICALS = 'dbfs:/mnt/adls/Tables/DBI_WEATHER_MONTHLY_HISTORICALS'
DBI_HOLIDAYS_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_HOLIDAYS_WEEKLY'
DBI_HOLIDAYS_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_HOLIDAYS_MONTHLY'
DBI_PRICING_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_PRICING_WEEKLY'
DBI_PRICING_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_PRICING_MONTHLY'
DBI_PROMO_PT_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_PROMO_PT_WEEKLY'
DBI_PROMO_PT_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_PROMO_PT_MONTHLY'
DBI_PROMO_ES_WEEKLY = 'dbfs:/mnt/adls/Tables/DBI_PROMO_ES_WEEKLY'
DBI_PROMO_ES_MONTHLY = 'dbfs:/mnt/adls/Tables/DBI_PROMO_ES_MONTHLY'
DBI_NIELSEN = 'dbfs:/mnt/adls/Tables/DBI_NIELSEN'
DBI_MEDIA = 'dbfs:/mnt/adls/Tables/DBI_MEDIA'
DBI_INVENTORY = 'dbfs:/mnt/adls/Tables/DBI_INVENTORY'

### Demand Brain Intermediate Tables ##################################################################################################
#MRD = model ready dataset
#DFU = Demand Forecasting Unit
#All DBA tables
dba_list=['DBA_MRD_EXPLORATORY',
          'DBA_MRD_PD_EXPLORATORY',
          'DBA_MRD_CLEAN',
          'DBA_MRD',
          'DBA_MODELIDS',
          'DBA_DRIVER_FORECASTS',
          'DBA_ACCURACY_DUPES',
          'DBA_MRD_SUB']
#notes 
#DBA_MRD_EXPLORATORY = Tables with all necessary merged data (not cleansed)
#DBA_MRD_CLEAN = Cleansed model ready data
#DBA_MODELIDS = Dataset to retain hierarcy information for every DFU
#DBA_DRIVER_FORECASTS = Forecast from driver (time series analysis)
#DBA_ACCURACY_DUPES = DBA Accuracy table - If duplicates are found between Pep / demandbrain accuracy tables, retained here

### Demand Brain Output Tables ########################################################################################################
dbo_list=['DBO_HYPERPARAMATER',
          'DBO_OUTLIERS',
          'DBO_SEGMENTS',
          'DBO_FORECAST',
          'DBO_COMPETITOR_VARS',
          'DBO_IDENTIFIED_COMPETITORS',
          'DBO_FORECAST_ROLLING',
          'DBO_FORECAST_TIME_SERIES',
          'DBO_FORECAST_TRAIN_TEST_SPLIT',
          'DBO_FORECAST_ROLLING_BACKTEST',
          'DBO_FORECAST_FUTURE_PERIOD',
          'DBO_ACCURACY_COMPARISON',
          'DBO_ACC_MERGED',
          'DBO_PICKLE_STAGE1_TRAIN_TEST_SPLIT',
          'DBO_PICKLE_STAGE1_ROLLING_BACKTEST ',
          'DBO_PICKLE_STAGE1_FUTURE_PERIOD']
#notes
#DBO_HYPERPARAMS= Hyperparameter Tuning Output Table
#DBO_OUTLIERS=Table of modeled hierarchy over time with outlier indicator
#DBO_SEGMENTS= Table of modeled hierarchy with business and statistical segments tagged
#DBO_FORECAST= Static one-time future forecast
#DBO_COMPETITOR_VARS =Pricing Features Tables
#DBO_FORECAST_ROLLING  =  Cross-validated rolling forecast
#DBO_ACCURACY_COMPARISON = Output path for demandbrain accuracies
#DBO_ACC_MERGED = Compared accuracies
required_tables=dba_list+dbo_list
#creates all required tables in loop
for cur_table in required_tables:
  exec(f"{cur_table}=concat_dbfs_path('{TABLES_ROOT}','{OPTIONAL_EXP_PATH}','{TIME_VAR}','{cur_table}')")
       


#Archive reference for old directories#################################################################################################
#path replaced by MVP sandbox directories
### Demand Brain Intermediate Tables ###
#MRD = model ready dataset
#DFU = Demand Forecasting Unit
#DBA_MRD_EXPLORATORY = path_configs.get('DBA_MRD_EXPLORATORY') #Table with all necessary merged data (not cleansed)
#DBA_MRD_PD_EXPLORATORY = path_configs.get('DBA_MRD_PD_EXPLORATORY')
# DBA_MRD_CLEAN = path_configs.get('DBA_MRD_CLEAN') #Cleansed model ready data
# DBA_MRD = path_configs.get('DBA_MRD') #Final model ready data with features included
#DBA_MODELIDS = path_configs.get('DBA_MODELIDS') #Dataset to retain hierarcy information for every DFU
# DBA_MODELIDS = "dbfs:/mnt/adls/Tables/DBA_MODELIDS" #TO-DO: Add to configs table
# DBA_DRIVER_FORECASTS = 'dbfs:/mnt/adls/Tables/DBA_DRIVER_FORECASTS'
### Hyperparameter Tuning Output Table
# DBO_HYPERPARAMATER = 'dbfs:/mnt/adls/Tables/DBO_HYPERPARAMS'
### Demand Brain Output Tables ###
#DBO_OUTLIERS = path_configs.get('DBO_OUTLIERS') #Table of modeled hierarchy over time with outlier indicator
#DBO_SEGMENTS = path_configs.get('DBO_SEGMENTS') #Table of modeled hierarchy with business and statistical segments tagged
#DBO_FORECAST = path_configs.get('DBO_FORECAST') #Static one-time future forecast
#DBO_FORECAST_ROLLING = path_configs.get('DBO_FORECAST_ROLLING') #Cross-validated rolling forecast
#DBO_FORECAST_TIME_SERIES = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_TIME_SERIES'
#DBO_FORECAST_TRAIN_TEST_SPLIT = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_TRAIN_TEST_SPLIT'
#DBO_FORECAST_ROLLING_BACKTEST = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_ROLLING_BACKTEST'
#DBO_FORECAST_FUTURE_PERIOD = 'dbfs:/mnt/adls/Tables/DBO_FORECAST_FUTURE_PERIOD'
#DBI_PEP_ACC_DATA = path_configs.get('DBI_PEP_ACC_DATA') #Comparison PepsiCo forecast
#DBO_ACCURACY_COMPARISON = path_configs.get('DBO_ACCURACY_COMPARISON') #Output path for demandbrain accuracies
#DBA_ACCURACY_DUPES = path_configs.get('DBA_ACCURACY_DUPES') # If duplicates are found between Pep / demandbrain accuracy tables,                                                            # they are retained here
#DBO_ACC_MERGED="dbfs:/mnt/adls/Tables/MB_test/accuracies_merged" 
#MEDIA_MERGE_FIELD = ["HRCHY_LVL_3_NM", "SRC_CTGY_1_NM", "BRND_NM", TIME_VAR] 
### Pricing Features Tables ###
#DBO_COMPETITOR_VARS = 'dbfs:/mnt/adls/Tables/competitor_vars'
#DBO_IDENTIFIED_COMPETITORS = 'dbfs:/mnt/adls/Tables/identified_competitors'

#Dictionary paths
# STATIC_MODEL_DICT_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/static_model_dict.csv"
# ROLLING_MODEL_DICT_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/rolling_model_dict.csv"
# SHAP_DICT_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/SHAP_value_dict.csv"

# DELTA_TABLE_PATH = "dbfs:/mnt/adls/Tables/"
# MRD_PATH_EXTERNAL = "dbfs:/mnt/adls/Tables/External_Joined_DBA_MRD"
# MRD_PATH_PD_EXTERNAL = "/dbfs/FileStore/PEP_Volume_Forecasting/external_joined_mrd.csv"

# MRD_PATH = "dbfs:/mnt/adls/Tables/DBA_MRD"
# MRD_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/mrd.csv"

# ##Anand Added it back
# MRD_PATH_CLEANSED = "dbfs:/mnt/adls/Tables/Cleansed_Joined_DBA_MRD"
# MRD_PATH_PD_CLEANSED = "/dbfs/FileStore/PEP_Volume_Forecasting/cleansed_joined_mrd.csv"

# PRED_PATH = "dbfs:/mnt/adls/Tables/DBA_PREDICTIONS"
# PRED_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/predictions.csv"

# PRED_XVAL_PATH = "dbfs:/mnt/adls/Tables/DBA_XVAL_PREDICTIONS"
# PRED_XVAL_PATH_PD = "/dbfs/FileStore/PEP_Volume_Forecasting/cross_val_predictions.csv"

# # TO-DO: These will be depreciated once modeling picks up hyperparams from mlflow
# XGB_HYPER_DICT_PATH = "/dbfs/FileStore/PEP_Volume_Forecasting/xgb_bayesian_hyperparam_dict.csv"
# LGBM_HYPER_DICT_PATH = "/dbfs/FileStore/PEP_Volume_Forecasting/lgbm_bayesian_hyperparam_dict.csv"
# RF_HYPER_DICT_PATH = "/dbfs/FileStore/PEP_Volume_Forecasting/rf_bayesian_hyperparam_dict.csv"

# LGBM_HYPER_DICT_PATH_STAGE2 = '/dbfs/FileStore/PEP_Volume_Forecasting/lgbm_grid_hyperparam_dict_stage2.csv'
# CATBOOST_HYPER_DICT_PATH_STAGE2 = '/dbfs/FileStore/PEP_Volume_Forecasting/catboost_grid_hyperparam_dict_stage2.csv'
# QUANT_GBM_HYPER_DICT_PATH_STAGE2 = '/dbfs/FileStore/PEP_Volume_Forecasting/quant_gbm_grid_hyperparam_dict_stage2.csv'

# COMMAND ----------

# DBTITLE 1,Global Configs
## Setting Directory & Pipeline 
WORKING_DIR = 'Sample_Directory_Path'
PIPELINE_NAME = 'DemandBrain_PEP_Shipments_Pipeline'

#Dates for saving files
SAVE_DATE = str((datetime.date(datetime.now())))
SAVE_MONTH_YEAR = str(datetime.now().strftime('%h')) + '_' + str(datetime.now().strftime('%Y'))

## Columns to join on (for "core" datasets)
PROD_MERGE_FIELD = merge_configs.get('PROD_MERGE_FIELD') 
CUST_MERGE_FIELD = merge_configs.get('CUST_MERGE_FIELD') 
LOC_MERGE_FIELD =  merge_configs.get('LOC_MERGE_FIELD')
MODEL_ID_HIER = [PROD_MERGE_FIELD, CUST_MERGE_FIELD, LOC_MERGE_FIELD]

## Columns to join on (for driver datasets)
MEDIA_MERGE_FIELD = ["HRCHY_LVL_3_NM", "SRC_CTGY_1_NM", "BRND_NM"] 
PROMO_MERGE_FIELD = ['PLANG_CUST_GRP_VAL', 'DMDUNIT']
INVENTORY_MERGE_FIELD = ["DMDUNIT", "LOC"]

#Reporting columns used in scripts
CATEGORY_FIELD = merge_configs.get('CATEGORY_FIELD')
BRAND_FIELD = merge_configs.get('BRAND_FIELD') 
MARKET_FIELD = merge_configs.get('MARKET_FIELD')

## For weather data pulls - Corey added on 9/1/2021
WEATHER_HISTORY_YEARS = 4
AGG_WEATHER_FEAT_BY_COUNTRY = True

# Setting Target variable name and qualifiers
target_var_flag = 'target_var_'
TARGET_VAR_ORIG = 'CASES'
TARGET_VAR_TREATMENT = '_log'
# TARGET_VAR = target_var_flag + TARGET_VAR_ORIG
TARGET_VAR = 'CASES'
TARGET_VAR_EQUIVALENT = 'DMND_HSTRY_SRC_UOM_QTY'
TARGET_VAR_ACCURACY = 'CASES_ORIG'


RUN_TYPE = "DYNAMIC"

## Setting time parameters
TIME_VAR_MONTH = 'Month_Of_Year'       
TIME_VAR_WEEK = 'Week_Of_Year' 

if TIME_VAR=="Month_Of_Year":
    START_PERIOD=201811
    END_PERIOD=202104
    CALENDAR_DATEVAR = "Month_start_date"
elif TIME_VAR=="Week_Of_Year":
    START_PERIOD=201842
    END_PERIOD=202121
    CALENDAR_DATEVAR = "Week_start_date"

# SS: I suspect these might be depreciated
## Set hierarchy elements across our 3 modeling dimensions

PRODUCT_HIER = hier_configs['PRODUCT_HIER'].dropna().unique().tolist()

# Palaash Note: Update column refrence below
CUSTOMER_HIER = hier_configs['CUSTOMER_HIER'].dropna().unique().tolist()
LOCATION_HIER = hier_configs['LOCATION_HIER'].dropna().unique().tolist()
COUNTRY_LIST = hier_configs['COUNTRY_LIST'].dropna().unique().tolist()
SHIP_DATA_COLS = hier_configs['SHIP_DATA_COLS'].dropna().unique().tolist()
LOC_DATA_COLS = hier_configs['LOC_DATA_COLS'].dropna().unique().tolist()
PROD_DATA_COLS = hier_configs['PROD_DATA_COLS'].dropna().unique().tolist()
CUST_DATA_COLS = hier_configs['CUST_DATA_COLS'].dropna().unique().tolist()
CORE_NUMERIC_COLS = hier_configs['CORE_NUMERIC_COLS'].dropna().unique().tolist()

#Probably depreciated
## Set level of the hierarchy at which you would like to model
PRODUCT_LEVEL = len(PRODUCT_HIER)
CUSTOMER_LEVEL = len(CUSTOMER_HIER)
LOCATION_LEVEL = len(LOCATION_HIER)

## Add other categoricals not represented above in hierarchies
# PRODUCT_HIER required for get_hierarchy
OTHER_CATEGORICALS = []
GROUPING_LEVEL = get_hierarchy()  ## updated this function to avoid an error in how old version was referenced (ie, now conforms to PEP)

# COMMAND ----------

# DBTITLE 1,Data Configs
UNIQUE_VAL_THRESH = data_configs.get('UNIQUE_VAL_THRESH') #Variables must have at least this many unique values or they are dropped
NAN_THRESH_PERC = data_configs.get('NAN_THRESH_PERC') 
NULL_THRESH = data_configs.get('NULL_THRESH') 
VIF_THRESHOLD = data_configs.get('VIF_THRESHOLD') 
TOP_HOL_THRESH = data_configs.get('TOP_HOL_THRESH') #Top # holidays to include in modeling

TOP_HOL_THRESH = int(TOP_HOL_THRESH)

#TO-DO: Transition to CSV (if they are still used, not if they are depreciated)
TIME_FEAT_BUNDLE = 'weekly'
COEFF_THRESH = 0.05
LOW_FEAT_IMP_CUTOFF = 20
SHAP_REF_MODEL = 'lightGBM_model_stage1'
HOLIDAY_YEARS_LIST = [2017, 2018, 2019, 2020, 2021, 2022, 2022, 2023]
ZERO_PRED_THRESHOLD = 10

# COMMAND ----------

# DBTITLE 1,Data Integration Config
TIME_COLS_TO_DROP = subtract_two_lists(["Week_Of_Year", "Month_Of_Year", "Week_start_date", "Month_start_date", "HSTRY_TMFRM_STRT_DT"], [TIME_VAR]) 

# COMMAND ----------

# DBTITLE 1,Cannibalization Configs
COMPETITOR_PRICE_VAR = "NET_PRICE_BAG"
COMPETITOR_NON_PROMO_PRICE_VAR = "LIST_PRICE_BAG"
COMPETITOR_FIELD = "Competitor_Flag"
COMPETITOR_SALES_LOWER_THRESHOLD = .1
COMPETITOR_HISTORY_THRESHOLD = 20
COMPETITOR_PEARSON_LOWER_THRESHOLD = 0
COMPETITOR_PEARSON_UPPER_THRESHOLD = 1
NUM_COMPETITOR_VARIABLES = 5

COMPETITOR_SORT_VAR = "CORR"  #"CORR" or TARGET_VAR
if COMPETITOR_SORT_VAR == TARGET_VAR:
  COMPETITOR_SORT_VAR = "sum_competitor_" + COMPETITOR_SORT_VAR
  
CLIENT_OWN_LIST = False

join_dict = {'SAVOURY SNACKS':["country_name","PLANG_CUST_GRP_VAL","LOC","SRC_CTGY_1_NM","SUBCAT","Competitor_Flag"],
             'CONFECTIONARY':["country_name","PLANG_CUST_GRP_VAL","LOC","SRC_CTGY_1_NM","SUBCAT","Competitor_Flag"],
             'FOODS':["country_name","PLANG_CUST_GRP_VAL","LOC","SRC_CTGY_1_NM","SUBCAT","Competitor_Flag"],
             'NON CARBONATED BEVERAGE(NCB)':["country_name","PLANG_CUST_GRP_VAL","LOC","SRC_CTGY_1_NM","SUBCAT","Competitor_Flag"],
             'CARBONATED SOFT DRINKS (CSD)':["country_name","PLANG_CUST_GRP_VAL","LOC","SRC_CTGY_1_NM","SUBCAT","Competitor_Flag"]             
            }
anti_dict = {'SAVOURY SNACKS':["DMDUNIT"],
             'CONFECTIONARY':["DMDUNIT"],
             'FOODS':["DMDUNIT"],
             'NON CARBONATED BEVERAGE(NCB)':["DMDUNIT"],
             'CARBONATED SOFT DRINKS (CSD)':["DMDUNIT"]             
            }

product_columns = {'HRCHY_LVL_1_ID':'DMDUNIT', 'PLANG_MTRL_GRP_NM': 'PLANG_MTRL_GRP_NM', 'SRC_CTGY_1_NM':'SRC_CTGY_1_NM',
                  'BRND_NM':'BRND_NM', 'SUBBRND_SHRT_NM':'SUBBRND_SHRT_NM', 'FLVR_NM':'FLVR_NM',
                  'PCK_CNTNR_SHRT_NM':'PCK_CNTNR_SHRT_NM', 'PLANG_MTRL_EA_PER_CASE_CNT':'PLANG_MTRL_EA_PER_CASE_CNT',
                  'SRC_CTGY_2_NM':'SUBCAT', 'PCK_SIZE_SHRT_NM' : 'PCK_SIZE_SHRT_NM'}

# COMMAND ----------

# DBTITLE 1,Cleansing Configs
## Create dictionary of imputation types
impute_configs = spark.createDataFrame(impute_configs)

#Variables and the imputation method (mean, median, etc.)
imputation_method_dict = convertDFColumnsToDict(impute_configs, "Variable", "Imputation_Method") 

#Variable and the level at which to impute (e.g., brand/week).  This in combination with the above will determine the imputation (e.g., price is imputed at the mean value for this brand, this week)
impute_level_dict = convertDFColumnsToDict(impute_configs, "Variable", "Level")
for key in impute_level_dict.keys():
  impute_level_dict[key] = convert_str_to_list(impute_level_dict[key], "~")
  
#TO-DO: Add configs to config file
imputation_method_dict = {'NET_PRICE_BAG': 'Mean', 
                          'LIST_PRICE_BAG': 'Mean'}
# imputation_method_dict = None

#TO-DO: Update configuration table for additional imputation types
impute_zeros_list = ['NET_PRICE_BAG','LIST_PRICE_BAG', 'ownSame_1_NET_PRICE_BAG', 'ownSame_2_NET_PRICE_BAG','ownSame_3_NET_PRICE_BAG','ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG']
impute_ffill_list = ['NET_PRICE_BAG','LIST_PRICE_BAG', 'ownSame_1_NET_PRICE_BAG', 'ownSame_2_NET_PRICE_BAG','ownSame_3_NET_PRICE_BAG','ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG']
impute_bfill_list = ['NET_PRICE_BAG','LIST_PRICE_BAG', 'ownSame_1_NET_PRICE_BAG', 'ownSame_2_NET_PRICE_BAG','ownSame_3_NET_PRICE_BAG','ownSame_4_NET_PRICE_BAG','ownSame_5_NET_PRICE_BAG']
groupcols_L1=[TIME_VAR]+['DMDUNIT','HRCHY_LVL_3_NM']
groupcols_L2=[TIME_VAR]+['SRC_CTGY_1_NM','HRCHY_LVL_3_NM']

# COMMAND ----------

# DBTITLE 1,Outlier Configs
## Outlier Configs
OUTLIER_THRESHOLD = outlier_configs.get('OUTLIER_THRESHOLD') #Contamination threshold for algorithm detecting outliers (pyod package)
OUTLIER_DETECT_LEVEL = outlier_configs.get('OUTLIER_DETECT_LEVEL') #Group at which to detect outliers, usually MODEL_ID level
OUTLIER_DETECT_LEVEL = OUTLIER_DETECT_LEVEL.split('~')
OUTLIER_METHOD = outlier_configs.get('OUTLIER_METHOD') #pyod Algorithm to use to detect outliers
# OUTLIER_VARS = outlier_configs.get('OUTLIER_VARS') #Variables used to detect outliers (always target variable, can also input variables such
#                                                    # as price, distribution)
# OUTLIER_VARS = OUTLIER_VARS.split('~')

OUTLIER_HANDLING_STRATEGY = outlier_configs.get('OUTLIER_HANDLING_STRATEGY') #Imputation for outliers (capping, interpolation, no method)
CAPPING_THRESHOLD = outlier_configs.get('CAPPING_THRESHOLD') #If capping is selected as straetgy, percentile used to cap outliers
INTERPOLATE_METHOD = outlier_configs.get('INTERPOLATE_METHOD') #If interpolation is selected as strategy, the interpolation method (linear 
                                                               # and time work well)

OUTLIER_VARS = [TARGET_VAR]
if TIME_VAR=="Week_Of_Year":
  MIN_ROWS_THRESHOLD = 10
elif TIME_VAR=="Month_Of_Year":
  MIN_ROWS_THRESHOLD = 6

#Convert outlier thresholds / algorithms to evaluation method
if OUTLIER_DETECT_LEVEL == "MAD":
  OUTLIER_THRESHOLD = int(OUTLIER_THRESHOLD)
else:
  OUTLIER_THRESHOLD = float(OUTLIER_THRESHOLD)

if OUTLIER_METHOD == 'MAD':
  method = eval(OUTLIER_METHOD)(threshold = OUTLIER_THRESHOLD)
else:
  method = eval(OUTLIER_METHOD)(contamination = OUTLIER_THRESHOLD)  
classifiers = {OUTLIER_METHOD : method}

# COMMAND ----------

# DBTITLE 1,Segmentation Configs
if TIME_VAR =='Week_Of_Year':
  CALENDAR_DATEVAR = "Week_start_date"
  SEGMENTATION_TIME_CUTOFF = 202121
elif TIME_VAR =='Month_Of_Year':
  CALENDAR_DATEVAR = "Month_start_date"
  SEGMENTATION_TIME_CUTOFF = 202104

## Stat Segmentation
STAT_SEGMENT_LEVEL=[]
STAT_SEGMENT_GROUP =list(seg_configs.get('STAT_SEGMENT_GROUP').split(",")) ## converts to string to list
STAT_SEGMENT_LEVEL =STAT_SEGMENT_LEVEL+[PROD_MERGE_FIELD]+[STAT_SEGMENT_GROUP][0]
STAT_SEG_ID = 'STAT_ID'
DTW_CLUSTER_NUM = 5
SEG_AGG_METHOD = sum

## Elements for "business" segmentation - Corey added on 5/10/2021

# SEGMENTATION_TIME_CUTOFF = int(seg_configs.get('SEGMENTATION_TIME_CUTOFF'))
QCUT_SEGMENTATION_LEVELS = [float(i) for i in seg_configs.get('QCUT_SEGMENTATION_LEVELS').replace(" ",'').split(',')]
QCUT_LABELS = seg_configs.get('QCUT_LABELS').replace(" ",'').replace("'",'').split(',')

QCUT_MATRIX_LEVELS = [float(i) for i in seg_configs.get('QCUT_MATRIX_LEVELS').replace(" ",'').split(',')]  
QCUT_MATRIX_LABELS = seg_configs.get('QCUT_MATRIX_LABELS').replace(" ",'').replace("'",'').split(',')

PEP_VALUE_THRESH_A = float(seg_configs.get('PEP_VALUE_THRESH_A'))
PEP_VALUE_THRESH_B = float(seg_configs.get('PEP_VALUE_THRESH_B'))

PEP_VOLATILITY_THRESH_A = float(seg_configs.get('PEP_VOLATILITY_THRESH_A'))
PEP_VOLATILITY_THRESH_B = float(seg_configs.get('PEP_VOLATILITY_THRESH_B'))

# COMMAND ----------

# DBTITLE 1,EDA Configs
#EDA Variables
# EDA_VARS = ["QTY","UDC_CATEGORY","UDC_BRAND","UDC_SUBBRAND","UDC_FLAVOUR","UDC_SHELF_LIFE","UDC_MARKETUNIT","UDC_CHANNEL","Wtd_Distribution_Proxy_L1","Wtd_Distribution_Proxy_L2","Wtd_Distribution_Proxy_L3","Wtd_Distribution_Proxy_L4","Wtd_Distribution_Proxy_L5","OUTLIER_IND","STARTDATE","UDC_CITY",
# "COUNTRY", "UDC_BRAND_GROUP", "UDC_PRODUCT_LINE", "UDC_SIZE", "Week_Of_Year", "Month_Of_Year", "UDC_PACK_CONTAINER"]

EDA_VARS = [TARGET_VAR,
 'SRC_CTGY_1_NM',
 'BRND_NM',
 'SUBBRND_SHRT_NM',
 'FLVR_NM',
 'UDC_SHELF_LIFE',
 'HRCHY_LVL_3_NM',
 'UDC_CHANNEL',
 'Wtd_Distribution_Proxy_L1',
 'Wtd_Distribution_Proxy_L2',
 'Wtd_Distribution_Proxy_L3',
 'Wtd_Distribution_Proxy_L4',
 'Wtd_Distribution_Proxy_L5',
 'OUTLIER_IND',
 'HSTRY_TMFRM_STRT_DT',
 'UDC_CITY',
 'PLANG_LOC_CTRY_ISO_CDV',
 'UDC_BRAND_GROUP',
 'UDC_PRODUCT_LINE',
 'UDC_SIZE',
 'Week_Of_Year',
 'Month_Of_Year',
 'UDC_PACK_CONTAINER']
EDA_SAMPLE_RATE = .99999

# COMMAND ----------

# DBTITLE 1,Driver Forecasting config
# DIST_PROXY_LEVEL_UNIV = [TIME_VAR, 'HRCHY_LVL_1_NM']
MIN_ROWS_THRESHOLD = 6
# DBA_DRIVER_FORECASTS = 'dbfs:/mnt/adls/Tables/DBA_DRIVER_FORECASTS'
covid_govt_policies_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_covid_govt_policies_clean_July2021.csv"

#Future assumptions that should be lagged from actuals
FORECAST_LAG_VARS = None

## COREY COMMENTS HERE - 7/20/2021
## The NUM_FWD_FRCST should be a user config, right?
## I refer to it as TIME_PERIODS_FORWARD in D07 - leaving distinct for now since I want it to be a config
## In my historical Train/Test split, I use: Holdout Period Used (Weeks) = (202114, 202121)
## In my Future pred, I would use 16 weeks from end of dataframe
## So, that would be: Forward-Looking Period Used (Weeks) = (202121, 202137)
## I am currently over-riding that with: FORWARD_RANGE = (202117, 202121) - so I can test functionality etc.
## PLEASE BE SURE TO SYNC/TIE WHAT YOU HAVE BELOW - in my mind, these should be configs

if TIME_VAR =='Week_Of_Year':
  NUM_FWD_FRCST = 16
  MODEL_FREQ = 'W'
  
  FORECAST_START_DATE = 202114
  FORECAST_START_DATES = []
  PERIODS = 4 # Adusting to include at least four weeks of rolling validation
  INTERVAL = 4 # This is the interval between FORECAST_START_DATES. If INTERVAL = 1 then FORECAST_START_DATES is continuous
  ONE_YEAR_LAG = 52
  TWO_YEAR_LAG = 104
  WEATHER_VAR = 'Week_Only'  
  CALENDAR_DATEVAR = "Week_start_date"
elif TIME_VAR =='Month_Of_Year':
  NUM_FWD_FRCST = 18
  MODEL_FREQ = 'M'
  
  FORECAST_START_DATE = 202008
  FORECAST_START_DATES = []
  PERIODS = 9
  INTERVAL = 1 # his is the interval between FORECAST_START_DATES. If INTERVAL = 1 then FORECAST_START_DATES is continuous
  ONE_YEAR_LAG = 12
  TWO_YEAR_LAG = 24
  WEATHER_VAR = 'Month_Only'
  CALENDAR_DATEVAR = "Month_start_date"
  
## Setting variables to be used in get_velocity_flag_pyspark function
if TIME_VAR == "Month_Of_Year":
  HIGH_VELOCTIY_TARGET_THRESH = 200*4
  HIGH_VELOCTIY_TIME_THRESH = 4/4
  LOW_VELOCITY_TARGET_THRESH = 25*4
  LOW_VELOCTIY_TIME_THRESH = 12/4
else:
  HIGH_VELOCTIY_TARGET_THRESH = 200
  HIGH_VELOCTIY_TIME_THRESH = 4
  LOW_VELOCITY_TARGET_THRESH = 25
  LOW_VELOCTIY_TIME_THRESH = 12

# COMMAND ----------

# DBTITLE 1,Feature Treatment Configs
COLS_TO_LOG = spark.createDataFrame(treatment_configs[treatment_configs["Treatment"]=='Log'])
COLS_TO_LOG = convertDFColumnsToList(COLS_TO_LOG,"Variable")

COLS_TO_INDEX = spark.createDataFrame(treatment_configs[treatment_configs["Index"]=='Yes'])
COLS_TO_INDEX = convertDFColumnsToList(COLS_TO_INDEX,"Variable")

# COMMAND ----------

# DBTITLE 1,Feature Engineering Configs
# #TO-DO: Why isn't this in the src files?
# # Also, this description doesn't make sense, what does this function do?
# def set_lag_period_list(input_list, min_lag_value):
#     '''
#     Dynamically sets lag periods greater than the minimum value set
#     This is to accommodate different future-looking forecast periods more automatically
#     '''
#     input_list = ensure_is_list(input_list)
#     output_list = sorted(list(set([each_lag if each_lag > min_lag_value else min_lag_value + each_lag for each_lag in input_list])))
#     return output_list

## Setting particular lags to be included
## Using the above function to tie this to minimum periods for prediction
DIST_PROXY_LEVEL_UNIV = [TIME_VAR, 'FCST_START_DATE', 'HRCHY_LVL_1_NM']

if TIME_VAR == "Month_Of_Year":
  LAGGED_PERIODS_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
else:
  LAGGED_PERIODS_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 52, 104] 

## Setting variables to be used in get_time_vars function
SHIPMENT_TIME_REF = CALENDAR_DATEVAR
TIME_VARS_TO_GENERATE = ["YearIndex", "MonthIndex", "Quarter", "QuarterYear", "LinearTrend", "CosX", "SinX"]

## Setting variable to be used in standardize_columns_pyspark function
STD_LEVEL = ["HRCHY_LVL_3_NM"]

# COMMAND ----------

## TODO - do we want to the tie the below to the user-selected hierarchy (above) more explicitly?
## Or, do we want to leave it as a separate option for the user to select?

# DIST_PROXY_LEVEL_UNIV = [TIME_VAR, 'UDC_CLIENT']
# DIST_PROXY_SELL_LEVELS_LIST = ['CTGY', 'BRND', 'SUBRND_CDV', 'FLVR_NM', 'DMDUNIT']

# DIST_PROXY_LEVEL_UNIV = [TIME_VAR, 'HRCHY_LVL_1_NM']
DIST_PROXY_SELL_LEVELS_LIST = ['SRC_CTGY_1_NM', 'BRND_NM', 'SUBBRND_SHRT_NM', 'FLVR_NM', 'DMDUNIT']

# COMMAND ----------

COLS_TO_STD = [
  'Wtd_Distribution_Proxy_L0', 'Wtd_Distribution_Proxy_L1', 'Wtd_Distribution_Proxy_L2',\
  'Wtd_Distribution_Proxy_L3', 'Wtd_Distribution_Proxy_L4',\
]

COLS_TO_QUANT_STD = []

# COMMAND ----------

# DBTITLE 1,Hyperparameter Configs
## Stage 1 Configurations ##
#Algos
stage_1_configs = hyper_configs[hyper_configs["Stage"]=="Stage 1"]
stage_1_configs = stage_1_configs[stage_1_configs["Run_Algo"]=="Yes"]
STAGE_1_ALGOS = stage_1_configs["Algorithm"].unique().tolist()

#Thresholds for grid search
STAGE_1_GRIDS = {}
for algos in STAGE_1_ALGOS:
  this_dict = stage_1_configs[stage_1_configs["Algorithm"]==algos]
  this_dict = this_dict[this_dict["Type"]=="Grid"]
  this_dict = this_dict[["Configuration", "Lower_Bound", "Upper_Bound"]]
  
  #Convert to dict of tuples
  this_dict = this_dict.set_index('Configuration', drop=True).to_dict('index')
  this_dict= {x: tuple(y.values()) for x, y in this_dict.items()}
  
  #Append to master dict
  STAGE_1_GRIDS[algos] = this_dict
  
#Thresholds for #models run in search
STAGE_1_MODELS = {}
for algos in STAGE_1_ALGOS:
  this_dict = stage_1_configs[stage_1_configs["Algorithm"]==algos]
  this_dict = this_dict[this_dict["Type"]=="Stop Criterion"]
  this_dict = this_dict[["Configuration", "Lower_Bound"]]
  
  #Convert to dict
  this_dict = this_dict.set_index('Configuration').T.to_dict('list')
  this_dict= {x: y[0] for x, y in this_dict.items()}
  
  #Append to master dict
  STAGE_1_MODELS[algos] = this_dict
  
int_vars = ['depth', 'max_depth', 'min_child_weight', 'max_leaves', 'n_estimators', 'depth', 'num_leaves']
STAGE_1_GRIDS = convert_dict_tuple_to_int(STAGE_1_GRIDS ,int_vars)

int_vars = ['boost_rounds', 'early_stop_rounds', 'init_points', 'n_iter']
STAGE_1_MODELS = convert_dict_vals_to_int(STAGE_1_MODELS ,int_vars)

STAGE_1_FUNC_LOOKUP = {'xgboost_model' : bayesian_opt_xgb_params,
                       'rforest_model' : bayesian_opt_rforest_params,
                       'lightGBM_model': bayesian_opt_lgbm_params,
                       'catboost_model': bayesian_opt_catboost_params,
                       'gbm_quantile_model': bayesian_opt_quantile_params                       
                      }

STAGE_1_FUNCS = {algo : STAGE_1_FUNC_LOOKUP.get(algo) for algo in STAGE_1_ALGOS}

# COMMAND ----------

## Stage 2 Configurations ##
#Algos
stage_2_configs = hyper_configs[hyper_configs["Stage"]=="Stage 2"]
stage_2_configs = stage_2_configs[stage_2_configs["Run_Algo"]=="Yes"]
STAGE_2_ALGOS = stage_2_configs["Algorithm"].unique().tolist()

#Thresholds for grid search
STAGE_2_GRIDS = {}
for algos in STAGE_2_ALGOS:
  this_dict = stage_2_configs[stage_2_configs["Algorithm"]==algos]
  this_dict = this_dict[this_dict["Type"]=="Grid"]
  this_dict = this_dict[["Configuration","Lower_Bound","Upper_Bound"]]
  
  #Convert to dict of tuples
  this_dict = this_dict.set_index('Configuration', drop=True).to_dict('index')
  this_dict= {x: tuple(y.values()) for x, y in this_dict.items()}
  
  #Append to master dict
  STAGE_2_GRIDS[algos] = this_dict

int_vars = ['max_depth','depth']
STAGE_2_GRIDS = convert_dict_tuple_to_int(STAGE_2_GRIDS ,int_vars)

# COMMAND ----------

# DBTITLE 1,Modeling Config
# LAGGED_PERIODS_1 was set as feature engineering config
if TIME_VAR == "Month_Of_Year":
  DYNAMIC_LAG_PERIODS = LAGGED_PERIODS_1 

elif TIME_VAR == "Week_Of_Year":
  DYNAMIC_LAG_PERIODS = LAGGED_PERIODS_1

else: print('User Error - please check the TIME_VAR being used for modeling!!')

# COMMAND ----------

# These flags define what will be be run in this pipeline
## Train/Test split vs. Rolling Review vs. Future Prediction 
RUN_ROLLING_XVAL = False
RUN_FUTURE_PREDICTION = True

## User to dictate whether models are re-trained following Train/Test split
## Keep as TRUE for now - issue with feature consistency when not retraining 
RETRAIN_FUTURE_MODELS = True

## User to dictate if we roll-up different lagged models to a single value
## Experiments indicated this led to a benefit in accuracy when tested OOS
AGGREGATE_LAGS_TO_PREDICT = True

## Defines the length of our OOS holdout for the Train-Test split part of this notebook
## Default should be to define this as 8-12 weeks from latest date
HOLDOUT_PERIOD_LEN = 8

## Alpha levels (ie, loss function inputs) to be potentially used for GBM Quantile models
## Including as new global features to avoid hard-coding values we might want to change
LOWER_ALPHA_LEVEL = 0.3
UPPER_ALPHA_LEVEL = 0.8
MID_ALPHA_LEVEL = 0.5
QUANTILE_ALPHA_2 = 0.85
QUANTILE_ALPHA_3 = 0.95

## Allows user to control 'volume' of retained lag values - cut down width and noise of FEATURES that are lagged
## Eg, if this value = 4 then Lag6 model will retain features lagged 7, 8, 9, 10 (ie, the 4 closest periods we can use)
LAGS_TO_KEEP = 4

## Level in the hierarchy/hierarchies at which we want to set model selection
## A 'best' model is then selected using OOS accuracies
BEST_MODEL_SELECTION_level1 = ['HRCHY_LVL_3_NM', 'SUBBRND_SHRT_NM']
BEST_MODEL_SELECTION_level2 = ['HRCHY_LVL_3_NM', 'BRND_NM']
BEST_MODEL_SELECTION_level3 = ['HRCHY_LVL_3_NM', 'SRC_CTGY_1_NM']
BEST_MODEL_SELECTION_level4 = ['HRCHY_LVL_3_NM']
BEST_MODEL_SELECTION = list(set(BEST_MODEL_SELECTION_level1 + BEST_MODEL_SELECTION_level2 
                                + BEST_MODEL_SELECTION_level3 + BEST_MODEL_SELECTION_level4))

## Dictates which tree-based model should be used for feature importance review following Train/Test split
FEAT_IMP_MODEL = 'lightGBM_model'

if TIME_VAR == 'Month_Of_Year':
  # defines which partition to use for backtesting/forcasting
  # adding this since our dataset has two partition in the mrd
  BACKTEST_PARTITION = FORECAST_START_DATE  #202009
  FORECAST_PARTITION = 202105 
  
  ## Note - these will only be used if RUN_ROLLING_XVAL = True 
  ROLLING_START_DATE = 202009 
  ROLLING_PERIODS = 2
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 18  
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [1,2, 4, 8, 12, 16, 18]
else:
  # defines which partition to use for backtesting/forcasting
  # adding this since our dataset has two partition in the mrd
  BACKTEST_PARTITION = FORECAST_START_DATE  #202114
  FORECAST_PARTITION = 202122
  
  ## Note - these will only be used if RUN_ROLLING_XVAL = True 
  ROLLING_START_DATE = 202114 
  ROLLING_PERIODS = 2
  
  ## Defines length of the time period forward 
  TIME_PERIODS_FORWARD = 16  
  
  ## This will dictate what models are actually run based on time periods forward. Allows user to control number of (and which) lag models to use
  DYNAMIC_LAG_MODELS = [1, 2, 4, 8, 12, 16]

# COMMAND ----------

# DBTITLE 1,Confidence Interval Config
# globals
CONFIDENCE_LEVEL = ['SRC_CTGY_1_NM'] #confidence intervals will be computed at this level
PREDICTION_COLUMN = 'final_prediction_value'

# please specify the percentile value for upper bound and lower bound
# make sure the limits are a multiple of 5
LOWER_LIMIT = 50
UPPER_LIMIT = 50

# COMMAND ----------

# DBTITLE 1,Default Hyperparameters
RUN_CATEGORIES = [4] # Faster run time

## Setting standard parameters for different models used
quantile_params = {
    'boosting_type': 'gbdt',     'objective': 'quantile',     'metric': {'quantile'},\
    'alpha': 0.50,     'num_leaves': 200,     'learning_rate': 0.10,\
    'feature_fraction': 0.65,     'bagging_fraction': 0.85,     'bagging_freq': 5,     'verbose': -1
}

catboost_params = {
   'depth': 8,    'learning_rate': 0.10,    'iterations': 200,\
   'subsample': 0.80,    'grow_policy': 'Depthwise',    'l2_leaf_reg': 5.0,\
}

rf_params = {
   'criterion': 'mse',    'n_estimators': 100,    'max_depth': 8,\
   'min_samples_split': 200,    'max_features': 0.65,    'max_samples': 0.85,\
   'n_jobs': -1
}

xgb_params = {
    'eta': 0.05,    'max_depth': 8,     'min_child_weight': 10,\
    'subsample': 0.80,     'gamma': 0.05,\
}

lgbm_params = {
    'boosting_type': 'gbdt',     'objective': 'regression',     'metric': {'l2', 'l1'},\
    'max_depth': 6,     'num_leaves': 100,     'learning_rate': 0.25,\
    'min_gain_to_split': 0.02,     'feature_fraction': 0.65,     'bagging_fraction': 0.85,\
    'bagging_freq': 5,     'verbose': -1
}

# COMMAND ----------

# ## Setting standard parameters for different models used
# ##Filtering for numerical parameters (stage1 parameters)
# stage_1_def_configs = hyper_def_configs[hyper_def_configs["Stage"]=="Stage 1"]
# stage_1_def_configs_num= stage_1_def_configs[hyper_def_configs["Type"]!='string']
# stage_1_def_configs_num = stage_1_def_configs_num[stage_1_def_configs_num["Run_Algo"]=="Yes"]
# stage_1_def_configs_num['Value'] = stage_1_def_configs_num['Value'].astype(float)

# ##Filtering for string parameters(stage1 parameters)
# stage_1_def_configs_str= stage_1_def_configs[hyper_def_configs["Type"]=='string']
# stage_1_def_configs_str = stage_1_def_configs_str[stage_1_def_configs_str["Run_Algo"]=="Yes"]

# ##Creating dictionary for numerical parameters
# num_dict_stage_1 = {k: f.groupby('Configuration')['Value'].apply(np.sum).to_dict()
#      for k, f in stage_1_def_configs_num.groupby('Algorithm')}

# ##Creating dictionary for string parameters
# str_dict_stage_1 = {k: f.groupby('Configuration')['Value'].apply(np.sum).to_dict()
#      for k, f in stage_1_def_configs_str.groupby('Algorithm')}

# ##Combining numerical and string dictionaries(stage1 parameters)
# for keys in num_dict_stage_1.keys():
#    if keys in str_dict_stage_1.keys():
#     num_dict_stage_1[keys].update(str_dict_stage_1[keys])
    
# ## Assigning modeling parameters to the different algorithms     
# xgb_params = num_dict_stage_1['xgboost']
# lgbm_params = num_dict_stage_1['lgbm']
# quantile_params =  num_dict_stage_1['quantile']
# rf_params = num_dict_stage_1['rf']
# catboost_params = num_dict_stage_1['catboost']
# knn_params = num_dict_stage_1['knn']
# svm_params = num_dict_stage_1['svm']
# lstm_params = num_dict_stage_1['lstm']

# ## Setting standard parameters for different models used
# ##Filtering for numerical parameters (stage2 parameters)
# stage_2_def_configs = hyper_def_configs[hyper_def_configs["Stage"]=="Stage 2"]
# stage_2_def_configs_num= stage_2_def_configs[stage_2_def_configs["Type"]!='string']
# stage_2_def_configs_num = stage_2_def_configs_num[stage_2_def_configs_num["Run_Algo"]=="Yes"]
# stage_2_def_configs_num['Value'] = stage_2_def_configs_num['Value'].astype(float)

# ##Filtering for string parameters(stage2 parameters)
# stage_2_def_configs_str= stage_2_def_configs[hyper_def_configs["Type"]=='string']
# stage_2_def_configs_str = stage_2_def_configs_str[stage_2_def_configs_str["Run_Algo"]=="Yes"]

# num_dict_stage_2 = {k: f.groupby('Configuration')['Value'].apply(np.sum).to_dict()
#      for k, f in stage_2_def_configs_num.groupby('Algorithm')}

# ##Combining numerical and string parameters(stage2 parameters)
# str_dict_stage_2 = {k: f.groupby('Configuration')['Value'].apply(np.sum).to_dict()
#      for k, f in stage_2_def_configs_str.groupby('Algorithm')}
# for keys in num_dict_stage_2.keys():
#    if keys in str_dict_stage_2.keys():
#     num_dict_stage_2[keys].update(str_dict_stage_2[keys])
    
# ## Assigning modeling parameters to the different algorithms     
# lgbm_params_stage2 =num_dict_stage_2['lgbm']
# catboost_params_stage2 = num_dict_stage_2['catboost']
# quantile_params_stage2 = num_dict_stage_2['quantile']

# COMMAND ----------

# ## Setting stage 1 models 
# stage1_models = {  
#     'gbm_quantile_model' : ((train_lightGBM, {key:(value if key != 'alpha' else 0.50) for key,
#                                      value in quantile_params.items()}, 500), predict_lightGBM),
#     'catboost_model' : ((train_catboost, catboost_params), predict_catboost),
#     'rforest_model' : ((train_random_forest, rf_params), predict_random_forest),
#     'lightGBM_model' : ((train_lightGBM, lgbm_params, 500), predict_lightGBM),
# }

# #     'naive_model_52weeks' : ((train_naive, '52'), predict_naive),
# #     'naive_model_104weeks' : ((train_naive, '104'), predict_naive), 
# #     'ENCV_model' : ((train_elastic_net_cv, 5, [0.1, 0.5, 0.7, 0.8, 0.9, 0.95]), predict_sklearn_model),
# #     'knn_model' : ((train_knn_regressor, knn_params), predict_knn_regressor),
# #     'ridge_model' : ((train_elastic_net_cv, 5, [0.01, 0.05, 0.10, 0.15]), predict_sklearn_model),
# #     'lasso_model' : ((train_elastic_net_cv, 5, [0.90, 0.95, 0.99, 1]), predict_sklearn_model),
# #     'xgb_model' : ((train_xgboost, xgb_params), predict_xgboost),


# ## Setting stage 2 models 
# stage2_models = {      
#     'lightGBM_model' : ((train_lightGBM, lgbm_params_stage2, 500), predict_lightGBM),
#     'gbm_quantile_model' : ((train_lightGBM, {key:(value if key != 'alpha' else 0.75) for key,
#                             value in quantile_params_stage2.items()}, 500), predict_lightGBM),
# }

# #     'lower_bound_model' : ((train_lightGBM, {key:(value if key != 'alpha' else 0.05) for key,
# #                             value in quantile_params_stage2.items()}, 500), predict_lightGBM),
    
# #     'upper_bound_model' : ((train_lightGBM, {key:(value if key != 'alpha' else 0.95) for key,
# #                             value in quantile_params_stage2.items()}, 500), predict_lightGBM),
# ##    'xgb_model' : ((train_xgboost, xgb_params), predict_xgboost),

# COMMAND ----------

# DBTITLE 1,Accuracy Report Configurations
ACCURACY_REPORT_TIME_LEVEL = accuracy_configs.get('ACCURACY_REPORT_TIME_LEVEL') #Time aggregation for accuracy reporting (e.g., week/month)
ACCURACY_REPORT_ITEM_LEVEL = accuracy_configs.get('ACCURACY_REPORT_ITEM_LEVEL') #Item aggregation for accuracy reporting (e.g., case count)
ACCURACY_REPORT_CLIENT_LEVEL = accuracy_configs.get('ACCURACY_REPORT_CLIENT_LEVEL') #Placeholder for if client reporting is required
ACCURACY_REPORT_DC_LEVEL = accuracy_configs.get('ACCURACY_REPORT_DC_LEVEL') #DC aggregation for accuracy reporting (e.g., LOC)
PEP_COMPARISON_PRED = accuracy_configs.get('PEP_COMPARISON_PRED') #Forecast prediction to use as a comparison to DemandBrain
FCST_LAG_VAR = accuracy_configs.get('FCST_LAG_VAR') #Variable name that indicates what lag a forecast was generated at
ERROR_METRICS_TO_REPORT = accuracy_configs.get('ERROR_METRICS_TO_REPORT') #Error metrics to output (e.g., APA, Bias)
VIZ_REPORT_LEVEL = accuracy_configs.get('VIZ_REPORT_LEVEL') #Level of aggregation of reporting to save visualizations/metrics to mlflow
# SAMPLING_VARS = ['oos_periods','sample'] #Automatically created DemandBrain variables in output forecast
SAMPLING_VARS = ['FCST_START_DATE']

ACCURACY_REPORT_TIME_LEVEL = ACCURACY_REPORT_TIME_LEVEL.split('~')
ACCURACY_REPORT_ITEM_LEVEL = ACCURACY_REPORT_ITEM_LEVEL.split('~')
ACCURACY_REPORT_DC_LEVEL = ACCURACY_REPORT_DC_LEVEL.split('~')
ACCURACY_REPORT_LEVEL = ACCURACY_REPORT_TIME_LEVEL + ACCURACY_REPORT_ITEM_LEVEL + ACCURACY_REPORT_DC_LEVEL + ['HRCHY_LVL_3_NM']
#TO-DO: Once we fully move to application layer on PEP Azure, report level will change
# ACCURACY_REPORT_LEVEL = ['Week_Of_Year','oos_periods','sample',
#                          "MU","SKULOC_CDV","BRND","SUBRND_CDV",
#                          "PCK_CNTNR_SHRT_NM","PLANG_PROD_KG_QTY","FLVR_NM","PROD_EA_PER_CASE_CNY"]


ERROR_METRICS_TO_REPORT = ERROR_METRICS_TO_REPORT.split('~')
VIZ_REPORT_LEVEL = VIZ_REPORT_LEVEL.split('~')
FCST_LAG_VAR = [FCST_LAG_VAR]
PEP_COMPARISON_PRED = ["DMNDFCST_QTY"]

# Indicate whether we use a previous merged df or if not and we're doing the merges between pep's and db's if we are storing the generated table
try:
  load_prev_merge = dbutils.widgets.get("load_prev_merge")
  if (load_prev_merge == "False"):
    load_prev_merge = False
  else:
    load_prev_merge = True
except:
  load_prev_merge = False

try:
  # 31 for weekly and 29 for monthly can be used as reference
  prev_merge_version = dbutils.widgets.get("prev_merge_version")
  if (prev_merge_version == "None"):
    prev_merge_version = None
  else:
    prev_merge_version = int(prev_merge_version)
except:
  prev_merge_version = None

try:
  save_curr_merge = dbutils.widgets.get("save_curr_merge")
  if (save_curr_merge == "False"):
    save_curr_merge = False
  else:
    save_curr_merge = True
except:
  save_curr_merge = True

try:
  backtest_fcst_version = dbutils.widgets.get("backtest_fcst_version")
  if (backtest_fcst_version == "None"):
    print("Switching to default backtest versions")
    if (TIME_VAR == "Week_Of_Year"):
      backtest_fcst_version = 50
    else:
      backtest_fcst_version = 34
  else:
    backtest_fcst_version = int(backtest_fcst_version)
except:
  backtest_fcst_version = None
  
# load_prev_merge = False
# prev_merge_version = None
# save_curr_merge = True

# If we are to choose the closest lag to the max forecast period (or if gaps in the lag models, select the latest periods such that a continuous time series is built)
only_closest_lag = True

# COMMAND ----------

# DBTITLE 1,Final forecast aggregation logics
# Identify categories that go with max of models approach (if empty default diagonal will be used)

fcst_categ_max ={"Week_Of_Year": {}, "Month_Of_Year": {}}

# COMMAND ----------

# DBTITLE 1,Neilson config
# Config for Neilsen mapping. 
NIELSON_MERGE_FIELD = [TIME_VAR, "DMDUNIT", "HRCHY_LVL_3_NM"]
level_at_which_to_impute = ["SRC_CTGY_1_NM", "HRCHY_LVL_3_NM", TIME_VAR] 
neilsen_cols_to_frcst = ['neil_Numeric_Distribution','neil_Wtd_Distribution', 'neil_Wtd_Distribution_Promo', 'neil_Wtd_Distribution_SEL', 'neil_Wtd_Distribution_SE', 'neil_Wtd_Distribution_L', 'neil_Wtd_Distribution_TPR', 'neil_Price_Per_Qty', 'neil_Promo_Price_Per_Volume', 'neil_percent_baseline_unit', 'neil_percent_baseline_volume', 'neil_percent_baseline_value']