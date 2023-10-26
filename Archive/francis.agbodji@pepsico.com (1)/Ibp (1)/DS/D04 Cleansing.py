# Databricks notebook source
# MAGIC %md 
# MAGIC ##04 - Data Cleansing
# MAGIC 
# MAGIC This script cleanses the data prior to automated feature engineering.  Cleansing includes:
# MAGIC * Filtering
# MAGIC * Imputations
# MAGIC * Outlier handling
# MAGIC * Transformations
# MAGIC * Indexing
# MAGIC 
# MAGIC Key source code (src) libraries to reference: feature_engineering, utilities

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
required_configs = [DBA_MRD_EXPLORATORY, DBO_SEGMENTS , DBO_OUTLIERS , DBA_MRD_CLEAN , #Input / Output Tables
                    TARGET_VAR , TIME_VAR ,  MODEL_ID_HIER ,
                    OUTLIER_HANDLING_STRATEGY , CAPPING_THRESHOLD , INTERPOLATE_METHOD ,
                    COLS_TO_LOG , COLS_TO_INDEX]

print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

# DBTITLE 1,Load Data
## Exploratory MRD
try:
##dev runs
#   mrd_df = load_delta(DBA_MRD_EXPLORATORY, 39) # Monthly
#   mrd_df = load_delta(DBA_MRD_EXPLORATORY, 38)   # Weekly

##Production run
  mrd_df = load_delta(DBA_MRD_EXPLORATORY)

except:
  dbutils.notebook.exit("Exploratory mrd load failed")
  
## Segments
try:
##dev runs
#   segments_df = load_delta(DBO_SEGMENTS, 30).distinct() # Monthly
#   segments_df = load_delta(DBO_SEGMENTS, 29).distinct()     # Weekly

##Production run
  segments_df = load_delta(DBO_SEGMENTS)
except:
  print("DBI_SEGMENTATION failed, models wil be run for all levels at once and segmentation will not be included as features")
  
## Outliers
try:
##dev runs
#   outliers_df = load_delta(DBO_OUTLIERS, 19) # weekly
#   outliers_df = load_delta(DBO_OUTLIERS, 20) # Monthly

##Production run
  outliers_df = load_delta(DBO_OUTLIERS)
except:
  print("DBO_OUTLIERS failed, models wil not cleanse outliers")
  
## Competitor / Cannibalization Features
try:
##dev runs
#   competitor_vars_df = load_delta(DBO_COMPETITOR_VARS, 13) #Monthly
#   competitor_vars_df = load_delta(DBO_COMPETITOR_VARS, 12) #Weekly

##Production run
  competitor_vars_df = load_delta(DBO_COMPETITOR_VARS)
except:
  print("DBO_COMPETITOR_VARS failed")
# Calendar dataframe
try:
  calendar = load_delta(DBI_CALENDAR)
except:
  print("DBO_COMPETITOR_VARS failed")

# COMMAND ----------

## Store unaltered target variable
total_sales_qc = int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0])
mrd_df = mrd_df.withColumn(TARGET_VAR + "_orig", col(TARGET_VAR))
print(total_sales_qc)
print(mrd_df.count())

# COMMAND ----------

# Making HSTRY_TMFRM_STRT_DT consistent to have no nulls 
calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct()
mrd_df = mrd_df.drop("HSTRY_TMFRM_STRT_DT")
mrd_df = mrd_df.join(calendar_df, on=TIME_VAR, how="left").withColumnRenamed(CALENDAR_DATEVAR, "HSTRY_TMFRM_STRT_DT")
display(mrd_df)

# COMMAND ----------

# DBTITLE 1,Merge Segments / Outliers / Features
## Create MODEL_ID to merge to segments / outliers
mrd_df = get_model_id(mrd_df, "MODEL_ID", MODEL_ID_HIER)
mrd_df.cache()

## Merge outliers dataframe 
if TIME_VAR == "Week_Of_Year":
  print(mrd_df.count())
  mrd_df = mrd_df.join(outliers_df, on=["MODEL_ID"] + [TIME_VAR], how="left")
  print(mrd_df.count())

#Merge prepped features
print(mrd_df.count())
mrd_df = mrd_df.join(competitor_vars_df, on=["MODEL_ID", TIME_VAR], how="left")
print(mrd_df.count())

#QC check
print(int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0]) == total_sales_qc)
print(int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0]))

# COMMAND ----------

## Merge segments (these are developed at a MODEL_ID level)
print(mrd_df.count())
mrd_df = mrd_df.join(segments_df.select(["MODEL_ID",
                                         "CV","ABC","XYZ","STAT_CLUSTER"]), on=["MODEL_ID"], how="left")

print(mrd_df.filter(col("ABC").isNull()).count())
mrd_df = impute_cols_ts(mrd_df, ["STAT_CLUSTER","ABC","XYZ","CV"], TIME_VAR, "ffill")
print(mrd_df.filter(col("ABC").isNull()).count())

mrd_df = mrd_df.filter(col("XYZ")!="Obsolete")
total_sales_qc = int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0])
## QC check
print(mrd_df.count())
print(int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0]) == total_sales_qc)
print(int(mrd_df.agg(F.sum(TARGET_VAR)).collect()[0][0]))

# COMMAND ----------

# DBTITLE 1,Filtering
## Subsetting by time variable based on globally-defined START/END
try:
  filtered_df = mrd_df.filter(col(TIME_VAR) >= START_PERIOD)
  filtered_df = filtered_df.filter(col(TIME_VAR) <= END_PERIOD)
  filtered_df.cache()
  print("Rows after filtering: " , filtered_df.count())
  filtered_sales_qc = int(filtered_df.agg(F.sum(TARGET_VAR)).collect()[0][0])

except:
  print('No filtering done')
  
## QC check
print(total_sales_qc - filtered_sales_qc)
print(filtered_sales_qc)

# COMMAND ----------

# DBTITLE 1,Imputations
treated_df = filtered_df
#If user specified zeros need to be imputed, set zero to Null so it can enter next block
try:
  for i in impute_zeros_list:
    treated_df = treated_df.withColumn(i,when(col(i)<=0,None).otherwise(col(i)))
except:
  print("Imputing zeros not required")

#Forward fill required null variables
try:
  treated_df = impute_cols_ts(treated_df, impute_ffill_list, [TIME_VAR], "ffill",'MODEL_ID')
except:
  print("Forward fill imputation not required")
  
#Backward fill required null variables
try:
  treated_df = impute_cols_ts(treated_df, impute_bfill_list, [TIME_VAR], "bfill",'MODEL_ID')
except:
  print("Backward fill imputation not required") 
  
# Fill null values with statistical value by group (e.g., avg by brand)
try:
  for i in imputation_method_dict:
    level_at_which_to_impute = groupcols_L1  
    treated_filter_df = treated_df.filter(col(i).isNotNull()).select(groupcols_L1 + [i])
    agg_df = treated_filter_df.groupBy(level_at_which_to_impute).agg(avg(i).alias("avg_"+i))
    treated_df = treated_df.join(agg_df, on=groupcols_L1, how="left")
    treated_df = treated_df.withColumn(i, when(col(i).isNull(), col("avg_" + i)).otherwise(col(i)).cast(DoubleType()))
    treated_df = treated_df.drop("avg_" + i) 

    treated_df = impute_cols_ts(treated_df, impute_ffill_list, [TIME_VAR], "ffill",'MODEL_ID')
    treated_df = impute_cols_ts(treated_df, impute_bfill_list, [TIME_VAR], "bfill",'MODEL_ID')

    level_at_which_to_impute = groupcols_L2
    #Impute at L2
    treated_filter_df = treated_df.filter(col(i).isNotNull()).select(groupcols_L2 + [i]+[TARGET_VAR])
    treated_filter_df=treated_filter_df.withColumn("SumProd",col(i)*col(TARGET_VAR))
    agg_df = treated_filter_df.groupBy(level_at_which_to_impute).agg(sum(col("SumProd")).alias("SumProd"), sum(col(TARGET_VAR)).alias(TARGET_VAR))
    agg_df= agg_df.withColumn("avg_"+i,col("SumProd")/col(TARGET_VAR))
    agg_df=agg_df.select(level_at_which_to_impute+["avg_"+i])
    treated_df = treated_df.join(agg_df, on=groupcols_L2, how="left")
    treated_df = treated_df.withColumn(i, when(col(i).isNull(), col("avg_" + i)).otherwise(col(i)).cast(DoubleType()))
    treated_df = treated_df.drop("avg_" + i) 

    treated_df = impute_cols_ts(treated_df, impute_ffill_list, [TIME_VAR], "ffill",'MODEL_ID')
    treated_df = impute_cols_ts(treated_df, impute_bfill_list, [TIME_VAR], "bfill",'MODEL_ID')
except:
  print("Forward fill imputation not required")
  
# Backward fill required null variables
# Fill null values with statistical value by group (e.g., avg by brand)
try:
  for i in imputation_method_dict:
      if len(intersect_two_lists([i], treated_df.columns)) > 0:
        #Get imputation strategy for this variable
        if imputation_method_dict.get(i) == "Mean":
          stat = [avg]
        elif imputation_method_dict.get(i) == "Min":
          stat = [min]
        elif imputation_method_dict.get(i) == "Max":
          stat = [max]
        elif imputation_method_dict.get(i) == "Median":
          #TO-DO: Need to call get_percentile function for median
          stat = [avg]
except:
  print("Statistical group imputation not required")

  
try:
  if imputation_method_dict != None:
    auto_default = impute_ffill_list + impute_bfill_list + list(imputation_method_dict.keys())
  else:
    auto_default = impute_ffill_list + impute_bfill_list
  auto_default = intersect_two_lists(auto_default, treated_df.columns)
  treated_df = treated_df.na.fill(value=0, subset=auto_default)
except:
  print('Automatic impute to zero not performed')
  
# QC check
print(treated_df.count())
print(int(treated_df.agg(F.sum(TARGET_VAR)).collect()[0][0]) == filtered_sales_qc)
print(int(treated_df.agg(F.sum(TARGET_VAR)).collect()[0][0]))

# COMMAND ----------

# DBTITLE 1,Outlier Handling
#OUTLIER_HANDLING_STRATEGY
#0: Capping
#1: Interpolation

if OUTLIER_HANDLING_STRATEGY == 0:
  print('Capping outliers')
  treated_df = calculate_percentile(treated_df, ["MODEL_ID"], TARGET_VAR, [CAPPING_THRESHOLD]) 
  treated_df = treated_df.withColumnRenamed(TARGET_VAR + str(CAPPING_THRESHOLD),"MaxSales")
  treated_df = treated_df.withColumn(TARGET_VAR,
                         when(col("OUTLIER_IND")==1,col("MaxSales")).otherwise(col(TARGET_VAR)))
  treated_df = treated_df.drop("MaxSales")

if OUTLIER_HANDLING_STRATEGY == 1:
  print('Interpolating outliers')
  ## Set outliers to NA so we can interpolate
  treated_df = treated_df.withColumn(TARGET_VAR,
                         when(col("OUTLIER_IND")==1,None).otherwise(col(TARGET_VAR)))
  
  auto_schema = treated_df.schema
  @pandas_udf(auto_schema, PandasUDFType.GROUPED_MAP)
  def interpolate_udf(data):
    return interpolate_col(data, TARGET_VAR, INTERPOLATE_METHOD)

  ## Interpolate target variable
  treated_df = treated_df.groupBy("MODEL_ID").apply(interpolate_udf)
  
treated_df.cache()
treated_df.count()

#QC check
print(treated_df.count())
print(int(treated_df.agg(F.sum(TARGET_VAR)).collect()[0][0]) == filtered_sales_qc)
print(int(treated_df.agg(F.sum(TARGET_VAR)).collect()[0][0]))

# COMMAND ----------

# Null Imputation
COLS_TO_IMPUTE_NULL = [f.name for f in treated_df.schema.fields if isinstance(f.dataType, DoubleType) | isinstance(f.dataType, IntegerType) | isinstance(f.dataType, LongType) & ~isinstance(f.dataType, StringType)]
COLS_TO_IMPUTE_NULL = subtract_two_lists(COLS_TO_IMPUTE_NULL, [x for x in treated_df.columns if '_index' in x])
treated_df = treated_df.na.fill(value=0, subset=COLS_TO_IMPUTE_NULL)

# COMMAND ----------

display(treated_df)

# COMMAND ----------

# DBTITLE 1,Downcasting
cols_to_downcast_index = [d[0] for d in treated_df.dtypes if d[0].lower().endswith('_index') and d[1] != 'int']
cols_to_downcast_flag = [d[0] for d in treated_df.dtypes if d[0].lower().endswith('_flag') and d[1] != 'int']
cols_to_downcast_bigint = [d[0] for d in treated_df.dtypes if d[1] == 'bigint']
col_stat_cluster = 'STAT_CLUSTER'
col_to_downcast_stat_cluster = [d[0] for d in treated_df.dtypes if d[0].lower() == col_stat_cluster.lower() and d[1] != 'int']
cols_to_downcast = col_to_downcast_stat_cluster + cols_to_downcast_index + cols_to_downcast_flag + cols_to_downcast_bigint


print(f'Cols to downcast: {len(cols_to_downcast)} ({np.round(len(cols_to_downcast) / len(treated_df.columns), 3)}%)')


for c in cols_to_downcast:
  treated_df = treated_df.withColumn(c, treated_df[c].cast(IntegerType()))

# COMMAND ----------

# DBTITLE 1,Saving Output
## Save to DBA_MRD_CLEAN
## Write as delta table to dbfs
treated_df = treated_df.drop("OUTLIER_IND", "OUTLIER_TYPE")
print(treated_df.count())

save_df_as_delta(treated_df, DBA_MRD_CLEAN, enforce_schema=False)
delta_info = load_delta_info(DBA_MRD_CLEAN)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Save MODEL_ID Details
#Save hierarchy information to DBA_MODELIDS

#Retain these variables in DBA_MODELIDS
id_vars=['DMDUNIT','PLANG_CUST_GRP_VAL', 'LOC',TIME_VAR, 'SRC_CTGY_1_NM','BRND_NM','SUBBRND_SHRT_NM','PCK_CNTNR_SHRT_NM',
           'PLANG_PROD_KG_QTY',
           'FLVR_NM','PLANG_MTRL_EA_PER_CASE_CNT','HRCHY_LVL_3_NM','STAT_CLUSTER','ABC','XYZ']
id_vars = list(set(id_vars))
id_vars = subtract_two_lists(id_vars, [TIME_VAR])
model_info = mrd_df.select(["MODEL_ID"]  +  id_vars).dropDuplicates()

#Keep latest snapshot of hierarchy variables
window = Window.partitionBy("MODEL_ID").orderBy(desc(TIME_VAR)) 
model_info = model_info.withColumn(TIME_VAR, monotonically_increasing_id())\
.withColumn('rank', rank().over(window))\
.filter(col('rank') == 1).drop('rank',TIME_VAR)

print(model_info.count())

#Save to dbfs
save_df_as_delta(model_info, DBA_MODELIDS, enforce_schema=False)
delta_info = load_delta_info(DBA_MODELIDS)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())
check_df=treated_df.filter(col('LIST_PRICE_BAG')<=0)
print('List Pricing is null for ', 100*(check_df.count()/treated_df.count()))

check_df=treated_df.filter(col('NET_PRICE_BAG')<=0)
print('NET Pricing is null for ', 100*(check_df.count()/treated_df.count()))

# COMMAND ----------

