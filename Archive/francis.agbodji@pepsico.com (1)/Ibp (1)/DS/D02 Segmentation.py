# Databricks notebook source
# MAGIC %md 
# MAGIC ##02 - Segmentation
# MAGIC 
# MAGIC This script creates business and statistical segmentation for reporting and features into the model
# MAGIC 
# MAGIC Key source code (src) libraries to reference: segmentation, data_cleansing, utilities

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

## Check that configurations exist for this script
required_configs = [
  DBA_MRD_EXPLORATORY, DBO_SEGMENTS, #Input/Output tables
  STAT_SEGMENT_GROUP, STAT_SEGMENT_LEVEL, SEGMENTATION_TIME_CUTOFF,
  TARGET_VAR, TIME_VAR,
  MARKET_FIELD, PROD_MERGE_FIELD, CUST_MERGE_FIELD, LOC_MERGE_FIELD,  PRODUCT_HIER, MODEL_ID_HIER
]

print(json.dumps(required_configs, indent=4))
if required_configs.count(None) > 0 :
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

## To uncomment below line in case of Monthly run 
# TIME_VAR = 'Month_Of_Year' 

# # pomil
# if TIME_VAR =='Week_Of_Year':
#   CALENDAR_DATEVAR = "Week_start_date"
#   SEGMENTATION_TIME_CUTOFF = 202121   ## Corey changed - to avoid leakage
# elif TIME_VAR =='Month_Of_Year':
#   CALENDAR_DATEVAR = "Month_start_date"
#   SEGMENTATION_TIME_CUTOFF = 202104

print('Target Var: {} \n Time Var: {} \n Model ID Hierarchy: {} \n SEGMENTATION_TIME_CUTOFF: {}'.format(TARGET_VAR, 
                                                                                                        TIME_VAR, 
                                                                                                        MODEL_ID_HIER,
                                                                                                        SEGMENTATION_TIME_CUTOFF)) 

# COMMAND ----------

# DBTITLE 1,Load Data
# mrd_orig = load_delta(DBA_MRD_EXPLORATORY, 32)  ## v32 has "Week" as TIME_VAR 
# if TIME_VAR == 'Month_Of_Year':
#   mrd_orig = load_delta(DBA_MRD_EXPLORATORY, 34)  ## v34 has "Month" as TIME_VAR

mrd_orig = load_delta(DBA_MRD_EXPLORATORY)

calendar = load_delta(DBI_CALENDAR)
mrd_orig = get_model_id(mrd_orig, STAT_SEG_ID, STAT_SEGMENT_LEVEL)

# COMMAND ----------

mrd = mrd_orig.filter(col(TIME_VAR) <= SEGMENTATION_TIME_CUTOFF) 

## Aggregate data for segmentation 
mrd = aggregate_data(mrd, [STAT_SEG_ID] + STAT_SEGMENT_LEVEL + [TIME_VAR], [TARGET_VAR], [SEG_AGG_METHOD]) 
mrd = mrd.withColumnRenamed(SEG_AGG_METHOD.__name__ + '_' + TARGET_VAR, TARGET_VAR) 
mrd.select(STAT_SEGMENT_LEVEL).dropDuplicates().count() 

# COMMAND ----------

# DBTITLE 1,Run Statistical Segmentation
##Setup information for Dynamic Time Warping (DTW) segmentation
## DTW is a distance metric that can be used in clustering algorithms (KNN, Hierarchical, etc.)
## This metric can compare time series that are not time aligned, unlike Euclidean distances

model_info_dict = dict(
    model_id     = STAT_SEG_ID,
    time_field   = TIME_VAR,
    target_field = TARGET_VAR,
    n_clusters   = DTW_CLUSTER_NUM
)
segmentation_cls = DTWModelInfo(**model_info_dict)
print(segmentation_cls.target_field)
print(segmentation_cls.time_field)

# COMMAND ----------

## Calculate dynamic time warping segment
schema = StructType([
StructField(STAT_SEG_ID, StringType()),
StructField('STAT_CLUSTER', DoubleType())])  ## name of created field

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def predict_dtw_segment_udf(data):
    return calculate_dtw_segment(data, segmentation_cls)

cluster_preds = mrd.groupBy(STAT_SEGMENT_GROUP).apply(predict_dtw_segment_udf)
cluster_preds.cache(), cluster_preds.count()

# COMMAND ----------

## Creating dataframe copy for below calculations
mrd_df = mrd_orig
mrd_df = mrd_df.withColumn("CASES",col("QTY")/col("PLANG_MTRL_EA_PER_CASE_CNT"))
mrd_df = mrd_df.withColumn("PLANG_PROD_KG_QTY", round(col("PLANG_PROD_KG_QTY")*lit(1000)))
mrd_df = mrd_df.withColumn("PLANG_MTRL_EA_PER_CASE_CNT", round(col("PLANG_MTRL_EA_PER_CASE_CNT")))
mrd_df = mrd_df.withColumn("PLANG_PROD_KG_QTY", col("PLANG_PROD_KG_QTY").cast('int'))

# COMMAND ----------

# pomil - 20210727 - Modified code to handle 53 weeks in leap year properly 
if CALENDAR_DATEVAR not in mrd.columns:
  calendar_df = calendar.select(TIME_VAR, CALENDAR_DATEVAR).distinct() 
  mrd_dates = mrd.join(calendar_df, on=TIME_VAR, how="left").select(TIME_VAR, CALENDAR_DATEVAR).distinct()

if 'week' in TIME_VAR.lower():
  mrd_dates = mrd_dates.withColumn("year_adjusted_date", F.date_sub(mrd_dates[CALENDAR_DATEVAR], 52*7))\
                       .withColumn("years3_adjusted_date", F.date_sub(mrd_dates[CALENDAR_DATEVAR], 52*7*3))\
                        .withColumn("year_adjusted_date", to_timestamp(col("year_adjusted_date")))\
                        .withColumn("years3_adjusted_date", to_timestamp(col("years3_adjusted_date")))
elif 'month' in TIME_VAR.lower():
  mrd_dates = mrd_dates.withColumn("year_adjusted_date", F.add_months(mrd_dates[CALENDAR_DATEVAR], -12))\
                       .withColumn("years3_adjusted_date", F.add_months(mrd_dates[CALENDAR_DATEVAR], -12*3))\
                        .withColumn("year_adjusted_date", to_timestamp(col("year_adjusted_date")))\
                        .withColumn("years3_adjusted_date", to_timestamp(col("years3_adjusted_date")))
max_year_adjusted_date = mrd_dates.select(max("year_adjusted_date")).collect()[0][0] 
max_years3_adjusted_date = mrd_dates.select(max("years3_adjusted_date")).collect()[0][0] 
last_date_value_start = calendar_df.filter(col(CALENDAR_DATEVAR)==max_year_adjusted_date).select(TIME_VAR).collect()[0][0]
last_date_volatility_start = calendar_df.filter(col(CALENDAR_DATEVAR)==max_years3_adjusted_date).select(TIME_VAR).collect()[0][0]
print("last_date_value_start:", last_date_value_start)
print("last_date_volatility_start:", last_date_volatility_start)

# COMMAND ----------

##declaring the group level -- item, client , dc, week/month
groupby_agg = [TIME_VAR]+ MODEL_ID_HIER
print(mrd_df.count()) 

## Cut the dataframe for the last 3 years to include item, client , dc, week,cases
mrd_df_history = mrd_df.filter((col(TIME_VAR) > last_date_volatility_start) & (col(TIME_VAR) <= SEGMENTATION_TIME_CUTOFF))\
               .select(MODEL_ID_HIER+[TIME_VAR,TARGET_VAR])
print(mrd_df_history.count())

mrd_df_filled = mrd_df_history
print(mrd_df_filled.count())

# COMMAND ----------

#filtering the data for 1 year for value segmentation
mrd_df_value=mrd_df_filled.filter((col(TIME_VAR) > last_date_value_start))

vol_agg_level =  MODEL_ID_HIER  ## no time elements in this (week or month)
vol_agg_level = subtract_two_lists(vol_agg_level, [MARKET_FIELD])
mrd_df_abc_agg = mrd_df_value.groupBy(MODEL_ID_HIER)\
                               .agg({TARGET_VAR:'avg'})\
                               .withColumnRenamed('avg(' + TARGET_VAR + ')', TARGET_VAR)

value_measure = get_cumsum_simple(mrd_df_abc_agg, vol_agg_level, TARGET_VAR)

## Setting up cascading thresholds to flag A vs B vs C
value_measure = value_measure.withColumn('ABC', lit('B'))
value_measure = value_measure.withColumn('ABC',when(col('cum_pct') <= PEP_VALUE_THRESH_A, 'A').otherwise(col('ABC')))
value_measure = value_measure.withColumn('ABC',when(col('cum_pct') >  PEP_VALUE_THRESH_B, 'C').otherwise(col('ABC')))

value_measure = change_schema_in_pyspark(value_measure, string_cols=MODEL_ID_HIER)
value_measure = get_model_id(value_measure, 'MODEL_ID', MODEL_ID_HIER)

# COMMAND ----------

## Note - groupby_agg includes time features - must include this to get a STD
mrd_df_xyz = mrd_df_filled.filter(((col(TIME_VAR) > last_date_volatility_start)) )\
                   .groupBy([TIME_VAR]+ MODEL_ID_HIER)\
                   .agg({TARGET_VAR:'sum'})\
                   .withColumnRenamed('sum(' + TARGET_VAR + ')', TARGET_VAR)

#Discard obsoletes (no TARGET VAR in the last year at all)
mrd_df_filled_obs=get_model_id(mrd_df_filled, 'MODEL_ID', MODEL_ID_HIER)
obsolete_ids = mrd_df_filled_obs.filter(col(TIME_VAR) > last_date_value_start).groupby('MODEL_ID').agg(sum('CASES')).filter(col("sum(CASES)") == 0).select('MODEL_ID')

#drop obsoletes from mrd_df_xyz
mrd_df_xyz_id=get_model_id(mrd_df_xyz, 'MODEL_ID', MODEL_ID_HIER)                                       
mrd_df_xyz_no_obs = mrd_df_xyz_id.filter(col("MODEL_ID").isin([row[0] for row in obsolete_ids.select('MODEL_ID').collect()]) == False).drop(col("MODEL_ID"))

## Volatility Measure (XYZ) - KEEP MONTH
## X = COV less than 0.3 // Y = COV 0.3 to 0.5 // Z = COV greater than 0.5
volatility_measure = calculate_cv_segmentation(mrd_df_xyz_no_obs, TARGET_VAR, MODEL_ID_HIER)
volatility_measure = volatility_measure.withColumn('XYZ', lit('Y'))
volatility_measure = volatility_measure.withColumn('XYZ', when(col('CV') <= PEP_VOLATILITY_THRESH_A, 'X').otherwise(col('XYZ')))
volatility_measure = volatility_measure.withColumn('XYZ', when(col('CV') >  PEP_VOLATILITY_THRESH_B, 'Z').otherwise(col('XYZ')))
volatility_measure = change_schema_in_pyspark(volatility_measure, string_cols=MODEL_ID_HIER)
volatility_measure = get_model_id(volatility_measure, 'MODEL_ID', MODEL_ID_HIER)

## Review the results
volatility_measure.groupBy("XYZ").count().show(truncate=False)
volatility_measure.groupBy("XYZ").mean("CV").show(truncate=False)

# COMMAND ----------

#Fill obsolete IDs again in the whole df with proper labels for CV and XYZ
mrd_df_xyz_model=get_model_id(mrd_df_xyz, 'MODEL_ID', MODEL_ID_HIER)
volatility_measure = mrd_df_xyz_model.select(['MODEL_ID']).join(volatility_measure, on='MODEL_ID', how="left")
volatility_measure=volatility_measure.na.fill({'CV':0, 'XYZ':'Obsolete' })

ABC_df = value_measure.select('MODEL_ID', 'ABC',"cum_pct")
XYZ_df = volatility_measure.select('MODEL_ID', 'XYZ','CV').distinct()

## ANAND FLAG - does it still make sense to join like this?

## Merging the ABC and XYZ dataframes 
biz_segments = ABC_df.join(XYZ_df, on='MODEL_ID', how="left")
display(biz_segments)

# COMMAND ----------

# DBTITLE 1,Advanced Business Segmentation
## Copying dataframe and setting up agg level
mrd_df = mrd_orig.filter(col(TIME_VAR) <= SEGMENTATION_TIME_CUTOFF)
business_segment_level = STAT_SEGMENT_GROUP + MODEL_ID_HIER 
desc_level = business_segment_level + [MARKET_FIELD]
print("Advanced Segmentation Agg Level:", desc_level)

## Creating aggregation views for mean and std of cases at 'desc_level' grouping
df_agg_mean = aggregate_data(mrd_df, desc_level, TARGET_VAR, [mean]).fillna(0)
df_agg_stddev = aggregate_data(mrd_df, desc_level, TARGET_VAR, [stddev]).fillna(0)

## Getting quantile values from the above
## Must use list and set to avoid duplicate values (esp for STD bins)
quantile_Mean = sorted(list(set(df_agg_mean.approxQuantile('mean_' + TARGET_VAR, QCUT_SEGMENTATION_LEVELS, relativeError=0.1))))
quantile_STD = sorted(list(set(df_agg_stddev.approxQuantile('stddev_' + TARGET_VAR, QCUT_SEGMENTATION_LEVELS, relativeError=0.1))))

## Using bucketizer to create Q-cuts based on above quantile values
bucketizerM = Bucketizer(splits=quantile_Mean, inputCol='mean_' + TARGET_VAR, outputCol='MEAN_qcuts')
bucketizerSTD = Bucketizer(splits=quantile_STD, inputCol='stddev_' + TARGET_VAR, outputCol='STD_qcuts')

## Creating binned (classification) versions of the above
df_binned_mean = bucketizerM.transform(df_agg_mean)
df_binned_stddev = bucketizerSTD.transform(df_agg_stddev)
mrd_binned = df_binned_mean.join(df_binned_stddev, on=desc_level, how="inner")

# COMMAND ----------

## Creating a more intuitive mapping for the above binning
t = {0:"Low", 1: "Low-Med", 2:"Mid", 3: "Med-High",4: "High"}
udf_foo = udf(lambda x: t[x], StringType())

mrd_binned = mrd_binned.withColumn("mean_bucket", udf_foo("MEAN_qcuts")).drop('MEAN_qcuts')
mrd_binned = mrd_binned.withColumn("stddev_bucket", udf_foo("STD_qcuts")).drop('STD_qcuts')

## Creating 4 segments - 2x2 quadrant using the above more specific bins
high_val_list = ['Med-High', 'High']
low_val_list = ['Low', 'Low-Med']

## Dividing the above into the 2x2
mrd_binned = mrd_binned.withColumn('adv_biz_seg',
                                   when((col('mean_bucket').isin (high_val_list)) & (col('stddev_bucket').isin (low_val_list)), lit("lar_nv"))\
                                    .when((col('mean_bucket').isin (high_val_list)) & (col('stddev_bucket').isin (high_val_list)), lit("lar_v"))\
                                    .when((col('mean_bucket').isin (low_val_list)) & (col('stddev_bucket').isin (low_val_list)), lit("sml_nv"))\
                                    .when((col('mean_bucket').isin (low_val_list)) & (col('stddev_bucket').isin (high_val_list)), lit("sml_v"))\
                                    .otherwise('other'))
mrd_binned=get_model_id(mrd_binned, 'MODEL_ID', MODEL_ID_HIER)

## ANAND FLAG - not sure if we can join on this at higher level of agg?
## Joining with the PEP segmentation ('biz_segments')
mrd_binned = mrd_binned.join(biz_segments, on='MODEL_ID', how="left")

# COMMAND ----------

## Merge advanced business segmentation to stat cluster
final_segments = get_model_id(mrd_binned, STAT_SEG_ID, STAT_SEGMENT_LEVEL)
final_segments = final_segments.join(cluster_preds, on=[STAT_SEG_ID], how="left")  ## To join to statistical segments
final_segments = final_segments.drop(*[STAT_SEG_ID])
final_segments = get_model_id(final_segments, 'MODEL_ID', MODEL_ID_HIER)   ## To join to transactional

## MODEL ID is our unique join, and keeping our other generated columns
cols_to_keep = [
  "MODEL_ID", "STAT_CLUSTER",  "cum_pct", "ABC", "XYZ", "CV",\
  "mean_CASES", "stddev_CASES", "mean_bucket", "stddev_bucket", "adv_biz_seg"
]

final_segments = final_segments.select(cols_to_keep)

# COMMAND ----------

# DBTITLE 1,Output Save & Review
## Write as delta table to dbfs
save_df_as_delta(final_segments, DBO_SEGMENTS, enforce_schema=False)

delta_info = load_delta_info(DBO_SEGMENTS)
set_delta_retention(delta_info, "90 days")
display(delta_info.history())

# COMMAND ----------

