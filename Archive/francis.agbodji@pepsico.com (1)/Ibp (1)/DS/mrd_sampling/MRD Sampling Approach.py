# Databricks notebook source
# MAGIC %md 
# MAGIC This script is to enable experimentations to obtain a representative sample of the mrd in order to facilitate faster run times with scalable results.
# MAGIC Output: List of sampled MODEL_ID's as a delta table column

# COMMAND ----------

# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

# To comment when done for weekly
TIME_VAR = "Month_Of_Year"
# TIME_VAR = "Week_Of_Year"

# COMMAND ----------

## Load the datasets
if TIME_VAR == "Week_Of_Year":
  mrd = load_delta(DBA_MRD, 45)   # version 45 - latest weekly data
  model_ids = load_delta(DBA_MODELIDS, 33)
  print(mrd.count())
else:
  mrd = load_delta(DBA_MRD, 48)   # version 48 - latest monthly data
  model_ids = load_delta(DBA_MODELIDS, 34)
  print(mrd.count())
  
ffcst_start_date = mrd.select(min("FCST_START_DATE")).collect()[0][0]

# COMMAND ----------

## Merge wrt model ids to obtain category and market/country
mrd = mrd.filter(col("FCST_START_DATE") == ffcst_start_date)    # For backtest forecast modelling data
mrd = mrd.join(model_ids.select("MODEL_ID", "HRCHY_LVL_3_NM", "SRC_CTGY_1_NM"), on = "MODEL_ID", how = "left")
mrd = mrd.withColumn("strat_level", concat(col("HRCHY_LVL_3_NM"), lit("_"), col("SRC_CTGY_1_NM")))
mrd.count()               

# COMMAND ----------

## Calculate ABC value Clusters at strat level of country-category
mrd_abc_agg = mrd.groupBy("MODEL_ID", "strat_level").agg({TARGET_VAR:'avg'}).withColumnRenamed('avg(' + TARGET_VAR + ')', TARGET_VAR)

## Derive total sales for time period of dataframe for each strat_level
total_sales = mrd_abc_agg.groupBy("strat_level").agg(sum(TARGET_VAR).alias("total_sum"))#.collect()[0][0] 

## Get cumulative sum by group
agg_sales = aggregate_data(mrd_abc_agg, ["strat_level", "MODEL_ID"], TARGET_VAR, [sum])    
windowval = (Window.partitionBy("strat_level").orderBy(desc('sum_' + TARGET_VAR)).rangeBetween(Window.unboundedPreceding,0))
agg_sales = agg_sales.withColumn('cum_sum', sum('sum_' + TARGET_VAR).over(windowval))

## Total sales for cumsum
agg_sales = agg_sales.join(total_sales, on = "strat_level", how = "left")

## Get cum % and output
value_measure = agg_sales.withColumn('cum_pct', col('cum_sum') / col('total_sum'))

## Setting up cascading thresholds to flag A vs B vs C
value_measure = value_measure.withColumn('ABC_new', lit('B'))
value_measure = value_measure.withColumn('ABC_new',when(col('cum_pct') <= PEP_VALUE_THRESH_A, 'A').otherwise(col('ABC_new')))
value_measure = value_measure.withColumn('ABC_new',when(col('cum_pct') >  PEP_VALUE_THRESH_B, 'C').otherwise(col('ABC_new')))

# COMMAND ----------

## Calculate XYZ COV Clusters

## Note - groupby_agg includes time features - must include this to get a STD
mrd_xyz = mrd.groupBy([TIME_VAR] + ["MODEL_ID"] + ["strat_level"]).agg({TARGET_VAR:'sum'}).withColumnRenamed('sum(' + TARGET_VAR + ')', TARGET_VAR)

volatility_measure = mrd_xyz.groupBy(["strat_level", "MODEL_ID"]).agg((stddev(col(TARGET_VAR))/mean(col(TARGET_VAR))).alias("CV"),
                                    (stddev(col(TARGET_VAR))).alias('STD'),
                                    (mean(col(TARGET_VAR)).alias('MEAN')))

thresh_A_df = volatility_measure.groupBy("strat_level").agg(percentile_approx(col('CV'), 0.25).alias("volatility_thresh_A"))
thresh_B_df = volatility_measure.groupBy("strat_level").agg(percentile_approx(col('CV'), 0.5).alias("volatility_thresh_B"))

volatility_measure = volatility_measure.join(thresh_A_df, on = "strat_level", how = "left").join(thresh_B_df, on = "strat_level", how = "left")
volatility_measure = volatility_measure.withColumn('XYZ_new', lit('Y'))
volatility_measure = volatility_measure.withColumn('XYZ_new', when(col('CV') <= col("volatility_thresh_A"), 'X').otherwise(col('XYZ_new')))
volatility_measure = volatility_measure.withColumn('XYZ_new', when(col('CV') >  col("volatility_thresh_B"), 'Z').otherwise(col('XYZ_new')))

# COMMAND ----------

## Count Check
mrd.select("strat_level", "MODEL_ID").distinct().count() == value_measure.count() == volatility_measure.count()

# COMMAND ----------

## Merge the new ABC and XYZ clusters
mrd = mrd.join(value_measure.select("strat_level", "MODEL_ID", "ABC_new"), on = ["strat_level", "MODEL_ID"], how = "left")\
         .join(volatility_measure.select("strat_level", "MODEL_ID", "XYZ_new"), on = ["strat_level", "MODEL_ID"], how = "left")
mrd = mrd.withColumn("strat_level_ABC_XYZ", concat(col("strat_level"), lit("_"), col("ABC_new"), lit("_"), col("XYZ_new")))
mrd.count()

# COMMAND ----------

unq_levels = convertDFColumnsToList(mrd.select("strat_level_ABC_XYZ").distinct(), "strat_level_ABC_XYZ")

# COMMAND ----------

model_ids_df = mrd.groupby('strat_level_ABC_XYZ').agg(F.collect_set(col('MODEL_ID')).alias('MODEL_ID_list'))
model_ids_df = model_ids_df.select('*',size('MODEL_ID_list').alias('mi_len'))

print(model_ids_df.count() == len(unq_levels))
print(model_ids_df.select(sum("mi_len")).collect()[0][0] == mrd.select("MODEL_ID").distinct().count())

# COMMAND ----------

grouped_mi = convertDFColumnsToList(model_ids_df.select("MODEL_ID_list"), "MODEL_ID_list")

# COMMAND ----------

## Keep n% model ids for each level in mi_dict
def sample_n_perc_mi(n, grouped_list):
  n_perc_mi = []
  k = 0
  for sub_list in grouped_list:
    n_perc_mi = n_perc_mi + random.sample(sub_list, int(np.round(len(sub_list)*n/100)))
    k = k + int(np.round(len(sub_list)*n/100))
  print (len(n_perc_mi) == k)
  print ("Number of model ids included in sample:", k)
  print (f"{n}% of all model ids:", int(np.round(mrd.select("MODEL_ID").distinct().count()*n/100)))
  
  return n_perc_mi

# COMMAND ----------

## Keeping 20% model ids for each level
chosen_mi_list = sample_n_perc_mi(20, grouped_mi)

# COMMAND ----------

chosen_mi = spark.createDataFrame(chosen_mi_list, StringType())
display(chosen_mi)

# COMMAND ----------

chosen_mi.count() == chosen_mi.distinct().count()

# COMMAND ----------

## Write as delta table to dbfs
chosen_mi_path = "dbfs:/mnt/adls/Tables/chosen_mi"        ## Versions considered currently: 4 for monthly, 9 for weekly
save_df_as_delta(chosen_mi, chosen_mi_path, enforce_schema=False)
delta_info = load_delta_info(chosen_mi_path)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

