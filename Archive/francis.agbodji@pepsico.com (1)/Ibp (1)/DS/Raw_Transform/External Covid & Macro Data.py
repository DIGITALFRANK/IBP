# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

# Checking locations for outputs of this script 
print(DBI_EXTERNAL_VARIABLES_WEEKLY)
print(DBI_EXTERNAL_VARIABLES_MONTHLY)


# COMMAND ----------

# Refreshed file locations - as of July 16 2021
covid_govt_policies_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_covid_govt_policies_clean_July2021.csv"
google_mobility_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_google_mobility_clean_July2021.csv"
population_growth_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_population_growth_details_clean_July2021.csv"
population_health_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_population_health_clean_July2021.csv"
population_levels_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_population_levels_clean_July2021.csv"
world_bank_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_world_bank_indicators_clean_July2021.csv"
covid_case_path = "/FileStore/tables/PEP_Volume_Forecasting/BQ_covid_case_details_clean_July2021.csv"

# COMMAND ----------

# Function to read in files
def read_file(file_location):
  return (spark.read.format("csv").option("inferSchema", "false").option("header", "true").option("sep", ",").load(file_location))

# COMMAND ----------

# Create custom calendar
from pyspark.sql import Row

df = spark.sparkContext.parallelize([Row(start_date='2016-01-01', end_date='2025-12-31')]).toDF()

df = df \
  .withColumn('start_date', F.col('start_date').cast('date')) \
  .withColumn('end_date', F.col('end_date').cast('date'))\
  .withColumn('cal_date', F.explode(F.expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
        .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
        .withColumn("Week_end_date",date_add("Week_start_date",6))\
        .withColumn('week_year',F.when((year(col('Week_start_date'))==year(col('cal_date'))) & (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
                    .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                          (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
                    .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                          (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
                    .otherwise(year('Week_end_date')))\
        .withColumn('month_year',year(col('cal_date')))\
        .withColumn('week',F.when((year(col('Week_start_date'))==year(col('Week_end_date'))),F.weekofyear(col("Week_end_date")))\
                           .otherwise(F.weekofyear(col("Week_end_date"))))\
        .withColumn('month',F.month("cal_date"))

calendar = df\
            .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
            .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
            .withColumn("flag",lit(1))\
            .drop('start_date','end_date','week','month','year','Week_year')

calendar_df = calendar.select("cal_date", "Week_Of_Year", "Month_Of_Year", "month_year").withColumnRenamed("month_year", "year").distinct()

# COMMAND ----------

display(calendar_df)

# COMMAND ----------

# Aggregation function
def timely_agg(df, agg_dict, agg_cols, country_col = "country", time_var = "Week_Of_Year"):  
  agg_df = df.groupby(country_col, time_var).agg(agg_dict)
  for i in agg_cols:
    if agg_dict[i] == "sum":
      agg_df =  agg_df.withColumnRenamed("sum(" + i + ")", i)
    elif agg_dict[i] == "mean":
      agg_df =  agg_df.withColumnRenamed("avg(" + i + ")", i)
    elif agg_dict[i] == "min":
      agg_df =  agg_df.withColumnRenamed("min(" + i + ")", i)
    elif agg_dict[i] == "max":
      agg_df =  agg_df.withColumnRenamed("max(" + i + ")", i)
    
  return agg_df

# COMMAND ----------

# Cleansing column names function
def cleanse_cols(df):
  df= df.select([F.col(col).alias(re.sub("[^0-9a-zA-Z$]+"," ",col)) for col in df.columns ])
  df = df.select([F.col(col).alias(re.sub(" ","_",col)) for col in df.columns])
  
  return df

# COMMAND ----------

# MAGIC %md #### Covid Govt Policies

# COMMAND ----------

covid_govt_policies = read_file(covid_govt_policies_path)
print(covid_govt_policies.count(), len(covid_govt_policies.columns))
print(convertDFColumnsToList(covid_govt_policies.select("country_name").distinct(), "country_name"))
print(covid_govt_policies.agg({"date": "min"}).collect()[0][0], covid_govt_policies.agg({"date": "max"}).collect()[0][0])
display(covid_govt_policies)

# COMMAND ----------

max_cols = ["confirmed_cases", "deaths"]
mean_cols = list(set(covid_govt_policies.columns) - set(["country_name", "date"]) - set(max_cols))

max_dict = {x: "max" for x in max_cols}
mean_dict = {x: "mean" for x in mean_cols}

all_dict = {**max_dict, **mean_dict}

for c in max_cols + mean_cols:
  covid_govt_policies = covid_govt_policies.withColumn(c, F.col(c).cast("integer"))

# COMMAND ----------

covid_govt_policies = covid_govt_policies.orderBy("country_name", "date", ascending=True)
display(covid_govt_policies)

# COMMAND ----------

covid_govt_policies = covid_govt_policies.na.fill(-1)  # imputing with -1 to account for absence of data
covid_govt_policies = covid_govt_policies.withColumn('date', F.col('date').cast('date'))
covid_govt_policies = covid_govt_policies.join(calendar_df, covid_govt_policies.date == calendar_df.cal_date, how = 'left')

# COMMAND ----------

cg_agg_week = timely_agg(covid_govt_policies, all_dict, max_cols+mean_cols, country_col = "country_name", time_var = "Week_Of_Year")
cg_agg_month = timely_agg(covid_govt_policies, all_dict, max_cols+mean_cols, country_col = "country_name", time_var = "Month_Of_Year")

for col in mean_cols:
  cg_agg_week = cg_agg_week.withColumn(col, round(col))
  cg_agg_month = cg_agg_month.withColumn(col, round(col))

# COMMAND ----------

display(cg_agg_week)

# COMMAND ----------

# MAGIC %md #### Google mobility

# COMMAND ----------

google_mobility = read_file(google_mobility_path)
print(google_mobility.count(), len(google_mobility.columns))
print(convertDFColumnsToList(google_mobility.select("country_region").distinct(), "country_region"))
print(google_mobility.agg({"date": "min"}).collect()[0][0], google_mobility.agg({"date": "max"}).collect()[0][0])
display(google_mobility)

# COMMAND ----------

google_mobility = google_mobility.withColumn('date', F.col('date').cast('date'))
google_mobility = google_mobility.join(calendar_df, google_mobility.date == calendar_df.cal_date, how = 'left')

# COMMAND ----------

mean_cols = [col for col in google_mobility.columns if "_percent" in col]
mean_dict = {x: "mean" for x in mean_cols}

gm_agg_week = timely_agg(google_mobility, mean_dict, mean_cols, country_col = "country_region", time_var = "Week_Of_Year")
gm_agg_month = timely_agg(google_mobility, mean_dict, mean_cols, country_col = "country_region", time_var = "Month_Of_Year")

# COMMAND ----------

# MAGIC %md #### Population Growth

# COMMAND ----------

population_growth = read_file(population_growth_path)
print(population_growth.count(), len(population_growth.columns))
print(convertDFColumnsToList(population_growth.select("country_name").distinct(), "country_name"))
print(population_growth.agg({"year": "min"}).collect()[0][0], population_growth.agg({"year": "max"}).collect()[0][0])
display(population_growth)

# COMMAND ----------

int_cols = list(set(population_growth.columns) - set(["country_code", "country_name", "year"]))  
for col_name in int_cols:
  population_growth = population_growth.withColumn(col_name, population_growth[col_name].cast(DoubleType()))

# COMMAND ----------

pg_agg_week = calendar_df.select("year", "Week_Of_Year").distinct().join(population_growth, on = "year", how = "left").drop("year", "country_code") 
pg_agg_month = calendar_df.select("year", "Month_Of_Year").distinct().join(population_growth, on = "year", how = "left").drop("year", "country_code")

mean_dict = {x: "mean" for x in int_cols}

pg_agg_week = timely_agg(pg_agg_week, mean_dict, int_cols, country_col = "country_name", time_var = "Week_Of_Year")
pg_agg_month = timely_agg(pg_agg_month, mean_dict, int_cols, country_col = "country_name", time_var = "Month_Of_Year")

# COMMAND ----------

display(pg_agg_week)

# COMMAND ----------

display(pg_agg_month)

# COMMAND ----------

# MAGIC %md #### Population Health

# COMMAND ----------

# List of important indicator_name for population health from Corey
health_indicator_list = [
  'Age dependency ratio (% of working-age population)',
  'Birth rate, crude (per 1,000 people)',
  'Death rate, crude (per 1,000 people)',
  'Number of under-five deaths',
  'Fertility rate, total (births per woman)',
  'Human capital index (HCI) (scale 0-1)',
  'Life expectancy at birth, total (years)',
  'Unemployment, total (% of total labor force)',
  'Labor force, female (% of total labor force)',
  'Urban population (% of total population)',
  'Urban population growth (annual %)',
  'GNI per capita, Atlas method (current US$)',
  'Population growth (annual %)',
  'Population, female (% of total population)',
  'Survival to age 65, female (% of cohort)',
  'Survival to age 65, male (% of cohort)',
]

# COMMAND ----------

population_health = read_file(population_health_path)
print(population_health.count(), len(population_health.columns))
print(convertDFColumnsToList(population_health.select("country_name").distinct(), "country_name"))
print(population_health.agg({"year": "min"}).collect()[0][0], population_health.agg({"year": "max"}).collect()[0][0])
display(population_health)

# COMMAND ----------

population_health = population_health.withColumn("value", population_health["value"].cast(DoubleType())) \
                                      .filter(population_health.indicator_name.isin(health_indicator_list))

ph_agg_week = calendar_df.select("year", "Week_Of_Year").distinct().join(population_health, on = "year", how = "inner").na.drop(subset="country_name")
ph_agg_week = ph_agg_week.groupBy("Week_Of_Year", "country_name").pivot("indicator_name").avg("value") 
ph_agg_week = cleanse_cols(ph_agg_week)

ph_agg_month = calendar_df.select("year", "Month_Of_Year").distinct().join(population_health, on = "year", how = "inner").na.drop(subset="country_name")
ph_agg_month = ph_agg_month.groupBy("Month_Of_Year", "country_name").pivot("indicator_name").avg("value") 
ph_agg_month = cleanse_cols(ph_agg_month)

# COMMAND ----------

display(ph_agg_month)

# COMMAND ----------

# MAGIC %md #### Population Levels

# COMMAND ----------

population_levels = read_file(population_levels_path)
print(population_levels.count(), len(population_levels.columns))
print(convertDFColumnsToList(population_levels.select("country_name").distinct(), "country_name"))
print(population_levels.agg({"year": "min"}).collect()[0][0], population_levels.agg({"year": "max"}).collect()[0][0])
display(population_levels)

# COMMAND ----------

def bucket_age_groups(indiv_age_group):
  separator_index = indiv_age_group.find('_')
  bin_floor =  int(indiv_age_group[:separator_index])
  bin_ceiling =  int(indiv_age_group[separator_index+1:])
  
  # rewriting age bins
  if (bin_floor == 100):
    return "75_100"
  if (bin_floor >= 0 and bin_ceiling <= 24):
    return "0_24"
  elif (bin_floor >= 25 and bin_ceiling <= 49):
    return "25_49"
  elif (bin_floor >= 50 and bin_ceiling <= 74):
    return "50_74"
  elif (bin_floor >= 75 and bin_ceiling <= 100):
    return "75_100"
  
udf_age_bucket = udf(bucket_age_groups, StringType()) # if the function returns an int
population_levels = population_levels.withColumn("age_bucket", udf_age_bucket("age_group")) 

# COMMAND ----------

display(population_levels)

# COMMAND ----------

age_buckets = convertDFColumnsToList(population_levels.select("age_bucket").distinct(), "age_bucket")
population_levels = population_levels.withColumn("midyear_population_male", population_levels["midyear_population_male"].cast(IntegerType())) \
                                      .withColumn("midyear_population_female", population_levels["midyear_population_female"].cast(IntegerType()))

pl_male = population_levels.groupBy("year", "country_name").pivot("age_bucket").sum("midyear_population_male")
pl_female = population_levels.groupBy("year", "country_name").pivot("age_bucket").sum("midyear_population_female")

# NOTE: This gives rise to constant population throughout weeks of a year
pl_agg_week_male = calendar_df.select("year", "Week_Of_Year").distinct().join(pl_male, on = "year", how = "inner").drop("year")   
pl_agg_week_female = calendar_df.select("year", "Week_Of_Year").distinct().join(pl_female, on = "year", how = "inner").drop("year")

mean_cols = list(set(pl_agg_week_male.columns) - set(["Week_Of_Year", "country_name"]))    
mean_dict = {x: "mean" for x in mean_cols}
pl_agg_week_male = timely_agg(pl_agg_week_male, mean_dict, mean_cols, country_col = "country_name", time_var = "Week_Of_Year")
pl_agg_week_female = timely_agg(pl_agg_week_female, mean_dict, mean_cols, country_col = "country_name", time_var = "Week_Of_Year")

pl_agg_month_male = calendar_df.select("year", "Month_Of_Year").distinct().join(pl_male, on = "year", how = "inner").drop("year")
pl_agg_month_female = calendar_df.select("year", "Month_Of_Year").distinct().join(pl_female, on = "year", how = "inner").drop("year")

mean_cols = list(set(pl_agg_month_female.columns) - set(["Month_Of_Year", "country_name"]))   
mean_dict = {x: "mean" for x in mean_cols}
pl_agg_month_male = timely_agg(pl_agg_month_male, mean_dict, mean_cols, country_col = "country_name", time_var = "Month_Of_Year")
pl_agg_month_female = timely_agg(pl_agg_month_female, mean_dict, mean_cols, country_col = "country_name", time_var = "Month_Of_Year")

for col in age_buckets:
  pl_agg_week_male = pl_agg_week_male.withColumnRenamed(col, col + "_midyear_population_male")
  pl_agg_week_female = pl_agg_week_female.withColumnRenamed(col, col + "_midyear_population_female")
  pl_agg_month_male = pl_agg_month_male.withColumnRenamed(col, col + "_midyear_population_male")
  pl_agg_month_female = pl_agg_month_female.withColumnRenamed(col, col + "_midyear_population_female")

# COMMAND ----------

pl_agg_week = pl_agg_week_male.join(pl_agg_week_female, on = ["Week_Of_Year", "country_name"], how = 'outer')
pl_agg_month = pl_agg_month_male.join(pl_agg_month_female, on = ["Month_Of_Year", "country_name"], how = 'outer')

# COMMAND ----------

display(pl_agg_week)

# COMMAND ----------

display(pl_agg_month)

# COMMAND ----------

# MAGIC %md #### World Bank

# COMMAND ----------

# List of important indicator_name for world bank data from Corey
wb_indicator_list = [
  'Adjusted net enrollment rate, primary (% of primary school age children)',
  'Adjusted net national income (annual % growth)',
  'Adjusted net national income per capita (annual % growth)',
  'Agricultural land (% of land area)',
  'Arable land (% of land area)',
  'Compensation of employees (% of expense)',
  'Consumer price index (2010 = 100)',
  'Current health expenditure (% of GDP)',
  'Export value index (2000 = 100)',
  'Export volume index (2000 = 100)',
  'GDP growth (annual %)',
  'GNI growth (annual %)',
  'Tax revenue (% of GDP)', 
]

# COMMAND ----------

world_bank = read_file(world_bank_path)
print (world_bank.count(), len(world_bank.columns))
print(convertDFColumnsToList(world_bank.select("country_name").distinct(), "country_name"))
print(world_bank.agg({"year": "min"}).collect()[0][0], world_bank.agg({"year": "max"}).collect()[0][0])
display(world_bank)

# COMMAND ----------

world_bank = world_bank.withColumn("value", world_bank["value"].cast(DoubleType())) \
                                      .filter(world_bank.indicator_name.isin(wb_indicator_list))

wb_agg_week = calendar_df.select("year", "Week_Of_Year").distinct().join(world_bank, on = "year", how = "inner").na.drop(subset="country_name")
wb_agg_week = wb_agg_week.groupBy("Week_Of_Year", "country_name").pivot("indicator_name").avg("value")
wb_agg_week= cleanse_cols(wb_agg_week)

wb_agg_month = calendar_df.select("year", "Month_Of_Year").distinct().join(world_bank, on = "year", how = "inner").na.drop(subset="country_name")
wb_agg_month = wb_agg_month.groupBy("Month_Of_Year", "country_name").pivot("indicator_name").avg("value")
wb_agg_month= cleanse_cols(wb_agg_month)

# COMMAND ----------

display(wb_agg_week)

# COMMAND ----------

display(wb_agg_month)

# COMMAND ----------

# MAGIC %md #### Covid Case

# COMMAND ----------

covid_case = read_file(covid_case_path)
print(covid_case.count(), len(covid_case.columns))
print(convertDFColumnsToList(covid_case.select("country_name").distinct(), "country_name"))
print(covid_case.agg({"date": "min"}).collect()[0][0], covid_case.agg({"date": "max"}).collect()[0][0])
display(covid_case)

# COMMAND ----------

sum_cols = ["new_confirmed", "new_confirmed_male", "new_confirmed_female"]
cum_cols = ["cumulative_confirmed", "cumulative_confirmed_male", "cumulative_confirmed_female"]
mean_cols = ["population", "average_temperature_celsius", "minimum_temperature_celsius", "maximum_temperature_celsius", "rainfall_mm", "snowfall_mm", \
             "dew_point", "relative_humidity"]

sum_dict = {x: "sum" for x in sum_cols}
cum_dict = {x: "max" for x in cum_cols}
mean_dict = {x: "mean" for x in mean_cols}

# all_dict = {**sum_dict, **cum_dict, **mean_dict}
all_dict = {**sum_dict, **cum_dict}

for c in sum_cols + cum_cols + mean_cols:
  covid_case = covid_case.withColumn(c, F.col(c).cast("integer"))

# COMMAND ----------

covid_case = covid_case.na.fill(0)
covid_case = covid_case.withColumn('date', F.col('date').cast('date'))
covid_case = covid_case.join(calendar_df, covid_case.date == calendar_df.cal_date, how = 'left')

# COMMAND ----------

display(covid_case)

# COMMAND ----------

from pyspark.sql.functions import col
display(covid_case.filter((col("country_name") == "Spain") & (col("Week_Of_Year") == 202119))\
        .select("cumulative_confirmed").orderBy("cumulative_confirmed"))

# COMMAND ----------

# covid_case = covid_case.drop(*mean_cols) #dropping irrelevant features

# COMMAND ----------

cc_agg_week = timely_agg(covid_case, all_dict, sum_cols+cum_cols, country_col = "country_name", time_var = "Week_Of_Year")
cc_agg_month = timely_agg(covid_case, all_dict, sum_cols+cum_cols, country_col = "country_name", time_var = "Month_Of_Year")

# COMMAND ----------

display(cc_agg_week)

# COMMAND ----------

display(cc_agg_month)

# COMMAND ----------

# MAGIC %md ### MERGE : Weekly Aggregations

# COMMAND ----------

merge_cols = ["country_name", "Week_Of_Year"]
# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(cg_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("cg_")==False:
    cg_agg_week = cg_agg_week.withColumnRenamed(i, "cg_"+i)
display(cg_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
gm_agg_week = gm_agg_week.withColumnRenamed("country_region", "country_name")
COLS_FOR_CATEGORIZATION = subtract_two_lists(gm_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("gm_")==False:
    gm_agg_week = gm_agg_week.withColumnRenamed(i, "gm_"+i)
display(gm_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(pg_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("pg_")==False:
    pg_agg_week = pg_agg_week.withColumnRenamed(i, "pg_"+i)
display(pg_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(ph_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("ph_")==False:
    ph_agg_week = ph_agg_week.withColumnRenamed(i, "ph_"+i)
display(ph_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(pl_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("pl_")==False:
    pl_agg_week = pl_agg_week.withColumnRenamed(i, "pl_"+i)
display(pl_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(wb_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("wb_")==False:
    wb_agg_week = wb_agg_week.withColumnRenamed(i, "wb_"+i)
display(wb_agg_week)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(cc_agg_week.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("cc_")==False:
    cc_agg_week = cc_agg_week.withColumnRenamed(i, "cc_"+i)
display(cc_agg_week)

# COMMAND ----------

weekly_agg_dfs = [cg_agg_week, gm_agg_week, pg_agg_week, ph_agg_week, pl_agg_week, wb_agg_week, cc_agg_week]

for df in weekly_agg_dfs:
  for i in df.columns:
    if "country" in i:
      print (i)
  print (df.count(), len(df.columns))
  print (df.agg({"Week_Of_Year": "min"}).collect()[0][0], df.agg({"Week_Of_Year": "max"}).collect()[0][0])

# COMMAND ----------


external_weekly_agg = cg_agg_week.join(gm_agg_week.withColumnRenamed("country_region", "country_name"), on = merge_cols, how = "outer") \
                                  .join(pg_agg_week, on = merge_cols, how = "outer") \
                                  .join(ph_agg_week, on = merge_cols, how = "outer") \
                                  .join(pl_agg_week, on = merge_cols, how = "outer") \
                                  .join(wb_agg_week, on = merge_cols, how = "outer") \
                                  .join(cc_agg_week, on = merge_cols, how = "outer") 

# COMMAND ----------

print(external_weekly_agg.count(), len(external_weekly_agg.columns))

# COMMAND ----------

display(external_weekly_agg.select("wb_Adjusted_net_enrollment_rate_primary_of_primary_school_age_children_"))

# COMMAND ----------

display(external_weekly_agg.orderBy('country_name', 'Week_Of_Year'))

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(external_weekly_agg, DBI_EXTERNAL_VARIABLES_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_EXTERNAL_VARIABLES_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# MAGIC %md ### MERGE : Monthly Aggregations

# COMMAND ----------

# Prefix for Driver Categorization 
merge_cols = ["country_name", "Month_Of_Year"]

COLS_FOR_CATEGORIZATION = subtract_two_lists(cg_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("cg_")==False:
    cg_agg_month = cg_agg_month.withColumnRenamed(i, "cg_"+i)
display(cg_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
gm_agg_month = gm_agg_month.withColumnRenamed("country_region", "country_name")

COLS_FOR_CATEGORIZATION = subtract_two_lists(gm_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("gm_")==False:
    gm_agg_month = gm_agg_month.withColumnRenamed(i, "gm_"+i)
display(gm_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(pg_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("pg_")==False:
    pg_agg_month = pg_agg_month.withColumnRenamed(i, "pg_"+i)
display(pg_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(ph_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("ph_")==False:
    ph_agg_month = ph_agg_month.withColumnRenamed(i, "ph_"+i)
display(ph_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(pl_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("pl_")==False:
    pl_agg_month = pl_agg_month.withColumnRenamed(i, "pl_"+i)
display(pl_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(wb_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("wb_")==False:
    wb_agg_month = wb_agg_month.withColumnRenamed(i, "wb_"+i)
display(wb_agg_month)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(cc_agg_month.columns, merge_cols)
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("cc_")==False:
    cc_agg_month = cc_agg_month.withColumnRenamed(i, "cc_"+i)
display(cc_agg_month)

# COMMAND ----------

for df in monthly_agg_dfs:
  for i in df.columns:
    if "country" in i:
      print (i)
  print (df.count(), len(df.columns))
  print (df.agg({"Month_Of_Year": "min"}).collect()[0][0], df.agg({"Month_Of_Year": "max"}).collect()[0][0])

# COMMAND ----------

external_monthly_agg = cg_agg_month.join(gm_agg_month.withColumnRenamed("country_region", "country_name"), on = merge_cols, how = "outer") \
                                  .join(pg_agg_month, on = merge_cols, how = "outer") \
                                  .join(ph_agg_month, on = merge_cols, how = "outer") \
                                  .join(pl_agg_month, on = merge_cols, how = "outer") \
                                  .join(wb_agg_month, on = merge_cols, how = "outer") \
                                  .join(cc_agg_month, on = merge_cols, how = "outer") 

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(external_monthly_agg, DBI_EXTERNAL_VARIABLES_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_EXTERNAL_VARIABLES_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# ## Quick QC of output
# display(external_weekly_agg)

# COMMAND ----------

