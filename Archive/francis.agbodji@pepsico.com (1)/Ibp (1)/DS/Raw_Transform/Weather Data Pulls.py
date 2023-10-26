# Databricks notebook source
# DBTITLE 1,Instantiate Notebook
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

## Checking locations for outputs of this script 
print(DBI_WEATHER_WEEKLY)
print(DBI_WEATHER_MONTHLY)

# COMMAND ----------

# DBTITLE 1,Import General Libraries
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from pyspark.sql import Row

# COMMAND ----------

# DBTITLE 1,Import Meteostat Libraries
import meteostat
from meteostat import Stations, Monthly, Daily, Hourly, units

# COMMAND ----------

## GENERAL RESOURCES / DOCUMENTATION FOR METEOSTAT
## https://dev.meteostat.net/terms.html#license
## https://dev.meteostat.net/python/#installation
## https://dev.meteostat.net/python/daily.html#api
## https://medium.com/meteostat/obtain-weather-data-for-any-location-with-python-c50a6909b271
## https://medium.com/meteostat/analyze-historical-weather-data-with-python-e188582f24ee
## https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.plot.html

# COMMAND ----------

## "Helper" function for time variable based aggregation
## Note - in the 9/1/2021 iteration of this code, this is not needed for df aggregation support

def timely_agg(df, agg_dict, agg_cols, country_col="country", time_var="Week_Of_Year"):   
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

# DBTITLE 1,"Helper" Function Definitions
## Function for some feature engineering - consecutive weeks of increasing or decreasing values
## Net-new and created on 9/1/2021 to capture general weather trends associated with a location

def consec_time_periods_increasing_value(input_df, count_col, count_reset_col, sort_cols_list):
  
  input_df = input_df.sort_values(by=sort_cols_list, ascending=True)
  s = (input_df[count_reset_col].ne(input_df[count_reset_col].shift()) | input_df[count_col].diff().lt(0)).cumsum()  ## lt = strictly less than
  consec_count_list = s.groupby(s).cumcount() + 1
  input_df['increase_count_' + count_col] = consec_count_list
  
  return input_df

##################################################
##################################################

def consec_time_periods_decreasing_value(input_df, count_col, count_reset_col, sort_cols_list):
  
  input_df = input_df.sort_values(by=sort_cols_list, ascending=True)
  s = (input_df[count_reset_col].ne(input_df[count_reset_col].shift()) | input_df[count_col].diff().gt(0)).cumsum()  ## gt = strictly greater than
  consec_count_list = s.groupby(s).cumcount() + 1
  input_df['decrease_count_' + count_col] = consec_count_list
  
  return input_df

# COMMAND ----------

# DBTITLE 1,Calendar Creation (for Mapping)
## Create custom calendar for date mapping purposes
## Necessary to map week start days in the same way as rest of our MRD data

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

calendar_df = calendar.select("cal_date", "Week_Of_Year", "Month_Of_Year").distinct()

# COMMAND ----------

display(calendar_df.orderBy(col('cal_date').desc()))

# COMMAND ----------

# DBTITLE 1,Data Set Up for (Weather) Station-Based Pulls
stations = Stations()

stations_spain = stations.region('ES').fetch().reset_index()
print('SPAIN: {}'.format(stations_spain.shape))

stations_port = stations.region('PT').fetch().reset_index()
print('PORTUGAL: {}'.format(stations_port.shape))

# COMMAND ----------

## Setting up list so multiple weather stations can be pulled at once (in one command)
## Keeping ES and PT separate for now, to enable a rollup by country (which occurs downstream)

spain_stations_list = list(set(stations_spain['id'].unique()))  ## list of Spain weather station IDs
port_stations_list = list(set(stations_port['id'].unique()))    ## list of Portugal weather station IDs

# COMMAND ----------

## Setting up the date pulls to be dynamic based on day code is run and user-dictated history needs

end_dt = pd.Timestamp('today').floor('D')                        ## automatically pulls based on code rundate
start_dt = end_dt - relativedelta(years=WEATHER_HISTORY_YEARS)   ## automatically generates based on today and config for historical depth

print('Weather Pull Start: {}'.format(start_dt))
print('Weather Pull End: {}'.format(end_dt))

# COMMAND ----------

# DBTITLE 1,Pull Daily Weather Station Data - for ES and PT
## Note - some weather stations will show an issue with loading ('cannot load ...')
## This should not be an issue, as long as it does not represent a significant % of country's stations

es_data = Daily(spain_stations_list, start=start_dt, end=end_dt)  ## to pull data by Day
es_data = es_data.normalize()  ## to auto-correct for sensor gaps
es_data = es_data.interpolate(limit=3)  ## must normalize before this!
es_data = es_data.convert(units.imperial)  ## to convert units
es_data = es_data.fetch().reset_index()  ## fetch + reset to give typical Pandas df

## Note - below was removed that would auto-aggregate to week across all stations
## This was done intentionally to better map to our calendar and enable day-based feature engineering
## es_data = es_data.aggregate(freq='1W', spatial=True)

## Review pulled dataset
es_data.describe()

# COMMAND ----------

## Note - some weather stations will show an issue with loading ('cannot load ...')
## This should not be an issue, as long as it does not represent a significant % of country's stations

pt_data = Daily(port_stations_list, start=start_dt, end=end_dt)
pt_data = pt_data.normalize()
pt_data = pt_data.interpolate(limit=3)
pt_data = pt_data.convert(units.imperial)  ## to convert units
pt_data = pt_data.fetch().reset_index()

## Note - below was removed that would auto-aggregate to week across all stations
## This was done intentionally to better map to our calendar and enable day-based feature engineering
## pt_data = pt_data.aggregate(freq='1W', spatial=True)

## Review pulled dataset
pt_data.describe()

# COMMAND ----------

es_data['country'] = 'Spain'
pt_data['country'] = 'Portugal'

combined_data = es_data.append(pt_data, ignore_index=True)                          ## 'stack' dataframes together
combined_data = combined_data.loc[:, combined_data.apply(pd.Series.nunique) != 1]   ## remove any cols with all same value

## Set up column for date details
## Will use this for null handling and feature engineering
combined_data['year'] = combined_data['time'].dt.year
combined_data['month'] = combined_data['time'].dt.month
combined_data['week'] = combined_data['time'].dt.week

## Using the above for construction of standard time variables
## Note - only planning to use these for feature engineering purposes
## combined_data['Week_Of_Year'] = combined_data['year'].astype(int).astype(str) + combined_data['week'].astype(int).astype(str)
## combined_data['Month_Of_Year'] = combined_data['year'].astype(int).astype(str) + combined_data['month'].astype(int).astype(str)
## combined_data[['Week_Of_Year', 'Month_Of_Year']] = combined_data[['Week_Of_Year', 'Month_Of_Year']].astype(int)

## Review output
print(es_data.shape[0] + pt_data.shape[0] == combined_data.shape[0])
print(combined_data.shape)
print(combined_data.dtypes.to_dict())
combined_data.describe()
# combined_data.head()

# COMMAND ----------

## Check and correct null values
print(combined_data.isna().sum().tolist())

## Setting up method for filling NA values based on the feature itself
## This was also informed by levels of null values seen in those columns
fill_na_zero = ['prcp', 'snow', 'tsun']
fill_na_mean = list(set(combined_data.columns) - set(fill_na_zero) - set(['time', 'station', 'country']))

## Filling with zeros
combined_data[fill_na_zero] = combined_data[fill_na_zero].fillna(0)

## Filling by grouping - mean as grouped by month and country
for each_col in fill_na_mean:
  print('Null handling for {}'.format(each_col))
  combined_data[each_col] = combined_data[each_col].fillna(combined_data.groupby(['country', 'month'])[each_col].transform('mean'))

## Dropping data cols for which there is only 1 value
combined_data = combined_data.loc[:, combined_data.apply(pd.Series.nunique) != 1]   ## remove any cols with all same value
  
## Review output
print(combined_data.isna().sum().tolist())
print('Corrected NaN Totals: {}'.format(combined_data.isna().sum().sum()))
print('Final Cols: {}'.format(combined_data.columns.tolist()))

# COMMAND ----------

# DBTITLE 1,Feature Engineering - b/c we must use weather details at the daily level
calendar_pd = calendar_df.toPandas()
calendar_pd['cal_date'] = calendar_pd['cal_date'].astype(str)

combined_data_feat = combined_data.copy()
combined_data_feat.rename(columns={'time':'cal_date'}, inplace=True)
combined_data_feat['cal_date'] = combined_data_feat['cal_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
combined_data_feat['cal_date'] = combined_data_feat['cal_date'].astype(str)

combined_data_feat

# COMMAND ----------

## Creating aggregation dictionary up to country-level
mean_cols = list(set(combined_data_feat.columns) - set(["country", "cal_date", "station", "Week_Of_Year", "Month_Of_Year", "year", "month", "week"]))
mean_dict = {x: 'mean' for x in mean_cols}

## Setting up ability to aggregate up to country and time first before the below calcs - stored in ./config
## This removes the station from our dataframe - likely better for what we are trying to capture 
if AGG_WEATHER_FEAT_BY_COUNTRY:
  combined_data_feat = combined_data_feat.groupby(["country", "cal_date"]).agg(mean_dict).reset_index()

## Merge with the calendar dataframe & review 
combined_data_feat = combined_data_feat.merge(calendar_pd, on="cal_date", how="left")
print('Merge Check:', combined_data_feat.isna().sum().sum() == 0)
combined_data_feat

# COMMAND ----------

# DBTITLE 1,Weekly Feature Engineering
## Source - weather data documentation - https://dev.meteostat.net/python/daily.html#data-structure

## FEATURE ENGINEERING WHEN ROLLING UP BY WEEK
feat_eng_grouping_cols = ['country', 'Week_Of_Year']

## Precipitation Counts - days in the week meeting each criteria
prcp_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['prcp'].apply(lambda x: x[x > 0.05].count()).reset_index()
snow_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['snow'].apply(lambda x: x[x > 0.05].count()).reset_index()
prcp_pd.rename(columns={'prcp':'count_nonzero_prcp'}, inplace=True)
snow_pd.rename(columns={'snow':'count_nonzero_snow'}, inplace=True)

## Temperature Counts - days in the week meeting each criteria
hot_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['tmax'].apply(lambda x: x[x > 90].count()).reset_index()
cold_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['tmin'].apply(lambda x: x[x < 35].count()).reset_index()
temp_pd1 = combined_data_feat.groupby(feat_eng_grouping_cols)['tavg'].apply(lambda x: x[x > 75].count()).reset_index()
temp_pd2 = combined_data_feat.groupby(feat_eng_grouping_cols)['tavg'].apply(lambda x: x[x < 50].count()).reset_index()
hot_pd.rename(columns={'tmax':'count_extreme_heat'}, inplace=True)
cold_pd.rename(columns={'tmin':'count_extreme_cold'}, inplace=True)
temp_pd1.rename(columns={'tavg':'count_heat'}, inplace=True)
temp_pd2.rename(columns={'tavg':'count_cold'}, inplace=True)

## Miscellaneous Counts - days in the week meeting each criteria
wind_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['wspd'].apply(lambda x: x[x > 8].count()).reset_index()
extreme_wind_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['wpgt'].apply(lambda x: x[x > 20].count()).reset_index()
pressure_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['pres'].apply(lambda x: x[x > 1020].count()).reset_index()
wind_pd.rename(columns={'wspd':'count_high_wind'}, inplace=True)
extreme_wind_pd.rename(columns={'wpgt':'count_extreme_wind'}, inplace=True)
pressure_pd.rename(columns={'pres':'count_high_pres'}, inplace=True)

## Precipitation Sums - sum across the week
prcp_sum_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['prcp'].sum().reset_index()
snow_sum_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['snow'].sum().reset_index()
prcp_sum_pd.rename(columns={'prcp':'sum_prcp_total'}, inplace=True)
snow_sum_pd.rename(columns={'snow':'sum_snow_total'}, inplace=True)

# COMMAND ----------

## Aggregate up to WEEK
weekly_weather_pd = combined_data_feat.groupby(feat_eng_grouping_cols).agg(mean_dict).reset_index()

## Weekly joins with the above feature engineering "temporary" dataframes
full_weekly_weather_pd = weekly_weather_pd.merge(prcp_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(snow_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(hot_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(cold_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(temp_pd1, how='left', on=feat_eng_grouping_cols)\
                                          .merge(temp_pd2, how='left', on=feat_eng_grouping_cols)\
                                          .merge(wind_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(extreme_wind_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(pressure_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(prcp_sum_pd, how='left', on=feat_eng_grouping_cols)\
                                          .merge(snow_sum_pd, how='left', on=feat_eng_grouping_cols)

## Review output
full_weekly_weather_pd.describe()

# COMMAND ----------

## Capturing TRENDS now that we have "final" weekly dataframe
full_weekly_weather_pd = full_weekly_weather_pd.pipe(consec_time_periods_increasing_value, 'tavg', 'country', feat_eng_grouping_cols)\
                                               .pipe(consec_time_periods_increasing_value, 'tmax', 'country', feat_eng_grouping_cols)\
                                               .pipe(consec_time_periods_decreasing_value, 'tmin', 'country', feat_eng_grouping_cols)\
                                               .pipe(consec_time_periods_increasing_value, 'prcp', 'country', feat_eng_grouping_cols)\
                                               .pipe(consec_time_periods_increasing_value, 'sum_prcp_total', 'country', feat_eng_grouping_cols)\

## Review output
## full_weekly_weather_pd[full_weekly_weather_pd['increase_count_prcp'] > 4]
full_weekly_weather_pd.describe()

# COMMAND ----------

# DBTITLE 1,Monthly Feature Engineering
## Source - weather data documentation - https://dev.meteostat.net/python/daily.html#data-structure

## FEATURE ENGINEERING WHEN ROLLING UP BY MONTH
feat_eng_grouping_cols = ['country', 'Month_Of_Year']

## Precipitation Counts - days in the month meeting each criteria
prcp_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['prcp'].apply(lambda x: x[x > 0.05].count()).reset_index()
snow_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['snow'].apply(lambda x: x[x > 0.05].count()).reset_index()
prcp_pd.rename(columns={'prcp':'count_nonzero_prcp'}, inplace=True)
snow_pd.rename(columns={'snow':'count_nonzero_snow'}, inplace=True)

## Temperature Counts - days in the month meeting each criteria
hot_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['tmax'].apply(lambda x: x[x > 90].count()).reset_index()
cold_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['tmin'].apply(lambda x: x[x < 30].count()).reset_index()
temp_pd1 = combined_data_feat.groupby(feat_eng_grouping_cols)['tavg'].apply(lambda x: x[x > 75].count()).reset_index()
temp_pd2 = combined_data_feat.groupby(feat_eng_grouping_cols)['tavg'].apply(lambda x: x[x < 50].count()).reset_index()
hot_pd.rename(columns={'tmax':'count_extreme_heat'}, inplace=True)
cold_pd.rename(columns={'tmin':'count_extreme_cold'}, inplace=True)
temp_pd1.rename(columns={'tavg':'count_heat'}, inplace=True)
temp_pd2.rename(columns={'tavg':'count_cold'}, inplace=True)

## Miscellaneous Counts - days in the month meeting each criteria
wind_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['wspd'].apply(lambda x: x[x > 8].count()).reset_index()
extreme_wind_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['wpgt'].apply(lambda x: x[x > 20].count()).reset_index()
pressure_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['pres'].apply(lambda x: x[x > 1020].count()).reset_index()
wind_pd.rename(columns={'wspd':'count_high_wind'}, inplace=True)
extreme_wind_pd.rename(columns={'wpgt':'count_extreme_wind'}, inplace=True)
pressure_pd.rename(columns={'pres':'count_high_pres'}, inplace=True)

## Precipitation Sums - sum across the month
prcp_sum_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['prcp'].sum().reset_index()
snow_sum_pd = combined_data_feat.groupby(feat_eng_grouping_cols)['snow'].sum().reset_index()
prcp_sum_pd.rename(columns={'prcp':'sum_prcp_total'}, inplace=True)
snow_sum_pd.rename(columns={'snow':'sum_snow_total'}, inplace=True)

# COMMAND ----------

## Aggregate up to MONTH
monthly_weather_pd = combined_data_feat.groupby(feat_eng_grouping_cols).agg(mean_dict).reset_index()

## Weekly joins with the above feature engineering "temporary" dataframes
full_monthly_weather_pd = monthly_weather_pd.merge(prcp_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(snow_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(hot_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(cold_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(temp_pd1, how='left', on=feat_eng_grouping_cols)\
                                            .merge(temp_pd2, how='left', on=feat_eng_grouping_cols)\
                                            .merge(wind_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(extreme_wind_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(pressure_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(prcp_sum_pd, how='left', on=feat_eng_grouping_cols)\
                                            .merge(snow_sum_pd, how='left', on=feat_eng_grouping_cols)

## Review output
full_monthly_weather_pd.describe()

# COMMAND ----------

## Capturing TRENDS now that we have "final" monthly dataframe
full_monthly_weather_pd = full_monthly_weather_pd.pipe(consec_time_periods_increasing_value, 'tavg', 'country', feat_eng_grouping_cols)\
                                                 .pipe(consec_time_periods_increasing_value, 'tmax', 'country', feat_eng_grouping_cols)\
                                                 .pipe(consec_time_periods_decreasing_value, 'tmin', 'country', feat_eng_grouping_cols)\
                                                 .pipe(consec_time_periods_increasing_value, 'prcp', 'country', feat_eng_grouping_cols)\
                                                 .pipe(consec_time_periods_increasing_value, 'sum_prcp_total', 'country', feat_eng_grouping_cols)\

## Review output
# full_monthly_weather_pd[full_monthly_weather_pd['increase_count_prcp'] > 4]
full_monthly_weather_pd.describe()

# COMMAND ----------

# DBTITLE 1,Convert to Spark & Prep for Save
weather_weekly = spark.createDataFrame(full_weekly_weather_pd)
weather_monthly = spark.createDataFrame(full_monthly_weather_pd)

# COMMAND ----------

## Validation of time periods included in each dataframe
sorted([i.Week_Of_Year for i in weather_weekly.select('Week_Of_Year').distinct().collect()])
display(weather_weekly)

# COMMAND ----------

## Validation of time periods included in each dataframe
sorted([i.Month_Of_Year for i in weather_monthly.select('Month_Of_Year').distinct().collect()])
display(weather_monthly)

# COMMAND ----------

## Setting prefixes for Driver Categorization - WEEKLY

COLS_FOR_CATEGORIZATION = subtract_two_lists(weather_weekly.columns, ['country', "Week_Of_Year"])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("weather_") == False:
    weather_weekly = weather_weekly.withColumnRenamed(i, "weather_" + i)

print('weekly weather columns:', weather_weekly.columns)

# COMMAND ----------

## Setting prefixes for Driver Categorization - MONTHLY

COLS_FOR_CATEGORIZATION = subtract_two_lists(weather_monthly.columns, ['country', "Month_Of_Year"])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("weather_") == False:
    weather_monthly = weather_monthly.withColumnRenamed(i, "weather_" + i)

print('monthly weather columns:', weather_monthly.columns)

# COMMAND ----------

## QC validation
print(weather_weekly.count() == weather_weekly.distinct().count() == weather_weekly.select("country", "Week_Of_Year").distinct().count())
print(weather_monthly.count() == weather_monthly.distinct().count() == weather_monthly.select("country", "Month_Of_Year").distinct().count())

# COMMAND ----------

# DBTITLE 1,Save as Delta Table
## Write as delta table to dbfs - WEEKLY
## DBI_WEATHER_WEEKLY - included in ./config file

save_df_as_delta(weather_weekly, DBI_WEATHER_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_WEATHER_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

## Write as delta table to dbfs - MONTHLY
## DBI_WEATHER_MONTHLY - included in ./config file

save_df_as_delta(weather_monthly, DBI_WEATHER_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_WEATHER_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Quick Review of Plots
weather_weekly.toPandas().plot(y=['weather_tavg', 'weather_tmin', 'weather_tmax'], kind='line')
plt.show()

# COMMAND ----------

weather_monthly.toPandas().plot(y=['weather_tavg', 'weather_tmin', 'weather_tmax'], kind='line')
plt.show()

# COMMAND ----------

# DBTITLE 1,Graveyard / QC Code
# import time
# time.sleep(1000)
# print('corey')

# COMMAND ----------

