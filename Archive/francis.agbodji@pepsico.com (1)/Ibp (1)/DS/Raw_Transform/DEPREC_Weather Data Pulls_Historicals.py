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
print(DBI_WEATHER_WEEKLY_HISTORICALS)
print(DBI_WEATHER_MONTHLY_HISTORICALS)

# COMMAND ----------

# DBTITLE 1,Import General Libraries
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
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

## display(calendar_df.orderBy(col('cal_date').desc()))

# COMMAND ----------

# DBTITLE 1,Data Set Up for (Weather) Station-Based Pulls
stations = Stations()

stations_spain = stations.region('ES').fetch().reset_index()
print('SPAIN: {}'.format(stations_spain.shape))

stations_port = stations.region('PT').fetch().reset_index()
print('PORTUGAL: {}'.format(stations_port.shape))

# COMMAND ----------

## Setting up list so multiple weather stations can be pulled at once (in one command)
## Keeping ES and PT separate for now, to enable a rollup by country

spain_stations_list = list(set(stations_spain['id'].unique()))  ## list of Spain weather station IDs
port_stations_list = list(set(stations_port['id'].unique()))    ## list of Portugal weather station IDs

# COMMAND ----------

start_dt = datetime(2015, 1, 1)             ## should not need any earlier than this - setting deep for HISTORICAL details
end_dt = pd.Timestamp('today').floor('D')   ## automatically pulls based on code rundate

print('Weather Pull Start: {}'.format(start_dt))
print('Weather Pull End: {}'.format(end_dt))

# end_dt = datetime(2021, 7, 1)  ## type mismatch
# end_dt = date.today()          ## manual approach

# COMMAND ----------

# MAGIC %md ### Pull Daily Data and Create Aggregated Weekly and Monthly Datasets

# COMMAND ----------

## Note - some weather stations will show an issue with loading ('cannot load ...')
## This should not be an issue, as long as it does not represent a significant % of country's stations

es_data = Daily(spain_stations_list, start=start_dt, end=end_dt)  ## to pull data by Day
es_data = es_data.normalize()  ## to auto-correct for sensor gaps
es_data = es_data.interpolate(limit=3)  ## must normalize before this!
es_data = es_data.convert(units.imperial)  ## to convert units
es_data = es_data.fetch().reset_index()  ## fetch + reset to give typical Pandas df

## Note - below was removed that would auto-aggregate to week across all stations
## This was done intentionally to better map to our calendar
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
## This was done intentionally to better map to our calendar
## pt_data = pt_data.aggregate(freq='1W', spatial=True)

## Review pulled dataset
pt_data.describe()

# COMMAND ----------

es_data['country'] = 'ES'
pt_data['country'] = 'PT'

combined_data = es_data.append(pt_data, ignore_index=True)                         ## 'stack' dataframes together
combined_data = combined_data.loc[:, combined_data.apply(pd.Series.nunique) != 1]  ## remove any cols with all same value
combined_data.head()

# COMMAND ----------

# es_data.time.nunique(), pt_data.time.nunique(), combined_data.time.nunique()
# es_data.time.min(), pt_data.time.min(), combined_data.time.min()
# es_data.time.max(), pt_data.time.max(), combined_data.time.max()
# combined_data.shape

# COMMAND ----------

## Check nulls
print(combined_data.isna().sum())
combined_data.fillna(0, inplace=True)
print('Corrected NaN Totals: {}'.format(combined_data.isna().sum().sum()))

# COMMAND ----------

## Checking for cols with all zero values
zero_cols = [c for c in combined_data.columns if combined_data[c].nunique() == 1 and combined_data[c].unique()[0] == 0]
cols_to_drop = ["station"] + zero_cols
cols_to_drop

# COMMAND ----------

## Generating PySPark dataframe at daily level
daily_weather = spark.createDataFrame(combined_data) 
daily_weather = daily_weather.withColumn('time', F.col('time').cast('date')).withColumnRenamed("time", "cal_date").drop(*cols_to_drop)
daily_weather = daily_weather.withColumn('country_name', F.when(F.col("country") == 'ES', "Spain").otherwise("Portugal")).drop('country')

# COMMAND ----------

## Join on the calendar created at the outset
## Then provides a simple 'Week_Of_Year' and 'Month_Of_Year' mapping

print(daily_weather.count())
daily_weather = daily_weather.join(calendar_df, on="cal_date", how="left").drop("cal_date")
print(daily_weather.count())

# COMMAND ----------

## Creating aggregation dictionary
mean_cols = list(set(daily_weather.columns) - set(["country_name", "Week_Of_Year", "Month_Of_Year"]))
mean_dict = {x: 'mean' for x in mean_cols}

## Aggregated using 'custom' function
weather_weekly = timely_agg(daily_weather, mean_dict, mean_cols, country_col="country_name", time_var="Week_Of_Year")
weather_monthly = timely_agg(daily_weather, mean_dict, mean_cols, country_col="country_name", time_var="Month_Of_Year")

for col_name in mean_cols:
  weather_weekly = weather_weekly.withColumn(col_name, round(col_name, 3))
  weather_monthly = weather_monthly.withColumn(col_name, round(col_name, 3))

# COMMAND ----------

## QC on aggregation approach
print(weather_weekly.count() == weather_weekly.distinct().count() == weather_weekly.select("country_name", "Week_Of_Year").distinct().count())
print(weather_monthly.count() == weather_monthly.distinct().count() == weather_monthly.select("country_name", "Month_Of_Year").distinct().count())

# COMMAND ----------

display(weather_weekly)
weather_weekly.dtypes

# COMMAND ----------

display(weather_monthly)
weather_monthly.dtypes

# COMMAND ----------

# DBTITLE 1,Now Aggregating by Week vs Month (no Year) - for Historical Reference
weather_weekly_hist = weather_weekly.withColumn('Week_Only', substring('Week_Of_Year', 5, 6))\
                                    .withColumn('Year_Only', substring('Week_Of_Year', 1, 4))
  
display(weather_weekly_hist)

# COMMAND ----------

weather_monthly_hist = weather_monthly.withColumn('Month_Only', substring('Month_Of_Year', 5, 6))\
                                      .withColumn('Year_Only', substring('Month_Of_Year', 1, 4))
  
display(weather_monthly_hist)

# COMMAND ----------

## Creating aggregation dictionary
mean_cols_week = list(set(weather_weekly_hist.columns) - set(["country_name", "Week_Of_Year", "Year_Only", "Week_Only"]))
mean_cols_month = list(set(weather_monthly_hist.columns) - set(["country_name", "Month_Of_Year", "Year_Only", "Month_Only"]))

mean_dict_week = {x: 'mean' for x in mean_cols_week}
mean_dict_month = {x: 'mean' for x in mean_cols_month}

## Aggregated using 'custom' function
weather_weekly_hist = timely_agg(weather_weekly_hist, mean_dict_week, mean_cols_week, country_col="country_name", time_var="Week_Only")
weather_monthly_hist = timely_agg(weather_monthly_hist, mean_dict_month, mean_cols_month, country_col="country_name", time_var="Month_Only")

for col_name in mean_cols_week: weather_weekly_hist = weather_weekly_hist.withColumn(col_name, round(col_name, 3))
for col_name in mean_cols_month: weather_monthly_hist = weather_monthly_hist.withColumn(col_name, round(col_name, 3))

# COMMAND ----------

display(weather_weekly_hist)

# COMMAND ----------

display(weather_monthly_hist)

# COMMAND ----------

delta_info = load_delta_info(DBI_WEATHER_WEEKLY_HISTORICALS)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

weather_weekly_hist = load_delta(DBI_WEATHER_WEEKLY_HISTORICALS)
display(weather_weekly_hist)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(weather_weekly_hist.columns, ['country_name', 'Week_start_date', 'FCST_START_DATE'])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("weather_")==False:
    weather_weekly_hist = weather_weekly_hist.withColumnRenamed(i, "weather_"+i)
display(weather_weekly_hist)

# COMMAND ----------

## Write as delta table to dbfs - HISTORICAL WEEKLY
## DBI_WEATHER_WEEKLY_HISTORICALS - included in ./config file

save_df_as_delta(weather_weekly_hist, DBI_WEATHER_WEEKLY_HISTORICALS, enforce_schema=False)
delta_info = load_delta_info(DBI_WEATHER_WEEKLY_HISTORICALS)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(weather_monthly_hist.columns, ['country_name', 'Month_Only'])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("weather_")==False:
    weather_monthly_hist = weather_monthly_hist.withColumnRenamed(i, "weather_"+i)
display(weather_monthly_hist)

# COMMAND ----------

## Write as delta table to dbfs - HISTORICAL MONTHLY
## DBI_WEATHER_MONTHLY_HISTORICALS - included in ./config file

save_df_as_delta(weather_monthly_hist, DBI_WEATHER_MONTHLY_HISTORICALS, enforce_schema=False)
delta_info = load_delta_info(DBI_WEATHER_MONTHLY_HISTORICALS)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# DBTITLE 1,Review of Plots
weather_weekly.toPandas().plot(y=['tavg', 'tmin', 'tmax'], kind='line')
plt.show()

weather_monthly.toPandas().plot(y=['tavg', 'tmin', 'tmax'], kind='line')
plt.show()

# COMMAND ----------

weather_weekly_hist_pd = weather_weekly_hist.toPandas()
weather_monthly_hist_pd = weather_monthly_hist.toPandas()

# COMMAND ----------

# weather_weekly_hist_pd.head()
# weather_monthly_hist_pd.head()

# COMMAND ----------

plt.figure(figsize=(8,4))
sns.lineplot(x='Week_Only', y='tavg', hue='country_name', data=weather_weekly_hist_pd)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,4))
sns.lineplot(x='Week_Only', y='prcp', hue='country_name', data=weather_weekly_hist_pd)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,4))
sns.lineplot(x='Month_Only', y='tavg', hue='country_name', data=weather_monthly_hist_pd)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,4))
sns.lineplot(x='Month_Only', y='prcp', hue='country_name', data=weather_monthly_hist_pd)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

plt.figure(figsize=(8,4))
sns.lineplot(x='Month_Only', y='snow', hue='country_name', data=weather_monthly_hist_pd)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# DBTITLE 1,Graveyard / QC Code
# cnp = load_delta(DBI_WEATHER_MONTHLY)
# display(cnp)

# COMMAND ----------

