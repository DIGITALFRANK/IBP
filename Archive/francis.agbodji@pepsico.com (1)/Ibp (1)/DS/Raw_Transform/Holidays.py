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
print(DBI_HOLIDAYS_WEEKLY)
print(DBI_HOLIDAYS_MONTHLY)

# COMMAND ----------

# daily calendar creation
from pyspark.sql import Row

df = spark.sparkContext.parallelize([Row(start_date='2016-01-01', end_date='2025-12-31')]).toDF()
df = df \
  .withColumn('start_date', F.col('start_date').cast('date')) \
  .withColumn('end_date', F.col('end_date').cast('date'))\
  .withColumn('cal_date', F.explode(F.expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
  .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
  .withColumn("Week_end_date",date_add("Week_start_date",6))\
  .withColumn('week_year',F.when((year(col('Week_start_date'))==year(col('cal_date'))) &          (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
              .otherwise(year('Week_end_date')))\
  .withColumn('month_year',year(col('cal_date')))\
  .withColumn('week',F.when((year(col('Week_start_date'))==year(col('Week_end_date'))),F.weekofyear(col("Week_end_date")))\
                     .otherwise(F.weekofyear(col("Week_end_date"))))\
  .withColumn('month',F.month("cal_date"))

calendar_df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',year(col('Week_start_date'))*lit(100)+F.month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .select('cal_date', 'Week_Of_Year','Month_Of_Year').withColumnRenamed("cal_date", "Holiday_Date")

# COMMAND ----------

raw_holidays = spark.read.format("csv").option("inferSchema", "false").option("header", "true")\
                          .option("sep", ",").load("/FileStore/tables/PEP_Volume_Forecasting/Iberian_Holidays_vF.csv")

es_hols = raw_holidays.filter(col("Country_Name") == "Spain")
pt_hols = raw_holidays.filter(col("Country_Name") == "Portugal")

# COMMAND ----------

es_hols_list = convertDFColumnsToList(es_hols.select("Holiday_Name").distinct(), "Holiday_Name") 
pt_hols_list = convertDFColumnsToList(pt_hols.select("Holiday_Name").distinct(), "Holiday_Name") 

for hol in es_hols_list:
  if "�" in hol:
    print ("ES:", hol)
      
for hol in pt_hols_list:
  if "�" in hol:
    print ("PT:", hol)

# COMMAND ----------

# Clean holidays with incomplete characters
es_hols = es_hols.withColumn('Holiday_Name', regexp_replace('Holiday_Name', 'Fiesta Nacional de Espa�a', 'Fiesta Nacional de Espana'))\
                  .withColumn('Holiday_Name', regexp_replace('Holiday_Name', 'D�a de la Constituci�n Espa�ola', 'Dia de la Constitucion Espanola'))

pt_hols = pt_hols.withColumn('Holiday_Name', regexp_replace('Holiday_Name', 'Implanta��o da Rep�blica', 'Implantacao da Republica'))\
                  .withColumn('Holiday_Name', regexp_replace('Holiday_Name', 'Restaura��o da Independ�ncia', 'Restauracao da Independencia'))

es_hols_list_clean = convertDFColumnsToList(es_hols.select("Holiday_Name").distinct(), "Holiday_Name") 
pt_hols_list_clean = convertDFColumnsToList(pt_hols.select("Holiday_Name").distinct(), "Holiday_Name") 
common_hols = intersect_two_lists(es_hols_list_clean, pt_hols_list_clean)

# COMMAND ----------

# Change holiday names for common ES and PT holidays
for c in common_hols:
  es_hols = es_hols.withColumn('Holiday_Name', F.when(F.col("Holiday_Name") == c, lit(f"es_{c}")).otherwise(F.col("Holiday_Name")))
  pt_hols = pt_hols.withColumn('Holiday_Name', F.when(F.col("Holiday_Name") == c, lit(f"pt_{c}")).otherwise(F.col("Holiday_Name")))

# COMMAND ----------

display(es_hols)

# COMMAND ----------

def get_date_format(df, col_name):  
  
  split_col = pyspark.sql.functions.split(df[col_name], '/')
 
  df = df.withColumn("month", split_col.getItem(0)).withColumn("day", split_col.getItem(1)).withColumn("year", split_col.getItem(2))
  df = df.withColumn("month", F.when(length(F.col("month")) == 1, concat(lit('0'), F.col("month"))).otherwise(F.col("month")))
  df = df.withColumn("day", F.when(length(F.col("day")) == 1, concat(lit('0'), F.col("day"))).otherwise(F.col("day")))
  df = df.withColumn(col_name, concat(F.col("year"), lit('-'), F.col("month"), lit('-'), F.col("day")))
  df = df.withColumn(col_name, to_date(col(col_name),"yyyy-MM-dd")).drop("month", "day", "year")
 
  return df

# COMMAND ----------

def create_holiday_df(country_hols, flag_col, time_var, original_hols_list):
  
  og_count = country_hols.count()
  
  # Get Week_Of_Year, Month_Of_Year columns
  country_hols = get_date_format(country_hols, "Holiday_Date")
  country_hols = country_hols.join(calendar_df, on = "Holiday_Date", how = "left").drop("Holiday_Date", "Country_Code")
  
  # Create dummies of holidays
  country_hols = get_dummies(country_hols, ["Holiday_Name"]).drop("Holiday_Name")
  
  # Clean column names
  country_hols = country_hols.select([F.col(col_name).alias(re.sub("[^0-9a-zA-Z_$]+","", col_name)) for col_name in country_hols.columns])
  print (og_count == country_hols.count())                                 # should be True
  
  hol_cols = [c for c in country_hols.columns if "Holiday_Name_" in c]
  print (len(hol_cols) == len(original_hols_list))                         # should be True

   # Creating holiday flag for country
  hol_flag_exp = '+'.join(hol_cols)
  country_hols = country_hols.withColumn(flag_col, expr(hol_flag_exp))
  
  # Aggregate upto time_var level  
  agg_dict = {x: 'sum' for x in hol_cols + [flag_col]}
  country_hols_agg = country_hols.groupBy(time_var).agg(agg_dict)

  for i in agg_dict.keys():
    country_hols_agg =  country_hols_agg.withColumnRenamed("sum(" + i + ")", i)
    
  return country_hols_agg

# COMMAND ----------

es_hols_weekly = create_holiday_df(es_hols, "spain_hol_flag", "Week_Of_Year", es_hols_list)
es_hols_monthly = create_holiday_df(es_hols, "spain_hol_flag", "Month_Of_Year", es_hols_list)

pt_hols_weekly = create_holiday_df(pt_hols, "portugal_hol_flag", "Week_Of_Year", pt_hols_list)
pt_hols_monthly = create_holiday_df(pt_hols, "portugal_hol_flag", "Month_Of_Year", pt_hols_list)

# COMMAND ----------

hols_weekly = es_hols_weekly.join(pt_hols_weekly, on = "Week_Of_Year", how = "outer")
hols_monthly = es_hols_monthly.join(pt_hols_monthly, on = "Month_Of_Year", how = "outer")

hols_weekly = hols_weekly.na.fill(value=0)
hols_monthly = hols_monthly.na.fill(value=0)

shared_cols_es = [c for c in es_hols_weekly.columns if "Holiday_Name_es_" in c]
shared_cols_pt = [c for c in pt_hols_weekly.columns if "Holiday_Name_pt_" in c]
print(len(shared_cols_es) == len(shared_cols_pt))

# COMMAND ----------

# Merge the common columns of both countries as one
from pyspark.sql.functions import greatest

for c in shared_cols_es:
  hols_weekly = hols_weekly.withColumn(c.replace("_es", ''), greatest(hols_weekly[c], hols_weekly[c.replace("_es", '_pt')]))
  hols_monthly = hols_monthly.withColumn(c.replace("_es", ''), greatest(hols_monthly[c], hols_monthly[c.replace("_es", '_pt')]))
  
hols_weekly = hols_weekly.drop(*shared_cols_es+shared_cols_pt)
hols_monthly = hols_monthly.drop(*shared_cols_es+shared_cols_pt)

# Change back hol columns to holiday names
for c in [c for c in hols_weekly.columns if "Holiday_Name_" in c]:
  hols_weekly = hols_weekly.withColumnRenamed(c, c.replace("Holiday_Name_", ""))
  hols_monthly = hols_monthly.withColumnRenamed(c, c.replace("Holiday_Name_", ""))

# COMMAND ----------

# Rearranging the columns for mrd merging purposes
holiday_cols = list(set(hols_weekly.columns) - set(["Week_Of_Year", "spain_hol_flag", "portugal_hol_flag"]))
hols_weekly = hols_weekly.select(["Week_Of_Year", "spain_hol_flag", "portugal_hol_flag"] + holiday_cols)
hols_monthly = hols_monthly.select(["Month_Of_Year", "spain_hol_flag", "portugal_hol_flag"] + holiday_cols)

# COMMAND ----------

print(hols_weekly.count() == hols_weekly.distinct().count() == hols_weekly.select("Week_Of_Year").distinct().count())
print(hols_monthly.count() == hols_monthly.distinct().count() == hols_monthly.select("Month_Of_Year").distinct().count())
print(len(hols_weekly.columns) - 3 == len(hols_monthly.columns) - 3 == len(list(set(es_hols_list + pt_hols_list))))

# COMMAND ----------

display(hols_weekly)

# COMMAND ----------

display(hols_monthly)

# COMMAND ----------

display(hols_weekly.describe())

# COMMAND ----------

display(hols_monthly.describe())

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(hols_weekly.columns, ["Week_Of_Year"])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("hol_")==False:
    hols_weekly = hols_weekly.withColumnRenamed(i, "hol_"+i)
display(hols_weekly)

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(hols_weekly, DBI_HOLIDAYS_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_HOLIDAYS_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(hols_monthly.columns, ["Month_Of_Year"])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("hol_")==False:
    hols_monthly = hols_monthly.withColumnRenamed(i, "hol_"+i)
display(hols_monthly)

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(hols_monthly, DBI_HOLIDAYS_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_HOLIDAYS_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

old_hols = load_delta(DBI_HOLIDAYS_MONTHLY)
display(old_hols)

# COMMAND ----------

