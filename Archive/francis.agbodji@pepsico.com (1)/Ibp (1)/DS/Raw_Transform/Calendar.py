# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

##Create pro-rated weekly calendar
from pyspark.sql.functions import datediff, to_date, lit
from pyspark.sql.functions import col, trim, lower
from pyspark.sql import functions as F
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

df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+F.month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .drop('start_date','end_date','week','month','year','Week_year')

calendar=df.groupBy("Month_Of_Year_WSD","Month_Of_Year","Week_Of_Year","Week_start_date","Week_end_date").agg(F.sum("flag").alias("Day_count"))\
                    .withColumn("month_ratio",col("Day_count")/lit(7))\
                    .withColumn("week_ratio",lit(1))
display(calendar)

# COMMAND ----------


calendar = calendar.withColumn("Year", F.floor(F.col("Week_Of_Year")/100))\
           .withColumn("Quarter", F.when(F.col("Week_Of_Year")%100 <= lit(52), F.ceil(F.col("Week_Of_Year")%100/13)).otherwise(lit(4))) \
           .withColumn("Quarter_Of_Year", concat_ws("", F.floor(F.col("Week_Of_Year")/100), lit("0"), "Quarter"))
calendar = calendar.withColumn('Month_start_date', F.concat(F.col('Month_Of_Year'), lit('01')))            

calendar = calendar.withColumn('Year',calendar['Year'].cast('int'))\
           .withColumn('Quarter',calendar['Quarter'].cast('int'))\
           .withColumn('Quarter_Of_Year',calendar['Quarter_Of_Year'].cast('int'))\
           .withColumn('Month_Of_Year',calendar['Month_Of_Year'].cast('int'))\
           .withColumn('Month_start_date', to_timestamp(calendar.Month_start_date, "yyyyMMdd"))
calendar.printSchema()
display(calendar)

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(calendar, DBI_CALENDAR, enforce_schema=False)
delta_info = load_delta_info(DBI_CALENDAR)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())


# COMMAND ----------

