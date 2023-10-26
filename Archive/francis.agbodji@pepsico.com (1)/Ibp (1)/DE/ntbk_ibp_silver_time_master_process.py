# Databricks notebook source
#Import
from pyspark.sql import functions as F
from pyspark.sql.functions import *
from pyspark.sql import Row

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("sourceStorageAccount", "")
dbutils.widgets.text("targetStorageAccount", "")

# COMMAND ----------

start_day = "monday"
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
loadType = dbutils.widgets.get("loadType")

no_of_months_prior = 120
no_of_months_post = 120

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$targetStorageAccount

# COMMAND ----------

data=[["1"]]
df = spark.createDataFrame(data,["id"])

#Creating start_date and end_date DF
df = df\
  .withColumn("current_date",current_date()) \
  .withColumn("start_date",add_months("current_date",-no_of_months_prior)) \
  .withColumn("end_date",add_months("current_date",no_of_months_post)) \
  .withColumn('cal_date', F.explode(F.expr('sequence(start_date, end_date, interval 1 day)'))) 

# COMMAND ----------

if start_day == "sunday":
  df = df \
    .withColumn('day_of_week', dayofweek(col('cal_date')))  \
    .withColumn("Week_start_date",to_timestamp(expr("date_sub(cal_date, day_of_week-1)"))) \
    .drop('id','current_date','day_of_week')
else:
  df = df \
    .withColumn("Week_start_date",date_trunc('week', col("cal_date"))) \
    .drop('id','current_date','day_of_week')
print(df.count())
display(df)

# COMMAND ----------

df1 = df.groupBy("Week_start_date").count().filter(col("count") == 7) \
        .withColumnRenamed("Week_start_date","Week_start_date_FD")

df = df.join(df1,df1.Week_start_date_FD == df.Week_start_date,"inner").select("start_date","end_date","cal_date","Week_start_date")

df.display()

# COMMAND ----------

df = df \
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

calendar.write.format("delta").option("overwriteSchema", "true").mode(loadType).save(tgtPath)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(calendar.count()))