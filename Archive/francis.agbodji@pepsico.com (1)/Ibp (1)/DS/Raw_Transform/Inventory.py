# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

## Establish connection to storage
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

# COMMAND ----------

## Check path for writing out the dataset to be created
print(DBI_INVENTORY)

# COMMAND ----------

## Function to read in files
def read_file(file_location, delim = ";"):
  return (spark.read.option("header","true").option("delimiter", delim).csv(file_location))

# COMMAND ----------

## Create Daily calendar
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
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+F.month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .select('cal_date','Week_start_date','Week_Of_Year','Month_Of_Year_WSD').distinct()

display(calendar_df)

# COMMAND ----------

## Read in the Supply Planning Data
dpi_adhoc_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/Adhoc/onetimefull-load/ibp-bronze/IBP/EDW/Prolink/SUPPLY PLANNING/"
raw_dpi_adhoc = read_file(dpi_adhoc_path)
display(raw_dpi_adhoc)
raw_dpi_adhoc.count()

# COMMAND ----------

## Trim spaces from MTRL_ID, LOC_ID
raw_dpi_adhoc = raw_dpi_adhoc.withColumn("MTRL_ID", trim(col("MTRL_ID"))) \
                             .withColumn("LOC_ID", trim(col("LOC_ID")))

# COMMAND ----------

## Retain relevant columns
qty_cols = ['PRJCTD_INVEN_QTY', 'PRJCTD_OH_SHRT_DMND_QTY']
cols_to_keep = ["MTRL_ID", "LOC_ID", "SPLY_PLANG_PRJCTN_GNRTN_DT", "SPLY_PLANG_PRJCTN_STRT_DT"] + qty_cols

dpi_adhoc = raw_dpi_adhoc.select(*cols_to_keep)

split_gnrtn_date = pyspark.sql.functions.split(dpi_adhoc["SPLY_PLANG_PRJCTN_GNRTN_DT"], ' ')
dpi_adhoc = dpi_adhoc.withColumn("gnrtn_date", split_gnrtn_date[0]).withColumn('gnrtn_date', F.col('gnrtn_date').cast('date')).drop("SPLY_PLANG_PRJCTN_GNRTN_DT")

split_start_date = pyspark.sql.functions.split(dpi_adhoc["SPLY_PLANG_PRJCTN_STRT_DT"], ' ')
dpi_adhoc = dpi_adhoc.withColumn("start_date", split_start_date[0]).withColumn('start_date', F.col('start_date').cast('date')).drop("SPLY_PLANG_PRJCTN_STRT_DT")
dpi_adhoc.count()

# COMMAND ----------

## Get Week_Of_Year, Month_Of_Year
dpi_adhoc_df = dpi_adhoc.join(calendar_df.withColumnRenamed("cal_date", "start_date"), on = "start_date", how = "left")
dpi_adhoc_df = dpi_adhoc_df.withColumnRenamed("Month_Of_Year_WSD", "Month_Of_Year")

dpi_adhoc_df = dpi_adhoc_df.withColumn("Day_Of_Week", datediff(col("start_date"), col("Week_start_date")) + 1).drop("Week_start_date")

dpi_adhoc_df.count()

# COMMAND ----------

## Subsetting for actual inventory
actual_inv = dpi_adhoc_df.filter(col("gnrtn_date") == col("start_date")).drop("PRJCTD_OH_SHRT_DMND_QTY").withColumnRenamed("PRJCTD_INVEN_QTY", "actual_inventory")
actual_inv = actual_inv.drop("start_date", "gnrtn_date").distinct()

inv_level = ["MTRL_ID", "LOC_ID", "Week_Of_Year"]
actual_inv = actual_inv.withColumn('max_day_of_week', F.max('Day_Of_Week').over(Window.partitionBy(inv_level)))
actual_inv = actual_inv.filter(col('max_day_of_week') == col('Day_Of_Week')).drop("Day_Of_Week", "max_day_of_week")

display(actual_inv)
print(actual_inv.select("MTRL_ID", "LOC_ID", "Week_Of_Year").distinct().count() == actual_inv.distinct().count() == actual_inv.count())
actual_inv.count()

# COMMAND ----------

## Get the max of gnrtn date at each "MTRL_ID", "LOC_ID", "Week_Of_Year" level
max_gnrtn_date = dpi_adhoc_df.groupBy("MTRL_ID", "LOC_ID", "Week_Of_Year").agg(max("gnrtn_date").alias("gnrtn_date"))
proj_df = dpi_adhoc_df.drop("start_date", "PRJCTD_INVEN_QTY").distinct()

## Subsetting for projected inventory
proj_inv = proj_df.join(max_gnrtn_date, on = max_gnrtn_date.columns, how = "inner").withColumnRenamed("PRJCTD_OH_SHRT_DMND_QTY", "projected_inventory")
proj_inv = proj_inv.drop("gnrtn_date").distinct()

inv_level = ["MTRL_ID", "LOC_ID", "Week_Of_Year"]
proj_inv = proj_inv.withColumn('max_day_of_week', F.max('Day_Of_Week').over(Window.partitionBy(inv_level)))
proj_inv = proj_inv.filter(col('max_day_of_week') == col('Day_Of_Week')).drop("Day_Of_Week", "max_day_of_week")

display(proj_inv)
print(proj_inv.select("MTRL_ID", "LOC_ID", "Week_Of_Year").distinct().count() == proj_inv.count())
proj_inv.count()

# COMMAND ----------

## Join actual and projected inventories
common_cols = subtract_two_lists(proj_inv.columns, ["projected_inventory"])
all_inv_rows = actual_inv.select(common_cols).union(proj_inv.select(common_cols)).distinct()
print(all_inv_rows.count())

dpi_adhoc_final = all_inv_rows.join(actual_inv, on = common_cols, how = "left").join(proj_inv, on = common_cols, how = "left")
dpi_adhoc_final = dpi_adhoc_final.withColumn("actual_inventory", dpi_adhoc_final["actual_inventory"].cast(IntegerType())) \
                                 .withColumn("projected_inventory", dpi_adhoc_final["projected_inventory"].cast(IntegerType()))

dpi_adhoc_final = dpi_adhoc_final.withColumn("actual_inventory",when(col("actual_inventory") < 0, lit(0)).otherwise(col("actual_inventory"))) \
                                 .withColumn("projected_inventory",when(col("projected_inventory") < 0, lit(0)).otherwise(col("projected_inventory")))
dpi_adhoc_final = dpi_adhoc_final.na.fill(value=0, subset=["actual_inventory", "projected_inventory"])

display(dpi_adhoc_final)
print(dpi_adhoc_final.select("MTRL_ID", "LOC_ID", "Week_Of_Year").distinct().count() == dpi_adhoc_final.count() == dpi_adhoc_final.distinct().count())
dpi_adhoc_final.count()

# COMMAND ----------

print(dpi_adhoc_final.select("MTRL_ID", "LOC_ID", "Week_Of_Year").distinct().count() == dpi_adhoc_final.count() == dpi_adhoc_final.distinct().count())

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(dpi_adhoc_final, DBI_INVENTORY, enforce_schema=False)
delta_info = load_delta_info(DBI_INVENTORY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

