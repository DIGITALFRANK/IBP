# Databricks notebook source
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

# Check path for output to be written out to
print(DBI_MEDIA)

# COMMAND ----------

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

# DBTITLE 1,Load the raw Media and AD Spends data
media_df_raw = spark.read.option("header","true").option("delimiter",";").csv("abfss://landing@cdodevadls2.dfs.core.windows.net/IBP/EDW/MediaOneTruth/Advertising-Media-Spend/ingestion_dt=2021-06-16/")

print(media_df_raw.count())
display(media_df_raw)

# COMMAND ----------

## Retain relevant columns
media_df = media_df_raw.select("MDMIX", "CHNL", "CMPGN_STRT", "CMPGN_END", "CTRY_ISO", "PEP_CTGY", "BRND", "SPNDNG_IN_USD")

print(media_df.count())
display(media_df)

# COMMAND ----------

# DBTITLE 1,Join with calendar to obtain Week_Of_Year
calendar = load_delta(DBI_CALENDAR)
calendar_df = calendar.select("Week_Of_Year", "Week_start_date").distinct()

media_df = media_df.withColumn('CMPGN_STRT', F.col('CMPGN_STRT').cast('date')) \
  .withColumn('CMPGN_END', F.col('CMPGN_END').cast('date'))

media_df = media_df.withColumn('day_of_week', F.dayofweek(F.col('CMPGN_STRT')))
media_df = media_df.selectExpr('*', 'date_sub(CMPGN_STRT, day_of_week-1) as CMPGN_START')

media_df = media_df.withColumn("A", lit(1))

calendar_df = calendar_df.withColumn("A", lit(1))
media_df = media_df.join(calendar_df, on="A", how='inner')
print(media_df.count())
media_df = media_df.filter((col("Week_Start_Date")>=col("CMPGN_START")) & (col("Week_Start_Date")<=col("CMPGN_END"))).drop("A")
calendar_df = calendar_df.drop("A", "CMPGN_STRT")
print(media_df.count())
display(media_df)

# COMMAND ----------

# DBTITLE 1,Pivot the channel types by spends
media_df2 = media_df.withColumn("Channel", when((media_df.MDMIX=="DIGITAL"), lit("DIGITAL")).otherwise(col("CHNL"))).drop("MDMIX", "CHNL")
media_df2 = media_df2.withColumn("SPNDNG_IN_USD",col("SPNDNG_IN_USD").cast('float'))
media_df2 = media_df2.groupBy("Week_Of_Year", "CTRY_ISO", "PEP_CTGY", "BRND").pivot("Channel").sum("SPNDNG_IN_USD")
media_df2 = media_df2.na.fill(0)
print(media_df2.count())
display(media_df2)

# COMMAND ----------

# DBTITLE 1,Clean Column names
media_df2 = media_df2.withColumnRenamed('CTRY_ISO', 'HRCHY_LVL_3_NM')\
.withColumnRenamed('PEP_CTGY', 'SRC_CTGY_1_NM')\
.withColumnRenamed('BRND', 'BRND_NM')

media_df2 = media_df2.withColumn("SRC_CTGY_1_NM", upper(col("SRC_CTGY_1_NM")))

new_names = media_df2.columns
new_names = [re.sub('[^A-Za-z0-9_]+', '', s) for s in new_names]
new_names = [re.sub(' ', '', s) for s in new_names]
new_names = [re.sub(',', '', s) for s in new_names]
new_names = [re.sub('=', '', s) for s in new_names]

media_df2 = media_df2.toDF(*new_names)

# Modfiy brand names
media_df2 = media_df2.withColumn('BRND_NM', regexp_replace('BRND_NM', 'PEPSI ZERO/MAX', 'PEPSI'))\
.withColumn('BRND_NM', regexp_replace('BRND_NM', 'DIET 7UP', '7UP'))

display(media_df2)

# COMMAND ----------

# Spend Validation
print("Pre processing total spend:", media_df.select(F.sum("SPNDNG_IN_USD")).collect()[0][0])
print("Post processing total spend:", media_df2.select(F.sum("DIGITAL")+F.sum("OOHDigital")+F.sum("OOHSpecial")+F.sum("OOHTraditional")+F.sum("OtherCinema")+F.sum("OtherPrint")+F.sum("OtherProduction")+F.sum("OtherRadio")+F.sum("OtherSponsorship")+F.sum("TVTraditional")).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Aggregate up spends
media_df2 = media_df2.groupBy(MEDIA_MERGE_FIELD).agg(*[sum(c).alias(c) for c in media_df2.columns if c not in MEDIA_MERGE_FIELD]) 

# COMMAND ----------

# DBTITLE 1,Replicating 2021 data for 2022
media_2022 = media_df2.join(calendar.select("Week_Of_Year", "Year").distinct(), on = "Week_Of_Year", how = "left").filter(col("Year") == 2021).drop("Year")
media_2022 = media_2022.withColumn("Week_Of_Year", col("Week_Of_Year") + lit(100))
media_2022 = media_2022.select(media_df2.columns)

media_df3 = media_df2.union(media_2022)

# COMMAND ----------

print(media_df3.count() == media_df2.count() + media_2022.count())
print(media_2022.columns == media_df2.columns == media_df3.columns)

# COMMAND ----------

# Prefix for Driver Categorization 
COLS_FOR_CATEGORIZATION = subtract_two_lists(media_df3.columns, ['HRCHY_LVL_3_NM', 'SRC_CTGY_1_NM', 'BRND_NM', 'Week_Of_Year'])
for i in COLS_FOR_CATEGORIZATION:
  if i.startswith("media_")==False:
    media_df3 = media_df3.withColumnRenamed(i, "media_"+i)
display(media_df3)

# COMMAND ----------

## Write as delta table to dbfs
# Unique at Week_Of_Year, HRCHY_LVL_3_NM, SRC_CTGY_1_NM, BRND_NM level 
save_df_as_delta(media_df3, DBI_MEDIA, enforce_schema=False)
delta_info = load_delta_info(DBI_MEDIA)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

