# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
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

#storing the parameters in variables
source_stgAccnt = dbutils.widgets.get("sourceStorageAccount")
target_stgAccnt = dbutils.widgets.get("targetStorageAccount")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
dpndntdatapath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+target_stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#Reading the data from the bronze path of DFU table
adv_deltaTable = DeltaTable.forPath(spark, srcPath)
adv_latest_version = adv_deltaTable.history().select(max(col('version'))).collect()[0][0]
#print(adv_latest_version)
#display(adv_deltaTable.history())

# COMMAND ----------

#Reading the adv&media source data from bonze layer
adv_df = spark.read.format("delta").option("versionAsOf", adv_latest_version).load(srcPath).drop("Dup_Check").select("MDMIX", "CHNL", "CMPGN_STRT", "CMPGN_END", "CTRY_ISO", "PEP_CTGY", "BRND", "SPNDNG_IN_USD")

# COMMAND ----------

#Creating Col List
colList = ["MDMIX", "CHNL", "CMPGN_STRT", "CMPGN_END", "CTRY_ISO", "PEP_CTGY", "BRND", "SPNDNG_IN_USD"]

# COMMAND ----------

#Replicating rows for Brand KAS into "KAS REFRESCOS" and "KASFRUIT"
adv_df_kas = adv_df.where(col('BRND')== "KAS")
adv_df_kas_v1 = adv_df_kas.withColumn('BRND',lit("KAS REFRESCOS"))
adv_df_kas_v2 = adv_df_kas.withColumn('BRND',lit("KASFRUIT"))
adv_df_v3 = (adv_df.where(col('BRND') != "KAS").select(colList)).union(adv_df_kas_v1.select(colList)).union(adv_df_kas_v2.select(colList))

# COMMAND ----------

#Substituting Brand name with new Value and renaming column
adv_df_final = adv_df_v3.withColumn("BRND",when(col("BRND") == "PEPSI ZERO/MAX",lit("PEPSI")) \
                                          .when(col("BRND")== "DIET 7UP",lit("7UP")).otherwise(col("BRND"))) \
                        .withColumnRenamed("PEP_CTGY","CTGY")

# COMMAND ----------

#Getting Start and End Date
start_date_val = adv_df_final.select(min(col("CMPGN_STRT")).cast('DATE').cast('STRING').alias("start_date_val")).first()[0]
end_date_val = adv_df_final.select(max(col("CMPGN_END")).cast('DATE').cast('STRING').alias("start_date_val")).first()[0]

# COMMAND ----------

print(start_date_val)
print(end_date_val)

# COMMAND ----------

# Creating Calendare DataFrame
df = spark.sparkContext.parallelize([Row(start_date=start_date_val, end_date=end_date_val)]).toDF()

df = df \
  .withColumn('start_date', col('start_date').cast('date')) \
  .withColumn('end_date', col('end_date').cast('date'))\
  .withColumn('cal_date', explode(expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
  .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
  .withColumn("Week_end_date",date_add("Week_start_date",6))\
  .withColumn('week_year',when((year(col('Week_start_date'))==year(col('cal_date'))) &          (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
              .otherwise(year('Week_end_date')))\
  .withColumn('month_year',year(col('cal_date')))\
  .withColumn('week',when((year(col('Week_start_date'))==year(col('Week_end_date'))),weekofyear(col("Week_end_date")))\
                     .otherwise(weekofyear(col("Week_end_date"))))\
  .withColumn('month',month("cal_date"))

df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .drop('start_date','end_date','week','month','year','Week_year')

calendar=df.groupBy("Month_Of_Year_WSD","Month_Of_Year","Week_Of_Year","Week_start_date","Week_end_date").agg(sum("flag").alias("Day_count"))\
                    .withColumn("month_ratio",col("Day_count")/lit(7))\
                    .withColumn("week_ratio",lit(1))
calendar_df = calendar.select("Week_Of_Year", "Week_start_date").withColumn("A", lit(1)).distinct()

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date(not needed as of now)
#max_value = adv_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
#print(max_value)
#adv_df2 = adv_df.filter(col("PROCESS_DATE")==max_value)
#display(stft_df2)

# COMMAND ----------

media_df = adv_df_final.withColumn('CMPGN_STRT',col('CMPGN_STRT').cast('date')) \
  .withColumn('CMPGN_END',col('CMPGN_END').cast('date'))

media_df_v1 = media_df.withColumn('day_of_week',dayofweek(col('CMPGN_STRT')))
media_df_v2 = media_df_v1.selectExpr('*', 'date_sub(CMPGN_STRT, day_of_week-1) as CMPGN_START') \
                         .withColumn("A", lit(1))

# COMMAND ----------

# Cross Join with Calendar in order to capture all the week and filtering data for valid week
media_df_cross = media_df_v2.join(calendar_df, on="A", how='inner')
media_filtered_df = media_df_cross.filter((col("Week_Start_Date")>=col("CMPGN_START")) & (col("Week_Start_Date")<=col("CMPGN_END")))

# COMMAND ----------

media_df_chnl= media_filtered_df.withColumn("Channel", when((media_filtered_df.MDMIX=="DIGITAL"), lit("DIGITAL")).otherwise(col("CHNL"))) \
                                .withColumn("SPNDNG_IN_USD",col("SPNDNG_IN_USD").cast('float')) \
                                .drop("MDMIX", "CHNL")

# COMMAND ----------

#Pivotting the data at Brand Level
media_df_pivot = media_df_chnl.groupBy("Week_Of_Year", "CTRY_ISO", "CTGY", "BRND").pivot("Channel").sum("SPNDNG_IN_USD")
media_df_final = media_df_pivot.na.fill(0) \
                          .withColumn("CTGY", upper(col("CTGY")))

# COMMAND ----------

# Renaming Column name in order to remove special character
import re
new_names = media_df_final.columns
new_names = [re.sub('[^A-Za-z0-9_]+', '', s) for s in new_names]
new_names = [re.sub(' ', '', s) for s in new_names]
new_names = [re.sub(',', '', s) for s in new_names]
new_names = [re.sub('=', '', s) for s in new_names]

silver_df=media_df_final.toDF(*new_names)
silver_df=silver_df.select(
silver_df.Week_Of_Year,
silver_df.CTRY_ISO,
silver_df.CTGY,
silver_df.BRND,
silver_df.OOHAmbient.alias('media_OOHAmbient'),
silver_df.TVTraditional.alias('media_TVTraditional'),
silver_df.OtherSponsorship.alias('media_OtherSponsorship'),
silver_df.OtherRadio.alias('media_OtherRadio'),
silver_df.OtherProduction.alias('media_OtherProduction'),
silver_df.OtherPrint.alias('media_OtherPrint'),
silver_df.OtherCinema.alias('media_OtherCinema'),
silver_df.OOHTraditional.alias('media_OOHTraditional'),
silver_df.OOHSpecial.alias('media_OOHSpecial'),
silver_df.OOHDigital.alias('media_OOHDigital'),
silver_df.DIGITAL.alias('media_DIGITAL'))
display(silver_df)

# COMMAND ----------

## DQ Check
#from pyspark import StorageLevel
#pklist2 = ['CMPGN_ID','WC_DT','CHNL','PLCMT_NAME','CMPGN_GNRTN','MEDA_TYP_NM','CHNL_DTL','FRMT','PLANG_TRGT','CHNL_DESC','FRMT_CATG','SPNSRSHP_TYP','CMPGN_TRGT_RGN_VAL','ACTL_PLAND']
#pklist2 = ','.join(str(e) for e in pklist2)
#if len(pklist2.split(','))>1:
  #ls = ["col('"+attr+"').isNull()" for attr in pklist2.split(',')]
  #null_cond = " | ".join(ls)
#else :
  #null_cond = "col('"+pklist2+"').isNull()"

#adv_dq = adv_df_agg.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pklist2.split(','))
                    #.orderBy(desc("DRVD_SPEND_USD")))).withColumn("Corrupt_Record",#when(eval(null_cond),lit("Primary Key is Null"))
                                                            #when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key"))).persist(StorageLevel.MEMORY_AND_DISK)
#adv_dq_pass = adv_dq.where(col("Corrupt_Record").isNull()).drop("DUP_CHECK","Corrupt_Record")

# COMMAND ----------

#Writing data innto silver delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = silver_df.alias("updates"),
    condition = cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  silver_df.write\
  .format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  silver_df.write\
  .format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

#Reading the DFU source data from bonze layer
# silverdfff = spark.read.format("delta").load(tgtPath)
# print(silverdfff.count())