# Databricks notebook source
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql import functions as F
from pyspark.sql import Row
import itertools
import re

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("dependentDatasetPath", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("stgAccount", "")
dbutils.widgets.text("isHistory", "")
dbutils.widgets.text("archivePath", "")
dbutils.widgets.text("archiveContainer", "")
dbutils.widgets.text("sourceFileFormat", "")
dbutils.widgets.text("sourceFileDelimiter", "")
dbutils.widgets.text("tbl_name", "")

# COMMAND ----------

#storing the parameters in variables
stg_Account = dbutils.widgets.get("stgAccount")
srcFormat = dbutils.widgets.get("sourceFileFormat")
srcDelimiter = dbutils.widgets.get("sourceFileDelimiter")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stg_Account+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stg_Account+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
archPath = "abfss://"+dbutils.widgets.get("archiveContainer")+"@"+stg_Account+".dfs.core.windows.net/"+dbutils.widgets.get("archivePath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
dpdPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stg_Account+".dfs.core.windows.net/"+dbutils.widgets.get("dependentDatasetPath")
tbl_name = dbutils.widgets.get("tbl_name")
tbl_name

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$stgAccount

# COMMAND ----------

dbutils.fs.ls(srcPath)

# COMMAND ----------

import re

def cleanColumn(tmpdf):
  cols_list = tmpdf.schema.names
  regex = r"[A-Za-z0-9\s/_-]"
  new_cols = []
  for col in cols_list:
    matches = re.finditer(regex, col, re.MULTILINE)
    name = []
    for matchNum, match in enumerate(matches, start=1):
      name.append(match.group())
      nn = "".join(name).replace(" ","_").replace("/","_")
      nn = nn.replace("__","_")
    tmpdf = tmpdf.withColumnRenamed(col, nn)
  return tmpdf

# COMMAND ----------

# landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_01-23-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_01-22-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_01-31-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_02-02-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_03-01-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_03-22-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_03-23-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_03-28-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_03-31-2020;landing/iberia/external/ibp/covid-19-prevalence-history/covid-19-prevalence_05-29-2020

# COMMAND ----------

srcPath_list =[x.path for x in dbutils.fs.ls(srcPath)]
for path in srcPath_list:
  print(path)
  if '_01-22-2020' in path:
    df_batch0 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
    #df_batch0.withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit("")).withColumn("",lit(""))
  if '_01-23-2020' in path:
    df_batch1 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_01-31-2020' in path:
    df_batch2 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_02-02-2020' in path:
    df_batch3 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_03-01-2020' in path:
    df_batch4 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_03-22-2020' in path:
    df_batch5 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_03-23-2020' in path:
    df_batch6 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_03-28-2020' in path:
    df_batch7 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_03-31-2020' in path:
    df_batch8 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_04-02-2020' in path:
    df_batch9 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_04-03-2020' in path:
    df_batch8 = df_batch8.union(cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)))
  if '_04-04-2020' in path:
    df_batch9 = df_batch9.union(cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)))
  if '_04-05-2020' in path:
    df_batch8 = df_batch8.union(cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)))
  if '_04-06-2020' in path:
    df_batch9 = df_batch9.union(cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)))
  if '_04-07-2020' in path:
    df_batch10 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  if '_05-29-2020' in path:
    df_batch11 = cleanColumn(spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path))
  

# COMMAND ----------

df_batch0 = df_batch0.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yyyy H:mm")," yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch1 = df_batch1.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch2 = df_batch2.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yyyy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch3 = df_batch3.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"yyyy-MM-dd'T'HH:mm:ss"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Lat",lit(""))\
.withColumn("Long_",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch4 = df_batch4.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"yyyy-MM-dd'T'HH:mm:ss"),"yyyy-MM-dd HH:mm:ss"))\
.withColumnRenamed("Latitude","Lat")\
.withColumnRenamed("Longitude","Long_")\
.withColumn("FIPS",lit(""))\
.withColumn("Admin2",lit(""))\
.withColumn("Active",lit(""))\
.withColumn("Combined_Key",lit(""))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch5 = df_batch5.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch6 = df_batch6.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch7 = df_batch7.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch8 = df_batch8.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch9 = df_batch9.withColumn("Last_Update", date_format(to_timestamp(col("Last_Update"),"M/d/yy H:mm"),"yyyy-MM-dd HH:mm:ss"))\
.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")

df_batch10 = df_batch10.withColumn("Incident_Rate",lit(""))\
.withColumn("Case_Fatality_Ratio",lit(""))\
.select("FIPS","Admin2","Province_State","Country_Region","Last_Update","Lat","Long_","Confirmed","Deaths","Recovered","Active","Combined_Key","Incident_Rate","Case_Fatality_Ratio")


# COMMAND ----------

final_df = df_batch0.union(df_batch1.union(df_batch2.union(df_batch3.union(df_batch4.union(df_batch5.union(df_batch6.union(df_batch7.union(df_batch8.union(df_batch9.union(df_batch10.union(df_batch11)))))))))))

# COMMAND ----------

final_df = final_df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

final_df.display()

# COMMAND ----------

#source record count
recCount = final_df.count()

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)):
  deltaTable = DeltaTable.forPath(spark, tgtPath)

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
  #deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = final_df.alias("updates"),
    condition = merge_cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  final_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
  final_df.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  final_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

if (DeltaTable.isDeltaTable(spark, tgtPath)):
  deltaTable = DeltaTable.forPath(spark, tgtPath)

# COMMAND ----------

tb_name = "sc_ibp_bronze.`"+tbl_name+"`"
DeltaTable.createIfNotExists(spark) \
    .tableName(tb_name) \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

#dbutils.notebook.exit("")

# COMMAND ----------

 #path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+srcPath
dbutils.fs.mv(srcPath, archPath,recurse=True)

# COMMAND ----------

# remove versions which are older than 336 hrs(30 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(720)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(recCount))