# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *

# COMMAND ----------

dbutils.widgets.text("stgAccount", "")
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("archivePath", "")
dbutils.widgets.text("archiveContainer", "")
dbutils.widgets.text("sourceFileFormat", "")
dbutils.widgets.text("sourceFileDelimiter", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("primaryKeyList", "")
dbutils.widgets.text("loadType", "")
dbutils.widgets.text("isHistory", "")

# COMMAND ----------

# Defining required variables
stgAccnt = dbutils.widgets.get("stgAccount")
srcFormat = dbutils.widgets.get("sourceFileFormat")
srcDelimiter = dbutils.widgets.get("sourceFileDelimiter")
srcPath = dbutils.widgets.get("sourcePath")
#srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
archPath = "abfss://"+dbutils.widgets.get("archiveContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("archivePath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
isHistory = dbutils.widgets.get("isHistory")
tgtPath

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$stgAccount

# COMMAND ----------

#Forming the Condition using Primary key list to uniquely identifying the list
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
  
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

if len(srcPath.split(';'))>1 or isHistory == 'True':
  if isHistory == 'True':
    srcPath_list = [x.path for x in dbutils.fs.ls("abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+srcPath)]
  else:
    srcPath_list =["abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+x for x in srcPath.split(';')]
  df = spark.createDataFrame([], StructType([]))
  cnt = 0
  for path in srcPath_list:
    if srcFormat == "CSV":
      tempdf = spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)
    elif srcFormat == "parquet":
      tempdf = spark.read.parquet(path)
    if(cnt == 0):
      df = tempdf
      cnt=cnt+1
    else :
      df = df.union(tempdf)
else :
  path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+srcPath
  if srcFormat == "CSV":
    df = spark.read.option("header","true").option("delimiter",srcDelimiter).csv(path)
  elif srcFormat == "parquet":
    df = spark.read.parquet(path)
  

#srcPath_list

# COMMAND ----------

df.count()

# COMMAND ----------

# # Create dataframe from source file
# if srcFormat == "CSV":
#   df = spark.read.option("header","true").option("delimiter",srcDelimiter).csv(srcPath)
# elif srcFormat == "parquet":
#   df = spark.read.parquet(srcPath)

# COMMAND ----------

# Adding Process_date attribute
df = df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

import re

def cleanColumn(tmpdf):
  cols_list = tmpdf.schema.names
  regex = r"[A-Za-z0-9\s_-]"
  new_cols = []
  for col in cols_list:
    matches = re.finditer(regex, col, re.MULTILINE)
    name = []
    for matchNum, match in enumerate(matches, start=1):
      name.append(match.group())
      nn = "".join(name).replace(" ","_")
      nn = nn.replace("__","_")
    tmpdf = tmpdf.withColumnRenamed(col, nn)
  return tmpdf

# COMMAND ----------

# def cleanColumn(tmpdf,colName,findChars,replaceChar):
#   new_col = colName
#   for char in findChars:
#     if char in colName:
#       new_col = new_col.replace(char,replaceChar)      
#   tmpdf = tmpdf.withColumnRenamed(colName, new_col)
#   return tmpdf

# COMMAND ----------

# chars = [' ',',',';','{','}','(',')','\n','\t','=']
# chars


# COMMAND ----------

# allColNames = df.schema.names
# chars = [' ',',',';','{','}','(',')','\n','\t','=','__']
# replaceWith ='_'
# for colName in allColNames:
#   df=cleanColumn(df,colName,chars,replaceWith)

# COMMAND ----------

df = cleanColumn(df)

# COMMAND ----------

#Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge' and pkList != 'NA' :
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = df.alias("updates"),
    condition = cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif loadType == 'overwrite':
  df.write.format("delta")\
  .mode('overwrite')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
else :
  df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)

# COMMAND ----------

#source record count
recCount = df.count()

# COMMAND ----------

#Moving the file to archive path
if len(srcPath.split(';'))>1:
  srcPath_list =["abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+x for x in srcPath.split(';')]
  archPath_list = ["abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+x for x in dbutils.widgets.get("archivePath").split(';')]
  for i in range(len(srcPath_list)):
    dbutils.fs.mv(srcPath_list[i], archPath_list[i],recurse=True)
else:
  path = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+srcPath
  dbutils.fs.mv(path, archPath,recurse=True)

# COMMAND ----------

#Exiting notebook with record count
dbutils.notebook.exit("recordCount : "+str(recCount))

# COMMAND ----------

