# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *
from datetime import date, timedelta
from pyspark.sql.window import Window

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

# COMMAND ----------

# Defining required variables
stgAccnt = dbutils.widgets.get("stgAccount")
srcFormat = dbutils.widgets.get("sourceFileFormat")
srcDelimiter = dbutils.widgets.get("sourceFileDelimiter")
#srcPath = dbutils.widgets.get("sourcePath")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
archPath = "abfss://"+dbutils.widgets.get("archiveContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("archivePath")
pkList = dbutils.widgets.get("primaryKeyList")
loadType = dbutils.widgets.get("loadType")
tgtPath
srcPath

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$stgAccount

# COMMAND ----------

spark.conf.set("fs.azure.sas.supplychain-ibp.cdodevextrblob.blob.core.windows.net","https://cdodevextrblob.blob.core.windows.net/?sv=2020-02-10&ss=bfqt&srt=sco&sp=rwdlacuptfx&se=2022-06-08T21:16:16Z&st=2021-06-09T13:16:16Z&spr=https&sig=ciFKbQI6uFpYKTh%2FOdnd2vw8qAxWUOmSX0LtPvfkt3Y%3D")

# COMMAND ----------

#srcPath = "abfss://landing@cdodevadls2.dfs.core.windows.net/ibp/SyndicatedDataNielsen/syndicate_nelson_portugal_sample.csv"
#srcPath = 'wasbs://supplychain-ibp@cdodevextrblob.blob.core.windows.net/iberia/access-db/syndicated-pos/datepart=/Syndicated_POS_Nielsen PT sin extraction.csv'
#srcPath ='wasbs://supplychain-ibp@cdodevextrblob.blob.core.windows.net/iberia/access-db/syndicated-pos/datepart=/1st round of data - Salty Snacks.csv'
# srcPath = 'wasbs://supplychain-ibp@cdodevextrblob.blob.core.windows.net/iberia/access-db/syndicated-pos/datepart=/'

# COMMAND ----------

import re

def remove_special_chars(cols_list):
  regex = r"[A-Za-z0-9\s-]"
  new_cols = []
  for col in cols_list:
    matches = re.finditer(regex, col, re.MULTILINE)
    name = []
    for matchNum, match in enumerate(matches, start=1):
      name.append(match.group())
      nn = "".join(name).replace(" ","_")
    new_cols.append(nn)
  return new_cols

# COMMAND ----------

def first_row_header(df_new):
  DFSchema = df_new.schema
  list_of_new_column_names = []
  #print(df_new.limit(1).collect()[0][0])
  for i in df_new.limit(1).collect()[0]:
    if list_of_new_column_names.count(i)>0:
      i =i+"_"+str(list_of_new_column_names.count(i))
      while i in list_of_new_column_names:
        if i in list_of_new_column_names:
          i =i.split("_")[0]+"_"+str(int(i.split("_")[1])+1)
        else :
          break
      list_of_new_column_names.append(i)
    else :
      list_of_new_column_names.append(i)
  
  for i,k in enumerate(DFSchema.fields):
    k.name = list_of_new_column_names[i]
    
  data = df_new.rdd
  firstRow = data.first()
  data = data.filter(lambda row:row != firstRow)
  new_df = spark.createDataFrame(data, DFSchema)
  df_new = spark.createDataFrame([], StructType([]))
  return new_df

# COMMAND ----------

def cleanColumn(tmpdf,colName,findChars,replaceChar):
  new_col = colName
  for char in findChars:
    if char in colName:
      new_col = new_col.replace(char,replaceChar)      
  tmpdf = tmpdf.withColumnRenamed(colName, new_col)
  return tmpdf

# COMMAND ----------

srcPath_list = dbutils.fs.ls(srcPath)
srcPath_list = [x.path for x in srcPath_list]
srcPath_list

# COMMAND ----------

pt_cnt=0
sn_cnt = 0
sn_bev_cnt=0
for path in srcPath_list:
  print(path)
  if 'Beverages' in path:
    tmpdf = spark.read.option("header", "true").option("delimiter","|").csv(path)
  else :
    tmpdf = spark.read.option("header", "true").csv(path)
  #print(tmpdf.limit(1).collect()[0][0])
  cols = [x for x in tmpdf.columns if "_c" not in x]
  ncols = ','.join(cols).split(",")
  
  start_year = tmpdf.select(tmpdf[ncols[1]]).collect()[0][0]
  
  new_df = first_row_header(tmpdf)
  
  standard_cols = new_df.columns[0:new_df.columns.index(start_year)]
  
  chars = [' ',',',';','{','}','(',')','\n','\t','=','__','/']
  replaceWith ='_'
  for colName in standard_cols:
    new_df=cleanColumn(new_df,colName,chars,replaceWith)
  
  standard_cols = new_df.columns[0:new_df.columns.index(start_year)]
  
  nw_cols = new_df.columns[new_df.columns.index(start_year):new_df.columns.index(start_year+"_1")]
  
  l=[]
  cols1 = ','.join(nw_cols).split(",")
  n = len(nw_cols)
  for i in range(n):
    cls=[]
    for j in range(len(ncols)-1):
      cls.append("`"+cols1[i]+"_"+str(j+1)+"`")
    cls_new = ','.join(cls).split(",")
    l.append("'{}'".format(nw_cols[i])+",`"+cols1[i]+"`"+","+','.join(cls_new))
  k = ",".join(l)
  
  new_cols = remove_special_chars(cols)
  
  df = new_df.selectExpr(*standard_cols,"stack({0}, {1}) as (WEEK,{2})".format(n,k,",".join(new_cols)))
  
  dname = path.split("/")[-1]
  if 'PT' in dname and pt_cnt > 0:
    print("inside sn if :"+str(df.count()))
    pt_df = pt_df.unionAll(df)
  elif 'PT' in dname and pt_cnt == 0:
    pt_df = df
    pt_cnt = pt_cnt+1
  elif 'Beverages' in dname and sn_bev_cnt > 0:
    sn_bev_df = sn_bev_df.unionAll(df)
  elif 'Beverages' in dname and sn_bev_cnt == 0:
    sn_bev_df = df
    sn_bev_cnt = sn_bev_cnt+1
  elif sn_cnt > 0:
    print("inside sn elif :"+str(df.count()))
    sn_df = sn_df.unionAll(df)
    sn_cnt = sn_cnt+1
  else :
    print("inside sn else :"+str(df.count()))
    sn_df =df
    sn_cnt = sn_cnt+1
    

# COMMAND ----------

try:
  pt_df = pt_df.withColumn("PROCESS_DATE",current_date())
  sn_df = sn_df.withColumn("PROCESS_DATE",current_date())
  sn_bev_df = sn_bev_df.withColumn("PROCESS_DATE",current_date())
except Exception as e:
  print(e)

# COMMAND ----------

pt_tgtPath=tgtPath+"/portugal"
sn_tgtPath=tgtPath+"/spain"
sn_bev_tgtPath=tgtPath+"/spain-beverages"

# COMMAND ----------

def write_to_adls(df,tgtPath):
  #Writing data innto delta lake
  if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge':
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
  elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'overwrite':
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

try:
  write_to_adls(pt_df,pt_tgtPath)
  write_to_adls(sn_df,sn_tgtPath)
  write_to_adls(sn_bev_df,sn_bev_tgtPath)
except Exception as e:
  print(e)

# COMMAND ----------

recCount = pt_df.count()+sn_df.count()+sn_bev_df.count()

# COMMAND ----------

#Moving the file to archive path
dbutils.fs.mv(srcPath, archPath,recurse=True)

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(recCount))
