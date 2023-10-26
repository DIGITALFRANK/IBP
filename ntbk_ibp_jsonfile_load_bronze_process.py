# Databricks notebook source
from pyspark.sql.functions import col, explode_outer, current_date
from pyspark.sql.types import *
from copy import deepcopy
from collections import Counter
from delta.tables import *
from pyspark.sql import SparkSession
from pyspark import SparkContext
import re
import json

# COMMAND ----------

#defining the widgets for accepting parameters from pipeline
dbutils.widgets.text("stgAccount", "")
dbutils.widgets.text("sourcePath", "")
dbutils.widgets.text("sourceContainer", "")
dbutils.widgets.text("archivePath", "")
dbutils.widgets.text("archiveContainer", "")
dbutils.widgets.text("sourceFileFormat", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("targetContainer", "")
dbutils.widgets.text("loadType", "")

# COMMAND ----------

# Defining required variables
stgAccnt = dbutils.widgets.get("stgAccount")
srcFormat = dbutils.widgets.get("sourceFileFormat")
srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("sourcePath")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")
archPath = "abfss://"+dbutils.widgets.get("archiveContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("archivePath")
loadType = dbutils.widgets.get("loadType")

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$stgAccount

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_json_common_classes

# COMMAND ----------

path=srcPath
if '/demographics/' in path:
  df=spark.read.option("multiLine","false").json(path)
else:
  df = spark.read.option("multiLine","true").json(path)
json_schema = df.schema

# COMMAND ----------

af = AutoFlatten(json_schema)

# COMMAND ----------

af.compute()

# COMMAND ----------

df1 = df

# COMMAND ----------

visited = set([f'.{column}' for column in df1.columns])
duplicate_target_counter = Counter(af.all_fields.values())
cols_to_select = df1.columns
for rest_col in af.rest:
    if rest_col not in visited:
        cols_to_select += [rest_col[1:]] if (duplicate_target_counter[af.all_fields[rest_col]]==1 and af.all_fields[rest_col] not in df1.columns) else [col(rest_col[1:]).alias(f"{rest_col[1:].replace('.', '>')}")]
        visited.add(rest_col)

df1 = df1.select(cols_to_select)


if af.order:
    for key in af.order:
        column = key.split('.')[-1]
        if af.bottom_to_top[key]:
            #########
            #values for the column in bottom_to_top dict exists if it is an array type
            #########
            df1 = df1.select('*', explode_outer(col(column)).alias(f"{column}_exploded")).drop(column)
            data_type = df1.select(f"{column}_exploded").schema.fields[0].dataType
            if not (isinstance(data_type, StructType) or isinstance(data_type, ArrayType)):
                df1 = df1.withColumnRenamed(f"{column}_exploded", column if duplicate_target_counter[af.all_fields[key]]<=1 else key[1:].replace('.', '>'))
                visited.add(key)
            else:
                #grabbing all paths to columns after explode
                cols_in_array_col = set(map(lambda x: f'{key}.{x}', df1.select(f'{column}_exploded.*').columns))
                #retrieving unvisited columns
                cols_to_select_set = cols_in_array_col.difference(visited)
                all_cols_to_select_set = set(af.bottom_to_top[key])
                #check done for duplicate column name & path
                cols_to_select_list = list(map(lambda x: f"{column}_exploded{'.'.join(x.split(key)[1:])}" if (duplicate_target_counter[af.all_fields[x]]<=1 and x.split('.')[-1] not in df1.columns) else col(f"{column}_exploded{'.'.join(x.split(key)[1:])}").alias(f"{x[1:].replace('.', '>')}"), list(all_cols_to_select_set)))
                #updating visited set
                visited.update(cols_to_select_set)
                rem = list(map(lambda x: f"{column}_exploded{'.'.join(x.split(key)[1:])}", list(cols_to_select_set.difference(all_cols_to_select_set))))
                df1 = df1.select(df1.columns + cols_to_select_list + rem).drop(f"{column}_exploded")        
        else:
            #########
            #values for the column in bottom_to_top dict do not exist if it is a struct type / array type containing a string type
            #########
            #grabbing all paths to columns after opening
            cols_in_array_col = set(map(lambda x: f'{key}.{x}', df1.selectExpr(f'{column}.*').columns))
            #retrieving unvisited columns
            cols_to_select_set = cols_in_array_col.difference(visited)
            #check done for duplicate column name & path
            cols_to_select_list = list(map(lambda x: f"{column}.{x.split('.')[-1]}" if (duplicate_target_counter[x.split('.')[-1]]<=1 and x.split('.')[-1] not in df1.columns) else col(f"{column}.{x.split('.')[-1]}").alias(f"{x[1:].replace('.', '>')}"), list(cols_to_select_set)))
            #updating visited set
            visited.update(cols_to_select_set)
            df1 = df1.select(df1.columns + cols_to_select_list).drop(f"{column}")

# COMMAND ----------

final_df = df1.select([field[1:].replace('.', '>') if duplicate_target_counter[af.all_fields[field]]>1 else af.all_fields[field] for field in af.all_fields])

# COMMAND ----------

display(final_df)

# COMMAND ----------

final_df = final_df.toDF(*[re.sub('[>]','_',col) for col in final_df.columns])
final_df = final_df.toDF(*[re.sub('[:]','-',col) for col in final_df.columns])
final_df = final_df.toDF(*[re.sub('[@,#]','',col) for col in final_df.columns])

# COMMAND ----------

# Adding Process_date attribute
final_df = final_df.withColumn("PROCESS_DATE",current_date())

# COMMAND ----------

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

final_df = cleanColumn(final_df)

# COMMAND ----------

display(final_df)

# COMMAND ----------

#Writing data innto delta lake
if (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'merge' and pkList != 'NA' :
  deltaTable = DeltaTable.forPath(spark, tgtPath)
  deltaTable.alias("target").merge(
    source = final_df.alias("updates"),
    condition = cond)\
  .whenMatchedUpdateAll()\
  .whenNotMatchedInsertAll().execute()
elif (DeltaTable.isDeltaTable(spark, tgtPath)) and loadType == 'insert':
  final_df.write.format("delta")\
  .mode('append')\
  .option("mergeSchema", "true")\
  .save(tgtPath)
elif loadType == 'overwrite':
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

#source record count
recCount = final_df.count()

# COMMAND ----------

#Moving the file to archive path
if len(srcPath.split(';'))>1:
  srcPath_list =["abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+x for x in srcPath.split(';')]
  archPath_list = ["abfss://"+dbutils.widgets.get("sourceContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+x for x in dbutils.widgets.get("archivePath").split(';')]
  for i in range(len(srcPath_list)):
    dbutils.fs.mv(srcPath_list[i], archPath_list[i],recurse=True)
else:
  dbutils.fs.mv(srcPath, archPath,recurse=True)

# COMMAND ----------

#Exiting notebook with record count
dbutils.notebook.exit("recordCount : "+str(recCount))