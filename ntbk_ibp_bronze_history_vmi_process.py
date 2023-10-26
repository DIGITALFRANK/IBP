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
dbutils.widgets.text("tbl_name", "")

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
isHistory = dbutils.widgets.get("isHistory")
tbl_name = dbutils.widgets.get("tbl_name")
tbl_name

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

#Reading the file details from source path
file_list = dbutils.fs.ls(srcPath)

# COMMAND ----------

##Reading the 9 CSV files from landing and merging them in one dataframe
#running a loop for the number of files present in the source path
for i in range(len(file_list)):
  print("Reading the folllowing file: "+file_list[i][1])
  df = spark.read.option("header","true").option("delimiter",srcDelimiter).csv(file_list[i][0])
  print("Number of columns in "+file_list[i][1]+": "+str(len(df.columns)))
  #dropping the extra columns from FG_CU_PT_MAD_01 file
  if file_list[i][1] == "FG_CU_PT_MAD_01":
    print("Number of columns in this file: "+str(len(df.columns)))
    c=0
    for i in range(44440,44516):
      print("Removing column "+str(i))
      df = df.drop(str(i))
      c = c+1
    print("Dropped a total of "+str(c)+" columns.")
    print("Number of columns currently in this file: "+str(len(df.columns)))
  #adding it to the merged df
  if i == 0:
    final_df = df
  else:
    final_df = final_df.unionByName(df)
print("Loop Completed")

# COMMAND ----------

print(final_df.count())
print(len(final_df.columns))

# COMMAND ----------

#checking the number of distinct location codes in the merged dataframe
display(final_df.select("Localidade").distinct())

# COMMAND ----------

#displaying number of records per location code
display(final_df.groupBy("Localidade").count())

# COMMAND ----------

#Removing the records with NULL in Localidade
final_df = final_df.filter(final_df.Localidade.isNotNull())
print(final_df.count())

# COMMAND ----------

display(final_df)

# COMMAND ----------

#cleaning column names
# descrição to descricao and Unid caixa to Unidcaixa
final_df = final_df.withColumnRenamed('descrição', 'descricao').withColumnRenamed('Unid caixa', 'Unidcaixa')
display(final_df)

# COMMAND ----------

#Adding process date column
final_df = final_df.withColumn("PROCESS_DATE",current_date())

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

tb_name = "sc_ibp_bronze.`"+tbl_name+"`"
DeltaTable.createIfNotExists(spark) \
    .tableName(tb_name) \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

#Moving the files to archive path
print(archPath)
dbutils.fs.mv(srcPath, archPath, recurse=True)

# COMMAND ----------

# remove versions which are older than 336 hrs(30 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(720)

# COMMAND ----------

#Exiting notebook with record count
dbutils.notebook.exit("recordCount : "+str(recCount))