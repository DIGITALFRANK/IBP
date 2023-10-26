# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

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

#Reading the data from the bronze path of Pricing table
invPrjc_deltaTable = DeltaTable.forPath(spark, srcPath)
invPrjc_deltaTable_version = invPrjc_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(invPrjc_deltaTable)
display(invPrjc_deltaTable.history())

# COMMAND ----------

#Reading the Inventory Projection source data from bronze layer
invPrjc_df = spark.read.format("delta").option("versionAsOf", invPrjc_deltaTable_version).load(srcPath)

# COMMAND ----------

print(invPrjc_df.count())
display(invPrjc_df)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = invPrjc_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
invPrjc_df = invPrjc_df.filter(col("PROCESS_DATE")==max_value)

# COMMAND ----------

#Reading the data from the bronze path of Distribution master table
distributionMaster_deltaTable = DeltaTable.forPath(spark, dpndntdatapath)
distributionMaster_deltaTable_version = distributionMaster_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(distributionMaster_deltaTable)
display(distributionMaster_deltaTable.history())

#Reading the Inventory Projection source data from bronze layer
distributionMaster_df = spark.read.format("delta").option("versionAsOf", distributionMaster_deltaTable_version).load(dpndntdatapath)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = distributionMaster_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
distributionMaster_df = distributionMaster_df.filter(col("PROCESS_DATE")==max_value)

# COMMAND ----------

invPrjc_df = invPrjc_df.withColumn('SCNR',when(to_date(col("SPLY_PLANG_PRJCTN_GNRTN_DT"))==to_date(col("SPLY_PLANG_PRJCTN_STRT_DT")),"Actuals").otherwise("Projected"))

invPrjc_df = invPrjc_df.withColumn('GNRTN_DT',to_date(col("SPLY_PLANG_PRJCTN_GNRTN_DT")))\
                       .withColumn('STRT_DT',to_date(col("SPLY_PLANG_PRJCTN_STRT_DT")))

invPrjc_df = invPrjc_df.withColumn('day_of_week', (when(dayofweek(col('STRT_DT')) == 1 ,6).otherwise(dayofweek(col('STRT_DT'))-2))).selectExpr('*', 'to_timestamp(date_sub(STRT_DT, day_of_week)) as N_STRTDT').drop('day_of_week','STRT_DT','PROCESS_DATE').withColumnRenamed("N_STRTDT","STRT_DT")

invPrjc_df = invPrjc_df.withColumn('WEEK_ID',concat(when((weekofyear(col("STRT_DT")) == 1) & (month(col("STRT_DT")) == 12),(year(col("STRT_DT"))+1)).otherwise(year(col("STRT_DT"))),\
                                                    (when(weekofyear(col("STRT_DT"))<10, concat(lit(0),weekofyear(col("STRT_DT")))).otherwise(weekofyear(col("STRT_DT"))))))\
                       .withColumn('MNTH_ID',concat(year(col("STRT_DT")),month(col("STRT_DT"))))
                       
invPrjc_df = invPrjc_df.withColumn('MNTH_ID',when(length(col("MNTH_ID"))<6,concat(year(col("STRT_DT")),lit("0"),month(col("STRT_DT")))).otherwise(col("MNTH_ID")))

# COMMAND ----------

display(invPrjc_df.select(min(col("STRT_DT")),max(col("STRT_DT"))))

# COMMAND ----------

display(invPrjc_df)

# COMMAND ----------

invPrjc_df_filtered = invPrjc_df.where(invPrjc_df["MTRL_ID"].rlike("^[a-zA-Z]")== False)
invPrjc_df_final=invPrjc_df_filtered.select("SCNR","MTRL_ID","DW_MTRL_ID","LOC_ID","DW_LOC_ID","GNRTN_DT","STRT_DT","WEEK_ID","MNTH_ID","PRJCTD_INVEN_QTY","PRJCTD_OH_SHRT_DMND_QTY")
silver_df_temp = invPrjc_df_final.withColumnRenamed("MTRL_ID","PROD_CD").withColumnRenamed("LOC_ID","LOC")

# COMMAND ----------

silver_df = (silver_df_temp.join(distributionMaster_df, silver_df_temp.LOC ==distributionMaster_df.PLANG_LOC_GRP_VAL, "left") 
                          .withColumn("MU",distributionMaster_df.PLANG_LOC_BU_NM)
                          .select("PROD_CD",
                                  "SCNR",
                                  "DW_MTRL_ID",
                                  "LOC",
                                  "DW_LOC_ID",
                                  col("GNRTN_DT").cast("TIMESTAMP"),
                                  col("STRT_DT").cast("TIMESTAMP"),
                                  col("WEEK_ID").cast("INT"),
                                  col("MNTH_ID").cast("INT"),
                                  col("PRJCTD_INVEN_QTY").cast("FLOAT"),
                                  col("PRJCTD_OH_SHRT_DMND_QTY").cast("FLOAT"),
                                  "MU"
                                 )
            )

display(silver_df)


# COMMAND ----------

print(silver_df.count())

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())

# COMMAND ----------

#Writing data innto delta lake
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

DeltaTable.createIfNotExists(spark) \
    .tableName("sc_ibp_silver.inventory_projection") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.inventory_projection

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

