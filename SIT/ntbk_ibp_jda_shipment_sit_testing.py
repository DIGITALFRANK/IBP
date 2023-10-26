# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount= "cdodevadls2"

# COMMAND ----------

sourcePath_his = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/jda-shipments/datepart=2021-10-22/"
sourcePath_inc = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/jda-shipments/datepart=2021-10-27/"
sourcePath_inc_latest = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/jda-shipments/datepart=2021-11-02/"
bronzePath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp/jda-shipments/"
silverPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/jda-shipments/"

# COMMAND ----------

source_df_his = spark.read.format("parquet").load(sourcePath_his)
source_df_inc = spark.read.format("parquet").load(sourcePath_inc_latest)
bronze_df = spark.read.format("delta").load(bronzePath)
silver_df = spark.read.format("delta").load(silverPath)

# COMMAND ----------

source_df_inc.select(max(col("STARTDATE"))).display()

# COMMAND ----------

bronze_df.select(max(col("STARTDATE"))).display()

# COMMAND ----------

silver_df.select(max(col("STRTDT"))).display()

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Count match between Source files and Bronze layer table
print("Source Layer Count(Historical) is "+str(source_df_his.count()))
print("Source Layer Count(Incremental) is "+str(source_df_inc.count()))
print("Bronze Layer Count(AfterMerge) is "+str(bronze_df.count()))

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Source file to Bronze layer column validation
src_col =  source_df_his.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))

# COMMAND ----------

print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa(History)
print("Count of Missing Rows in Bronze are " + str(+(source_df_his.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df_his.select(src_col))).count()))

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa(Incremental)
print("Count of Missing Rows in Bronze(for incremental load) are " + str(+(source_df_inc.select(src_col).exceptAll(bronze_df.select(src_col))).count()))

# COMMAND ----------

# DBTITLE 1,Source - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['DMDUNIT','DMDGROUP','LOC','STARTDATE','DUR','TYPE','EVENT']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

src_dq = source_df_his.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                      .orderBy(desc("STARTDATE")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(src_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("DMDUNIT")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['DMDUNIT','DMDGROUP','LOC','STARTDATE','DUR','TYPE','EVENT']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

brnz_dq = bronze_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("STARTDATE")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(brnz_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("DMDUNIT")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

depeDatSetPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp/dfu"
productPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/edw/ibp/product-master"

# COMMAND ----------

dep_deltaTable = DeltaTable.forPath(spark, depeDatSetPath)
dependData_vers = dep_deltaTable.history().select(max(col('version'))).collect()[0][0]
dependData_DF = spark.read.format("delta").option("versionAsOf", dependData_vers).load(depeDatSetPath)

# COMMAND ----------

dep_df = dependData_DF.where(col("DMNDFCST_UNIT_LVL_VAL")=='SB-S-FL-ITEM_CLIENT_DC')
print(dep_df.count())

# COMMAND ----------

product_master = spark.read.format("delta").load(productPath)

# COMMAND ----------

bronze_df_fltr = bronze_df.filter("(TYPE = 1 and EVENT like 'CH%') or (TYPE = 1 and EVENT = ' ')")
cond = [bronze_df_fltr.DMDUNIT == dep_df.PLANG_MTRL_GRP_VAL, bronze_df_fltr.DMDGROUP == dep_df.PLANG_CUST_GRP_VAL, bronze_df_fltr.LOC == dep_df.PLANG_LOC_GRP_VAL]
slvr_test_fst = bronze_df_fltr.join(dep_df,cond).withColumn("MKT_UNIT",col("DMDGROUP").substr(1, 2))
slvr_test_ltr = slvr_test_fst.join(product_master.select("PLANG_MTRL_GRP_VAL","SRC_CTGY_1_NM"), slvr_test_fst.DMDUNIT==product_master.PLANG_MTRL_GRP_VAL, "left")
slvr_test_fnl = slvr_test_ltr.withColumn("UDC_UOM", regexp_replace(col("UDC_UOM")," ","EA"))
slvr_test = slvr_test_fnl.groupBy("DMDUNIT","DMDGROUP","STARTDATE","UDC_UOM","DUR","HISTSTREAM","TYPE","LOC","MKT_UNIT","SRC_CTGY_1_NM").agg(sum("QTY").alias("QTY"))

# COMMAND ----------

print("bronze Layer Count is "+str(slvr_test.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

# DBTITLE 1,Silver layer column validation.
silver_column_mdl = ['PROD_CD','CUST_GRP','LOC','MKT_UNIT','CTGY','STRTDT','QTY','UOM','DURTN','HISTSTREM','TYP','WEEK_OF_YEAR','PROCESS_DATE']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

# DBTITLE 1,Silver - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['PROD_CD','CUST_GRP','LOC','STRTDT']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

silver_dq = silver_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("DURTN")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------



# COMMAND ----------

display(silver_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PROD_CD")).alias("Count")))

# COMMAND ----------

ship_edw = spark.read.format("delta").load("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/shipment-actuals/")
ship_edw.display()

# COMMAND ----------

# slvr_test_new = slvr_test.drop("QTY","UDC_UOM").select(col("DMDUNIT").alias("PROD_CD"),col("DMDGROUP").alias("CUST_GRP"),"LOC","MKT_UNIT",col("SRC_CTGY_1_NM").alias("CTGY"),col("STARTDATE").alias("STRTDT").cast("timestamp"),col("DUR").alias("DURTN").cast("int"),col("HISTSTREAM").alias("HISTSTREM"),col("TYPE").alias("TYP"))
# ship_edw_new = ship_edw.drop("QTY","UOM","WEEK_OF_YEAR","PROCESS_DATE")
# common_df = slvr_test_new.intersectAll(ship_edw_new)
# common_df.count()

# COMMAND ----------

slvr_test_new = slvr_test.select(col("DMDUNIT").alias("PROD_CD"),col("DMDGROUP").alias("CUST_GRP"),"LOC",col("STARTDATE").alias("STRTDT").cast("timestamp"))
ship_edw_new = ship_edw.select("PROD_CD","CUST_GRP","LOC","STRTDT")
commond_df = slvr_test_new.intersectAll(ship_edw_new)
jda_extra = slvr_test_new.subtract(commond_df)
edw_extr = ship_edw_new.subtract(commond_df)

# COMMAND ----------

print(commond_df.count())
print(jda_extra.count())
print(edw_extr.count())

# COMMAND ----------

jda_extra.display()

# COMMAND ----------

edw_extr.display()