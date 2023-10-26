# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Users/bishnumohan.tiwary.contractor@pepsico.com/Pipeline-Notebook/NTBK_ADLS_CRED

# COMMAND ----------

sourcePath_his = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp-poc/shipment-actuals/datepart=2021-07-27/"
sourcePath_inc = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp-poc/shipment-actuals/datepart=2021-07-28/"
bronzePath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp-poc/shipment-actuals/"
silverPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/shipment-actuals/"

# COMMAND ----------

source_df_his = spark.read.format("parquet").load(sourcePath_his).drop("MIN_DATE")
source_df_inc = spark.read.format("parquet").load(sourcePath_inc).drop("DELTA_DATE")
bronze_df = spark.read.format("delta").load(bronzePath).drop("DELTA_DATE")
silver_df = spark.read.format("delta").load(silverPath)

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
primaryKey = ['PLANG_MTRL_GRP_VAL','PLANG_CUST_GRP_VAL','PLANG_LOC_GRP_VAL','HSTRY_TMFRM_STRT_DT']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

src_dq = source_df_his.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("DW_UPDT_DTM")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(src_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PLANG_MTRL_GRP_VAL")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['PLANG_MTRL_GRP_VAL','PLANG_CUST_GRP_VAL','PLANG_LOC_GRP_VAL','HSTRY_TMFRM_STRT_DT']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

brnz_dq = bronze_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("DW_UPDT_DTM")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(brnz_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PLANG_MTRL_GRP_VAL")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

depeDatSetPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp-poc/dfu"

# COMMAND ----------

dep_deltaTable = DeltaTable.forPath(spark, depeDatSetPath)
dependData_vers = dep_deltaTable.history().select(max(col('version'))).collect()[0][0]
dependData_DF = spark.read.format("delta").option("versionAsOf", dependData_vers).load(depeDatSetPath)

# COMMAND ----------

dep_df = dependData_DF.where(col("DMNDFCST_UNIT_LVL_VAL")=='SB-S-FL-ITEM_CLIENT_DC')
print(dep_df.count())

# COMMAND ----------

cond = [bronze_df.PLANG_MTRL_GRP_VAL == dep_df.PLANG_MTRL_GRP_VAL, bronze_df.PLANG_CUST_GRP_VAL == dep_df.PLANG_CUST_GRP_VAL, bronze_df.PLANG_LOC_GRP_VAL == dep_df.PLANG_LOC_GRP_VAL]
slvr_test = bronze_df.join(dep_df,cond).groupBy(bronze_df.PLANG_MTRL_GRP_VAL,bronze_df.PLANG_CUST_GRP_VAL,bronze_df.PLANG_LOC_GRP_VAL).agg(count("HSTRY_TMFRM_STRT_DT").alias("count"))

# COMMAND ----------

display(slvr_test.agg(sum(col("count")).alias("Bronze Layer Count after Transformation")))
print("Silver Layer Count is "+str(silver_df.count()))

# COMMAND ----------

# DBTITLE 1,Silver layer column validation.
silver_column_mdl = ['PROD_CD','CUST_GRP','LOC','MKT_UNIT','CTGY','STRTDATE','QTY','UOM','DURTN','HISTSTREM','TYP']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))

# COMMAND ----------

# DBTITLE 1,Silver - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['PROD_CD','CUST_GRP','LOC','STRTDATE']
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

display(silver_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PROD_CD")).alias("Count")))

# COMMAND ----------

