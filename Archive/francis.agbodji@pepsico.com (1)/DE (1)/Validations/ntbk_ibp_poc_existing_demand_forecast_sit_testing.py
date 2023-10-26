# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Users/bishnumohan.tiwary.contractor@pepsico.com/Pipeline-Notebook/NTBK_ADLS_CRED

# COMMAND ----------

# DBTITLE 1,Replace the datepart= <Date of pipeline run> in sourcepath
sourcePath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/prolink-edw/ibp-poc/dfu-to-sku-forecast/datepart=2021-08-02/"
bronzePath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/prolink-edw/ibp-poc/dfu-to-sku-forecast/"
silverPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/prolink-edw/ibp-poc/dfu-to-sku-forecast/"

# COMMAND ----------

# DBTITLE 1,Creating the Data frame for each layer
source_df = spark.read.format("parquet").load(sourcePath)
bronze_df = spark.read.format("delta").load(bronzePath)
silver_df = spark.read.format("delta").load(silverPath)

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Count match between Source files and Bronze layer table
print("Source Layer Count is "+str(source_df.count()))
print("Bronze Layer Count is "+str(bronze_df.count()))

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Source file to Bronze layer column validation
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

# DBTITLE 1,Source to Bronze column Compare
print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))

# COMMAND ----------

# DBTITLE 1,Bronze to Source Column Compare
print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

# DBTITLE 1,Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa.
print("Count of Missing Rows in Bronze are " + str(+(source_df.select(src_col).exceptAll(bronze_df.select(src_col))).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select(src_col).exceptAll(source_df.select(src_col))).count()))


# COMMAND ----------

# DBTITLE 1,Source - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['PLANG_CUST_GRP_VAL']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

src_dq = source_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("DW_UPDT_DTM")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(src_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PLANG_CUST_GRP_VAL")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze Layer - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['PLANG_CUST_GRP_VAL']
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

display(brnz_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("PLANG_CUST_GRP_VAL")).alias("Count")))

# COMMAND ----------

# DBTITLE 1,Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
print("Bronze Layer Count is "+str(bronze_df.count()))
print("Silver Layer Count is "+str(silver_df.count()))



# COMMAND ----------

# DBTITLE 1,Silver Layer Column Validation
silver_column_mdl = ['CUST_GRP','CUST_GRP_NM','LVL','SCTR','BU','MU','CHNL','CLNT']
silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))


# COMMAND ----------

# DBTITLE 1,Silver Layer - Primary Key check; NULL Check; Duplicate Check.
primaryKey = ['CUST_GRP']
primaryKey = ','.join(str(e) for e in primaryKey)
if len(primaryKey.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in primaryKey.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+primaryKey+"').isNull()"

slvr_dq = silver_df.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(primaryKey.split(','))
                    .orderBy(desc("CUST_GRP")))).withColumn("Corrupt_Record",when(eval(null_cond),lit("Primary Key is Null"))
                                                           .when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))

# COMMAND ----------

display(slvr_dq.where(col("Corrupt_Record").isNotNull()).groupBy(col("Corrupt_Record")).agg(count(col("CUST_GRP")).alias("Count")))

# COMMAND ----------

