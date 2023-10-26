# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount='cdodevadls2'

# COMMAND ----------

srcPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/vmi/"

# COMMAND ----------

#Reading the delta history from the silver path of VMI table
vmi_deltaTable = DeltaTable.forPath(spark, srcPath)
vmi_latest_version = vmi_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(vmi_latest_version)
display(vmi_deltaTable.history())

# COMMAND ----------

silver_df = spark.read.format('delta').option("versionAsOf", vmi_latest_version).load(srcPath)

# COMMAND ----------

print(silver_df.count())
display(silver_df)

# COMMAND ----------

f_df = silver_df.filter(silver_df.CUST_GRP.startswith('PT_OT'))

# COMMAND ----------

f_df.createOrReplaceTempView("silver")

# COMMAND ----------

spark.sql("select distinct vmi_flg from silver").show()

# COMMAND ----------

#PK Duplicate check
spark.sql("""
select PROD_CD, CUST_GRP, LOC, AVAILDT, EXPDT, PRJCT, count(*) as cnt
from silver
group by PROD_CD, CUST_GRP, LOC, AVAILDT, EXPDT, PRJCT
having cnt>1
""").show()

# COMMAND ----------

#PK Null check
spark.sql("""
select * from silver
where PROD_CD is Null
or CUST_GRP is Null
or LOC is Null
or AVAILDT is Null
or EXPDT is Null
or PRJCT is Null
""").show()

# COMMAND ----------

#Max VAILDT
spark.sql("select max(AVAILDT) from silver").show()
#Min VAILDT
spark.sql("select min(AVAILDT) from silver").show()

# COMMAND ----------

spark.sql("select distinct QUARANTINE from silver").show()

# COMMAND ----------

print(silver_df.count())

# COMMAND ----------

# MAGIC  %sql 
# MAGIC  select 955965+26862396