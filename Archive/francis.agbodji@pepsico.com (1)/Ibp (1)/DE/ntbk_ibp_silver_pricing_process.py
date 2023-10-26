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
dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
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
prcng_deltaTable = DeltaTable.forPath(spark, srcPath)
prcng_deltaTable_version = prcng_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(prcng_deltaTable)
display(prcng_deltaTable.history())

# COMMAND ----------

#Reading the Pricing source data from bronze layer
prcng_df = spark.read.format("delta").option("versionAsOf", prcng_deltaTable_version).load(srcPath)

# COMMAND ----------

print(prcng_df.count())
display(prcng_df)

# COMMAND ----------

#Getting the max process date from the bronze data and then filtering on the max process date
max_value = prcng_df.agg({"PROCESS_DATE": "max"}).collect()[0][0]
print(max_value)
prcng_df = prcng_df.filter(col("PROCESS_DATE")==max_value)
#prcng_df = prcng_df.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df.CD_CUSTOMER_SOURCE))
#display(stft_df2)

# COMMAND ----------

dpndntdatapath_list = dpndntdatapath.split(";")
for path in dpndntdatapath_list:
  srcPath = "abfss://"+dbutils.widgets.get("sourceContainer")+"@"+source_stgAccnt+".dfs.core.windows.net/"+path
  if '/sap/' in path:
    customerMappingDF_Beverages = spark.read.format('delta').load(srcPath)
  if '/gtm-oracle/' in path:
    customerMappingDF_Snacks = spark.read.format('delta').load(srcPath)
  if '/as400/' in path:
    customerMappingDF_Alvalle = spark.read.format('delta').load(srcPath)
  if 'product-case-mapping' in path:
    product_snacks = spark.read.format('delta').load(srcPath)
  if 'product-mapping' in path:
    product_bevs = spark.read.format('delta').load(srcPath)
  if 'product-master' in path:
    product_master = spark.read.format('delta').load(srcPath)

# COMMAND ----------

prcng_df_es = prcng_df.filter(prcng_df.CD_COUNTRY=='138')
prcng_df_pt = prcng_df.filter(prcng_df.CD_COUNTRY=='119')
prcng_df_es = (prcng_df_es.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df_es.CD_CUSTOMER_SOURCE))
                          .withColumn('COUNTRY', lit("ES"))
              )
prcng_df_pt = (prcng_df_pt.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df_pt.CD_CUSTOMER_SOURCE))
                          .withColumn('CD_CUSTOMER_SOURCE', expr("right(CD_CUSTOMER_SOURCE, length(CD_CUSTOMER_SOURCE)-2)"))
                           .withColumn('COUNTRY', lit("PT"))
              )
prcng_df = prcng_df_es.union(prcng_df_pt)

# COMMAND ----------

# product_snacks = (spark.read
#                         .option("header","true")
#                         .option("delimiter",",")
#                         .csv("abfss://landing@cdodevadls2.dfs.core.windows.net/edw/ibp/product-master/CASE.csv")
#                        )

# product_bevs = (spark.read
#                         .option("header","true")
#                         .option("delimiter",",")
#                         .csv("abfss://landing@cdodevadls2.dfs.core.windows.net/edw/ibp/product-master/PRODUCT.csv")
#                        )

# COMMAND ----------

df1 = (customerMappingDF_Beverages.withColumn("Source",lit("SAP"))
                                  .select(col("Customer_Id").alias("Customer_Code")
                                         ,col("Demand_Group").alias("DMDGroup")
                                         ,col("Customer_Name").alias("Desc")
                                         ,col("Source"))
      )
df1 = df1.withColumn('Customer_Code', trim(df1.Customer_Code))

df2 = (customerMappingDF_Snacks.withColumn("Source",lit("GTM"))
                              .select(col("Customer_Id").alias("Customer_Code")
                                     ,col("Demand_Group").alias("DMDGroup")
                                     ,col("Customer_Name").alias("Desc")
                                     ,col("Source"))
      )
df2 = df2.withColumn('Customer_Code', trim(df2.Customer_Code))


df3 = (customerMappingDF_Alvalle.withColumn("Source",lit("AS400"))
                                .withColumn("Desc",lit("N/A"))
                                .select(col("Customer_ID").alias("Customer_Code")
                                       ,col("Demand_Group").alias("DMDGroup")
                                       ,col("Customer_Name").alias("Desc")
                                       ,col("Source"))
      )
df3 = df3.withColumn('Customer_Code', trim(df3.Customer_Code))

Customer_Groups_Combined_DF = df3.union(df1.union(df2))
Customer_Groups_Combined_DF = Customer_Groups_Combined_DF.withColumn("MU", expr("left(DMDGroup, 2)"))

# COMMAND ----------

pricing_join_cust = prcng_df.join(Customer_Groups_Combined_DF,  (prcng_df.CD_CUSTOMER_SOURCE == Customer_Groups_Combined_DF.Customer_Code) & (prcng_df.COUNTRY == Customer_Groups_Combined_DF.MU), 'left')
pricing_join_cust = pricing_join_cust.withColumn("CT",expr("substring(CD_PROD_SELLING, 1, 1)"))
display(pricing_join_cust)

# COMMAND ----------

ProlinkProductMaster = product_master.filter("SYS_ID = 564 and right(PLANG_MTRL_GRP_VAL, 2) in ('04','05','06')")
ProlinkProductMaster = ProlinkProductMaster.withColumn('DMDUNIT', trim(product_master.PLANG_MTRL_GRP_VAL))
ProlinkProductMaster = ProlinkProductMaster.withColumn("Selling_SKU",expr("substring(DMDUNIT, 1, length(DMDUNIT)-3)"))
ProlinkProductMaster.select("SRC_CTGY_1_NM").distinct().display()

# COMMAND ----------

### Calc Pricing Snacks and Grains
pricing_join_cust_F = pricing_join_cust.filter("left(CD_PROD_SELLING,1) = 'F'")

# Now join to Snacks, Grains Mapping file
pricing_snacks_mapped = pricing_join_cust_F.join(product_snacks, pricing_join_cust_F.CD_PROD_SELLING == product_snacks.CD_PRODUCT_SELLING , 'left')
pricing_snacks_mapped = pricing_snacks_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('SNACKS','GRAINS')"), pricing_snacks_mapped.CD_PRODUCT_CASE == ProlinkProductMaster.Selling_SKU , 'left')
# Rename
pricing_snacks_mapped = pricing_snacks_mapped.select(["DMDGroup", "CD_CUSTOMER","CD_CUSTOMER_SOURCE","CD_PROD_SELLING","PLANG_MTRL_GRP_VAL","CD_COUNTRY","ID_TIME_WEEK","ID_TIME_MONTH","ME_GROSS_SALES_ACT","ME_TOTAL_DISCOUNTS_ACT","ME_SALES_UNITS_BAG_ACT"]) 

# COMMAND ----------

### Product Join for Bevs.
pricing_join_cust_B = pricing_join_cust.filter("left(CD_PROD_SELLING,1) = 'B'")

product_bevs = product_bevs.withColumnRenamed("CD_PROD_SELLING", "CD_PRODUCT_SELLING").select("CD_PRODUCT_SELLING","CD_PRODUCT")
# Now join to Snacks, Grains and Juices Mapping file
pricing_bevs_mapped = pricing_join_cust_B.join(product_bevs ,pricing_join_cust_B.CD_PROD_SELLING == product_bevs.CD_PRODUCT_SELLING, 'left')
pricing_bevs_mapped = pricing_bevs_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('BEVERAGES','JUICE')"), pricing_bevs_mapped.CD_PRODUCT == ProlinkProductMaster.Selling_SKU , 'left')
# # Rename
pricing_bevs_mapped = pricing_bevs_mapped.select(["DMDGroup", "CD_CUSTOMER","CD_CUSTOMER_SOURCE","CD_PROD_SELLING","PLANG_MTRL_GRP_VAL","CD_COUNTRY","ID_TIME_WEEK","ID_TIME_MONTH","ME_GROSS_SALES_ACT","ME_TOTAL_DISCOUNTS_ACT","ME_SALES_UNITS_BAG_ACT"]) 

# COMMAND ----------

### Product Join for Juices
# 06 lookup
product_alvalle = spark.read.csv("/FileStore/tables/temp/Alvelle_Mapping.csv", header="true", inferSchema="true")
pricing_join_cust_A = pricing_join_cust.filter("left(CD_PROD_SELLING,1) = 'A'")

product_alvalle = product_alvalle.withColumnRenamed("CD_PROD_SELLING", "CD_PRODUCT_SELLING").withColumnRenamed("06 lookup ", "PLANG_MTRL_GRP_VAL").select("CD_PRODUCT_SELLING","PLANG_MTRL_GRP_VAL")
# Now join to Snacks, Grains and Juices Mapping file
pricing_alvalle_mapped = pricing_join_cust_A.join(product_alvalle ,pricing_join_cust_A.CD_PROD_SELLING == product_alvalle.CD_PRODUCT_SELLING, 'left')
# pricing_alvalle_mapped = pricing_alvalle_mapped.join(ProlinkProductMaster.filter("SRC_CTGY_1_NM in ('JUICES')"), pricing_alvalle_mapped.CD_PRODUCT == ProlinkProductMaster.Selling_SKU , 'left')
# Rename
pricing_alvalle_mapped = pricing_alvalle_mapped.select(["DMDGroup", "CD_CUSTOMER","CD_CUSTOMER_SOURCE","CD_PROD_SELLING","PLANG_MTRL_GRP_VAL","CD_COUNTRY","ID_TIME_WEEK","ID_TIME_MONTH","ME_GROSS_SALES_ACT","ME_TOTAL_DISCOUNTS_ACT","ME_SALES_UNITS_BAG_ACT"]) 


# COMMAND ----------

IBP_SL_PRCNG = pricing_bevs_mapped.union(pricing_snacks_mapped.union(pricing_alvalle_mapped))
IBP_SL_PRCNG = IBP_SL_PRCNG.withColumn("ID_TIME_WEEK",concat(substring(col('ID_TIME_WEEK'),0,4),substring(col('ID_TIME_WEEK'),7,2)))
 
IBP_SL_PRCNG.display()

# COMMAND ----------

IBP_SL_PRCNG = IBP_SL_PRCNG.groupby('DMDGroup','PLANG_MTRL_GRP_VAL','ID_TIME_WEEK','ID_TIME_MONTH').agg({'ME_GROSS_SALES_ACT': 'sum','ME_TOTAL_DISCOUNTS_ACT': 'sum','ME_SALES_UNITS_BAG_ACT': 'sum'})
IBP_SL_PRCNG = IBP_SL_PRCNG.withColumn("LIST_PRICE",col("sum(ME_GROSS_SALES_ACT)")/col("sum(ME_SALES_UNITS_BAG_ACT)"))\
.withColumn("NET_PRICE",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col("sum(ME_SALES_UNITS_BAG_ACT)"))


# COMMAND ----------

IBP_SL_PRCNG = IBP_SL_PRCNG.withColumnRenamed('DMDGroup','CUST_GRP')\
                        .withColumnRenamed('PLANG_MTRL_GRP_VAL','PROD_CD').withColumnRenamed('ID_TIME_MONTH','MONTH_OF_YEAR').withColumnRenamed('sum(ME_GROSS_SALES_ACT)','ME_GROSS_SALES_ACT_SUM').withColumnRenamed('sum(ME_SALES_UNITS_BAG_ACT)','ME_SALES_UNITS_BAG_ACT_SUM').withColumnRenamed('sum(ME_TOTAL_DISCOUNTS_ACT)','ME_TOTAL_DISCOUNTS_ACT_SUM')\
                             .withColumnRenamed('ID_TIME_WEEK','WEEK_OF_YEAR')\
                             .withColumnRenamed('LIST_PRICE','ACTL_GRSS_LIST_PRC')\
                             .withColumnRenamed('NET_PRICE','ACTL_NET_LIST_PRC')\
                            # .withColumn("DURTN",lit("WEEKLY"))


IBP_SL_PRCNG = IBP_SL_PRCNG.select("CUST_GRP","PROD_CD","MONTH_OF_YEAR","WEEK_OF_YEAR","ACTL_GRSS_LIST_PRC","ACTL_NET_LIST_PRC","ME_TOTAL_DISCOUNTS_ACT_SUM","ME_SALES_UNITS_BAG_ACT_SUM","ME_GROSS_SALES_ACT_SUM")
IBP_SL_PRCNG.display()

# COMMAND ----------

IBP_SL_PRCNG.count()

# COMMAND ----------

## DQ Check
pklist2 = pkList.split(';')
pklist2 = ','.join(str(e) for e in pklist2)
if len(pklist2.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in pklist2.split(',')]
  null_cond = " | ".join(ls)
else :
  null_cond = "col('"+pklist2+"').isNull()"

# COMMAND ----------

prcng_dq = IBP_SL_PRCNG.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pklist2.split(','))
                    .orderBy(desc("ACTL_GRSS_LIST_PRC")))).withColumn("Corrupt_Record",when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))
silver_df = prcng_dq.where(col("Corrupt_Record").isNull()).drop("Corrupt_Record","DUP_CHECK")
prcng_dq_not_pass = prcng_dq.where(col("Corrupt_Record").isNotNull())
print(prcng_dq_not_pass.count())

# COMMAND ----------

print(silver_df.count())

# COMMAND ----------

# silver_df = prcng_dq_pass.groupby(col('CUST_GRP'),col('PROD_CD'),col('STRTDT'),col('DURTN')).agg(avg(col('ACTL_GRSS_LIST_PRC')).alias('ACTL_GRSS_LIST_PRC_AVG'),avg(col('ACTL_NET_LIST_PRC')).alias('ACTL_NET_LIST_PRC'))

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

dbutils.notebook.exit("recordCount : "+str(silver_df.count()))

# COMMAND ----------

