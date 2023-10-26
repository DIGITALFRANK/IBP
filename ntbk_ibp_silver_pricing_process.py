# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

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
    product_case = spark.read.format('delta').load(srcPath)
  if 'product-mapping' in path:
    product_mapping = spark.read.format('delta').load(srcPath)
  if 'product-master' in path:
    product_master = spark.read.format('delta').load(srcPath)
  if 'customer-mapping-pricing' in path:
    new_map = spark.read.format("csv").option("header",True).load(srcPath)

# COMMAND ----------

#Pricing Dataframe - creating separate for spain and protugal and then unioning them
prcng_df_es = prcng_df.filter(prcng_df.CD_COUNTRY=='138')
prcng_df_pt = prcng_df.filter(prcng_df.CD_COUNTRY=='119')
prcng_df_es = (prcng_df_es.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df_es.CD_CUSTOMER_SOURCE))
                          .withColumn('COUNTRY', lit("ES"))
              )
prcng_df_pt = (prcng_df_pt.withColumn('CD_CUSTOMER', trim(prcng_df_pt.CD_CUSTOMER))
                           .withColumn('COUNTRY', lit("PT"))
              )
#prcng_df = prcng_df_es.union(prcng_df_pt)-----------Line Commented out in new code

# COMMAND ----------

# DBTITLE 1,Spain Customer Mapping
# new sap
sap = spark.read.options(header = 'True', inferSchema='True').csv('dbfs:/FileStore/tables/temp/SAP_customer_list.csv').withColumn("Source", lit("SAP")).withColumnRenamed('Denom.5','DMDGroup').select('Cliente','DMDGroup','Source').filter('DMDGroup is Not Null').drop_duplicates()
#sap.count()

# COMMAND ----------

# new AS400-G2M
as400 = spark.read.options(header = 'True', inferSchema='True').csv('dbfs:/FileStore/tables/temp/AS400_G2M_customer_link.csv').withColumnRenamed("Cliente G2M","Cliente").withColumn("Source", lit("AS400-G2M")).withColumnRenamed('Demand Group','DMDGroup').select('Cliente','DMDGroup','Source').filter('DMDGroup is Not Null').drop_duplicates()
#as400.count()

# COMMAND ----------

# previous G2M
g2m = customerMappingDF_Snacks.withColumn("Source",lit("G2M")).withColumnRenamed("Customer_Id", "Cliente").withColumnRenamed("Demand_Group","DMDGroup")\
                              .withColumn("MU", expr("left(DMDGroup, 2)")).filter("MU == 'ES'").select('Cliente','DMDGroup','Source')\
                              .filter('DMDGroup is Not Null').drop_duplicates()
#g2m.count()

# COMMAND ----------

# union all mapping files
full = sap.union(as400).union(g2m)
# extract the customer codes that have 2 DMDGroups from different system
check = full.groupBy('Cliente').count().filter("count > 1").drop('count')
extract = prcng_df_es.join(check, (prcng_df_es.CD_CUSTOMER_SOURCE == check.Cliente), how = 'inner')\
                     .union(prcng_df_es.join(check, (prcng_df_es.CD_CUSTOMER_SOURCE == check.Cliente), how = 'leftanti').join(check, (prcng_df_es.CD_CUSTOMER == check.Cliente), how = 'inner'))
# create Source column
extract = extract.withColumn("Source", when(col('CD_PROD_SELLING').startswith('A'), 'AS400-G2M').otherwise('G2M'))
mapped_1 = extract.join(full, on = ['Cliente','Source'], how = 'inner')
# checking
print(extract.count(), mapped_1.count())
mapped_1.display()

# COMMAND ----------

# exclude the duplicates Customer Codes from the full mapping files
Customer_Groups_Combined_DF = full.join(check, on = 'Cliente', how = 'leftanti')
#Joining Pricing and Customer - Spain
# exclude extract from prcng_df_es
prcng_df_es_new = prcng_df_es.join(check, (prcng_df_es.CD_CUSTOMER_SOURCE == check.Cliente)|(prcng_df_es.CD_CUSTOMER == check.Cliente), how = 'leftanti')
# mapping the rest
pricing_join_cust_es_1 = prcng_df_es_new.join(Customer_Groups_Combined_DF,  (prcng_df_es_new.CD_CUSTOMER_SOURCE == Customer_Groups_Combined_DF.Cliente), how = 'left')
# Using CD_CUSTOMER to map additional DMDGROUP
not_mapped = pricing_join_cust_es_1.filter('DMDGroup is NULL').select(prcng_df_es_new.columns)
new_map_es = not_mapped.join(Customer_Groups_Combined_DF,not_mapped.CD_CUSTOMER==Customer_Groups_Combined_DF.Cliente,how = 'left')
pricing_join_cust_es = pricing_join_cust_es_1.filter('DMDGroup is NOT NULL').unionByName(new_map_es).unionByName(mapped_1).withColumnRenamed("COUNTRY","MU")
# checking
print(pricing_join_cust_es.count(), prcng_df_es.count())
print(pricing_join_cust_es.filter("DMDGroup is Null").count())
display(pricing_join_cust_es)

# COMMAND ----------

# DBTITLE 1,PT David's Customer Mapping File
new_map = new_map.drop_duplicates()
new_map = new_map.select("CD_CUSTOMER","Actual Prolink Dmd Group")
new_map.display()

# COMMAND ----------

print('PT SalesBI Count:',prcng_df_pt.count(),'PT SalesBI Distinct CD_CUSTOMER Count:',prcng_df_pt.drop_duplicates(['CD_CUSTOMER']).count())
#mapping customers for PT
pt_dts = prcng_df_pt.join(new_map, on = 'CD_CUSTOMER', how ='left')
print('PT SalesBI Mapped Count:',pt_dts.filter(col("Actual Prolink Dmd Group").isNotNull()).count(),'PT SalesBI Distinct Mapped CD_CUSTOMER Count:',pt_dts.filter(col("Actual Prolink Dmd Group").isNotNull()).drop_duplicates(['CD_CUSTOMER']).count())

# COMMAND ----------

pt_dts = pt_dts.withColumnRenamed("Actual Prolink Dmd Group","DMDGroup").withColumnRenamed("COUNTRY","MU").withColumn("Source",lit("G2M")).withColumn("Cliente",col("CD_CUSTOMER"))

# COMMAND ----------

col_sp = ['CD_PROD_SELLING','ID_TIME_WEEK','ID_TIME_MONTH','CD_CUSTOMER_SOURCE','CD_CUSTOMER','CD_SOURCE_SYSTEM','CD_COUNTRY','CD_PROD_CATEGORY','CD_PROD_SUB_CATEGORY','CD_PROD_BRAND', 'CD_PROD_SUB_BRAND','CD_PROD_FLAVOR','CD_PROD_PACK_SIZE','CD_PROD_SIZE','ME_NET_SALES_ACT','ME_SALES_UNITS_P_CASES_ACT','ME_SALES_UNITS_BAG_ACT','ME_SALES_UNITS_RAW_ACT','ME_STALES_EUR_ACT', 'ME_TOTAL_DISCOUNTS_ACT','ME_GROSS_SALES_ACT','ME_VOLUME_L_ACT','ME_VOLUME_KG_ACT','ME_ON_INVOICE_ACT','PROCESS_DATE','Cliente','DMDGroup','Source','MU']

col_pt = ['CD_PROD_SELLING','ID_TIME_WEEK','ID_TIME_MONTH','CD_CUSTOMER_SOURCE','CD_CUSTOMER','CD_SOURCE_SYSTEM','CD_COUNTRY','CD_PROD_CATEGORY','CD_PROD_SUB_CATEGORY','CD_PROD_BRAND', 'CD_PROD_SUB_BRAND','CD_PROD_FLAVOR','CD_PROD_PACK_SIZE','CD_PROD_SIZE','ME_NET_SALES_ACT','ME_SALES_UNITS_P_CASES_ACT','ME_SALES_UNITS_BAG_ACT','ME_SALES_UNITS_RAW_ACT','ME_STALES_EUR_ACT', 'ME_TOTAL_DISCOUNTS_ACT','ME_GROSS_SALES_ACT','ME_VOLUME_L_ACT','ME_VOLUME_KG_ACT','ME_ON_INVOICE_ACT','PROCESS_DATE','Cliente','DMDGroup','Source','MU']

# COMMAND ----------

# union PT and Spain
pricing_join_cust = pricing_join_cust_es.selectExpr(col_sp).union(pt_dts.selectExpr(col_pt))
#Update the MU column coming from Customer data and using CD_COUNTRY to create MU
pricing_join_cust = pricing_join_cust.withColumn("MU", when(pricing_join_cust.CD_COUNTRY == "138", lit("ES")).when(pricing_join_cust.CD_COUNTRY == "119", lit("PT")))
display(pricing_join_cust)

# COMMAND ----------

# checking the customer mapping, after and before
pricing_join_cust.count(), prcng_df.count()

# COMMAND ----------

##Product Mapping

#Filtering the product maping data for Spain and Portugal
product_mapping_fil = product_mapping.filter((product_mapping.CD_PROD_COUNTRY == "138") | (product_mapping.CD_PROD_COUNTRY == "119"))
product_case_fil = product_case.filter((product_case.CD_PROD_COUNTRY == "138") | (product_case.CD_PROD_COUNTRY == "119"))

# COMMAND ----------

# preparing distinct product mapping table source wise
product_sap_1 = (product_mapping_fil.filter("CD_PRODUCT is not null and CD_PROD_ORIGIN = 'SAP'")
                              .withColumn("PROD_CD", expr("concat(CD_PRODUCT,'_04')"))
                              .select("CD_PROD_SELLING","PROD_CD", lit("SAP").alias("SOURCE"))
                 )
print('From Product Mapping:')
print("SAP:", product_sap_1.count(), 'SAP Distinct:',product_sap_1.distinct().count())
# need to add another filter logic, and CD
product_g2m_1 = (product_mapping_fil.filter("CD_PRODUCT is not null and CD_PROD_ORIGIN = 'G2M'")
                              .withColumn("PROD_CD", expr("concat(CD_PRODUCT,'_05')"))
                              .select("CD_PROD_SELLING","PROD_CD", lit("G2M").alias("SOURCE"))
                 )
print("G2M:", product_g2m_1.count(), 'G2M Distinct:', product_g2m_1.distinct().count())
# product case table
product_sap_2 = (product_case_fil.filter("CD_PRODUCT is not null and CD_PROD_ORIGIN = 'SAP'")
                              .withColumn("PROD_CD", expr("concat(CD_PRODUCT,'_04')"))
                              .withColumnRenamed("CD_PRODUCT_SELLING","CD_PROD_SELLING")
                              .select("CD_PROD_SELLING","PROD_CD", lit("SAP").alias("SOURCE"))
                 )
print('From Product Case Mapping:')
print("SAP:", product_sap_2.count(), 'SAP Distinct: ',product_sap_2.distinct().count())

product_sap = product_sap_1.union(product_sap_2).dropDuplicates()
print("SAP Distinct Total:", product_sap.count())

product_g2m_2 = (product_case_fil.filter("CD_PRODUCT is not null and CD_PROD_ORIGIN = 'G2M'")\
                                .withColumn("PROD_CD", expr("concat(CD_PRODUCT,'_05')"))\
                                .withColumnRenamed("CD_PRODUCT_SELLING","CD_PROD_SELLING")\
                                .select("CD_PROD_SELLING","PROD_CD", lit("G2M").alias("SOURCE"))\
                 )
print("G2M:", product_g2m_2.count(), 'G2M Distinct: ',product_g2m_2.distinct().count())

product_g2m_3 = (product_case_fil.filter("CD_PRODUCT_CASE is not null")
                                .withColumn("PROD_CD", expr("concat(CD_PRODUCT_CASE,'_05')"))
                                .withColumnRenamed("CD_PRODUCT_SELLING","CD_PROD_SELLING")
                                .select("CD_PROD_SELLING","PROD_CD", lit("G2M").alias("SOURCE"))
                 )
print("G2M:", product_g2m_3.count(), 'G2M Distinct: ',product_g2m_3.distinct().count())

product_g2m = product_g2m_1.union(product_g2m_2).union(product_g2m_3).dropDuplicates()
print("G2M Distinct Total:", product_g2m.count())
# as400
product_as400 = (product_case_fil.filter("CD_PCASE_ALTERNATE_CODE_1 is not null")
                                  .withColumn("PROD_CD", expr("concat(CD_PCASE_ALTERNATE_CODE_1,'_06')"))
                                  .withColumnRenamed("CD_PRODUCT_SELLING","CD_PROD_SELLING")
                                  .select("CD_PROD_SELLING","PROD_CD", lit("AS400").alias("SOURCE"))
                   )
print("AS400:", product_as400.count(), 'AS400 Distinct:', product_as400.distinct().count())

# union all and drop duplicates to have a full mapping table
product_mapping_full = product_sap.union(product_g2m).union(product_as400).dropDuplicates()
print('Distinct Total:', product_mapping_full.count())

# COMMAND ----------

# Mapping
Pricing_Harmonsized = pricing_join_cust.join(product_mapping_full, on = 'CD_PROD_SELLING', how = 'left')\
                                       .withColumnRenamed('DMDGroup','CUST_GRP')\
                                       .select("ID_TIME_WEEK","ID_TIME_MONTH","PROD_CD","CUST_GRP","ME_TOTAL_DISCOUNTS_ACT","ME_GROSS_SALES_ACT","ME_SALES_UNITS_BAG_ACT","MU")
print(Pricing_Harmonsized.count())
display(Pricing_Harmonsized)

# COMMAND ----------

#Price Calculation
Pricing_Harmonsized = Pricing_Harmonsized.groupby("ID_TIME_WEEK","ID_TIME_MONTH","PROD_CD","CUST_GRP", "MU").agg({'ME_GROSS_SALES_ACT': 'sum','ME_TOTAL_DISCOUNTS_ACT': 'sum','ME_SALES_UNITS_BAG_ACT': 'sum'})
Pricing_Harmonsized = Pricing_Harmonsized.withColumn("LIST_PRICE",col("sum(ME_GROSS_SALES_ACT)")/col("sum(ME_SALES_UNITS_BAG_ACT)"))
Pricing_Harmonsized = Pricing_Harmonsized.withColumn("NET_PRICE",(col("sum(ME_GROSS_SALES_ACT)")-col("sum(ME_TOTAL_DISCOUNTS_ACT)"))/col("sum(ME_SALES_UNITS_BAG_ACT)"))

display(Pricing_Harmonsized)

# COMMAND ----------

#Creating final silver df
IBP_SL_PRCNG = Pricing_Harmonsized.withColumnRenamed('sum(ME_TOTAL_DISCOUNTS_ACT)','ME_TOTAL_DISCOUNTS_ACT_SUM')\
.withColumnRenamed('sum(ME_SALES_UNITS_BAG_ACT)','ME_SALES_UNITS_BAG_ACT_SUM')\
.withColumnRenamed('sum(ME_GROSS_SALES_ACT)','ME_GROSS_SALES_ACT_SUM')\
.withColumnRenamed('LIST_PRICE','ACTL_GRSS_LIST_PRC')\
.withColumnRenamed('NET_PRICE','ACTL_NET_LIST_PRC')\
.withColumnRenamed('ID_TIME_WEEK','WEEK_OF_YEAR')\
.withColumnRenamed('ID_TIME_MONTH','MONTH_OF_YEAR')

# COMMAND ----------

#Formatting WEEK_OF_YEAR
IBP_SL_PRCNG = IBP_SL_PRCNG.withColumn("WEEK_OF_YEAR", concat(substring(col('WEEK_OF_YEAR'),0,4),substring(col('WEEK_OF_YEAR'),7,2)).cast(IntegerType()))\
                           .withColumn("MONTH_OF_YEAR", col('MONTH_OF_YEAR').cast(IntegerType()))
display(IBP_SL_PRCNG)

# COMMAND ----------

IBP_SL_PRCNG.count()

# COMMAND ----------

## DQ Check
pklist2 = pkList.split(';')
pklist2 = ','.join(str(e) for e in pklist2)
if len(pklist2.split(','))>1:
  ls = ["col('"+attr+"').isNull()" for attr in pklist2.split(',')]
  null_cond = " | ".join(ls)
  print(null_cond)
else :
  null_cond = "col('"+pklist2+"').isNull()"
  print(null_cond)

# COMMAND ----------

#Duplicate removal on PK
prcng_dq = IBP_SL_PRCNG.withColumn("DUP_CHECK",row_number().over(Window.partitionBy(pklist2.split(','))
                    .orderBy(desc("ACTL_GRSS_LIST_PRC")))).withColumn("Corrupt_Record",when(col("DUP_CHECK") > 1 , lit("Duplicate Row based on Primary Key")))
silver_df = prcng_dq.where(col("Corrupt_Record").isNull()).drop("Corrupt_Record","DUP_CHECK")
prcng_dq_not_pass = prcng_dq.where(col("Corrupt_Record").isNotNull())
print(prcng_dq_not_pass.count())

# COMMAND ----------

display(prcng_dq_not_pass)

# COMMAND ----------

silver_count = silver_df.count()
print(silver_count)

# COMMAND ----------

#Null check on PK - CUST_GRP;PROD_CD;MONTH_OF_YEAR;WEEK_OF_YEAR;MU 
print("Number of records with NULL in CUST_GRP: "+str(silver_df.filter(silver_df.CUST_GRP.isNull()).count()))
print("Number of records with NULL in PROD_CD: "+str(silver_df.filter(silver_df.PROD_CD.isNull()).count()))
print("Number of records with NULL in MONTH_OF_YEAR: "+str(silver_df.filter(silver_df.MONTH_OF_YEAR.isNull()).count()))
print("Number of records with NULL in WEEK_OF_YEAR: "+str(silver_df.filter(silver_df.WEEK_OF_YEAR.isNull()).count()))
print("Number of records with NULL in MU: "+str(silver_df.filter(silver_df.MU.isNull()).count()))

# COMMAND ----------

#Removing the records with nulls in primary key columns
silver_df = silver_df.filter(silver_df.CUST_GRP.isNotNull()).filter(silver_df.PROD_CD.isNotNull()).filter(silver_df.MU.isNotNull()).filter(silver_df.MONTH_OF_YEAR.isNotNull()).filter(silver_df.WEEK_OF_YEAR.isNotNull())

# COMMAND ----------

silver_count = silver_df.count()
print(silver_count)

# COMMAND ----------

silver_df = silver_df.withColumn("PROCESS_DATE",current_timestamp())\
.withColumn("ME_TOTAL_DISCOUNTS_ACT_SUM",col("ME_TOTAL_DISCOUNTS_ACT_SUM").cast('float'))\
.withColumn("ME_SALES_UNITS_BAG_ACT_SUM",col("ME_SALES_UNITS_BAG_ACT_SUM").cast('float'))\
.withColumn("ME_GROSS_SALES_ACT_SUM",col("ME_GROSS_SALES_ACT_SUM").cast('float'))\
.withColumn("ACTL_GRSS_LIST_PRC",col("ACTL_GRSS_LIST_PRC").cast('float'))\
.withColumn("ACTL_NET_LIST_PRC",col("ACTL_NET_LIST_PRC").cast('float'))

# COMMAND ----------

display(silver_df)

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
    .tableName("sc_ibp_silver.pricing") \
    .location(tgtPath) \
    .execute()

# COMMAND ----------

# remove versions which are older than 336 hrs(14 days)
deltaTable = DeltaTable.forPath(spark, tgtPath)
deltaTable.vacuum(336)

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE sc_ibp_silver.pricing

# COMMAND ----------

dbutils.notebook.exit("recordCount : "+str(silver_count))

# COMMAND ----------

