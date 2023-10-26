# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred

# COMMAND ----------

# DBTITLE 1,Reading the source and Bronze path tables Update date part to date of pipeline execution
edl_bronze_path_incr = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp/pricing"
edl_bronze_path_hist = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/bronze/iberia/europe-dl/ibp/pricing_hist"
edl_source_path_1 = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/pricing/datepart=2021-09-03/"
edl_source_path_2 = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/pricing/datepart=2021-09-21/"
edl_source_path_3 = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/pricing/datepart=2021-10-07/"
edl_source_path_4 = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/archive/iberia/europe-dl/ibp/pricing/datepart=2021-10-11/"

# COMMAND ----------

source_df_1 = spark.read.format("parquet").load(edl_source_path_1).drop('YEAR')
source_df_2 = spark.read.format("parquet").load(edl_source_path_2)
source_df_3 = spark.read.format("parquet").load(edl_source_path_3)
source_df_4 = spark.read.format("parquet").load(edl_source_path_4)
source_df=source_df_1.unionByName(source_df_2.unionByName(source_df_3.unionByName(source_df_4)))

# COMMAND ----------

bronze_df_incr=spark.read.format("delta").load(edl_bronze_path_incr)
bronze_df_hist=spark.read.format("delta").load(edl_bronze_path_hist)
bronze_df=bronze_df_incr.unionByName(bronze_df_hist)

# COMMAND ----------

#Source and Bronze Layer Count Validation for EDL
#print("Source Layer Count is "+str(
display(source_df.select(countDistinct('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE')))
print("Below Bronze Layer Count is ")
print(bronze_df.count())

# COMMAND ----------

#Source and Bronze layer column validation for EDL
src_col =  source_df.columns
brnz_col = bronze_df.columns

# COMMAND ----------

print("Missing Column in bronze:", (set(src_col).difference(brnz_col)))

# COMMAND ----------

print("Additional column in Bronze :", (set(brnz_col).difference(src_col)))

# COMMAND ----------

print(len(source_df.columns))
print(len(bronze_df.columns))

# COMMAND ----------

#Source to Bronze Layer - Minus check for source file vs Bronze layer and vice versa for EDW
print("Count of Missing Rows in Bronze are " + str(+(source_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct().exceptAll(bronze_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct())).count()))
print("Count of New Rows in Bronze are " + str(+(bronze_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct().exceptAll(source_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct())).count()))

# COMMAND ----------

#EDL Source Layer Primary Key Uniqueness check
print("source count")
print(source_df.count())
print("source grouby count on PK fields")
print(source_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct().count())
#EDL Bronze Layer Primary Key Uniqueness check
print("bronze count")
print(bronze_df.count())#.filter("process_date = '2021-10-11'").count())
print("bronze group by count on PK fields")
print(bronze_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').distinct().count())
#.filter("process_date = '2021-10-11'")

# COMMAND ----------

#EDL Source Layer PK Null check
source_df = source_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE')
source_df_agg = source_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in source_df.columns])
display(source_df_agg)

bronze_df = bronze_df.select('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE')
bronze_df_agg = bronze_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in bronze_df.columns])
display(bronze_df_agg)


# COMMAND ----------

#EDL Source Layer PK Duplicate check
source_df.groupby('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDl Bronze Layer PK Duplicate check
bronze_df.groupby('ID_TIME_WEEK','CD_PROD_SELLING','CD_CUSTOMER_SOURCE').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#Bronze to Silver Layer - Count match between Bronze layer table and Silver layer table
silver_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp/pricing"
silver_df = spark.read.format("delta").load(silver_path)

# COMMAND ----------

bronze_df_incr=spark.read.format("delta").load(edl_bronze_path_incr)
bronze_df_hist=spark.read.format("delta").load(edl_bronze_path_hist)
bronze_df=bronze_df_incr.unionByName(bronze_df_hist)

# COMMAND ----------

print("Bronze Layer Count is "+str(bronze_df.count()))#filter(" process_date = '2021-10-11' ")
print("Silver Layer Count is "+str(silver_df.count()))#.filter(" process_date = '2021-10-11' ")

# COMMAND ----------

dpndntdatapath = 'bronze/iberia/as400/ibp/customer-master;bronze/iberia/sap/ibp/customer-master;bronze/iberia/gtm-oracle/ibp/customer-master;bronze/iberia/edw/ibp/product-master/;bronze/iberia/europe-dl/ibp/product-case-mapping;bronze/iberia/europe-dl/ibp/product-mapping;bronze/iberia/config/customer-mapping-pricing-pt/'
dpndntdatapath_list = dpndntdatapath.split(";")
for path in dpndntdatapath_list:
  srcPath = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/"+path
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
prcng_df_es = bronze_df.filter(bronze_df.CD_COUNTRY=='138')
prcng_df_pt = bronze_df.filter(bronze_df.CD_COUNTRY=='119')
prcng_df_es = (prcng_df_es.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df_es.CD_CUSTOMER_SOURCE))
                          .withColumn('COUNTRY', lit("ES"))
              )
prcng_df_pt = (prcng_df_pt.withColumn('CD_CUSTOMER_SOURCE', trim(prcng_df_pt.CD_CUSTOMER_SOURCE))
                          .withColumn('CD_CUSTOMER_SOURCE', expr("right(CD_CUSTOMER_SOURCE, length(CD_CUSTOMER_SOURCE)-2)")) #remove the leading '90' in CD_CUSTOMER_SOURCE
                           .withColumn('COUNTRY', lit("PT"))
              )
#prcng_df = prcng_df_es.union(prcng_df_pt)-----------Line Commented out in new code

# COMMAND ----------

#Customer Maping
df1 = (customerMappingDF_Beverages.withColumn("Source",lit("SAP")).withColumn("Source_Cat",lit("Beverage"))
                                  .select(col("Customer_Id").alias("Customer_Code")
                                         ,col("Demand_Group").alias("DMDGroup")
                                         ,col("Customer_Name").alias("Desc")
                                         ,col("Source_Cat")
                                         ,col("Source"))
      )
df1 = df1.withColumn('Customer_Code', trim(df1.Customer_Code))

df2 = (customerMappingDF_Snacks.withColumn("Source",lit("GTM")).withColumn("Source_Cat",lit("Snacks"))
                              .select(col("Customer_Id").alias("Customer_Code")
                                     ,col("Demand_Group").alias("DMDGroup")
                                     ,col("Customer_Name").alias("Desc")
                                     ,col("Source_Cat")
                                     ,col("Source"))
      )
df2 = df2.withColumn('Customer_Code', trim(df2.Customer_Code))

Customer_Groups_Combined_DF =df1.union(df2)
Customer_Groups_Combined_DF = Customer_Groups_Combined_DF.withColumn("MU", expr("left(DMDGroup, 2)")).filter("MU == 'ES'") #-----adding for MU and filtering for ES

# COMMAND ----------

new_map = new_map.drop_duplicates()
new_map = new_map.select("CD_CUSTOMER","Actual Prolink Dmd Group","DE_CUSTOMER_NAME")
new_map.display()

print('PT SalesBI Count:',prcng_df_pt.count(),'PT SalesBI Distinct CD_CUSTOMER Count:',prcng_df_pt.drop_duplicates(['CD_CUSTOMER']).count())
#mapping customers for PT
pt_dts = prcng_df_pt.join(new_map, on = 'CD_CUSTOMER', how ='left')
print('PT SalesBI Mapped Count:',pt_dts.filter(col("Actual Prolink Dmd Group").isNotNull()).count(),'PT SalesBI Distinct Mapped CD_CUSTOMER Count:',pt_dts.filter(col("Actual Prolink Dmd Group").isNotNull()).drop_duplicates(['CD_CUSTOMER']).count())

pt_dts = pt_dts.withColumnRenamed("Actual Prolink Dmd Group","DMDGroup").withColumn("MU",lit("PT")).withColumn("CT",expr("substring(CD_PROD_SELLING, 1, 1)")).withColumnRenamed("DE_CUSTOMER_NAME","Des").withColumn("Source",lit("GTM")).withColumn("Source_Cat",lit("")).withColumn("Customer_Code",col("CD_CUSTOMER"))

#Joining Pricing and Customer - Spain
pricing_join_cust_es_1 = prcng_df_es.join(Customer_Groups_Combined_DF,  (prcng_df_es.CD_CUSTOMER_SOURCE == Customer_Groups_Combined_DF.Customer_Code), how = 'left')
# Using CD_CUSTOMER to map additional DMDGROUP
not_mapped = pricing_join_cust_es_1.filter('DMDGroup is NULL').select(prcng_df_es.columns)
new_map_es = not_mapped.join(Customer_Groups_Combined_DF,not_mapped.CD_CUSTOMER==Customer_Groups_Combined_DF.Customer_Code,how = 'left')
pricing_join_cust_es = pricing_join_cust_es_1.filter('DMDGroup is NOT NULL').union(new_map_es).withColumn("CT",expr("substring(CD_PROD_SELLING, 1, 1)"))
display(pricing_join_cust_es)

col_sp = ['CD_PROD_SELLING','ID_TIME_WEEK','ID_TIME_MONTH','CD_CUSTOMER_SOURCE','CD_CUSTOMER','CD_SOURCE_SYSTEM','CD_COUNTRY','CD_PROD_CATEGORY','CD_PROD_SUB_CATEGORY','CD_PROD_BRAND', 'CD_PROD_SUB_BRAND','CD_PROD_FLAVOR','CD_PROD_PACK_SIZE','CD_PROD_SIZE','ME_NET_SALES_ACT','ME_SALES_UNITS_P_CASES_ACT','ME_SALES_UNITS_BAG_ACT','ME_SALES_UNITS_RAW_ACT','ME_STALES_EUR_ACT', 'ME_TOTAL_DISCOUNTS_ACT','ME_GROSS_SALES_ACT','ME_VOLUME_L_ACT','ME_VOLUME_KG_ACT','ME_ON_INVOICE_ACT','PROCESS_DATE','COUNTRY','Customer_Code','DMDGroup','Desc','Source_Cat','Source','MU','CT']

col_pt = ['CD_PROD_SELLING','ID_TIME_WEEK','ID_TIME_MONTH','CD_CUSTOMER_SOURCE','CD_CUSTOMER','CD_SOURCE_SYSTEM','CD_COUNTRY','CD_PROD_CATEGORY','CD_PROD_SUB_CATEGORY','CD_PROD_BRAND', 'CD_PROD_SUB_BRAND','CD_PROD_FLAVOR','CD_PROD_PACK_SIZE','CD_PROD_SIZE','ME_NET_SALES_ACT','ME_SALES_UNITS_P_CASES_ACT','ME_SALES_UNITS_BAG_ACT','ME_SALES_UNITS_RAW_ACT','ME_STALES_EUR_ACT', 'ME_TOTAL_DISCOUNTS_ACT','ME_GROSS_SALES_ACT','ME_VOLUME_L_ACT','ME_VOLUME_KG_ACT','ME_ON_INVOICE_ACT','PROCESS_DATE','COUNTRY','Customer_Code','DMDGroup','Des','Source_Cat','Source','MU','CT']


# union PT and Spain
pricing_join_cust = pricing_join_cust_es.selectExpr(col_sp).union(pt_dts.selectExpr(col_pt))
#Update the MU column coming from Customer data and using CD_COUNTRY to create MU
pricing_join_cust = pricing_join_cust.withColumn("MU", when(pricing_join_cust.CD_COUNTRY == "138", lit("ES")).when(pricing_join_cust.CD_COUNTRY == "119", lit("PT")))
display(pricing_join_cust)

# COMMAND ----------

##Product Mapping

#Filtering the product maping data for Spain and Portugal
product_mapping_fil = product_mapping.filter((product_mapping.CD_PROD_COUNTRY == "138") | (product_mapping.CD_PROD_COUNTRY == "119"))
product_case_fil = product_case.filter((product_case.CD_PROD_COUNTRY == "138") | (product_case.CD_PROD_COUNTRY == "119"))

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


# Mapping
Pricing_Harmonsized = pricing_join_cust.join(product_mapping_full, on = 'CD_PROD_SELLING', how = 'left')\
                                       .withColumnRenamed('DMDGroup','CUST_GRP')\
                                       .select("ID_TIME_WEEK","ID_TIME_MONTH","PROD_CD","CUST_GRP","ME_TOTAL_DISCOUNTS_ACT","ME_GROSS_SALES_ACT","ME_SALES_UNITS_BAG_ACT","MU")
#print(Pricing_Harmonsized.count())
display(Pricing_Harmonsized)

# COMMAND ----------

display(Pricing_Harmonsized.where(col("CUST_GRP").isNotNull()).agg(sum(col("ME_SALES_UNITS_BAG_ACT").cast('float'))))

# COMMAND ----------

display(silver_df.select(sum(col("ME_SALES_UNITS_BAG_ACT_SUM").cast('float'))))

# COMMAND ----------

#Silver Layer Column Validation
silver_column_mdl = ["CUST_GRP","PROD_CD","MONTH_OF_YEAR",	"WEEK_OF_YEAR",	"MU","ACTL_GRSS_LIST_PRC","ACTL_NET_LIST_PRC","ME_TOTAL_DISCOUNTS_ACT_SUM","ME_SALES_UNITS_BAG_ACT_SUM","ME_GROSS_SALES_ACT_SUM"]

silver_df_col = silver_df.columns
print("Missing Column in silver:", (set(silver_column_mdl).difference(silver_df_col)))
print("Extra Column in silver:", (set(silver_df_col).difference(silver_column_mdl)))


# COMMAND ----------

#EDL silver Layer Primary Key Uniqueness check
print(silver_df.count())
print(silver_df.select('CUST_GRP','PROD_CD','MONTH_OF_YEAR','WEEK_OF_YEAR').distinct().count())

#EDl silver Layer PK Duplicate check
silver_df.groupby('CUST_GRP','PROD_CD','MONTH_OF_YEAR','WEEK_OF_YEAR').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

#EDL Silver Layer PK Null check
silver_df = silver_df.select('CUST_GRP','PROD_CD','MONTH_OF_YEAR','WEEK_OF_YEAR')
silver_df_agg = silver_df.agg(*[F.count(F.when(F.isnull(c), c)).alias(c) for c in silver_df.columns])
display(silver_df_agg)

# COMMAND ----------

#EDl silver Layer PK Duplicate check
silver_df.groupby('CUST_GRP','PROD_CD','MONTH_OF_YEAR','WEEK_OF_YEAR').count().where('count > 1').sort('count', ascending=False).show()

# COMMAND ----------

