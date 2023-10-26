# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window
from pyspark.sql import Row
# clean column names and combine tables
from pyspark.sql.functions import desc
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import *

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

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$sourceStorageAccount

# COMMAND ----------

#join condition for merge operation
if len(pkList.split(';'))>1:
  ls = ["target."+attr+" = updates."+attr for attr in pkList.split(';')]
  cond = " and ".join(ls)
else :
  cond = "target."+pkList+" = updates."+pkList
cond

# COMMAND ----------

#Reading the data from the bronze path of SFA table
promtion_deltaTable = DeltaTable.forPath(spark, srcPath)
promtion_deltaTable_version = promtion_deltaTable.history().select(max(col('version'))).collect()[0][0]
print(promtion_deltaTable)
display(promtion_deltaTable.history())

# COMMAND ----------

#Reading the SFA source data from bonze layer
SFA_DF = spark.read.format("delta").option("versionAsOf", promtion_deltaTable_version).load(srcPath)

# COMMAND ----------

SFA_DF.count()

# COMMAND ----------

display(SFA_DF)

# COMMAND ----------

dpndntdatapath = dbutils.widgets.get("dependentDatasetPath")
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
    product_SNACK = spark.read.format('delta').load(srcPath)
  if 'product-mapping' in path:
    product_BEV = spark.read.format('delta').load(srcPath)
  if 'product-master' in path:
    productDF = spark.read.format('delta').load(srcPath)
  if 'edw/ibp-poc/customer-master' in path:
    cust = spark.read.format('delta').load(srcPath)
    

# COMMAND ----------

# daily calendar creation
from pyspark.sql import Row

df = spark.sparkContext.parallelize([Row(start_date='2016-01-01', end_date='2025-12-31')]).toDF()
df = df \
  .withColumn('start_date', col('start_date').cast('date')) \
  .withColumn('end_date', col('end_date').cast('date'))\
  .withColumn('cal_date', explode(expr('sequence(start_date, end_date, interval 1 day)'))) 

df = df \
  .withColumn("Week_start_date",date_trunc('week', col("cal_date")))\
  .withColumn("Week_end_date",date_add("Week_start_date",6))\
  .withColumn('week_year',when((year(col('Week_start_date'))==year(col('cal_date'))) &          (year(col('Week_end_date'))==year(col('cal_date'))),year(col('cal_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(52)),year(col('Week_start_date')))\
              .when((year(col('Week_start_date'))!=year(col('Week_end_date'))) &\
                    (weekofyear(col('Week_end_date'))==lit(53)),year(col('Week_start_date')))\
              .otherwise(year('Week_end_date')))\
  .withColumn('month_year',year(col('cal_date')))\
  .withColumn('week',when((year(col('Week_start_date'))==year(col('Week_end_date'))),weekofyear(col("Week_end_date")))\
                     .otherwise(weekofyear(col("Week_end_date"))))\
  .withColumn('month',month("cal_date"))

calendar_df=df\
  .withColumn('Week_Of_Year',df.week_year*lit(100)+df.week)\
  .withColumn('Month_Of_Year',df.month_year*lit(100)+df.month)\
  .withColumn('Month_Of_Year_WSD',year(col('Week_start_date'))*lit(100)+month("Week_start_date"))\
  .withColumn("flag",lit(1))\
  .select('cal_date','Week_start_date','Week_end_date','Week_Of_Year','Month_Of_Year_WSD')

# COMMAND ----------

# clean column names and combine tables
from pyspark.sql.functions import desc
from pyspark.sql.functions import lit
from pyspark.sql.functions import col
from pyspark.sql.functions import *

# rename and select fields
df1 = customerMappingDF_Beverages.withColumn("Source_Cat",lit("Beverage")).select(col("Customer_Id")
          .alias("Customer_Code"),
                 col("Demand_Group")
          .alias("DMDGroup"),
                 col("Customer_Name")
          .alias("Desc"),"Source_Cat")

df1 = df1.withColumn('Customer_Code', trim(df1.Customer_Code))

df2 = customerMappingDF_Snacks.withColumn("Source_Cat",lit("Snacks")).select(col("Customer_Id")
          .alias("Customer_Code"),
                 col("Demand_Group")
          .alias("DMDGroup"),
                 col("Customer_Name")
          .alias("Desc"),"Source_Cat")

df2 = df2.withColumn('Customer_Code', trim(df2.Customer_Code))


df3 = customerMappingDF_Alvalle.withColumn("Source_Cat",lit("Alvalle")).select(col("Customer_ID")
          .alias("Customer_Code"),col("Demand_Group")
          .alias("DMDGroup"),
           col("Customer_Name")
          .alias("Desc"),
          "Source_Cat")

df3 = df3.withColumn('Customer_Code', trim(df3.Customer_Code))

# this is the three customer tables appended onto one another
Customer_Groups_Combined_DF = df3.union(df1.union(df2))

# COMMAND ----------

product_AV = spark.read.csv("/FileStore/tables/temp/Alvelle_Mapping.csv", header="true", inferSchema="true")

# COMMAND ----------

SFA_DF_mapped = SFA_DF.join(Customer_Groups_Combined_DF, col("Promo_Cadena") == Customer_Groups_Combined_DF.Desc, "left")

# COMMAND ----------

SFA_DF_filter = SFA_DF.select("dmdgroups", "Producto_ID", "Promo_Start", "Promo_Finish", "Promo_Display_Type", "Promo_Product_Pack_ID")
print(SFA_DF_filter.count())
SFA_DF_filter = SFA_DF_filter.withColumnRenamed('dmdgroups','DMDGROUP')\
                             .withColumnRenamed('Producto_ID','Product_ID')\
                             .withColumnRenamed('Promo_Start','Start_Date')\
                             .withColumnRenamed('Promo_Finish','End_Date')\
                             .withColumnRenamed('Promo_Display_Type','Promo_Display_Type')\
                             .withColumnRenamed('Promo_Product_Pack_ID','Prod_Selling_SKU').distinct()
print(SFA_DF_filter.count())
display(SFA_DF_filter)

# COMMAND ----------

def get_date_format(df, col_name):  
  
  split_col = split(df[col_name], '/')
 
  df = df.withColumn("month", split_col.getItem(0)).withColumn("day", split_col.getItem(1)).withColumn("year", split_col.getItem(2))
  df = df.withColumn("month", when(length(col("month")) == 1, concat(lit('0'), col("month"))).otherwise(col("month")))
  df = df.withColumn("day", when(length(col("day")) == 1, concat(lit('0'), col("day"))).otherwise(col("day")))
  df = df.withColumn(col_name, concat(col("year"), lit('-'), col("month"), lit('-'), col("day")))
  df = df.withColumn(col_name, to_date(col(col_name),"yyyy-MM-dd")).drop("month", "day", "year")
 
  return df

# COMMAND ----------

es_promo = get_date_format(get_date_format(SFA_DF_filter, "Start_Date"), "End_Date")
print (es_promo.count())

# Join with calendar to get TIME_VAR, WEEK start and end dates
es_promo = es_promo.crossJoin(calendar_df).filter((col('cal_date')>=col('Start_Date')) & (col('cal_date')<=col('End_Date')))                    
print (es_promo.count())
es_promo = es_promo.withColumn("Day_Of_Week", datediff(col("cal_date"), col("Week_start_date")) + 1)
es_promo = es_promo.withColumn("weekend_flag", when(col("Day_Of_Week") >= 6, lit(1)).otherwise(0))

# COMMAND ----------

# Create additional features
promo_day_count_level = list(set(es_promo.columns) - set(["cal_date", "Month_Of_Year_WSD", "Day_Of_Week", "weekend_flag"]))
es_promo  = es_promo.withColumn('no_of_promo_days', max('Day_Of_Week').over(Window.partitionBy(promo_day_count_level)) - \
                                min('Day_Of_Week').over(Window.partitionBy(promo_day_count_level)) + lit(1))
es_promo  = es_promo.withColumn('no_of_weekend_promo_days', sum('weekend_flag').over(Window.partitionBy(promo_day_count_level)))\
                     .drop("cal_date", "Day_Of_Week", "weekend_flag").withColumnRenamed("Month_Of_Year_WSD", "Month_Of_Year")
print (es_promo.count())
es_promo = es_promo.dropDuplicates()
print (es_promo.count())

# COMMAND ----------

display(es_promo)  # 10536, 2021-01-02 - 2021-01-13, 1509, 2021-01-01, 2021-01-31

# COMMAND ----------

dmdunit_mapping_SNACK = product_SNACK.select("CD_PRODUCT_SELLING","CD_PRODUCT_CASE")
dmdunit_mapping_BEV = product_BEV.select(col("CD_PROD_SELLING").alias("CD_PRODUCT_SELLING"),col("CD_PRODUCT").alias("CD_PRODUCT_CASE"))
dmdunit_mapping_AV = product_AV.select(col("CD_PROD_SELLING").alias("CD_PRODUCT_SELLING"),col("DMDUnit").alias("CD_PRODUCT_CASE"))

#Split the data by category
es_promo_split = es_promo.withColumn("CT",expr("substring(Prod_Selling_SKU, 1, 1)"))
SFA_SNACK_DISTINCT = es_promo_split.filter(es_promo_split.CT == "F").select('Prod_Selling_SKU').distinct()
SFA_ALVELLE_DISTINCT = es_promo_split.filter(es_promo_split.CT == "A").select('Prod_Selling_SKU').distinct()
SFA_BEV_DISTINCT = es_promo_split.filter(es_promo_split.CT == "B").select('Prod_Selling_SKU').distinct()

SFA_SNACK_DISTINCT = SFA_SNACK_DISTINCT.join(dmdunit_mapping_SNACK, SFA_SNACK_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_SNACK.CD_PRODUCT_SELLING , how='left')
SFA_BEV_DISTINCT = SFA_BEV_DISTINCT.join(dmdunit_mapping_BEV, SFA_BEV_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_BEV.CD_PRODUCT_SELLING , how='left')
SFA_ALVELLE_DISTINCT = SFA_ALVELLE_DISTINCT.join(dmdunit_mapping_AV, SFA_ALVELLE_DISTINCT.Prod_Selling_SKU == dmdunit_mapping_AV.CD_PRODUCT_SELLING , how='left')

SFA_ALL_DISTINCT = SFA_SNACK_DISTINCT.union(SFA_BEV_DISTINCT.union(SFA_ALVELLE_DISTINCT))

print(SFA_SNACK_DISTINCT.count())
print(SFA_BEV_DISTINCT.count())
print(SFA_ALVELLE_DISTINCT.count())
print(SFA_ALL_DISTINCT.count() == SFA_SNACK_DISTINCT.count() + SFA_BEV_DISTINCT.count() + SFA_ALVELLE_DISTINCT.count())

# COMMAND ----------

display(SFA_SNACK_DISTINCT)
display(SFA_BEV_DISTINCT)
display(SFA_ALVELLE_DISTINCT)


# COMMAND ----------

display(SFA_ALL_DISTINCT)

# COMMAND ----------

display(es_promo_split)

# COMMAND ----------

productDF_DMDUNIT = productDF.withColumn("DMDunit_Product_master",expr("substring(PLANG_MTRL_GRP_VAL, 1, length(PLANG_MTRL_GRP_VAL)-3)")) 
SFA_ALL_DISTINCT_DMDUNIT = SFA_ALL_DISTINCT.join(productDF_DMDUNIT, SFA_ALL_DISTINCT.CD_PRODUCT_CASE == productDF_DMDUNIT.DMDunit_Product_master , how='left')\
                                    .select("Prod_Selling_SKU", "PLANG_MTRL_GRP_VAL").distinct()

# COMMAND ----------

display(SFA_ALL_DISTINCT_DMDUNIT)

# COMMAND ----------

print(es_promo_split.count())
es_promo_DMD = es_promo_split.join(SFA_ALL_DISTINCT_DMDUNIT, on = "Prod_Selling_SKU", how = "left").withColumnRenamed("PLANG_MTRL_GRP_VAL", "DMDUNIT").drop("CT")

# Create unique promo id level
es_promo_id = es_promo_DMD.withColumn("promo_id", concat(col("DMDGROUP"), col("DMDUNIT"), col("start_date"), col("end_date"), col("Promo_Display_Type")))


# COMMAND ----------

def get_dummies(df, columns, delim=None):
  """
  Creates dummy variables for unique values of list of columns

  Parameters
  ----------
  df : PySpark dataframe
      Input dataset
  columns : List
      List of columns which you want to create dummy variables
  delim : Str
      String containing delimeter that should separate input column values - indicators will be created for all values separated by delim

  Returns
  -------
  df : PySpark dataframe
      Input dataset with appended dummy fields
  """
  #TO-DO: Get rid of loop
  if isinstance(columns, str):
      columns = [columns]

  for c in columns:
    if c in df.columns:
      unique_values = df.select(c).dropDuplicates()
      unique_values = convertDFColumnsToList(unique_values, c)
      unique_values = ['NULL' if x==None else x for x in unique_values]
      unique_values = [str(x) if isinstance(x, str)==False  else x for x in unique_values] #Handle numeric unique types
      if delim is None:
        indicator_dict = {i: [(c+'_'+i).replace(" ","_")] for i in unique_values} #Create dictionary from values
      else:
        indicator_dict = {i: [(c+'_'+j).replace(" ","_") for j in str.split(i,delim)] for i in unique_values} #Sep delim dictionary

      #Initialize to 0
      for ind in list(itertools.chain(*indicator_dict.values())):
        df = df.withColumn(ind, lit(0))

      #Create indicators
      for this_key in indicator_dict.keys():
          for this_indicator in indicator_dict[this_key]:
            if this_key == 'NULL':
                df = df.withColumn(this_indicator, when(col(c).isNull(), lit(1)).otherwise(col(this_indicator)))
            else:
              df = df.withColumn(this_indicator, when(col(c)==this_key, lit(1)).otherwise(col(this_indicator)))

  return(df)


# COMMAND ----------

display(es_promo_id)

# COMMAND ----------

print(es_promo_id.select("Prod_Selling_SKU").distinct().count())
print(es_promo_id.select("Prod_Selling_SKU", "DMDUNIT").distinct().count())

# COMMAND ----------

es_promo_id = get_dummies(es_promo_id, ["Promo_Display_Type"])

# COMMAND ----------

def agg_promo(time_var):
  
  df = es_promo_id.drop("Week_start_date", "Week_end_date", "start_date", "end_date", "Promo_Display_Type").drop_duplicates()
  
  if time_var == "Month_Of_Year":
    df = df.drop("Week_Of_Year").drop_duplicates()
    
  group_cols = ["DMDGROUP", "DMDUNIT", time_var, "promo_id"]  
  sum_cols = [c for c in df.columns if "no_of" in c]
  max_cols = [c for c in df.columns if "Promo_Display_Type_" in c]
  
  sum_dict = {x: "sum" for x in sum_cols}
  max_dict = {x: "max" for x in max_cols}
  all_dict = {**sum_dict, **max_dict}
  
  df = df.groupBy(group_cols).agg(all_dict).drop("promo_id")
  
  for i in sum_cols+max_cols:
    if all_dict[i] == "sum":
      df =  df.withColumnRenamed("sum(" + i + ")", i)
    elif all_dict[i] == "max":
      df =  df.withColumnRenamed("max(" + i + ")", i)
      
  df = df.groupBy(["DMDGROUP", "DMDUNIT", time_var]).agg({x: "max" for x in sum_cols + max_cols})
  
  for i in sum_cols+max_cols:
    df = df.withColumnRenamed("max(" + i + ")", i)
    df = df.withColumnRenamed(i, i.replace("_Type", "").replace("/_", ""))
    
  df = df.withColumnRenamed("Promo_Display_Box_Pallet_¼_Pallet", "Promo_Display_Box_Pallet_Quarter_Pallet")\
          .withColumnRenamed("Promo_Display_½_Pallet", "Promo_Display_Half_Pallet")
  return df

# COMMAND ----------

es_promo_agg_weekly = agg_promo("Week_Of_Year")
es_promo_agg_monthly = agg_promo("Month_Of_Year")

# COMMAND ----------

print(es_promo_agg_weekly.count() == es_promo_agg_weekly.distinct().count() == es_promo_agg_weekly.select("DMDGROUP", "DMDUNIT", "Week_Of_Year").distinct().count())
print(es_promo_agg_weekly.count())
print(es_promo_agg_monthly.count() == es_promo_agg_monthly.distinct().count() == es_promo_agg_monthly.select("DMDGROUP", "DMDUNIT", "Month_Of_Year").distinct().count())
print(es_promo_agg_monthly.count())

# COMMAND ----------

display(es_promo_agg_weekly)

# COMMAND ----------

display(es_promo_agg_monthly)

# COMMAND ----------

es_promo_agg_weekly.filter(col("DMDGROUP").isNotNull()).count()/es_promo_agg_weekly.count()

# COMMAND ----------

es_promo_agg_monthly.filter(col("DMDGROUP").isNotNull()).count()/es_promo_agg_monthly.count()

# COMMAND ----------

es_promo_agg_weekly.filter(col("DMDUNIT").isNotNull()).count()/es_promo_agg_weekly.count()

# COMMAND ----------

es_promo_agg_monthly.filter(col("DMDUNIT").isNotNull()).count()/es_promo_agg_monthly.count()

# COMMAND ----------

# MAGIC %md ### Mapping the DTS DemandGroups to Servicio

# COMMAND ----------

cust = load_delta(DBI_CUSTOMER)
cust_es_dts = cust.filter(col("HRCHY_LVL_2_ID") == "ES_DTS").withColumn("DMDGROUP", lit("SERVICIO DC")).select("DMDGROUP", "PLANG_CUST_GRP_VAL").distinct()

# COMMAND ----------

display(cust.select("PLANG_CUST_GRP_VAL").distinct())

# COMMAND ----------

prev_count = es_promo_agg_weekly.filter(col("DMDGROUP") == "SERVICIO DC").count()
es_promo_agg_weekly = es_promo_agg_weekly.join(cust_es_dts, on = "DMDGROUP", how = "left")
es_promo_agg_weekly = es_promo_agg_weekly.withColumn("DMDGROUP", F.when(F.col("DMDGROUP") == "SERVICIO DC", \
                                                                        F.col("PLANG_CUST_GRP_VAL")).otherwise(F.col("DMDGROUP"))).drop("PLANG_CUST_GRP_VAL")
print(prev_count*2 == es_promo_agg_weekly.filter(col("DMDGROUP") == "ES_DTS_DISTRIBUTORS").count() + es_promo_agg_weekly.filter(col("DMDGROUP") == "ES_DTS_OTHERS").count())

# COMMAND ----------

prev_count = es_promo_agg_monthly.filter(col("DMDGROUP") == "SERVICIO DC").count()
es_promo_agg_monthly = es_promo_agg_monthly.join(cust_es_dts, on = "DMDGROUP", how = "left")
es_promo_agg_monthly = es_promo_agg_monthly.withColumn("DMDGROUP", F.when(F.col("DMDGROUP") == "SERVICIO DC", \
                                                                        F.col("PLANG_CUST_GRP_VAL")).otherwise(F.col("DMDGROUP"))).drop("PLANG_CUST_GRP_VAL")
print(prev_count*2 == es_promo_agg_monthly.filter(col("DMDGROUP") == "ES_DTS_DISTRIBUTORS").count() + es_promo_agg_monthly.filter(col("DMDGROUP") == "ES_DTS_OTHERS").count())

# COMMAND ----------

print(es_promo_agg_weekly.count() == es_promo_agg_weekly.distinct().count() == es_promo_agg_weekly.select("DMDGROUP", "DMDUNIT", "Week_Of_Year").distinct().count())
print(es_promo_agg_weekly.count())
print(es_promo_agg_monthly.count() == es_promo_agg_monthly.distinct().count() == es_promo_agg_monthly.select("DMDGROUP", "DMDUNIT", "Month_Of_Year").distinct().count())
print(es_promo_agg_monthly.count())

# COMMAND ----------

display(es_promo_agg_weekly)

# COMMAND ----------

display(es_promo_agg_monthly)

# COMMAND ----------

# MAGIC %md ### Write out the datasets

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(es_promo_agg_weekly, DBI_PROMO_ES_WEEKLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_ES_WEEKLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

## Write as delta table to dbfs
save_df_as_delta(es_promo_agg_monthly, DBI_PROMO_ES_MONTHLY, enforce_schema=False)
delta_info = load_delta_info(DBI_PROMO_ES_MONTHLY)
set_delta_retention(delta_info, '90 days')
display(delta_info.history())

# COMMAND ----------

