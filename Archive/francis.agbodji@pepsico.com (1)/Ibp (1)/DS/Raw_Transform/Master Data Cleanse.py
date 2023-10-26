# Databricks notebook source
# MAGIC %md 
# MAGIC ##01 - Data Cleansing
# MAGIC 
# MAGIC TO COMPLETE DESCRIPTION WHEN PIPELINE MORE FINALIZED

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
# MAGIC %run ./../src/libraries

# COMMAND ----------

# MAGIC %run ./../src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./../src/load_src

# COMMAND ----------

# MAGIC %run ./../src/config

# COMMAND ----------

#connecting to blob
run_scope_and_key()

# COMMAND ----------

# DBTITLE 1,Load Core Datasets
## Shipments data extract

try:
#   shipments_pd = load_data(SHIPMENT_PATH,format="csv",sep=";").toPandas()
    shipments = load_data(SHIPMENT_PATH,data_format="csv",sep=";")
except:
  dbutils.notebook.exit("Shipments load failed")
  
## Hierarchies across our 3 dimensions 
try:
#   prod_pd = load_data(PRODUCT_PATH,format="csv",sep=";").toPandas()
    prod = load_data(PRODUCT_PATH,data_format="csv",sep=";")
except:
  dbutils.notebook.exit("Product master load failed")
    
try:
#   cust_pd = load_data(CUSTOMER_PATH,format="csv",sep=";").toPandas()
    cust = load_data(CUSTOMER_PATH,data_format="csv",sep=";")
except:
  dbutils.notebook.exit("Customer master load failed")
  
try:
#   loc_dc_pd = load_data(LOCATION_PATH,format="csv",sep=";").toPandas()
  loc_dc = load_data(LOCATION_PATH,data_format="csv",sep=";")
except:
  dbutils.notebook.exit("Location master load failed")
  
try:
  print("Shipment",shipments.count(),len(shipments.columns))
  print("Product",prod.count(),len(prod.columns))
  print("Customer",cust.count(),len(cust.columns))
  print("Location",loc_dc.count(),len(loc_dc.columns))
except:
  dbutils.notebook.exit("Data shape query failed")

# COMMAND ----------

display(cust)

# COMMAND ----------

# DBTITLE 1,Shipments: Review & Cleanse
## "Automated" checks on this dataset - before user adjustments
auto_check_list = []

auto_check_list.append(shipments.select('DMND_HSTRY_STREM_NM').distinct().count() == 1)
auto_check_list.append(shipments.select('DMND_HSTRY_STREM_NM').distinct().first()[0] == 'SHIPMENTS')

auto_check_list.append(shipments.select('DMND_HSTRY_TYP_CDV').distinct().count() == 1)
auto_check_list.append(shipments.select('DMND_HSTRY_TYP_CDV').distinct().first()[0] == 1)

auto_check_list.append(shipments.select('HSTRY_DURTN_MIN_QTY').distinct().count() == 1)
auto_check_list.append(shipments.select('HSTRY_DURTN_MIN_QTY').distinct().first()[0] == 10080)

auto_check_list.append(shipments.select('DMND_HSTRY_EVNT_NM').distinct().count() == 1)
auto_check_list.append(shipments.select('DMND_HSTRY_EVNT_NM').distinct().first()[0] == ' ')

if False in auto_check_list:
  print('User validation required!')

print(auto_check_list)  ## all should read TRUE

# COMMAND ----------

## Stripping down to base columns - min memory and ease merging 
ship_clean = shipments.select(SHIP_DATA_COLS)

## Checking both row/col dimensions
print(ship_clean.count() == shipments.count(), len(ship_clean.columns) == len(SHIP_DATA_COLS))

# COMMAND ----------

# DBTITLE 1,Product Master: Review & Cleanse
prod_clean = prod.drop('DW_PLANG_MTRL_UNIT_ID', 'SYS_ID')\
                 .withColumnRenamed('PLANG_MTRL_GRP_VAL','DMDUNIT')

## Checks - should be TRUE
prod_clean.select('DMDUNIT').distinct().count() == prod.select('PLANG_MTRL_GRP_VAL').distinct().count()

# COMMAND ----------

## Dropping single-value and null columns to reduce noise
dict = eval(prod_clean.select([approx_count_distinct(c).alias(c) for c in prod_clean.columns])\
                       .withColumn('dict', F.to_json(F.struct(prod_clean.columns)))\
                       .select('dict')\
                      .collect()[0].dict)
keep_columns=list({key: value for key, value in dict.items() if value>1}.keys())
prod_clean=prod_clean.select(keep_columns)

## Column drop check
print(keep_columns, prod_clean.count(), len(prod_clean.columns))

# COMMAND ----------

prod_clean= prod_clean.select(PROD_DATA_COLS).withColumnRenamed('DESCR','DESCR_PROD')

## Checking both col dimension
print(len(prod_clean.columns) == len(PROD_DATA_COLS))

# COMMAND ----------

# DBTITLE 1,Customer Master: Review & Cleanse
## "Automated" checks on this dataset - before user adjustments
auto_check_list = []

cust_clean = cust.filter((col('PLANG_CUST_HRCHY_TYP_NM') == 'CLIENT') & (col('PLANG_CUST_GRP_TYP_NM') == 'CLIENT'))

auto_check_list.append(cust_clean.select('PLANG_CUST_HRCHY_TYP_NM').distinct().count() == 1)
auto_check_list.append(cust_clean.select('PLANG_CUST_HRCHY_TYP_NM').distinct().first()[0] == 'CLIENT')

auto_check_list.append(cust_clean.select('PLANG_CUST_GRP_TYP_NM').distinct().count() == 1)
auto_check_list.append(cust_clean.select('PLANG_CUST_GRP_TYP_NM').distinct().first()[0] == 'CLIENT')

if False in auto_check_list:
  print('User validation required!')

print(auto_check_list)  ## all should read TRUE

# COMMAND ----------

## Dropping single-value and null columns to reduce noise
dict = eval(cust_clean.select([approx_count_distinct(c).alias(c) for c in cust_clean.columns])\
                       .withColumn('dict', F.to_json(F.struct(cust_clean.columns)))\
                       .select('dict')\
                      .collect()[0].dict)
keep_columns=list({key: value for key, value in dict.items() if value>1}.keys())
cust_clean=cust_clean.select(keep_columns)

## Column drop check
print(keep_columns, cust_clean.count(), len(cust_clean.columns))

# COMMAND ----------

cust_clean = cust_clean.select(CUST_DATA_COLS).withColumnRenamed('PLANG_CUST_GRP_NM','DESCR_CUST')

## Checking both col dimension
print(len(cust_clean.columns)== len(CUST_DATA_COLS))

# COMMAND ----------

# DBTITLE 1,Location/DC Master: Review & Cleanse
## These checks are outdated - were used for original Location/DC Hierarchy that we were provided
## Adjusted the below checks to fit our new Location/DC Hierarchy

## "Automated" checks on this dataset - before user adjustments
auto_check_list = []

loc_clean = loc_dc.drop('SYS_ID')\
                        .withColumnRenamed('PLANG_LOC_GRP_VAL','LOC')

loc_clean = loc_clean.filter(col('PLANG_LOC_BU_NM').isin(COUNTRY_LIST))
auto_check_list.append(loc_clean.select('PLANG_LOC_BU_NM').distinct().count() == len(COUNTRY_LIST))

loc_clean = loc_clean.filter(col('PLANG_LOC_GRP_TYP_NM')== 'DC')
auto_check_list.append(loc_clean.select('PLANG_LOC_GRP_TYP_NM').distinct().count() == 1)
auto_check_list.append(loc_clean.select('PLANG_LOC_GRP_TYP_NM').distinct().first()[0] == 'DC')

if False in auto_check_list:
  print('User validation required!')

print(auto_check_list)  ## all should read TRUE

# COMMAND ----------

## Dropping single-value columns to reduce noise
dict = eval(loc_clean.select([approx_count_distinct(c).alias(c) for c in loc_clean.columns])\
                       .withColumn('dict', F.to_json(F.struct(loc_clean.columns)))\
                       .select('dict')\
                      .collect()[0].dict)
keep_columns=list({key: value for key, value in dict.items() if value>1}.keys())
loc_clean=loc_clean.select(keep_columns)

## Column drop check
print(keep_columns, loc_clean.count(),len(loc_clean.columns)- len(keep_columns))

# COMMAND ----------

## Dropping some high-null columns - not sure why there are reading in as null
# loc_clean_pd.dropna(thresh=len(loc_clean) - NULL_THRESH, axis=1, inplace=True)

loc_clean = loc_clean.select(LOC_DATA_COLS).withColumnRenamed('DESCR','DESCR_LOC')

## Checking both col dimension
print(len(loc_clean.columns) == len(LOC_DATA_COLS))


# COMMAND ----------

# DBTITLE 1,Drop Noise
## Dropping columns based on first-hand look at the hierarchies and data
## These represent "noisy" or duplicative columns based on Corey's manual review

prod_cols_to_drop = [col_name for col_name in prod_clean.columns if '_prod' in col_name]  ## these are duplicative columns
cust_cols_to_drop = [col_name for col_name in cust_clean.columns if '_cust' in col_name]  ## these are duplicative columns
loc_cols_to_drop = [col_name for col_name in loc_clean.columns if '_loc' in col_name]    ## these are duplicative columns

# final_cols_to_drop = list(set(prod_cols_to_drop + cust_cols_to_drop + loc_cols_to_drop))
if prod_cols_to_drop:
   prod_simple = prod_clean.drop(prod_cols_to_drop)
else :
  prod_simple = prod_clean
if cust_cols_to_drop:
   cust_simple = cust_clean.drop(cust_cols_to_drop)
else :
  cust_simple = cust_clean
if loc_cols_to_drop:
   loc_simple = loc_clean.drop(loc_cols_to_drop)
else :
  loc_simple = loc_clean

## Review results after column drops
print('Trimmed Product DF shape:', prod_simple.count(),len(prod_simple.columns))
print('Trimmed Customer DF shape:', cust_simple.count(),len(cust_simple.columns))
print('Trimmed Location DF shape:', loc_simple.count(),len(loc_simple.columns))




prod_simple_df=prod_simple.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in prod_simple.columns])
expression = '+'.join(prod_simple_df.columns)
nan_prod = prod_simple_df.withColumn('sum_cols', expr(expression)).select('sum_cols').collect()[0][0]

cust_simple_df=cust_simple.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in cust_simple.columns])
expression = '+'.join(cust_simple_df.columns)
nan_cust = cust_simple_df.withColumn('sum_cols', expr(expression)).select('sum_cols').collect()[0][0]

loc_simple_df=loc_simple.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in loc_simple.columns])
expression = '+'.join(loc_simple_df.columns)
nan_loc = loc_simple_df.withColumn('sum_cols', expr(expression)).select('sum_cols').collect()[0][0]


print('% elements as NaN in Product:', nan_prod / (prod_simple.count() * len(prod_simple.columns)))
print('% elements as NaN in Customer:', nan_cust/ (cust_simple.count() * len(cust_simple.columns)))
print('% elements as NaN in Location:', nan_loc / (loc_simple.count() * len(loc_simple.columns)))

print(prod_simple.count() == prod_clean.count())
print(cust_simple.count() == cust_clean.count())
print(loc_simple.count() == loc_clean.count())

## Ensure all columns are present to be dropped
## set(final_cols_to_drop) - set(master_pd.columns)
## [col_name for col_name in master_simple_pd if 'UDC_ROWID' in col_name] 

# COMMAND ----------

# DBTITLE 1,Schema Updates
## Resetting columns to match appropriate datatype - NUMERIC
## This was originally performed via manual review to validate true numeric cols
CORE_NUMERIC_COLS = ['DMND_HSTRY_QTY','PLANG_PROD_KG_QTY','PLANG_PROD_8OZ_QTY','PLANG_MTRL_EA_PER_CASE_CNT']
change_ship_cols = intersect_two_lists(CORE_NUMERIC_COLS, ship_clean.columns)
change_prod_cols = intersect_two_lists(CORE_NUMERIC_COLS, prod_simple.columns)
change_cust_cols = intersect_two_lists(CORE_NUMERIC_COLS, cust_simple.columns)
change_loc_cols = intersect_two_lists(CORE_NUMERIC_COLS, loc_simple.columns)

if len(change_ship_cols) >0:
  for c in change_ship_cols:
    ship_clean=ship_clean.withColumn(c, col(c).cast(FloatType()))
if len(change_prod_cols) >0:
  for c in change_prod_cols:
    prod_simple=prod_simple.withColumn(c, col(c).cast(FloatType()))
if len(change_cust_cols) >0:
  for c in change_cust_cols:
    cust_simple=cust_simple.withColumn(c, col(c).cast(FloatType()))
if len(change_loc_cols) >0:
  for c in change_cust_cols:
    loc_simple=loc_simple.withColumn(c, col(c).cast(FloatType()))



# COMMAND ----------

# Renaiming some columns 
ship_clean = ship_clean.withColumnRenamed('PLANG_MTRL_GRP_VAL','DMDUNIT').withColumnRenamed('PLANG_LOC_GRP_VAL','LOC')\
                  .withColumnRenamed('DMND_HSTRY_QTY','QTY')
prod_simple = prod_simple.withColumnRenamed('HRCHY_LVL_2_NM','FLVR_NM')
cust_simple = cust_simple.withColumnRenamed('HRCHY_LVL_2_NM','UDC_CHANNEL')
loc_simple = loc_simple.withColumnRenamed('HRCHY_LVL_2_NM','REGION')

# COMMAND ----------

# Validation 
print((prod_simple.select(countDistinct("DMDUNIT")).collect()[0][0]) >= (ship_clean.select(countDistinct("DMDUNIT")).collect()[0][0]))
print((cust_simple.select(countDistinct("PLANG_CUST_GRP_VAL")).collect()[0][0]) >= (ship_clean.select(countDistinct("PLANG_CUST_GRP_VAL")).collect()[0][0]))
print((loc_simple.select(countDistinct("LOC")).collect()[0][0]) >= (ship_clean.select(countDistinct("LOC")).collect()[0][0]))

print("DMDUNIT Distinct count in Prod Master:", prod_simple.select(countDistinct("DMDUNIT")).collect()[0][0])
print("DMDUNIT Distinct count in Shipment:", ship_clean.select(countDistinct("DMDUNIT")).collect()[0][0])

print("PLANG_CUST_GRP_VAL Distinct count in Cust Master:", cust_simple.select(countDistinct("PLANG_CUST_GRP_VAL")).collect()[0][0])
print("PLANG_CUST_GRP_VAL Distinct count in Shipment:", ship_clean.select(countDistinct("PLANG_CUST_GRP_VAL")).collect()[0][0])

print("LOC Distinct count in Loc Master:", loc_simple.select(countDistinct("LOC")).collect()[0][0])
print("LOC Distinct count in Shipment:", ship_clean.select(countDistinct("LOC")).collect()[0][0])

# COMMAND ----------

# DBTITLE 1,Output
#Shipments
save_df_as_delta(ship_clean, DBI_SHIPMENTS, enforce_schema=False)

# COMMAND ----------

#Products
save_df_as_delta(prod_simple, DBI_PRODUCTS, enforce_schema=False)

# COMMAND ----------

#Customers
save_df_as_delta(cust_simple, DBI_CUSTOMER, enforce_schema=False)

# COMMAND ----------

#Loc
save_df_as_delta(loc_simple, DBI_LOC, enforce_schema=False)

# COMMAND ----------

# DBTITLE 1,Code given by DE team 
# filteredShipmentDF = spark.sql("""
# with CTE_DFU as (
#   SELECT PLANG_MTRL_GRP_VAL,PLANG_CUST_GRP_VAL,PLANG_LOC_GRP_VAL,DMNDFCST_UNIT_LVL_VAL,DMND_POST_DT,
#           ROW_NUMBER() OVER(PARTITION BY  PLANG_MTRL_GRP_VAL,PLANG_CUST_GRP_VAL,PLANG_LOC_GRP_VAL ORDER BY DMND_POST_DT DESC) AS ROW_NUM
#   FROM DFU
# )
# SELECT H.*
# FROM Shipment H
# INNER JOIN ( Select * from CTE_DFU where ROW_NUM = 1 ) DFU 
# ON H.PLANG_MTRL_GRP_VAL=DFU.PLANG_MTRL_GRP_VAL 
# AND H.PLANG_CUST_GRP_VAL=DFU.PLANG_CUST_GRP_VAL 
# AND H.PLANG_LOC_GRP_VAL=DFU.PLANG_LOC_GRP_VAL 
# WHERE DMNDFCST_MKT_UNIT_CDV IN ('ES','PT' )
# AND DFU.DMNDFCST_UNIT_LVL_VAL='SB-S-FL-ITEM_CLIENT_DC'
# AND (H.DMND_HSTRY_TYP_CDV=1 AND NULLIF(TRIM(DMND_HSTRY_EVNT_NM),'') IS NULL)
# AND H.HSTRY_TMFRM_STRT_DT >= ADD_MONTHS(CURRENT_DATE,-12)  
# """)

# # AND H.HSTRY_TMFRM_STRT_DT >= ADD_MONTHS(CURRENT_DATE,-36)  
# # AND H.PLANG_CUST_GRP_VAL IN ('ES_DTS_OTHERS','ES_DTS_DISTRIBUTORS','ES_OT_OTCD_REG','ES_OT_OTHERS','ES_OT_MERCADONA','ES_OT_DIA','ES_OT_EROSKI','ES_OT_ALCAMPO','ES_OT_CARREFOUR','PT_O