# Databricks notebook source
# DBTITLE 1,Import & Instantiate
# MAGIC %run /Ibp/DS/src/libraries

# COMMAND ----------

# MAGIC %run /Ibp/DS/src/load_src_parallel

# COMMAND ----------

# MAGIC %run /Ibp/DS/src/load_src

# COMMAND ----------

# MAGIC %run /Ibp/DS/src/config

# COMMAND ----------

# DBTITLE 1,Data Load & Review
DBA_MRD_version = 45  ## confirm this represents latest version
mrd_df = load_delta(DBA_MRD, DBA_MRD_version)

delta_info = load_delta_info(DBA_MRD)
display(delta_info.history())

# COMMAND ----------

# mrd_df.select('OtherProduction').distinct().describe().display()
# mrd_df.select('Weighted_Distribution_Proxy_L0_lag14').distinct().display()

# COMMAND ----------

mrd_df.toPandas().info()

# COMMAND ----------

mrd_df_cast = mrd_df

# COMMAND ----------

cols_to_downcast_index = [d[0] for d in mrd_df_cast.dtypes if d[0].endswith('_index') and d[1] != 'int']
cols_to_downcast_flag = [d[0] for d in mrd_df_cast.dtypes if d[0].endswith('_flag') and d[1] != 'int']
cols_to_downcast_bigint = [d[0] for d in mrd_df_cast.dtypes if d[1] == 'bigint']
cols_to_downcast = ['STAT_CLUSTER'] + cols_to_downcast_index + cols_to_downcast_flag + cols_to_downcast_bigint


print(len(cols_to_downcast), len(cols_to_downcast) / len(mrd_df_cast.columns))


for c in cols_to_downcast:
  mrd_df_cast = mrd_df_cast.withColumn(c, mrd_df_cast[c].cast(IntegerType()))

# COMMAND ----------

[c[1] for c in mrd_df.dtypes if c[0] == 'STAT_CLUSTER']
[c[1] for c in mrd_df_cast.dtypes if c[0] == 'STAT_CLUSTER']

# COMMAND ----------

mrd_df_cast.toPandas().info()

# COMMAND ----------

'STAT_CLUSTER', 'double', 'int'
'_index' -> int
'_flag' -> int
bigint -> int

# COMMAND ----------

token = '_LOC'

for d in mrd_df.dtypes:
  if token.lower() in d[0].lower():
    print( mrd_df.select(d[0]).distinct().display() )

# COMMAND ----------

## Review our dtypes
[d for d in mrd_df.dtypes if not d[0].endswith('_index') and not d[0].lower().endswith('_flag') and d[1] != 'int']

## BigInt = 'long' = [-9223372036854775808, 9223372036854775807]
## Double = precision float

# COMMAND ----------

# DBTITLE 1,Testing Downcasting
## Create df copy to adjust/alter/test below
mrd_df_cast = mrd_df

# COMMAND ----------

[col_name for col_name in mrd_df_cast.columns if 'index' in col_name]

# COMMAND ----------

## Find flags for features we can definitely downcast

## 'index' can be read as int
## 'flag' or 'Flag' can be read as int
## anything read as bigint can be int

bigint_downcast_cols = []

for each_element in mrd_df.dtypes:
  temp_col_name = each_element[0]
  temp_col_type = each_element[1]
  if temp_col_type == 'bigint':
    bigint_downcast_cols.append(temp_col_name)

index_int_cols = [col_name for col_name in mrd_df_cast.columns if 'index' in col_name and '100' not in col_name]
flag_int_cols =  [col_name for col_name in mrd_df_cast.columns if 'flag' in col_name or 'Flag' in col_name]

all_int_cols = index_int_cols + flag_int_cols + bigint_downcast_cols

len(all_int_cols), all_int_cols

# COMMAND ----------

for each_int_col in all_int_cols:
  mrd_df_cast = mrd_df_cast.withColumn(each_int_col, mrd_df_cast[each_int_col].cast(IntegerType()))

# COMMAND ----------

mrd_df_cast.dtypes

# COMMAND ----------

# DBTITLE 1,Miscellaneous Column Validation/Review
mrd_df_cast.select(col('distinct_PLANG_CUST_GRP_VAL_BRND_NM')).describe().show()

# COMMAND ----------

mrd_df.select(col('distinct_PLANG_CUST_GRP_VAL_BRND_NM')).describe().show()