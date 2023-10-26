# Databricks notebook source
#imports
from pyspark.sql.functions import *
from delta.tables import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_poc_adls_cred

# COMMAND ----------

slv_advertising_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/advertising-and-media-spend"
slv_customer_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/customer-master"
slv_dfu_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/dfu"
slv_demand_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/dfu-to-sku-forecast"
slv_distribution_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/distribution-master"
slv_pricing_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/pricing"
slv_product_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/product-master"
slv_shipment_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/shipment-actuals"
slv_store_foot_delta_path = "abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/silver/iberia/ibp-poc/store-foot-traffic"

# COMMAND ----------

adv_df = spark.read.format("delta").load(slv_advertising_delta_path)
print(adv_df.count())
#adv_df.show(5)
display(adv_df, 5)

# COMMAND ----------



# COMMAND ----------

cust_df = spark.read.format("delta").load(slv_customer_delta_path)
print(cust_df.count())
display(cust_df, 5)

# COMMAND ----------



# COMMAND ----------

dfu_df = spark.read.format("delta").load(slv_dfu_delta_path)
print(dfu_df.count())
display(dfu_df, 5)

# COMMAND ----------



# COMMAND ----------

demand_df = spark.read.format("delta").load(slv_demand_delta_path)
print(demand_df.count())
display(demand_df, 5)

# COMMAND ----------



# COMMAND ----------

dist_df = spark.read.format("delta").load(slv_distribution_delta_path)
print(dist_df.count())
display(dist_df, 5)

# COMMAND ----------



# COMMAND ----------

pricing_df = spark.read.format("delta").load(slv_pricing_delta_path)
print(pricing_df.count())
display(pricing_df, 5)

# COMMAND ----------



# COMMAND ----------

product_df = spark.read.format("delta").load(slv_product_delta_path)
print(product_df.count())
display(product_df, 5)

# COMMAND ----------



# COMMAND ----------

shipment_df = spark.read.format("delta").load(slv_shipment_delta_path)
print(shipment_df.count())
display(shipment_df, 5)

# COMMAND ----------



# COMMAND ----------

store_foot_df = spark.read.format("delta").load(slv_store_foot_delta_path)
print(store_foot_df.count())
display(store_foot_df, 5)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

