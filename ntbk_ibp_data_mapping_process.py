# Databricks notebook source
# MAGIC %md # Config Notebook

# COMMAND ----------

#imports
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import *

# COMMAND ----------

#widgets
dbutils.widgets.text("configDatasetName", "")
dbutils.widgets.text("targetPath", "")
dbutils.widgets.text("stgAccount", "")
dbutils.widgets.text("targetContainer", "")

# COMMAND ----------

#creating variables to capture widget values
stgAccnt = dbutils.widgets.get("stgAccount")
configDatasetName = dbutils.widgets.get("configDatasetName")
tgtPath = "abfss://"+dbutils.widgets.get("targetContainer")+"@"+stgAccnt+".dfs.core.windows.net/"+dbutils.widgets.get("targetPath")+"/"+configDatasetName

# COMMAND ----------

print(tgtPath)

# COMMAND ----------

# MAGIC %run /Ibp/DE/ntbk_ibp_adls_cred $storageAccount=$stgAccount

# COMMAND ----------

#function to create the customer-locations mapping dataset
def map_customer_locations():
  #defining a dictionary with the mapped values
  dict_cust_loc = {'PT_OT_AUCHAN':'PL_PT_CAR_01', 'PT_OT_ITM':'PL_PT_CAR_01', 'PT_OT_OTHERS':'PL_PT_CAR_01', 'PT_OT_SONAE':'PL_PT_CAR_01', 'PT_OT_PINGO_DOCE':'PL_PT_CAR_01', 'PT_OT_DIA':'PL_PT_CAR_01', 'PT_OT_LIDL':'PL_PT_CAR_01', 'PT_DTS_OTHERS':'PL_PT_CAR_01', 'PT_EX_CABO_VERDE':'PL_PT_CAR_01', 'PT_EX_ANGOLA':'PL_PT_CAR_01', 'PT_EX_MARRUECOS':'PL_PT_CAR_01', 'PT_EX_MAURITIUS':'PL_PT_CAR_01', 'PT_DTS_MIRANDELA':'FG_CU_PT_MIR_01', 'PT_DTS_GAIA':'FG_CU_PT_GAI_01', 'PT_DTS_VISEU':'FG_CU_PT_VIS_01', 'PT_DTS_COVILHA':'FG_CU_PT_COV_01', 'PT_DTS_COIMBRA':'FG_CU_PT_COI_01', 'PT_DTS_OBIDOS':'FG_CU_PT_OBI_01', 'PT_DTS_LISBOA':'FG_CU_PT_LIS_01', 'PT_DTS_SOUSEL':'FG_CU_PT_SOU_01', 'PT_DTS_MADEIRA':'FG_CU_PT_MAD_01'}
  
  mapped_dict = list(map(list, dict_cust_loc.items()))
  customer_locations_df = spark.createDataFrame(mapped_dict, ["customer_group", "location_code"])
  
  #adding process date
  customer_locations_df = customer_locations_df.withColumn("PROCESS_DATE",current_date())
  
  display(customer_locations_df)
  
  #write it in bronze layer
  customer_locations_df.write.format('delta').mode('overwrite').option("mergeSchema", "true").save(tgtPath)
  
  print("Completed")

# COMMAND ----------

#Calling the function for the dataset
if configDatasetName == 'customer-locations':
  map_customer_locations()
else:
  print("Function for this is not defined")

# COMMAND ----------

