# Databricks notebook source
dbutils.widgets.text("storageAccount","")

# COMMAND ----------

storageAccount = dbutils.widgets.get("storageAccount")

# COMMAND ----------

tenant_id       = dbutils.secrets.get(scope="cdo-ibp-kvinst-scope",key="cdo-ibp-dbk-tenant-id")
client_id       = dbutils.secrets.get(scope="cdo-ibp-kvinst-scope",key="cdo-ibp-dbk-client-id")
client_secret   = dbutils.secrets.get(scope="cdo-ibp-kvinst-scope",key="cdo-ibp-dbk-spn")
client_endpoint = f'https://login.microsoftonline.com/{tenant_id}/oauth2/token'
#storage_account = "cdodevadls2"
storage_account = dbutils.widgets.get("storageAccount")

# COMMAND ----------

storage_account_uri = f"{storage_account}.dfs.core.windows.net"  
spark.conf.set(f"fs.azure.account.auth.type.{storage_account_uri}", "OAuth")
spark.conf.set(f"fs.azure.account.oauth.provider.type.{storage_account_uri}",
               "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set(f"fs.azure.account.oauth2.client.id.{storage_account_uri}", client_id)
spark.conf.set(f"fs.azure.account.oauth2.client.secret.{storage_account_uri}", client_secret)
spark.conf.set(f"fs.azure.account.oauth2.client.endpoint.{storage_account_uri}", client_endpoint)
#spark.conf.set("fs.azure.createRemoteFileSystemDuringInitialization","true")

# COMMAND ----------

