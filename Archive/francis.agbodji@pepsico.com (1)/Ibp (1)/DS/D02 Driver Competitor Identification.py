# Databricks notebook source
# MAGIC %md 
# MAGIC ##02 - Competitor Identification
# MAGIC 
# MAGIC This modules creates competitor variables to be used in modeling. There are two possible ways to obtain competitor variables:
# MAGIC 
# MAGIC > **Automatic Generation**: This method automatically finds the most likely competitor set of products given a set of hierarchy configurations and statistical thresholds.
# MAGIC >> 1.  Products are filtered to a set that are within provided hierarchy configurations (e.g., competitors should be same category, pack size)
# MAGIC >> 2.  Products are filtered using statistical thresholds. E.g.
# MAGIC   * Filter Low sales competitors (competitors with <10% of sales compared to own product)
# MAGIC   * Filter Low history (<20 rows of history where both own and competitive products were sold)
# MAGIC   * Keep highly correlated pairs (competitor price vs. own demand)
# MAGIC   * Filter to top X competitors (based on competitor sales or correlation)
# MAGIC >> 3.  The data for the top competitors is then shaped into variables so that it can be merged into mrd
# MAGIC 
# MAGIC > **Client provided list**: This method directly uses competitor pairs provided by the client and only prepares the data into variables for mrd
# MAGIC 
# MAGIC TO-DOs:
# MAGIC * 1) Reference separate syndicated data for Promo Distribution, cross competitor identification, etc.

# COMMAND ----------

# DBTITLE 1,Instantiate with Notebook Imports
# MAGIC %run ./src/libraries

# COMMAND ----------

# MAGIC %run ./src/load_src_parallel

# COMMAND ----------

# MAGIC %run ./src/load_src

# COMMAND ----------

#initiate widget if needed, otherwise does nothing        
check_or_init_dropdown_widget("TIME_VAR_LOCAL","Week_Of_Year",["Week_Of_Year","Month_Of_Year"])  

# COMMAND ----------

# MAGIC %run ./src/config

# COMMAND ----------

#Check configurations exist for this script
try:
  required_configs = [
    CLIENT_OWN_LIST,
    COMPETITOR_PRICE_VAR,
    COMPETITOR_NON_PROMO_PRICE_VAR,
    DBA_MRD_EXPLORATORY 
    ]
  print(json.dumps(required_configs, indent=4))

  if CLIENT_OWN_LIST == False:
    required_configs = [
      COMPETITOR_FIELD, 
      COMPETITOR_SALES_LOWER_THRESHOLD,
      COMPETITOR_HISTORY_THRESHOLD,
      COMPETITOR_PEARSON_LOWER_THRESHOLD,
      COMPETITOR_PEARSON_UPPER_THRESHOLD,
      NUM_COMPETITOR_VARIABLES,
      COMPETITOR_SORT_VAR,
      join_dict,
      anti_dict
    ]
    print(json.dumps(required_configs, indent=4))
except:
  dbutils.notebook.exit("Missing required configs")

# COMMAND ----------

# DBTITLE 1,Load Data
## Loading in dev runs
# mrd = load_delta(DBA_MRD_EXPLORATORY, 38) #Weekly version 
# if TIME_VAR=="Month_Of_Year":
#   mrd = load_delta(DBA_MRD_EXPLORATORY, 39) #Monthly

# Production Run
mrd = load_delta(DBA_MRD_EXPLORATORY)

pingo_prod = spark.read.option("header","true").option("delimiter",",").csv("abfss://bronze@cdodevadls2.dfs.core.windows.net/IBP/EDW/PINGO_DOCE/Product.csv")

prod = spark.read.option("header","true").option("delimiter",";")\
              .csv("abfss://supplychain-ibp@cdodevadls2.dfs.core.windows.net/Adhoc/onetimefull-load/ibp-bronze/IBP/EDW/Prolink/Product Master/ingestion_dt=2021-06-04/")
prod = prod.select(pingo_prod.drop("DMDUnit").columns)
prod = prod.withColumn('DMDUnit', trim(prod.PLANG_MTRL_GRP_VAL))
prod = prod.withColumn("DMDUnit",expr("substring(DMDUnit, 1, length(DMDUnit)-3)"))

prod = prod.select(list(product_columns.keys()))\
                   .select([col(x[0]).alias(x[1]) for x in product_columns.items()]) #subset and rename column
keep_vars = ["DMDUNIT","SUBCAT","PCK_SIZE_SHRT_NM"]
prod = prod.select(keep_vars)

print(mrd.count())
mrd = mrd.join(prod, on=["DMDUNIT"], how="left")
print(mrd.count())

# COMMAND ----------

# DBTITLE 1,Foundational Pricing Variables
#Foundational price varibles

#Depth
ratio_dict = {'Discount_Depth':{'top_var':COMPETITOR_PRICE_VAR,'bottom_var':COMPETITOR_NON_PROMO_PRICE_VAR}}
mrd = calculate_ratio(mrd,ratio_dict) 
mrd = mrd.withColumn("Discount_Depth", (1 - col("Discount_Depth"))*100)
mrd = mrd.withColumn("Discount_Depth", when(col("Discount_Depth") < 0, 0).otherwise(col("Discount_Depth")))
mrd = mrd.withColumn("Discount_Depth", when(col("Discount_Depth").isNull(), 0).otherwise(col("Discount_Depth")))

#Dollar Discount
mrd = mrd.withColumn("Dollar_Discount", col(COMPETITOR_NON_PROMO_PRICE_VAR)-col(COMPETITOR_PRICE_VAR))

#Dummy competitor flag identifier until Syndicated data is setup
mrd = mrd.withColumn("Competitor_Flag", lit(0))


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ##Own Competitor Identification 
# MAGIC 
# MAGIC This section finds "own competitors" (PEP manufactured) and develops the variables to feed into modeling.  A client can provide the competitor list directly and this code will develop the variables.  Alternatively, this code can be used to generate an automatic competitor list and then create variables to merge into the modeling dataset.

# COMMAND ----------

#OWN COMPETITORS - CLIENT PROVIDED LIST
if CLIENT_OWN_LIST == True:
  #Get data for merging to mrd
  cannibalization_info_dict = dict(
        model_id              = "MODEL_ID",
        date_field            = TIME_VAR,
        comp_group_field      = "comp_type",
        sales_data            = mrd)
  cannibalization_cls = CompetitorInfo(**cannibalization_info_dict)
  cannibalization_cls.set_comp_pairs(client_provided_competitors)
  cannibalization_cls.get_pair_data([COMPETITOR_PRICE_VAR])
  cannibalization_cls.get_final_competitor_variables(COMPETITOR_PRICE_VAR)

# COMMAND ----------

#OWN COMPETITORS - AUTOMATIC LIST
if CLIENT_OWN_LIST == False:
  #Create cannibalization own competitors
  cannibalization_info_dict = dict(
        model_id              = "MODEL_ID",
        date_field            = TIME_VAR,
        comp_group_field      = "comp_type",
        sales_data            = mrd,
        same_dict             = join_dict,
        diff_dict             = anti_dict,
        configuration_level   = "SUBCAT",
        competitor_flag_field = COMPETITOR_FIELD)
  cannibalization_cls = CompetitorInfo(**cannibalization_info_dict)

  #Get base pairs
  cannibalization_cls.get_competitor_join_cols()
  cannibalization_cls.get_competitor_pairs()
  ownSame = cannibalization_cls.competitor_pairs.withColumn("comp_type", lit("ownSame"))
  cannibalization_cls.set_comp_pairs(ownSame) 
  
  #Filter own competitors to most likely list
  ### Grab summary statistics by competitor pairs
  cannibalization_cls.get_pair_data([TARGET_VAR,COMPETITOR_PRICE_VAR])

  #Filter 1: Filter low sales by competitor (noise)
  summary_vars = ["client_" + TARGET_VAR, "competitor_" + TARGET_VAR , "competitor_" + TARGET_VAR]
  summary_type = [sum, sum, countDistinct]
  filtered_list = aggregate_data(cannibalization_cls.comp_data,["MODEL_ID","COMP_MODEL_ID","comp_type"],
                                 summary_vars,summary_type)
  ratio_dict = {'competitor_pct_sales':{'top_var'   : "sum_competitor_" + TARGET_VAR,
                                        'bottom_var': "sum_client_" + TARGET_VAR}}
  filtered_list = calculate_ratio(filtered_list,ratio_dict)
  filtered_list = filtered_list.filter(filtered_list.competitor_pct_sales > COMPETITOR_SALES_LOWER_THRESHOLD)

  #Filter 2: Filter low number of common weeks
  filtered_list = filtered_list.withColumnRenamed("countDistinct_competitor_" + TARGET_VAR, "num_common_rows")
  filtered_list = filtered_list.filter(filtered_list.num_common_rows > COMPETITOR_HISTORY_THRESHOLD)

  #Filter 3: Filter low correlation between competitor price and own volume sales
  corr = calculate_corr(cannibalization_cls.comp_data,
                        "client_" + TARGET_VAR , "competitor_" + COMPETITOR_PRICE_VAR,
                        ["MODEL_ID","comp_MODEL_ID"])
  filtered_list = filtered_list.join(corr, on=["MODEL_ID","COMP_MODEL_ID"], how="left")
  filtered_list = filtered_list.filter(filtered_list.CORR >= float(COMPETITOR_PEARSON_LOWER_THRESHOLD))
  filtered_list = filtered_list.filter(filtered_list.CORR <= float(COMPETITOR_PEARSON_UPPER_THRESHOLD))
  filtered_list.cache()
  print(filtered_list.count())

  # Filter to top X competitors (by correlation)
  cannibalization_cls.set_comp_pairs(filtered_list) #Update class pairs to filtered list
  cannibalization_cls.rank_competitors(COMPETITOR_SORT_VAR, "desc") #Rank by correlation
  cannibalization_cls.filter_ranked_competitors(NUM_COMPETITOR_VARIABLES) #Keep top X correlated pairs

  
  
  #Get mrd data
  cannibalization_cls.get_final_competitor_variables(COMPETITOR_PRICE_VAR)
  identified_own_competitors = cannibalization_cls.competitor_pairs
  own_competitors_df = cannibalization_cls.comp_data

  

# COMMAND ----------

# DBTITLE 1,Cross Competitor Identification
# COREY FLAG - but nothing to do here yet (until Nielsen) 
#Need nielsen data matched with shipments for cross competitor identification

# COMMAND ----------

# DBTITLE 1,Append Outputs
try:
  identified_competitors = identified_own_competitors.union(identified_cross_competitors)
  #competitor_vars = own_competitors_df.union(cross_competitors_df) #TO-DO: This should be a join not union
except:
  identified_competitors = identified_own_competitors
  competitor_vars = own_competitors_df

# COMMAND ----------

#Join competitor variables to foundational price variables
competitor_vars_joined = mrd.select(["MODEL_ID",TIME_VAR]).dropDuplicates()
competitor_vars_joined = competitor_vars_joined.join(mrd.select(["MODEL_ID",TIME_VAR, "Discount_Depth" ,"Dollar_Discount"]), 
                                                     on=["MODEL_ID",TIME_VAR], how="left")
competitor_vars_joined = competitor_vars_joined.join(competitor_vars, on=["MODEL_ID",TIME_VAR], how="left")
competitor_vars = competitor_vars_joined

# COMMAND ----------

# DBTITLE 1,Output
#Write as delta table to dbfs
save_df_as_delta(identified_competitors, DBO_IDENTIFIED_COMPETITORS, enforce_schema=False)
delta_info = load_delta_info(DBO_IDENTIFIED_COMPETITORS)
set_delta_retention(delta_info, "90 days")
display(delta_info.history())

# COMMAND ----------

#Write as delta table to dbfs
save_df_as_delta(competitor_vars, DBO_COMPETITOR_VARS, enforce_schema=False)
delta_info = load_delta_info(DBO_COMPETITOR_VARS)
set_delta_retention(delta_info, "90 days")
display(delta_info.history())

# COMMAND ----------

