# Databricks notebook source
#Competitor Cannibalization Suite.  This module identifies competitors based on whether the competitor is in a "same" (e.g., category)
#or "different" (e.g., customer) hierarchy.  There are supplementary functions to:
#   1) allow you to filter this pair list based on data trends (e.g., correlations, # common weeks, etc.)
#   2) override the automatically generated list with a client provided list of competitors
#   3) grab the data for the final competitors and transpose into variables (e.g., competitor 1 price)
#TO-DO: Update Tests for Class Method

class CompetitorInfo:
    """
    Class to contain the required information for finding relevant competitors.
    Required Parameters:
        model_id : String
            Name of model id field (should be one variable with concatenated hierarchy)
        date_field : String
            Name of date field
        comp_group_field : String
            Name of the field containing competitor group (e.g., own competitors, cross competitors)
        sales_data : PySpark DataFrame
            Transactional data for both competitor and own transactions
    """

    def __init__(self, **kwargs):

        # These arguments are required by all cannibalization functions.  Their presence is
        # checked by check_required_attrs_received, with an error being raised if
        # they aren't set.
        self._required_attrs = [
            'model_id',
            'date_field',
            'comp_group_field',
            'sales_data'
        ]

        self.__dict__.update(kwargs)
        self._check_required_attrs_received()
        self.comp_model_id = "comp_" + self.model_id #Field name created by class to identify competitor ID

    def _check_required_attrs_received(self):
        self.missing_attrs = [attr for attr in self._required_attrs if attr not in self.__dict__]
        if self.missing_attrs:
            missing = ', '.join(self.missing_attrs)
            err_msg = f'The following parameters are required but were not provided: {missing}'
            raise TypeError(err_msg)

    def get_competitor_join_cols(self):
      """
      Develops cartesian products of all pairs and appends two columns (join_col, anti_join_col).  These columns are used
      to filter pairs that are within the "same_dict" and outside of "diff_dict".
        - same_dict {key group : hierarchy (hierarchy which should be the same for this group)}
          E.g.: {'Ice Cream':["Customer","Category","Competitor_Flag"]}
        - diff_dict {key group : hierarchy (hierarchy which should be different for this group)}
          E.g.: {'Ice Cream':["SKU"]}
        - Configuration_level is the hierarchy of the keys in same/diff
      """
      #Restrict data to those with key groups
      self.competitor_pairs = self.sales_data
      self.competitor_pairs = self.competitor_pairs.filter(col(self.configuration_level).isin([*self.same_dict]))
      self.competitor_pairs = self.competitor_pairs.filter(col(self.configuration_level).isin([*self.diff_dict]))

      #Create join columns
      unique_values = list(itertools.chain(*self.same_dict.values()))
      unique_values = list(set(list(set(unique_values)) + [self.configuration_level]))
      self.competitor_pairs = self.competitor_pairs.withColumn("join_col",
                                                               concatColsByGroup(self.competitor_pairs[unique_values],
                                                                                 self.configuration_level,
                                                                                 self.same_dict))

      #Create anti-join columns
      unique_values = list(itertools.chain(*self.diff_dict.values()))
      unique_values = list(set(list(set(unique_values)) + [self.configuration_level]))
      self.competitor_pairs = self.competitor_pairs.withColumn("anti_join_col",\
                                                               concatColsByGroup(self.competitor_pairs[unique_values],
                                                                                 self.configuration_level,
                                                                                 self.diff_dict))

    def get_competitor_pairs(self):
      """
      Filters competitor pair data according to join/anti-join columns
      """

      #Filter products to within similar product hierarchies
      if "join_col" not in self.competitor_pairs.columns:
        err_msg = f'No join column was provided'
        raise TypeError(err_msg)
      elif "anti_join_col" not in self.competitor_pairs.columns:
        err_msg = f'No anti-join column was provided'
        raise TypeError(err_msg)
      else:
        #Performs a cross join of products that have a similar "join_col".  Retains fields needed to do anti-join at subsequent stages.
        own_products = self.competitor_pairs.filter(col(self.competitor_flag_field)==0)
        own_products = own_products.select(self.model_id,"join_col","anti_join_col").dropDuplicates()
        comp_products = self.competitor_pairs.select(self.model_id,"join_col","anti_join_col").dropDuplicates()
        comp_products = comp_products.withColumnRenamed(self.model_id,self.comp_model_id)
        comp_products = comp_products.withColumnRenamed("anti_join_col","anti_join_col_comp")
        self.competitor_pairs = own_products.join(comp_products, ["join_col"], "inner")

      #Filter Anti-Join
      self.competitor_pairs = self.competitor_pairs.filter(self.competitor_pairs.anti_join_col != self.competitor_pairs.anti_join_col_comp)
      self.competitor_pairs = self.competitor_pairs.select(self.model_id,self.comp_model_id)

    def set_comp_pairs(self, comp_pairs):
      """
      Overrides competitor pair list with manually fed dataframe.  This is used when a client wishes to override the automatic
      competitor generation with their own competitor list or when overriding base pair list with filtered list.
      """
      self.competitor_pairs = comp_pairs


    def get_pair_data(self, data_vars):
        """
        Returns specified variables for both own and cross competitors in cartesian form.  This is used to filter
        the base pair list based on business criterion (e.g., x number of common weeks, correlation above threshold).
        """
        #Join own/cross transactional data
        keep_vars = [self.model_id,self.date_field] + data_vars
        own_data = self.sales_data.select(keep_vars).dropDuplicates()
        cross_data = own_data.withColumnRenamed(self.model_id,"COMP_MODEL_ID")

        #Add prefix to fields (client_ for own data and comp_ for cross data)
        new_names = [self.comp_model_id,self.date_field] + add_prefix_to_list(data_vars, "competitor_")
        cross_data = cross_data.toDF(*new_names)
        new_names = [self.model_id,self.date_field] + add_prefix_to_list(data_vars, "client_")
        own_data = own_data.toDF(*new_names)

        self.comp_data = self.competitor_pairs.join(own_data, on=[self.model_id], how="left").dropDuplicates()
        self.comp_data = self.comp_data.join(cross_data, on=[self.comp_model_id,self.date_field], how="left")

    def rank_competitors(self, rank_var, desc_type):
        """
        Ranks competitor pairs based on passed variable (correlation, sales, etc.)
        """
        #Remove rows where the rank variable is empty
        self.competitor_pairs = self.competitor_pairs.where(col(rank_var).isNotNull())

        #Rank according to desc_type - Uses row number to break ties
        if desc_type == "desc":
          self.competitor_pairs =  self.competitor_pairs.withColumn("rank", row_number()\
                                                          .over(Window.partitionBy(self.model_id,self.comp_group_field)\
                                                                .orderBy(desc(rank_var))))
        elif desc_type == "asc":
          self.competitor_pairs =  self.competitor_pairs.withColumn("rank", row_number()\
                                                          .over(Window.partitionBy(self.model_id,self.comp_group_field)\
                                                                .orderBy(asc(rank_var))))

    def filter_ranked_competitors(self, threshold):
      """
      Filters competitor pair list based on a rank variable
      """
      self.competitor_pairs = self.competitor_pairs.filter(self.competitor_pairs.rank <= threshold)

    def get_final_competitor_variables(self, comp_var):
      """
      This function creates variables (e.g., "competitor1_AvgPrice","competitor2_AvgPrice") based on the final competitor
      rankings so that it can be merged into the model ready dataset.
      """
      #Rename compType to new variable name
      self.competitor_pairs = self.competitor_pairs.withColumn(self.comp_model_id + "2",
                                                               concat(col(self.comp_group_field), lit("_"),col("rank"),lit("_"), lit(comp_var)))

      #Merge in variables to competitor pairs
      self.comp_data = self.competitor_pairs.join(self.comp_data, on=[self.model_id,self.comp_model_id], how="left")
      self.comp_data = self.comp_data.select("MODEL_ID",self.date_field,self.comp_model_id + "2","competitor_" + comp_var).dropDuplicates()
      self.comp_data = self.comp_data.withColumnRenamed("competitor_" + comp_var, comp_var)

      #Pivot so that competitor data are now variables
      self.comp_data = self.comp_data.groupBy(self.model_id,self.date_field).pivot(self.comp_model_id + "2").max(comp_var)

    def get_competitor_pair_hier(self, summary_vars, hier_vars):
      """
      This function returns key hierarchy and summary statistics for the final competitor pairs.  This table is often used in
      PEA dashboards to explain what competitors were selected for each model_id
      """
      summary_df = self.competitor_pairs.select(summary_vars)
      unique_hier = self.sales_data.select(hier_vars).dropDuplicates()
      self.output_hier = summary_df.join(unique_hier, on=self.model_id, how="left")
      cross_hier_names = add_prefix_to_list(unique_hier.columns, "COMP_")
      cross_hier = unique_hier.toDF(*cross_hier_names)
      self.output_hier = self.output_hier.join(cross_hier, on=self.comp_model_id, how="left")

def concatColsByGroup(df, group_col, cols_to_concat_dict):
  """
  Concatenates columns that vary by group

  Parameters
  ----------
  df : PySpark dataframe
  group_col: Str
      Name of key group column in df
  cols_to_concat_dict: Dictionary
      Dictionary with key group value and list of columns to concatenate for that group

  Returns
  -------
  new_col : PySpark Column
  """
  new_col = coalesce(*[when(df[group_col] == key, concat(*value)) for key, value in cols_to_concat_dict.items()])
  return new_col