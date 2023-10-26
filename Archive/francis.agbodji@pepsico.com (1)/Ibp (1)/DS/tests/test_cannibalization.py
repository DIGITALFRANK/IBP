# Databricks notebook source
#Create dummy data
dummyDat = sqlContext.createDataFrame([
    ("Walmart10001","2020-10-03" , 100, "Walmart", "Ice Cream", 10001, 0, 3.5),
    ("Walmart10001","2020-10-04" , 200, "Walmart", "Ice Cream", 10001, 0, 4.5),
    ("Walmart10001","2020-10-05" , 300, "Walmart", "Ice Cream", 10001, 0, 6.5),
    ("Walmart10002","2020-10-03" , 100, "Walmart", "Ice Cream", 10002, 0, 3.5),
    ("Walmart10002","2020-10-04" , 200, "Walmart", "Snacks", 10002, 0, 4.5),
    ("Walmart10002","2020-10-05" , 300, "Walmart", "Snacks", 10002, 0, 6.5),
    ("Walmart10003","2020-10-03" , 200, "Walmart", "Snacks", 10003, 0, 2.9),
    ("Walmart10003","2020-10-04" , 400, "Walmart", "Snacks", 10003, 0, 2.8),
    ("Walmart10003","2020-10-05" , 600, "Walmart", "Snacks", 10003, 0, 5.9)
], ["MODEL_ID","DATE", "Sales", "Customer", "Category", "SKU", "Competitor_Flag", "Price"])

join_dict = {'Ice Cream':["Customer","Category","Competitor_Flag"],
             'Snacks':["Customer"],
            }
anti_dict = {'Ice Cream':["SKU"],
             'Snacks':["SKU"],
            }

def test_cannibalization_dict():
    #Test cannibalization dictionary errors if required fields are missing
    try:
            cannibalization_info_dict = dict(
                date_field            = "DATE",
                comp_group_field      = "comp_type",
                sales_data            = dummyDat,
                same_dict             = join_dict,
                diff_dict             = anti_dict,
                configuration_level   = "Category",
                competitor_flag_field = "Competitor_Flag")
            cannibalization_cls = cannibalization.CompetitorInfo(**cannibalization_info_dict)
    except:
        assert(1)

def test_cannibalization_basepairs():
    #Test cannibalization dictionary errors if required fields are missing
    cannibalization_info_dict = dict(
        model_id              = "MODEL_ID",
        date_field            = "DATE",
        comp_group_field      = "comp_type",
        sales_data            = dummyDat,
        same_dict             = join_dict,
        diff_dict             = anti_dict,
        configuration_level   = "Category",
        competitor_flag_field = "Competitor_Flag")
    cannibalization_cls = cannibalization.CompetitorInfo(**cannibalization_info_dict)

    cannibalization_cls.get_competitor_join_cols()
    cannibalization_cls.get_competitor_pairs()
    actual_output = cannibalization_cls.competitor_pairs.withColumn("comp_type", lit("crossSame"))

    schema = StructType([StructField("MODEL_ID", StringType(), True),
                            StructField("comp_MODEL_ID", StringType(), True),
                            StructField("comp_type", StringType(), False)])
    expected_output = sqlContext.createDataFrame([
        ("Walmart10003","Walmart10002" , "crossSame"),
        ("Walmart10002","Walmart10003" , "crossSame"),
        ("Walmart10001","Walmart10002" , "crossSame"),
        ("Walmart10002","Walmart10001" , "crossSame")
    ], schema)
    assert_df_equality(expected_output, actual_output)

# def test_cannibalization_rank_competitors():
#     #Test cannibalization dictionary errors if required fields are missing
#     cannibalization_info_dict = dict(
#         model_id              = "MODEL_ID",
#         date_field            = "DATE",
#         comp_group_field      = "comp_type",
#         sales_data            = dummyDat,
#         same_dict             = join_dict,
#         diff_dict             = anti_dict,
#         configuration_level   = "Category",
#         competitor_flag_field = "Competitor_Flag")
#     cannibalization_cls = cannibalization.CompetitorInfo(**cannibalization_info_dict)
#
#     cannibalization_cls.get_competitor_join_cols()
#     cannibalization_cls.get_competitor_pairs()
#     actual_output = cannibalization_cls.competitor_pairs.withColumn("comp_type", lit("crossSame"))
#     #Get mrd data
#     cannibalization_cls.get_final_competitor_variables("Price")
#     disp = cannibalization_cls.comp_data.toPandas()
#     display(disp)