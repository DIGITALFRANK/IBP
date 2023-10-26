# Databricks notebook source
# DBTITLE 1,Python Library
import glob
import os
import warnings
import itertools
import sys
from functools import reduce
from itertools import chain
import gc
import pickle
import holidays
import json
from collections import Counter
import functools
import re
import ast
from typing import Iterable
import datetime
from datetime import datetime, timedelta
import datetime as dt


# dataframe operation and visualisation
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pylab 
import scipy.stats as stats
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.font_manager as font_manager
import seaborn as sns
import shap
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from pmdarima.arima import auto_arima
from matplotlib.lines import Line2D
import sweetviz as sv

# modeling library
from bayes_opt import BayesianOptimization
import xgboost as xgb
import lightgbm
import catboost
from catboost import CatBoostRegressor, Pool
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pyod.models.abod import ABOD
from pyod.models.knn import KNN
from pyod.models.mad import MAD
from pyod.models.cblof import CBLOF
from pyod.models.copod import COPOD 
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest


#Dynamic Time Warping (DTW) Segmentation
from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import scale

# COMMAND ----------

# DBTITLE 1,Pyspark Library
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrameStatFunctions as statFunc
from pyspark.ml.feature import Bucketizer, Binarizer
from pyspark.sql import DataFrame, SQLContext
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, IndexToString, MinMaxScaler, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkConf, SparkContext
from pyspark.sql.window import Window
from pyspark.ml import Pipeline

# from pyspark.ml.feature import Binarizer
# from pyspark.sql import functions
# from pyspark.sql.functions import expr
# from pyspark.sql.functions import concat, col, lit,  sequence, to_date, explode, to_timestamp, coalesce, greatest
# from pyspark.sql.functions import udf

# COMMAND ----------

# DBTITLE 1,Mlflow
from delta.tables import *
import mlflow
from mlflow.tracking import MlflowClient
import yaml