# Databricks notebook source
# DBTITLE 1,Pandas Exploratory
def print_missing_percentage(pd_df):
  """
  Print an array showing the percent of missing entries in each column.
  
  Parameters
  ----------
  pd_df : pandas DataFrame
      DataFrame with columns whose percent of missing entries is desired.
  """
  print(pd_df.isna().mean().round(4) * 100)
  

def get_color_palette(palette):
  """
  Get standard seaborn color palette
  
  If palette is int, returns blues palette with palette number of colors
  If palette is str, returns color palette of that name
  If palette is a seaborn ColorPalette, returns palette
  Otherwise returns None
  """
  
  if not palette:
    palette = get_deloitte_digital_color_palette()
  
  
  if isinstance(palette, int):
    return sns.color_palette("Blues", palette)
  elif isinstance(palette, str):
    return sns.color_palette(palette)
  elif isinstance(palette, sns.palettes._ColorPalette):
    return palette
  else: 
    return None

  
def get_default_tick_marks(designator):
  """
  Set default value for tick marks
  """
  if designator == "x":
    return {"rotation":0, "size":10}
  if designator == "y":
     return {"rotation":0, "size":10}
  else:
    raise ValueError("Please designate x or y.")
    
  
def plot_box_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10, show_outliers=False, 
                  fig_size=(10,10), x_label=None, y_label=None, fontproperties=None, **kwargs):
  """
  Plot a boxplot in seaborn
  TODO clean this function up; standardize arguments between plotting functions
  """  
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
    
  plt.figure(figsize=fig_size)
  plot = sns.boxplot(x=x_var, y=y_var, data=pd_df, hue=hue, showfliers=show_outliers, **kwargs)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  plt.xlabel(x_label, fontproperties=fontproperties)
  plt.ylabel(y_label, fontproperties=fontproperties)
  
  sns.despine()
  
  return plot
      
  
def plot_bar_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10,
                  fig_size=(10,10), y_limit_list=None, x_label=None, y_label=None, fontproperties=None):
  """
  Plot a barplot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.barplot(x=x_var, y=y_var, data=pd_df, hue=hue)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
  
  if y_limit_list:
    plt.ylim(y_limit_list[0], y_limit_list[1])
    
  if x_label:
    plt.xlabel(x_label, fontproperties=fontproperties)

  if y_label:
    plt.ylabel(y_label, fontproperties=fontproperties)
    
  sns.despine()
  
  return plot
    
    
def plot_scatter_plot(pd_df, x_var, y_var, hue=None, size=None, stylehue=None, title=None, title_size=10,
                      fig_size=(10,10), x_label=None, y_label=None, fontproperties=None, **kwargs):
  """
  Plot a boxplot in seaborn
  """  
  x_ticks=get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.scatterplot(x=x_var, y=y_var, data=pd_df, hue=hue, style=stylehue, size=size, **kwargs)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  plt.xlabel(x_label, fontproperties=fontproperties)
  plt.ylabel(y_label, fontproperties=fontproperties)
  
  sns.despine()
  
  return plot


def plot_joint_plot(pd_df, x_var, y_var, kind="scatter", title=None, title_size=10, fig_size=(10,10),
                    y_limit_list=None, x_label=None, y_label=None, fontproperties=None, **kwargs):
  """
  Plot a joint plot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.jointplot(x=x_var, y=y_var, kind=kind, data=pd_df, **kwargs)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
  
  if y_limit_list:
    plt.ylim(y_limit_list[0], y_limit_list[1])
    
  if x_label:
    plt.xlabel(x_label, fontproperties=fontproperties)

  if y_label:
    plt.ylabel(y_label, fontproperties=fontproperties)
    
  sns.despine()
  
  return plot
    
    
def plot_violin_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10, show_outliers=False,
                   fig_size=(10,10), x_label=None, y_label=None, fontproperties=None, **kwargs):
  """
  Plot a violin plot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.violinplot(x=x_var, y=y_var, data=pd_df, hue=hue, showfliers=show_outliers, **kwargs)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  if x_label:
    plt.xlabel(x_label, fontproperties=fontproperties)

  if y_label:
    plt.ylabel(y_label, fontproperties=fontproperties)
    
  sns.despine()
  
  return plot


def plot_line_plot(pd_df, x_var, y_var, hue=None, title=None, title_size=10,
                   fig_size=(10,10), x_label=None, y_label=None, fontproperties=None, **kwargs):
  """
  Plot a line plot in seaborn
  """
  x_ticks = get_default_tick_marks("x")
  y_ticks = get_default_tick_marks("y")
  
  plt.figure(figsize=fig_size)
  plot = sns.lineplot(x=x_var, y=y_var, data=pd_df, hue=hue, **kwargs)
  plt.title(title, size=title_size, fontproperties=fontproperties)
  
  plt.xticks(rotation=x_ticks["rotation"], size=x_ticks["size"])
  plt.yticks(rotation=y_ticks["rotation"], size=y_ticks["size"])
    
  if x_label:
    plt.xlabel(x_label, fontproperties=fontproperties)

  if y_label:
    plt.ylabel(y_label, fontproperties=fontproperties)
  
  sns.despine()
    
  return plot


def plot_corr_heatmap(pd, x_vars_as_list, colormap=None):
  """
  Returns a correlation heatmap that can be called using display()
  """
  if not colormap:
    colormap = colormap=get_single_digital_color_palette('yellow', n=10)[::-1]
  
  def getCorrelations(pd, x_vars_as_list):
    temp_pd = pd[x_vars_as_list]
    corr_mat = temp_pd.corr()
    return corr_mat  
  
  heatmap = sns.heatmap(getCorrelations(pd, x_vars_as_list), cmap=colormap, center=0, annot=True)
  
  sns.despine()
  
  return heatmap

# COMMAND ----------

# DBTITLE 1,Pandas Color Palettes
def convert_rgb_256_to_float(rgb_256):
  """
  Takes RGB in typical (1-255, 1-255, 1-255) format and returns RGB in matplotlib-friendly (0-1, 0-1, 0-1) format
  """
  
  assert len(rgb_256) == 3, 'Exactly three codes (RGB) are required'
  
  for color in rgb_256:
    assert 0 <= color <= 255, 'One of the RGB codes is not between 0 and 256'
  
  return tuple(color/256.0 for color in rgb_256)


def mix_colors(color_1_hex, color_2_hex, mix=0.5):
  """
  Mix two colors (represented in hex) with proportion of color_2_hex = mix  
  """
  color_1_rgb = np.array(colors.to_rgb(color_1_hex))
  color_2_rgb = np.array(colors.to_rgb(color_2_hex))
  return colors.to_hex((1 - mix) * color_1_rgb + mix * color_2_rgb)


def get_deloitte_digital_colors_rgb():
  """
  Returns dictionary of Deloitte Digital colors and their RGB 256 codes
  """
  return {'green' : (134, 242, 0),
          'blue' : (52, 240, 255),
          'yellow' : (253, 211, 0),
          'teal' : (62, 250, 197)}


def get_deloitte_digital_color_palette(option='normal'):
  """
  Returns the Deloitte Digital color palette
  
  Parameter option can be set as 'normal' (original colors), 'light' (colors 30% lighter), or 'dark' (colors 10% darker)
  """
  
  assert option in ('light', 'normal', 'dark')
  
  white_hex = '#FFFFFF'
  black_hex = '#000000'
  
  deloitte_digital_colors_rgb_float = [convert_rgb_256_to_float(rgb_256) for rgb_256 in get_deloitte_digital_colors_rgb().values()]
  deloitte_digital_colors_hex = [colors.to_hex(rgb_float) for rgb_float in deloitte_digital_colors_rgb_float]
  
  if option == 'light':
    deloitte_digital_colors_hex = [mix_colors(color_hex, white_hex, 0.3) for color_hex in deloitte_digital_colors_hex]
  elif option == 'dark':
    deloitte_digital_colors_hex = [mix_colors(color_hex, black_hex, 0.1) for color_hex in deloitte_digital_colors_hex]
    
  return sns.color_palette(deloitte_digital_colors_hex)


def get_single_digital_color_palette(color='teal', option='light', n=6):
  """
  Returns a seaborn color palette using one of the four Deloitte Digital colors
  Color palettes are best used with discrete number of plots (e.g., 4 boxplots)
  
  Parameters
  ----------
  color : str (default = 'teal')
      Deloitte Digital color to use for palette ('green', 'blue', 'yellow', or 'teal')
  option : str (default = 'light')
      Option to change color for each additional color in the palette
      If option = 'light' or 'dark', palette colors get increasingly light/dark, starting with the original Deloitte Digital color
      If option = 'same', then all palette colors are equal to the Deloitte Digital color
  n : int (default = 6)
      Number of colors to create in palette
  
  Returns
  -------
  color_palette : seaborn.color_palette
      Palette of n colors
  """
  
  deloitte_digital_colors_rgb = get_deloitte_digital_colors_rgb()
  
  assert color in deloitte_digital_colors_rgb.keys(), 'Select color among \'green\', \'blue\', \'yellow\', and \'teal\''

  color_rgb = deloitte_digital_colors_rgb[color]
  color_hex = colors.to_hex(convert_rgb_256_to_float(color_rgb))

  white_hex = '#FFFFFF'
  black_hex = '#000000'
  
  if option == 'light':
    mix_color_hex = white_hex
  elif option == 'dark':
    mix_color_hex = black_hex
  elif option == 'same':
    mix_color_hex = color_hex
  else:
    raise ValueError('option must be \'light\', \'dark\', or \'same\'')
  
  
  palette_colors = []
  for i in range(n):
    palette_colors.append(mix_colors(color_hex, mix_color_hex, i / n))
  
  return sns.color_palette(palette_colors)


def create_digital_colormap(color1='yellow', color2='teal'):
  """
  Creates a matplotlib colormap using two of the Deloitte Digital colors
  colormaps are best used to plot continuous values (e.g., map of MAPA)
  
  Parameters
  ----------
  color1 : str (default = 'yellow')
  color2 : str (default = 'teal')
      Both color1 and color2 must be Deloitte Digital colors ('green', 'blue', 'yellow', or 'teal') 
  
  Returns
  -------
  cmap : matplotlib.colors.LinearSegmentedColormap
      colormap starting at color1 and ending at color2
  """
  
  deloitte_digital_colors_rgb = get_deloitte_digital_colors_rgb()
  
  assert color1 in deloitte_digital_colors_rgb, \
         '\'' + color1 + '\' is not a digital color. Select \'green\', \'blue\', \'yellow\', or \'teal\''
  assert color1 in deloitte_digital_colors_rgb, \
         '\'' + color1 + '\' is not a digital color. Select \'green\', \'blue\', \'yellow\', or \'teal\''

  color1_rgb = deloitte_digital_colors_rgb[color1]
  color2_rgb = deloitte_digital_colors_rgb[color2]
  
  color1_rgb_float = convert_rgb_256_to_float(color1_rgb)
  color2_rgb_float = convert_rgb_256_to_float(color2_rgb)

  cmap = colors.LinearSegmentedColormap.from_list(color1 + '-' + color2, [color1_rgb_float, color2_rgb_float])
  
  return cmap

# COMMAND ----------

# DBTITLE 1,PySpark Exploratory
def print_missing_percentage_pyspark(df):
  """
  Print an array showing the percent of missing entries in each column of a PySpark DataFrame.
  
  Parameters
  ----------
  df : PySpark DataFrame
      DataFrame with columns whose percent of missing entries is desired.
  """
  rows = df.count()
  summary = df.describe().filter(col("summary") == "count")
  final_df = summary.select(*((lit(rows)-col(c)).alias(c) for c in df.columns)).show()
  display(final_df)
  
  
def print_missing_pyspark(pyspark_df):
  """
  Print number of nulls in each column in a PySpark DataFrame
  Not as intuitive as pd_df.isna().sum() (sadly)
  """
  display(pyspark_df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in pyspark_df.columns]))

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,PySpark Exploratory

def find_time_gaps_pyspark(input_df, date_format = "yyyy-ww"):
  """
  Check for gaps in the time hierarchy
  NOTE: The internals of this function assumes you're using the YYYY-WW format for TIME_VAR.
  """
  
  from pyspark.sql.window import Window

  grouping_vars = get_hierarchy()
  
  overWindow = Window.partitionBy(grouping_vars).orderBy(TIME_VAR) 
  
  output_df = input_df.withColumn("gap", col(TIME_VAR) - lag(TIME_VAR, 1)\
                    .over(overWindow))\
                    .groupBy(grouping_vars + [TIME_VAR])\
                    .max("gap")

  
  return output_df

# COMMAND ----------

# DBTITLE 1,Workbench (DJMP): Missingness
#TODO are all of your imports in /libraries?
# (DJMP) Was hesitant to update /src/libraries, so I've added them as a workbench in that file

#TODO showMissingMap() was pretty cool; should that be included in this code?
# (DJMP) Good idea - added those here
def print_missing_row_percentage(pd_df):
  print('Percent of rows with any missing data: %3.3f%%' %
        (pd_df.isna().any(axis = 1).shape[0] / pd_df.shape[0] * 100))

  
def show_missingness_map(pd_df):
  # DONE label y-axis and remove row #s
  sns.set(font_scale = 0.5)
  fig = sns.heatmap(pd_df.isnull(), cbar = False)
  fig.set(xlabel = 'Features', ylabel = 'Rows', yticklabels = [])
  plt.tight_layout()
  display(fig)
  print('Note: black = present, taupe = missing')

#TODO can we auto-set the list of features_to_one_hot_encode rather than specifiy it as an argument
# (DJMP) By "auto-set," do you mean identifying which variables are categorical and one-hot encoding those? Or setting those as a global variable? Or something else?

#TODO curious if the 'option' parameter is really interested
# (DJMP) Did you mean interesting? Maybe some background would be helpful. When option = 'missing_only', the model predicts the missingness of a variable based on
#        whether other variables are missing (e.g., var A is missing when var B is missing). When option = 'values', the model predicts the missingness of a variable
#        based on whether othe variables are missing AND the values of the other other variables, not just if they are missing or not (e.g., var A is missing when var C = 10).  
def prep_df_for_missingness(pd_df, feature_name, features_to_one_hot_encode = [], option = 'missing_only', test_size = 0.2, random_state = 1):
  assert feature_name in pd_df.columns.values, 'feature_name is not a column in pd_df'
  assert features_to_one_hot_encode in pd_df.columns.values, 'One or more of the features in features_to_one_hot_encode is not a column in pd_df'
  assert feature_name not in features_to_one_hot_encode, 'feature_name is one of the features in features_to_one_hot_encode'
  assert option in ['missing_only', 'values'], \
    'option is not \'missing_only\' (to use only whether feature is missing or not) or \'values\' (using values to one-hot encode also)'
  
  missing_feature_name = feature_name + '_missing'
  
  #missing_pd = pd_df.copy()
  #TODO why copy this? this df could be very large
  # (DJMP) Agreed - commented out that line (for when you review again) and updated the line below
  missing_pd = pd_df.isna().astype('int')
  
  #TODO why rename these columns? seems to make your life more difficult.
  # (DJMP) Based on my transformation above (1 or 0 if value is NA), renaming columns to clarify
  missing_pd.columns = [str(col) + '_missing' for col in missing_pd.columns]

  orig_column_names = list(pd_df.columns)
  missing_column_names = list(missing_pd.columns)
  
  #TODO not sure I understand the values vs. missing_only
  # (DJMP) See above
  #TODO are the get_dummies needed if we use a more robust modeling algo that can handle categoricals (e.g., lightGBM)?
  # (DJMP) That is correct
  if (option == 'values'):
    values_pd = pd_df.copy().drop(feature_name, axis = 1)
    if len(features_to_one_hot_encode) > 0:
      values_pd_onehot = pd.get_dummies(values_pd[features_to_one_hot_encode])
      values_pd = pd.concat([values_pd.drop(features_to_one_hot_encode, axis = 1),
                             values_pd_onehot], axis = 1)
    missing_pd = pd.concat([missing_pd, values_pd], axis = 1)

  missing_features_train, missing_features_test, missing_target_train, missing_target_test = train_test_split(missing_pd.drop(feature_name, axis = 1),
                                                                                                              missing_pd[feature_name],
                                                                                                              test_size = test_size,
                                                                                                              random_state = random_state)
  #TODO is RANDOM_STATE defined somewhere?
  # (DJMP) That was from my original workbook - I've updated it to reflect that it's a variable that the user can set

  return missing_features_train, missing_features_test, missing_target_train, missing_target_test
  
#TODO is there a reason to use DecisionTreeClassifier over other, more robust methods? visualization might be one reason
# (DJMP) Used DecisionTreeClassifier since it's very explainable and easy to POC. Happy to explore other classifiers
def train_missingness_tree(missing_features_train, missing_target_train, random_state = 1):
  missingness_tree = DecisionTreeClassifier(random_state = RANDOM_STATE,
                                            class_weight = 'balanced')
  missingness_tree.fit(missing_features_train, missing_target_train)

  return missingness_tree

#TODO could rename this function to designate that it could be used to impute missing values in the main dataframe
# (DJMP) Right now, all it does is predict which features are missing based on the missingness of the others (and their values, if that option is selected).
#        It doesn't impute values right now, but we could take this a step further and predict what those values would be based on
#        other rows with similar values or missing data.
def predict_missingness_tree(missing_features_test, missingness_tree):
  missingness_pred = missingness_tree.predict(missing_features_test)
  return missingness_pred

#TODO: add function that returns image of tree

#DONE: add functions for random forest
def train_missingness_rf(missing_features_train, missing_target_train, random_state = 1):
  missingness_forest = RandomForestClassifier(random_state = RANDOM_STATE,
                                              class_weight = 'balanced')
  missingness_forest.fit(missing_features_train, missing_target_train)

  return missingness_forest

def predict_missingness_rf(missing_features_test, missingness_forest):
  missingness_pred = missingness_forest.predict(missing_features_test)
  return missingness_pred

def print_missingness_tree_report(missing_target_test, missingness_pred):
  print(classification_report(missing_target_test, missingness_pred))
#DONE why return a value here? (removed)