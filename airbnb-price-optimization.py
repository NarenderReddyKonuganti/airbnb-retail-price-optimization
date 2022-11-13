# Databricks notebook source
# MAGIC %md
# MAGIC # Homework: Introductory Regression
# MAGIC 
# MAGIC ## Predict Air B&B prices
# MAGIC * Download the San Francisco **listings.csv.gz** from http://insideairbnb.com/get-the-data.html
# MAGIC   * Search for San Francisco on the page, and then right click to download the data
# MAGIC * Read the uncompressed csv file
# MAGIC * Select a subset of columns for regression
# MAGIC   * you will predict the *price* column
# MAGIC * Cast column values to double or int
# MAGIC   * price will need to be parsed as a double from the currency format (e.g., $100.00)
# MAGIC     * Check your capitalization!
# MAGIC * Check the data using DataFrame.summary()
# MAGIC * Filter out empty or erroneous data
# MAGIC * Graph some raw data (e.g., bar chart)
# MAGIC * Split the data into training and test
# MAGIC * Create a regression model from the training data
# MAGIC * Test the regression model on the test data
# MAGIC * Evaluate the model
# MAGIC 
# MAGIC Look for the **TODO** notes below
# MAGIC * This notebook is a subset from an answer notebook
# MAGIC * You may choose to approach the problem differently, but address all the **ToDos**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Remove any old files

# COMMAND ----------

# MAGIC %sh
# MAGIC rm -rf /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022/*
# MAGIC rm -rf /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022/*
# MAGIC pwd

# COMMAND ----------

# MAGIC %md
# MAGIC ## ToDo: Download data
# MAGIC * Get the link for wget from here: http://insideairbnb.com/get-the-data.html
# MAGIC   * Right-click on the San Francisco listings.csv.gz link to Copy Link
# MAGIC   * Paste the link after wget below

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022
# MAGIC wget http://data.insideairbnb.com/united-states/ca/san-francisco/2022-09-07/data/listings.csv.gz
# MAGIC wget http://data.insideairbnb.com/united-states/ca/san-francisco/2022-09-07/data/reviews.csv.gz
# MAGIC gunzip /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022/listings.csv.gz
# MAGIC gunzip /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022/reviews.csv.gz
# MAGIC cd /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022
# MAGIC wget http://data.insideairbnb.com/united-states/ca/san-francisco/2022-06-03/data/listings.csv.gz
# MAGIC wget http://data.insideairbnb.com/united-states/ca/san-francisco/2022-06-03/data/reviews.csv.gz
# MAGIC gunzip /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022/listings.csv.gz
# MAGIC gunzip /Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022/reviews.csv.gz

# COMMAND ----------

# DBTITLE 1,list the files
# MAGIC %sh
# MAGIC cd Jun2022
# MAGIC mv listings.csv listings_jun2022.csv
# MAGIC mv reviews.csv reviews_jun2022.csv
# MAGIC cd ../Sep2022
# MAGIC mv listings.csv listings_sep2022.csv
# MAGIC mv reviews.csv reviews_sep2022.csv

# COMMAND ----------

# Ideally, do this
# But, the CSV reader is not as good as the Pandas CSV reader
#lDF = spark.read.csv(path='file:///databricks/driver/listings.csv',header='true', inferSchema ='true', sep=',', mode='DROPMALFORMED')

# COMMAND ----------

# DBTITLE 1,Read the data
import pandas as pd
# Read all as String, https://stackoverflow.com/questions/16988526/pandas-reading-csv-as-string-type
listings_sep22 = pd.read_csv('/Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022/listings_sep2022.csv',converters={i: str for i in range(100)}) 
reviews_sep22 = pd.read_csv('/Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Sep2022/reviews_sep2022.csv',converters={i: str for i in range(100)}) 
# From pandas to DataFrame
df_listings_sep22 = sqlContext.createDataFrame(listings_sep22)
df_reviews_sep22 = sqlContext.createDataFrame(reviews_sep22)

# COMMAND ----------

# Read all as String, https://stackoverflow.com/questions/16988526/pandas-reading-csv-as-string-type
listings_jun22 = pd.read_csv('/Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022/listings_jun2022.csv',converters={i: str for i in range(100)}) 
reviews_jun22 = pd.read_csv('/Workspace/Repos/narenderreddy548@gmail.com/airbnb-repo/Jun2022/reviews_jun2022.csv',converters={i: str for i in range(100)}) 
# From pandas to DataFrame
df_listings_jun22 = sqlContext.createDataFrame(listings_jun22)
df_reviews_jun22 = sqlContext.createDataFrame(reviews_jun22)

# COMMAND ----------

df_listings_sep22.printSchema()
df_reviews_sep22.printSchema()
df_listings_jun22.printSchema()
df_reviews_jun22.printSchema()

# COMMAND ----------

listings_sep22

# COMMAND ----------

reviews_sep22

# COMMAND ----------

listings_jun22

# COMMAND ----------

reviews_jun22

# COMMAND ----------

# MAGIC %md
# MAGIC ## ToDo: Select a subset of columns to work with
# MAGIC * select **price** and more than 5 other columns

# COMMAND ----------

reviews_jun22.columns

# COMMAND ----------

import pandas as pd
listings= pd.concat([listings_jun22, listings_sep22])
listings

# COMMAND ----------

import pandas as pd
reviews= pd.concat([reviews_jun22, reviews_sep22])
reviews

# COMMAND ----------

import random
len(listings['last_scraped'].unique())
# s = pd.Series(np.random.randn())
# a = pd.DataFrame(np.random.randn(0,1), columns=list('Occupancy Rate'))
# df['Occupancy Rate'] = random.uniform(0, 1)
listings['property_type'].unique()

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
%matplotlib inline

# u'zipcode',u'location_price',,u'instant_bookable', u'host_is_superhost',u'host_response_rate',
selected_features = [u'price',u'accommodates',u'host_response_time',
       u'bathrooms', u'bedrooms', u'beds',
       u'minimum_nights', u'maximum_nights', 
       u'availability_365',
       u'number_of_reviews', u'review_scores_rating',u'review_scores_cleanliness', u'review_scores_checkin',
       u'review_scores_communication', u'review_scores_location',
       u'review_scores_value',u'amenities', 'room_type', 'property_type']
listings = listings.loc[:, selected_features]
listings = listings.apply(lambda x:x.fillna(x.value_counts().index[0]))
listings.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ToDo: Convert some string types to ...
# MAGIC * Follow the pattern below to convert types

# COMMAND ----------

from pyspark.sql.functions import regexp_extract,col
# Get just number for price
df = df.withColumn('price', regexp_extract(col('price'), '\$?(\d*\.?\d*)', 1))
#df = df.withColumn('cleaning_fee', regexp_extract(col('cleaning_fee'), '\$?(\d*\.?\d*)', 1))
# Now cast...
df = df.withColumn('price', df['price'].cast('double'))
df = df.withColumn('beds', df['beds'].cast('int'))
df = df.withColumn('bedrooms', df['bedrooms'].cast('int'))
# ...

# COMMAND ----------

display(df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ToDo: Spot check data
# MAGIC * use df.summary() to review the data

# COMMAND ----------

# Consider this pattern...
print(df.where('price =0').count())
print(df.where('bedrooms = 0').count())
# ...

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## TODO
# MAGIC * Only keep rows where bedrooms > 0
# MAGIC * Only keep rows where price >0 and price < 900000000
# MAGIC   * Ensure no wierd prices
# MAGIC * Remove rows where there are null values
# MAGIC   * use dropna()
# MAGIC * **After** regression, come back here to see if you can improve the model fit by removing errornous rows

# COMMAND ----------

# Add some PySpark DataFrame filters to remove some data
# df = ...
df.count()

# COMMAND ----------

display(df.select('neighbourhood','bedrooms','beds','price'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Graph count by neighbourhood

# COMMAND ----------

dfNeighboorhood = df.groupby('neighbourhood').agg({'neighbourhood' : 'count'})\
                    .withColumnRenamed("count(neighbourhood)", "count")\
                    .orderBy("count")
pdf = dfNeighboorhood.toPandas()

#%matplotlib inline # Use display,  see https://docs.databricks.com/user-guide/visualizations/matplotlib-and-ggplot.html
import matplotlib.pyplot as plt   #Data visualisation libraries 
import seaborn as sns  # https://seaborn.pydata.org/generated/seaborn.barplot.html
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
sns.set(style="white", font_scale=.5)
# head for top n rows
ax = sns.barplot(data=pdf, y='neighbourhood', x='count')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ToDo: Prepare test data
# MAGIC * Split the data into train_data and test_data

# COMMAND ----------

train_data,test_data  = df.randomSplit([0.6, 0.4], 24)   # proportions [], seed for random
# 
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression
# MAGIC * Predict price with the other columns
# MAGIC * Use RFormula to create a linear regression model
# MAGIC * Prepare the train_data and test_data for use with the RFormula
# MAGIC   * Start with a copy of the cell for the same purpose, found in the Spark example

# COMMAND ----------

from pyspark.ml.feature import RFormula 
columns = df.columns
# Not using price (label) or neighbourhood in features
columns.remove('price')
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply regression to the data
# MAGIC * Create a LinearRegression with the train_preparedDF data
# MAGIC * Apply that model to the test_preparedDF
# MAGIC   * Start with a copy of the cell for the same purpose, found in the Spark example

# COMMAND ----------

# DBTITLE 1,See the PySpark regression from the other notebook
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol ="label", featuresCol ="features")
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Show the predicted results

# COMMAND ----------

# There are more labels than predictions.
# Here, get pairs where there is both a label and prediction
# collect gets the data from the data grid and places the results in a list (label, prediction)
# zip(* ) converts the tuple list into two lists
y_test,predictions = zip(*labeledPredictions.collect())
fig, ax = plt.subplots()
plt.scatter(y_test,predictions)
display(fig)

# COMMAND ----------

lrModel = train_fittedLR
# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lrModel.coefficients))
print("Intercept: %s" % str(lrModel.intercept))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the model
# MAGIC * Print the RMSE and r2 for the model
# MAGIC   * Start with a copy of the cell for the same purpose, found in the Spark example

# COMMAND ----------

# Summarize the model over the training set and print out some metrics
# ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## ToDo
# MAGIC Think about the following:
# MAGIC * Can the model fit (R^2) be improved? How?
# MAGIC   * Can you select better features (columns)?
# MAGIC   * Are there errors in some of the data?
# MAGIC     * Consider the price column. Do all the values look accurate? Can you filter out rows that have errors? Would the model fit improve?
# MAGIC * What's the best model fit (R^2) that you can obtain?

# COMMAND ----------


