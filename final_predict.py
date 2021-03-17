from datetime import timedelta

from pyparsing import col
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark.sql.functions import array, explode, lit, struct, window
from pyspark.sql import DataFrame
from typing import Iterable
from pyspark.sql import functions as f
import numpy as np

spark = SparkSession.builder.getOrCreate()

path = "turbine_power_data.xlsx"

# Loading data from excel, specifying required properties
df_excel = spark.read.format("com.crealytics.spark.excel") \
    .option("header", "true") \
    .option("dataAddress", "'power'!A7") \
    .option("timestampFormat", "MM-dd-yyyy mm:ss") \
    .option("treatEmptyValuesAsNulls", "false") \
    .option("inferSchema", "true") \
    .load(path)

# Removing all null and corrupted data
df_filtered = df_excel.na.drop()

# Specifying the data type of each column
df_filtered_refined = df_filtered.selectExpr(
    "cast(time as timestamp) time",
    "cast(power_turbine1 as double) power_turbine1",
    "cast(power_turbine2 as double) power_turbine2",
    "cast(power_turbine1 as double) power_turbine3")

# Melting the data frame from wide to long format
df_transformed = df_filtered_refined \
    .selectExpr("time",
                "stack(3,'power_turbine1',power_turbine1,'power_turbine2',power_turbine2,'power_turbine3',power_turbine3)") \
    .withColumnRenamed("col0", "turbine") \
    .withColumnRenamed("col1", "power")

# Re-sampling and aggregating on 10 minutes interval
df_granularity = df_transformed.groupBy(window("time", "10 minute")).sum("power")
print(df_granularity.head(n=5))
df_granularity.printSchema()

df_selected = df_granularity.withColumn('time', f.col('window.end')) \
    .withColumn('power', f.col('sum(power)'))
df_selected = df_selected.drop('window')
df_selected = df_selected.drop('sum(power)')
print(df_selected.head())
df_selected.printSchema()

"""
Moving Average class
to fit the simple averaging model
based on the specified window size.
"""
class MovingAverage:
    def __init__(self, hrs):
        """
        Constructor taking initial moving average
        window duration in hours
        :param hrs:
        """
        self.hrs = hrs
        self.model = None

    def hours(self, i):
        """
        Helper method to convert hours into seconds
        :param i: Number of hours
        :return: the number of seconds
        """
        return i * 3600

    def fit(self, df, time_col, value_col):
        """
        This method calculates the moving average
        values based on the number of hours duration.
        :param df: the data frame
        :param time_col: the time column on which window is applied to get the range
        :param value_col: the value which needs to be averaged over the time window range
        :return:
        """
        w = Window.orderBy(f.col(time_col).cast('long')).rangeBetween(-self.hours(self.hrs), 0)
        self.model = df.withColumn(value_col, f.avg(value_col).over(w))

    def predict(self, next_hour=1):
        """
        This method forecast the next values based on
        the specified parameter, it calculates the
        number of steps to be predicted based on
        the next_hours and initial window size.
        :param next_hour: the values to be predicted for next_hours
        :return: the dataframe containing values (forecasted, lower/upper confidence level) for the next_hours.
        """
        steps = int(self.hours(self.hrs) / 600)
        next_steps = int(self.hours(next_hour) / 600)
        time_vector = []
        forecast = []
        predicted_value = None
        for i in range(next_steps):
            last_df = self.model.orderBy(f.desc("time")).take(steps)
            last_df = [x[1] for x in last_df]
            last_one = self.model.orderBy(f.desc("time")).take(1)
            last_one = [x[0] for x in last_one]
            df_stats = sum(last_df) / len(last_df)
            sd = float(np.std(last_df))
            n_sqrt = float(np.sqrt(len(last_df)))
            z = 1.96 # z-value for 95% confidence interval
            t_value = last_one[0] + timedelta(seconds=600)
            time_vector.append(t_value)
            forecast.append(df_stats)
            lower_level = df_stats - z * sd / n_sqrt
            upper_level = df_stats + z * sd / n_sqrt
            predicted_new = [(t_value, df_stats)]
            schema = StructType([StructField("time", TimestampType(), True), StructField("power", DoubleType(), True)])
            df_new = spark.createDataFrame(predicted_new, schema=schema)
            df_new = df_new.selectExpr(
                "cast(time as timestamp) time",
                "cast(power as double) power")
            self.model = self.model.union(df_new)

            schema_predicted = StructType([StructField("time", TimestampType(), False),
                                           StructField("power", DoubleType(), True),
                                           StructField("lower_level", DoubleType(), True),
                                           StructField("upper_level", DoubleType(), True)])
            predicted_new_level = [(t_value, df_stats, lower_level, upper_level)]
            df_new_predicted = spark.createDataFrame(predicted_new_level, schema=schema_predicted)
            if predicted_value is None:
                predicted_value = df_new_predicted
            else:
                predicted_value = predicted_value.union(df_new_predicted)
        return predicted_value


# Creating moving average object
model = MovingAverage(1) # 1 specify to consider the last 1 hour values to be averaged out to find the next value
model.fit(df_selected, 'time', 'power') # Calling fit method and specifying corresponding columns
fitted_values = model.predict(next_hour=1) # Calling predict method to predict values for next 1 hour
fitted_values.show()

fitted_values.write.csv('result.csv')
