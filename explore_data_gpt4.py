# Databricks notebook source
# DBTITLE 1,Conectar zonas del datalake
# MAGIC %run /Workspace/Users/afrestrepa@eafit.edu.co/poc_delta_table_streaming/mount_point_datalake

# COMMAND ----------

# DBTITLE 1,cargar tabla muestra streaming para análisis de datos
df = spark.read.format("delta").load("/mnt/bronze/reddit")
display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df = df.withColumnRenamed("Post Date", "Post_Date") \
                        .withColumnRenamed("User Posting", "User_Posting")

# COMMAND ----------

# DBTITLE 1,limpiar los post de caracteres especiales, doble espacio, emojis
from pyspark.sql.functions import regexp_replace, trim

# Remove special characters, double spaces, emojis, and newlines
df_cleaned = df.withColumn("User_Posting", regexp_replace("User_Posting", "[^\\x00-\\x7F]+", "")) \
               .withColumn("User_Posting", regexp_replace("User_Posting", "\\s+", " ")) \
               .withColumn("User_Posting", regexp_replace("User_Posting", "[\\n\\r]", " ")) \
               .withColumn("User_Posting", trim("User_Posting"))

display(df_cleaned)

# COMMAND ----------

# DBTITLE 1,contar el numero de token por post
import tiktoken
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType

# Define a UDF to count tokens using tiktoken
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

count_tokens_udf = udf(count_tokens, IntegerType())

# COMMAND ----------

# DBTITLE 1,contar el numero de post y tokens por posts agrupados por dia
from pyspark.sql.functions import size, split, count, sum, avg, to_date

#df_with_words_count = df.withColumn("words_count", size(split(df["User Posting"], " ")))
df_with_token_count = df_cleaned.withColumn("token_count_tiktoken", count_tokens_udf(df_cleaned["User_Posting"]))
df_with_date = df_with_token_count.withColumn("Post_Date", to_date(df_with_token_count["Post_Date"]))
df_grouped = df_with_date.groupBy("Post_Date", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count")
)

display(df_grouped)

# COMMAND ----------

# DBTITLE 1,contar el numero de post y tokens por posts agrupados por mes
from pyspark.sql.functions import date_format

df_with_month = df_with_date.withColumn("Month", date_format(df_with_date["Post_Date"], "yyyy-MM"))
df_grouped_by_month = df_with_month.groupBy("Month", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count")
)

display(df_grouped_by_month)

# COMMAND ----------

# DBTITLE 1,aplicar modelo a post (filtrar por numero de tokens >=10)
from pyspark.sql.functions import when, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import requests


# Define a UDF to query the model endpoint and extract prediction and prediction_prob
def query_model(text):
    API_URL = "https://si5rdemrbopl05yv.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": text,
        "parameters": {}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()[0]
    return (result['prediction'], result['prediction_prob'])

schema = StructType([
    StructField("prediction", StringType(), True),
    StructField("prediction_prob", FloatType(), True)
])

query_model_udf = udf(query_model, schema)

# Apply the UDF conditionally based on token count
df_with_model_output = df_with_token_count.withColumn(
    "model_output",
    when(df_with_token_count["token_count_tiktoken"] > 10, query_model_udf(df_with_token_count["User_Posting"]))
)

# Extract prediction and prediction_prob into separate columns
df_with_predictions = df_with_model_output.select(
    "Subreddit",
    "Thread",
    "Author",
    "Post_Date",
    "User_Posting",
    "token_count_tiktoken",
    col("model_output.prediction").alias("prediction"),
    col("model_output.prediction_prob").alias("prediction_prob")
)
df_with_predictions.write.format("delta").mode("overwrite").save("/mnt/gold/reddit/post_predictions")
display(df_with_predictions)

# COMMAND ----------

# DBTITLE 1,cargar df desde tabla con predicciones en gold
df_with_predictions = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions")
display(df_with_predictions)

# COMMAND ----------

from pyspark.sql.functions import date_format, count, sum, avg

df_with_month = df_with_predictions.withColumn("Month", date_format(df_with_predictions["Post_Date"], "yyyy-MM"))
df_grouped_by_month = df_with_month.groupBy("Month", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count"),
    count(when(col("prediction") == "1", True)).alias("prediction_1")
)
display(df_grouped_by_month)
