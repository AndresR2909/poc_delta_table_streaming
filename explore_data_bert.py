# Databricks notebook source
# DBTITLE 1,Conectar zonas del datalake
# MAGIC %run /Workspace/Users/afrestrepa@eafit.edu.co/poc_delta_table_streaming/mount_point_datalake

# COMMAND ----------

from dotenv import load_dotenv
import os
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# COMMAND ----------

# DBTITLE 1,cargar tabla muestra streaming para análisis de datos
df = spark.read.format("delta").load("/mnt/bronze/reddit")
display(df)

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

df_with_token_count = df.withColumn("token_count_tiktoken", count_tokens_udf(df["User Posting"]))

# COMMAND ----------

# DBTITLE 1,aplicar modelo a post (filtrar por numero de tokens >=10)
from pyspark.sql.functions import when, col
from pyspark.sql.types import StringType
import requests

# Define a UDF to query the model endpoint and extract prediction
def query_model(text):
    API_URL = "https://zxdya8c7y59kegiu.us-east-1.aws.endpoints.huggingface.cloud"
    headers = {
        "Accept" : "application/json",
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json" 
    }
    payload = {
        "inputs": text,
        "parameters": {}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()[0]
    return result['clasiffication']

query_model_udf = udf(query_model, StringType())

# Apply the UDF conditionally based on token count
df_with_model_output = df_with_token_count.withColumn(
    "prediction",
    when(df_with_token_count["token_count_tiktoken"] > 10, query_model_udf(df_with_token_count["User Posting"]))
)

# Select relevant columns
df_with_predictions = df_with_model_output.select(
    "Subreddit",
    "Thread",
    "Author",
    "Post Date",
    "User Posting",
    "token_count_tiktoken",
    "prediction"
)
display(df_with_predictions)

# COMMAND ----------

df_with_predictions = df_with_predictions.withColumnRenamed("Post Date", "Post_Date") \
                                         .withColumnRenamed("User Posting", "User_Posting")

df_with_predictions.write.format("delta").mode("overwrite").save("/mnt/gold/reddit/post_predictions_bert")

# COMMAND ----------

# DBTITLE 1,cargar df desde tabla con predicciones en gold
df_with_predictions = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions_bert")
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
