# Databricks notebook source
# MAGIC %run /Workspace/Users/afrestrepa@eafit.edu.co/poc_delta_table_streaming/mount_point_datalake

# COMMAND ----------

file_path ="/mnt/bronze/reddit/"
streamingInputDF = (
    spark.readStream
    .format("delta")
    .load(file_path)
    .filter("EventEnqueuedUtcTime >= '2024-10-22'")
)

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, FloatType
import json
from src.open_ia_chat import OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate
import logging

open_ia_deployments = [{"azure":False,"deployment":"gpt-4o-mini"},{"azure":False,"deployment":"gpt-4o-2024-08-06"}]

def generate_clasification(text):
    open_ia_model = OpenAIChatModel(**open_ia_deployments[1])
    
    system_msg = """Clasifica el siguiente post con "1" si identificas ideación o comportamiento suicida, y con "0" si no. Además, proporciona la probabilidad de la clase seleccionada sea la correcta. Responde estrictamente en el siguiente formato JSON: {{ "clase": <0 o 1>, "probabilidad": <número entre 0 y 1> }}""".strip()

    zero_shot_prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{text}")
    ])
    llm = open_ia_model.chat_model_client
    chain = zero_shot_prompt | llm
    try:
        response = chain.invoke(
            {
                "text": text,
            }
        )
        return json.loads(response.content)
    except Exception as e:
        logging.info(e)
        return { "clase": -1, "probabilidad": 0 }

# COMMAND ----------

from pyspark.sql.functions import regexp_replace, trim, udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
import tiktoken

# Define a UDF to count tokens using tiktoken
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

count_tokens_udf = udf(count_tokens, IntegerType())

# Define a UDF to query the model endpoint and extract prediction
def query_model(text):
    return generate_clasification(text)

query_model_udf = udf(query_model, StructType([
    StructField("clase", IntegerType(), True),
    StructField("probabilidad", FloatType(), True)
]))

# Apply cleaning to the User_Posting column, count tokens, and apply the custom model UDF
streamingPostReddit = (
  streamingInputDF
    .withColumn("User_Posting", regexp_replace("User Posting", "[^\\x00-\\x7F]+", "")) 
    .withColumn("User_Posting", regexp_replace("User Posting", "\\s+", " ")) 
    .withColumn("User_Posting", regexp_replace("User Posting", "[\\n\\r]", " ")) 
    .withColumn("User_Posting", trim("User Posting")) 
    .withColumn("token_count_tiktoken", count_tokens_udf(col("User_Posting")))
    .filter(col("token_count_tiktoken") > 10)
    .withColumn("model_output", query_model_udf(col("User_Posting")))
    .withColumn("clase", col("model_output.clase"))
    .withColumn("probabilidad", col("model_output.probabilidad"))
    .filter(col("clase") == 1)
)

display(streamingPostReddit)

# COMMAND ----------

# from pyspark.sql.functions import window, col, size, split, count, sum

# streamingCountsDF = (
#   streamingInputDF
#     .groupBy(window(streamingInputDF.EventEnqueuedUtcTime, "5 minute"))
#     .agg(
#         sum(size(split(col("User Posting"), " "))).alias("word_count"),
#         count("User Posting").alias("post_count")
#     )
# )

# display(streamingCountsDF)

# COMMAND ----------

query = (
  streamingPostReddit
    .writeStream
    .format("memory")        # memory = store in-memory table (for testing only)
    .queryName("post_reddit")     # counts = name of the in-memory table
    .outputMode("complete")  # complete = all the counts should be in the table
    .start()
)

# COMMAND ----------

# MAGIC %sql select * as time from post_reddit order by time asc
