# Databricks notebook source
# MAGIC %run /Workspace/Users/afrestrepa@eafit.edu.co/poc_delta_table_streaming/mount_point_datalake

# COMMAND ----------

# DBTITLE 1,Streaming DataFrame that reads from the Delta table and filters posts based on the EventEnqueuedUtcTime
file_path ="/mnt/bronze/reddit/"
streamingInputDF = (
    spark.readStream
    .format("delta")
    .load(file_path)
    .filter("EventEnqueuedUtcTime >= '2024-10-22'")
)

# COMMAND ----------

# DBTITLE 1,UDF: Generate a classification for the given text using an OpenAI model.
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import json
from src.open_ia_chat import OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate
import logging

open_ia_deployments = [{"azure":False,"deployment":"gpt-4o-mini"},{"azure":False,"deployment":"gpt-4o-2024-08-06"}]

def generate_clasification(text):
    """
    Generates a classification for the given text using an OpenAI model.

    Args:
        text (str): The text to classify.

    Returns:
        dict: A dictionary containing the classification result with keys:
            - "clase" (int): 1 if suicidal ideation or behavior is identified, 0 otherwise, -1 if an error occurs.
            - "probabilidad" (float): The probability that the classification is correct, 0 if an error occurs.
    """
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

# DBTITLE 1,Apply udf quey model and count_tokens to de stream
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
    .withColumn("User_Posting", regexp_replace("User Posting", "[^\\x00-\\x7F]+", ""))  # Remove non-ASCII characters
    .withColumn("User_Posting", regexp_replace("User Posting", "\\s+", " "))  # Replace multiple spaces with a single space
    .withColumn("User_Posting", regexp_replace("User Posting", "[\\n\\r]", " "))  # Remove new line and carriage return characters
    .withColumn("User_Posting", trim("User Posting"))  # Trim leading and trailing spaces
    .withColumn("token_count_tiktoken", count_tokens_udf(col("User_Posting")))  # Count tokens using tiktoken
    .filter(col("token_count_tiktoken") > 10)  # Filter posts with more than 10 tokens
    .withColumn("model_output", query_model_udf(col("User_Posting")))  # Apply the custom model UDF
    .withColumn("clase", col("model_output.clase"))  # Extract 'clase' from model output
    .withColumn("probabilidad", col("model_output.probabilidad"))  # Extract 'probabilidad' from model output
    .filter(col("clase") == 1)  # Filter posts classified as having suicidal ideation or behavior
)

display(streamingPostReddit)

# COMMAND ----------

# DBTITLE 1,write stream
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
