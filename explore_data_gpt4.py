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
df_with_token_count = df_cleaned.withColumn("token_count_tiktoken", count_tokens_udf(df_cleaned["User_Posting"]))

# COMMAND ----------

# DBTITLE 1,modelo chatgp4
from src.open_ia_chat import OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate

open_ia_deployments = [{"azure":False,"deployment":"gpt-4o-mini"},{"azure":False,"deployment":"gpt-4o-2024-08-06"}]

open_ia_model = OpenAIChatModel(**open_ia_deployments[1])

# COMMAND ----------

open_ia_model.chat_model_client.invoke("suma 2 mas 2").content

# COMMAND ----------

#system_msg = """
#Clasifica el siguiente post con "1" si identificas ideación o comportamiento suicida, y con "0" si no. Solo responde con "0" o "1" sin agregar ningún comentario #adicional.
#Post:""".strip()
#zero_shot_prompt = ChatPromptTemplate.from_messages([("system", system_msg,),("human", "{text}"),])

# COMMAND ----------


from src.open_ia_chat import OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate
import logging
import json

open_ia_deployments = [{"azure":False,"deployment":"gpt-4o-mini"},{"azure":False,"deployment":"gpt-4o-2024-08-06"}]

def generate_clasification(text):
    open_ia_model = OpenAIChatModel(**open_ia_deployments[1])
    
    # system_msg = """Clasifica el siguiente texto con "1" si identificas ideación o comportamiento suicida, y con #"0" si no. Además, proporciona la probabilidad de la clase seleccionada sea la correcta. Responde estrictamente #en el siguiente formato JSON: {{ "clase": <0 o 1>, "probabilidad": <número entre 0 y 1> }}""".strip()

    system_msg = """
    Analiza el siguiente texto y clasifícalo de la siguiente manera:
    - Asigna "1" si identificas ideación o comportamiento suicida
    - Asigna "0" si no identificas contenido relacionado con suicidio
    Además, proporciona una probabilidad entre 0 y 1 que refleje tu nivel de confianza en la clasificación.
    Responde ÚNICAMENTE en el siguiente formato JSON:
    {{
        "clase": <0 o 1>,
        "probabilidad": <número entre 0 y 1>
    }}
    Texto a analizar:
    """.strip()

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
        print(response.content)
        return { "clase": -1, "probabilidad": 0 }

# COMMAND ----------

text = "Si usted es de los que comenta todo en el hp cine, o quiere parecer inteligentísimo diciendo lo que va a pasar en la película... le tengo una sugerencia:"
response = generate_clasification(text)
response

# COMMAND ----------

# DBTITLE 1,aplicar modelo a post (filtrar por numero de tokens >=10)
from pyspark.sql.functions import when, col, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import json

# Define a UDF to query the model endpoint and extract prediction
def query_model(text):
    return generate_clasification(text)

query_model_udf = udf(query_model, StructType([
    StructField("clase", StringType(), True),
    StructField("probabilidad", FloatType(), True)
]))



# Apply the UDF conditionally based on token count
df_with_model_output = df_with_token_count.withColumn(
    "model_output",
    when(df_with_token_count["token_count_tiktoken"] > 10, query_model_udf(df_with_token_count["User_Posting"]))
).cache()


# Extract prediction and prediction_prob into separate columns
df_with_predictions = df_with_model_output.select(
    "Subreddit",
    "Thread",
    "Author",
    "Post_Date",
    "User_Posting",
    "token_count_tiktoken",
    col("model_output.clase").alias("clasification"),
    col("model_output.probabilidad").alias("prediction_prob")
)
df_with_predictions.write.format("delta").mode("overwrite").save("/mnt/gold/reddit/post_predictions_gpt4o")
display(df_with_predictions)

# COMMAND ----------

df_with_predictions = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions_gpt4o")
display(df_with_predictions)

# COMMAND ----------

from pyspark.sql.functions import date_format, count, sum, avg, when, col

df_with_month = df_with_predictions.withColumn("Month", date_format(df_with_predictions["Post_Date"], "yyyy-MM"))
df_grouped_by_month = df_with_month.groupBy("Month", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count"),
    count(when(col("clasification") == "1", True)).alias("prediction_1")
)
display(df_grouped_by_month)
