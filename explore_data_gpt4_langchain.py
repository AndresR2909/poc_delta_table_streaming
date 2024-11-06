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

# DBTITLE 1,eliminar registros repetidos
df = df.dropDuplicates(["Post ID", "Subreddit", "User Posting", "Author", "Post Date"])
display(df)

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

from pyspark.sql.functions import year, month

df_with_token_count_grouped = df_with_token_count.groupBy(
    year("Post_Date").alias("year"),
    month("Post_Date").alias("month")
).count()

display(df_with_token_count_grouped)

# COMMAND ----------

from pyspark.sql.functions import year, month

df_with_token_count_grouped = df_with_token_count.groupBy(
    year("Post_Date").alias("year"),
).count()

display(df_with_token_count_grouped)

# COMMAND ----------

df_with_token_count_grouped.agg({"count": "sum"}).alias("sum_count").show()

# COMMAND ----------

# DBTITLE 1,modelo chatgp4
from src.open_ia_chat import OpenAIChatModel
from langchain_core.prompts import ChatPromptTemplate

open_ia_deployments = [{"azure":False,"deployment":"gpt-4o-mini"},{"azure":False,"deployment":"gpt-4o-2024-08-06"}]

open_ia_model = OpenAIChatModel(**open_ia_deployments[1])

# COMMAND ----------

open_ia_model.chat_model_client.invoke("suma 2 mas 2").content

# COMMAND ----------

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import logging
import json

# Definir los despliegues disponibles
open_ia_deployments = [
    {"azure": False, "deployment": "gpt-4o-mini"},
    {"azure": False, "deployment": "gpt-4o-2024-08-06"}
]


# Definir la función de generación de clasificación
def generate_classification(text):
    from pydantic import BaseModel, Field
    # Definir el modelo de clasificación utilizando Pydantic
    class Classification(BaseModel):
        clase: int = Field(description="Asigna 1 si hay ideación suicida, 0 si no",enum=[1, 0])
        probabilidad: float = Field(description="Probabilidad entre 0 y 1 del resultado de la clase")
    # Definir el prompt de clasificación
    tagging_prompt = ChatPromptTemplate.from_template(
        """
        Analiza el siguiente texto y clasifícalo de la siguiente manera:
        - Asigna "1" si identificas ideación o comportamiento suicida
        - Asigna "0" si no identificas contenido relacionado con suicidio
        Además, proporciona una probabilidad entre 0 y 1 que refleje tu nivel de confianza en la clasificación.
        Responde en el siguiente formato JSON:
        {{
            "clase": <0 o 1>,
            "probabilidad": <número entre 0 y 1>
        }}
        
        Texto a analizar:
        {input}
        """
    )
    try:
        open_ia_model = OpenAIChatModel(**open_ia_deployments[1])
        # Instanciar el modelo OpenAI adecuado
        llm = open_ia_model.chat_model_client.with_structured_output(Classification)
        
        # Generar la cadena de clasificación
        tagging_chain = tagging_prompt | llm
        
        # Ejecutar el análisis
        response = tagging_chain.invoke({"input": text})
        
        # Devolver el resultado en formato JSON
        return response.dict()
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return {"clase": -1, "probabilidad": 0}

# Ejemplo de uso
text_to_analyze = "No puedo más, siento que no vale la pena seguir adelante."
classification_result = generate_classification(text_to_analyze)
print(json.dumps(classification_result, indent=2))

# COMMAND ----------

text = "Si usted es de los que comenta todo en el hp cine, o quiere parecer inteligentísimo diciendo lo que va a pasar en la película... le tengo una sugerencia:"
response = generate_classification(text)
response

# COMMAND ----------

# DBTITLE 1,aplicar modelo a post (filtrar por numero de tokens >=10)
from pyspark.sql.functions import when, col, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType
import json

def query_model(text):
    return generate_classification(text)

query_model_udf = udf(
    query_model, 
    StructType([
        StructField("clase", StringType(), True),
        StructField("probabilidad", FloatType(), True)
    ])
)

# Apply the UDF conditionally based on token count
df_with_model_output = df_with_token_count.withColumn(
    "model_output",
    when(
        df_with_token_count["token_count_tiktoken"] > 10, 
        query_model_udf(df_with_token_count["User_Posting"])
    )
)

# Avoid repeating API calls
df_with_model_output.cache()

# Extract prediction and prediction_prob into separate columns
df_with_predictions = df_with_model_output.select(
    "Subreddit",
    "Thread",
    "Author",
    "Post_Date",
    "User_Posting",
    "token_count_tiktoken",
    col("model_output.clase").alias("classification"),
    col("model_output.probabilidad").alias("prediction_prob")
)

df_with_predictions.write.format("delta").mode("overwrite").save("/mnt/gold/reddit/post_predictions_gpt4o_v2")
display(df_with_predictions)

# COMMAND ----------

df_with_predictions = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions_gpt4o_v2")
display(df_with_predictions)

# COMMAND ----------

from pyspark.sql.functions import date_format, count, sum, avg

df_with_month = df_with_predictions.withColumn("Month", date_format(df_with_predictions["Post_Date"], "yyyy-MM"))
df_grouped_by_month = df_with_month.groupBy("Month", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count"),
    count(when(col("classification") == "1", True)).alias("prediction_1")
)
display(df_grouped_by_month)
