# Databricks notebook source
# Definir una función personalizada que llame al modelo en Hugging Face
def custom_model_inference(price):
    url = "https://api-inference.huggingface.co/models/tu-modelo"
    headers = {"Authorization": "Bearer tu-token"}
    payload = {"inputs": price}
    
    response = requests.post(url, headers=headers, json=payload)
    result = response.json()
    
    # Suponiendo que el resultado es un valor numérico
    return result[0]['score']

# Registrar la función UDF
custom_model_udf = udf(custom_model_inference, DoubleType())

# Aplicar la función personalizada en el DataFrame
streamingCountsDF = (
  streamingInputDF
    .groupBy(window(streamingInputDF.EventEnqueuedUtcTime, "5 minute"))
    .agg(custom_model_udf(streamingInputDF.price).alias("custom_model_price"))
)

display(streamingCountsDF)

# COMMAND ----------

path = "reddit/stream_delta_table/"
file_path = "/mnt/bronze/"+path
df_btc_price = spark.read.format("delta").load(file_path)

# COMMAND ----------

df_btc_price.show()

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

path = "reddit/stream_delta_table/"
file_path = f"abfss://bronze@sadatalakeproyectommds.dfs.core.windows.net/{path}"
streamingInputDF = (
    spark.readStream
    .format("delta")
    .load(file_path)
    .filter("EventEnqueuedUtcTime >= '2024-10-01'")
)

# COMMAND ----------

from pyspark.sql.functions import window

streamingCountsDF = (
  streamingInputDF
    .groupBy(
      window(streamingInputDF.EventEnqueuedUtcTime, "5 minute"))
    .avg("price")
)

# COMMAND ----------

query = (
  streamingCountsDF
    .writeStream
    .format("memory")        # memory = store in-memory table (for testing only)
    .queryName("mean")     # counts = name of the in-memory table
    .outputMode("complete")  # complete = all the counts should be in the table
    .start()
)

# COMMAND ----------

# MAGIC %sql select `avg(price)`, date_format(window.end, "MMM-dd HH:mm") as time from mean order by time asc
