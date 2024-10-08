# Databricks notebook source
# spark.conf.set(
#     "fs.azure.account.key.<storage-account>.dfs.core.windows.net",
#     dbutils.secrets.get(scope="<scope>", key="<storage-account-access-key>"))

# spark.read.load("abfss://<container-name>@<storage-account-name>.dfs.core.windows.net/<path-to-data>")

# dbutils.fs.ls("abfss://<container-name>@<storage-account-name>.dfs.core.windows.net/<path-to-data>")

# COMMAND ----------

storageAccountName = "nobre de la cuenta"
storageAccountAccessKey = "key de la cuenta"
blobContainerName = "bronze"
mountPoint = "/mnt/bronze/"
if not any(mount.mountPoint == mountPoint for mount in dbutils.fs.mounts()):
  try:
    dbutils.fs.mount(
      source = "wasbs://{}@{}.blob.core.windows.net".format(blobContainerName, storageAccountName),
      mount_point = mountPoint,
      extra_configs = {'fs.azure.account.key.' + storageAccountName + '.blob.core.windows.net': storageAccountAccessKey}
    )
    print("mount succeeded!")
  except Exception as e:
    print("mount exception", e)

# COMMAND ----------

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

dbutils.fs.mounts()

# COMMAND ----------

dbutils.fs.unmount('/mnt/bronze/')

# COMMAND ----------

path = "reddit/stream_delta_table/"
#file_path = f"abfss://bronze@sadatalakeproyectommds.dfs.core.windows.net/{path}"
file_path = "/mnt/bronze/"+path
df_btc_price = spark.read.format("delta").load(file_path)

# COMMAND ----------

df_btc_price.show()

# COMMAND ----------

dbutils.secrets.listScopes()

# COMMAND ----------

#dbutils.secrets.list('secrets_eafit_mmds')

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
