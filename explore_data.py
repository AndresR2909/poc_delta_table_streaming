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

from pyspark.sql.functions import col

df_filtered = df.filter(col("EventProcessedUtcTime") >= "2024-11-06")
display(df_filtered)

# COMMAND ----------

from pyspark.sql.functions import year, month

df_grouped = df_filtered.groupBy(
    year("Post Date").alias("year"),
    month("Post Date").alias("month")
).count()

display(df_grouped)

# COMMAND ----------

df_grouped.agg({"count": "sum"}).alias("sum_count").show()

# COMMAND ----------

df_filtered_duplicates = df_filtered.dropDuplicates(["Post ID", "Subreddit", "User Posting", "Author", "Post Date"])
display(df_filtered_duplicates)

# COMMAND ----------

from pyspark.sql.functions import year, month

df_grouped = df_filtered_duplicates.groupBy(
    year("Post Date").alias("year"),
    month("Post Date").alias("month")
).count()

display(df_grouped)

# COMMAND ----------

df_grouped.agg({"count": "sum"}).alias("sum_count").show()

# COMMAND ----------

# DBTITLE 1,leer tabla de gold con predicciones
df_with_predictions = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions_gpt4o_v2")
display(df_with_predictions)

# COMMAND ----------

spark.sql("CREATE SCHEMA IF NOT EXISTS reddit")
df_with_predictions.write.format("delta").mode("overwrite").saveAsTable("reddit.post_predictions_gpt4o_v2")

# COMMAND ----------

df_with_predictions_old = spark.read.format("delta").load("/mnt/gold/reddit/post_predictions_gpt4o")
display(df_with_predictions_old)

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS reddit.post_predictions_gpt4o")
df_with_predictions_old.write.format("delta").mode("overwrite").saveAsTable("reddit.post_predictions_gpt4o")

# COMMAND ----------

from pyspark.sql.functions import date_format, count, sum, avg, when, col

df_with_month = df_with_predictions.withColumn("Month", date_format(df_with_predictions["Post_Date"], "yyyy-MM"))
df_grouped_by_month = df_with_month.groupBy("Month", "Subreddit").agg(
    count("User_Posting").alias("post_count"),
    sum("token_count_tiktoken").alias("total_token_count"),
    avg("token_count_tiktoken").alias("avg_token_count"),
    count(when(col("classification") == "1", True)).alias("prediction_1")
)
display(df_grouped_by_month)
