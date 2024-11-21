# Delta Table Streaming

Este repositorio contiene notebooks databricks (.py) y notebooks para el análisis y procesamiento de datos de streaming de post de reddit usando Delta Lake.

## Notebooks

- [EDA_trainig_data.ipynb](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/EDA_trainig_data.ipynb): Este notebook demuestra varios pasos de exploración y preprocesamiento de datos, incluyendo la tokenización, análisis de frecuencia y técnicas de visualización usando bibliotecas de Python como NLTK, Pandas, Matplotlib y Seaborn.

## Notebooks Databricks

- [EDA_data_reddit.py](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/EDA_data_reddit.py): Este script se conecta a las zonas del datalake, carga una tabla de muestra de streaming para análisis de datos, y realiza varias operaciones de filtrado, agrupación y creación de tablas en PySpark.
- [explore_data_xgboost.py](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/explore_data_xgboost.py): Este script se conecta al datalake, carga una tabla de muestra de streaming, limpia los datos, cuenta tokens y posts, y aplica modelos a los posts.
- [explore_data_bert.py](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/explore_data_bert.py): Este script se conecta al datalake, carga una tabla de muestra de streaming, limpia los datos, cuenta tokens, y aplica modelos a los posts.
- [stream_data_lake.py](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/stream_data_lake.py): Este script crea un DataFrame de streaming que lee de la tabla Delta, aplica funciones UDF para generar clasificaciones y contar tokens, y filtra los posts basados en el tiempo.
- [explore_data_llama.py](https://github.com/AndresR2909/poc_delta_table_streaming/blob/main/explore_data_llama.py): Este script se conecta al datalake, carga una tabla de muestra de streaming, limpia los datos, cuenta tokens, y aplica modelos a los posts.
