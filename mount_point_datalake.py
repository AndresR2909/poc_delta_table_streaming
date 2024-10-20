# Databricks notebook source
# spark.conf.set(
#     "fs.azure.account.key.<storage-account>.dfs.core.windows.net",
#     dbutils.secrets.get(scope="<scope>", key="<storage-account-access-key>"))

# spark.read.load("abfss://<container-name>@<storage-account-name>.dfs.core.windows.net/<path-to-data>")

# dbutils.fs.ls("abfss://<container-name>@<storage-account-name>.dfs.core.windows.net/<path-to-data>")

# COMMAND ----------

#pip install python-dotenv

# COMMAND ----------

from dotenv import load_dotenv
import os
load_dotenv()

# COMMAND ----------

storageAccountName = os.getenv('STORAGE_ACCOUNT_NAME')
storageAccountAccessKey = os.getenv('STORAGE_ACCOUNT_ACCESS_KEY')

# COMMAND ----------


for point in ['/mnt/bronze/', '/mnt/silver/','/mnt/gold/']:
    try:
        dbutils.fs.unmount(point)
        print("unmount succeeded!")
    except Exception as e:
        print("unmount exception:",point)
    

# COMMAND ----------

# mount zones
for point in ['/mnt/bronze/', '/mnt/silver/','/mnt/gold/']:
  blobContainerName = point.split("/")[-2]
  mountPoint = point
  if not any(mount.mountPoint == mountPoint for mount in dbutils.fs.mounts()):
    try:
      dbutils.fs.mount(
        source = "wasbs://{}@{}.blob.core.windows.net".format(blobContainerName, storageAccountName),
        mount_point = mountPoint,
        extra_configs = {'fs.azure.account.key.' + storageAccountName + '.blob.core.windows.net': storageAccountAccessKey}
      )
      print(f"mount succeeded!: {blobContainerName}")
    except Exception as e:
      print("mount exception", e)

# COMMAND ----------

#dbutils.secrets.list('secrets_eafit_mmds')

# COMMAND ----------

#dbutils.fs.mounts()
