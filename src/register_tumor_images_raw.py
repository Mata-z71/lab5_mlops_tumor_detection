# src/register_tumor_images_raw.py
#
# Phase 1.3 - Register Bronze data asset in Azure ML:
#   tumor_images_raw -> points to raw/tumor_images in ADLS Gen2

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Data

# 1) Connect to workspace using .azureml/config.json
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# 2) Define the Data asset
data_asset = Data(
    name="tumor_images_raw",
    version="1",
    description="Brain tumor MRI images (yes/no) in Bronze layer (raw/tumor_images).",
    type="uri_folder",
    path=(
        "abfss://lakehouse@tumorlab560300294.dfs.core.windows.net/"
        "raw/tumor_images/"
    ),
)

# 3) Create or update asset
created = ml_client.data.create_or_update(data_asset)
print(f"✅ Registered data asset: {created.name}, version: {created.version}")
