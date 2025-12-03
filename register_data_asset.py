# src/register_tumor_images_raw.py
#
# Register tumor_images_raw as a Data Asset in Azure ML

from azure.identity import InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data

# Connect to workspace
ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# Create Data asset definition
data_asset = Data(
    name="final_lab05_tumor_images_raw",
    version="1",
    description="Brain tumor MRI images (yes/no) in Bronze layer.",
    type="uri_folder",
    path=(
        "abfss://lakehouse@tumorlab560300294.dfs.core.windows.net/"
        "raw/tumor_images/"
    ),
)

# Register the data asset
created = ml_client.data.create_or_update(data_asset)
print(f"✅ Registered data asset: {created.name}, version: {created.version}")
