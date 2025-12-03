# src/fs_entity.py
#
# Phase 3 – Create Feature Store Entity for image_id

from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import FeatureStoreEntity, DataColumn, DataColumnType

# 1) Connect to your workspace / feature store
#    For the lab we reuse the same config you’ve been using.
ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# 2) Define the Feature Store Entity
tumor_entity = FeatureStoreEntity(
    name="TumorImage",              # or "tumor_image" if you prefer
    version="1",
    index_columns=[
        DataColumn(name="image_id", type=DataColumnType.STRING)
    ],
    description="Entity for MRI tumor images, keyed by image_id.",
    tags={"lab": "dsai3202", "phase": "3"},
)

# 3) Create / update the entity in the feature store
poller = ml_client.feature_store_entities.begin_create_or_update(tumor_entity)
result = poller.result()

print("=====================================")
print("✅ Feature Store Entity created!")
print(f"Name:    {result.name}")
print(f"Version: {result.version}")
print("=====================================")
