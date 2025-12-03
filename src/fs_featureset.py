# src/fs_featureset.py
#
# Phase 3 – Register Feature Set for MRI tumor features (new SDK version)

from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import (
    FeatureSet,
    FeatureSetSpecification,
    Feature,
)

ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# Path to your Silver Parquet output from Phase 2
silver_features_path = "azureml://datastores/workspaceblobstore/paths/silver/features.parquet"

# ===== Feature Spec: extract all columns automatically =====
spec = FeatureSetSpecification(
    features=[
        Feature(name="*", type="string")   # Wildcard → interpret all columns automatically
    ],
    source=silver_features_path,
)

# ===== Feature Set definition =====
tumor_features_fs = FeatureSet(
    name="tumor_features",
    version="1",
    description="GLCM + filter features (Silver Layer) for tumor classification",
    specification=spec,
    entities=["TumorImage:1"],   # <-- entity created earlier
    tags={"lab": "dsai3202", "phase": "3"},
)

# ===== Register the feature set =====
created = ml_client.feature_sets.create_or_update(tumor_features_fs)

print("=====================================")
print("✅ Feature Set created successfully!")
print(f"Name:    {created.name}")
print(f"Version: {created.version}")
print("=====================================")
