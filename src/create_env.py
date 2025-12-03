# src/create_env.py
#
# Creates a custom Azure ML environment for Lab 5
# This environment supports OpenCV, scikit-image, pyarrow, pandas, numpy, mlflow

from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.entities import Environment

ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# ==== Create Environment Object ====
env = Environment(
    name="tumor-lab5-env",
    description="Environment for Lab 5 - GLCM feature extraction",
    tags={"lab": "dsai3202", "phase": "silver"},
    conda_file={
        "name": "tumor-lab5-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.10",
            "pip",
            {
                "pip": [
                    "numpy",
                    "pandas",
                    "scikit-image",
                    "opencv-python",
                    "pyarrow",
                    "mlflow",
                ]
            },
        ],
    },
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20231024.v1",
)

# ==== Register the Environment ====
created_env = ml_client.environments.create_or_update(env)

print("=====================================")
print("✅ Environment created successfully!")
print(f"Name: {created_env.name}")
print(f"Version: {created_env.version}")
print("=====================================")
