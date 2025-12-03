# run_extract_features_job.py
#
# Phase 2 – Silver Layer: Submit feature extraction as a simple CommandJob
# NO function-style component call, so NO unexpected keyword errors.

from azure.ai.ml import MLClient, command, Input, Output
from azure.identity import InteractiveBrowserCredential

COMPUTE_TARGET = "Standard-E4ds-v4"
# -------------------------------------------------------

# 1) Connect to Azure ML workspace
ml_client = MLClient.from_config(
    credential=InteractiveBrowserCredential()
)

# 2) Define the CommandJob directly
job = command(
    display_name="Silver - Tumor Image Feature Extraction",
    experiment_name="lab5_silver_feature_extraction",

    code="./src",  # folder that contains extract_features.py

    command=(
        "python extract_features.py "
        "--input_data ${{inputs.input_data}} "
        "--output_features ${{outputs.output_features}}"
    ),

    environment="azureml:tumor-lab5-env:1",

    inputs={
        "input_data": Input(
            type="uri_folder",
            path="azureml:tumor_images_raw:1"
        ),
    },

    outputs={
        "output_features": Output(
            type="uri_file",
            path="azureml://datastores/workspaceblobstore/paths/silver/features.parquet"
        ),
    },

    compute=COMPUTE_TARGET,
)

# 3) Submit job
returned_job = ml_client.jobs.create_or_update(job)

print("==============================================")
print("✅ Silver feature extraction job submitted!")
print("Job name:", returned_job.name)
print("Status:", returned_job.status)
print("==============================================")
