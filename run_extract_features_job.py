from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, Input
from azure.ai.ml import load_component

# 🔧 FILL THESE with your own values
SUBSCRIPTION_ID = "<YOUR_SUBSCRIPTION_ID>"
RESOURCE_GROUP = "<YOUR_RESOURCE_GROUP>"
WORKSPACE_NAME = "<YOUR_WORKSPACE_NAME>"

def main():
    # 1) Connect to your Azure ML workspace
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )

    # 2) Load the component definition from YAML
    extract_features = load_component("components/extract_features_component.yml")

    # 3) Create a job from the component
    job = extract_features(
        input_data=Input(
            type="uri_folder",
            # 👇 Use your actual data asset name + version
            path="azureml:tumor_image_raw:1",
        ),
        # Azure ML will create output_parquet in default datastore
        output_parquet=None,
    )

    # 4) Submit the job
    returned_job = ml_client.jobs.create_or_update(
        job,
        experiment_name="lab5_extract_features",
    )

    print(f"Job submitted. Name: {returned_job.name}")
    print(f"Status: {returned_job.status}")


if __name__ == "__main__":
    main()
