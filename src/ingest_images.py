# Phase 1 – Bronze Layer: Data Ingestion
# Uploads MRI images to ADLS Gen2 under raw/tumor_images/

import argparse
import os
from pathlib import Path
from azure.storage.filedatalake import DataLakeServiceClient


def get_datalake_client(account_name, account_key):
    return DataLakeServiceClient(
        account_url=f"https://{account_name}.dfs.core.windows.net",
        credential=account_key
    )


def upload_directory(local_dir, file_system_client, target_prefix):
    local_dir = Path(local_dir)

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.startswith("."):
                continue

            local_path = Path(root) / file
            rel_path = local_path.relative_to(local_dir)
            adls_path = f"{target_prefix}/{rel_path.as_posix()}"

            file_client = file_system_client.get_file_client(adls_path)

            print(f"[UPLOAD] {local_path} → {adls_path}")
            with open(local_path, "rb") as f:
                file_client.upload_data(f, overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_data_dir", required=True,
                        help="Local path containing brain_tumor_dataset/yes and no")
    parser.add_argument("--account_name", required=True)
    parser.add_argument("--account_key", required=True)
    parser.add_argument("--file_system", required=True,
                        help="Your ADLS container (e.g., lakehouse)")
    parser.add_argument("--target_prefix", default="raw/tumor_images",
                        help="Where to upload inside your ADLS container")
    args = parser.parse_args()

    service_client = get_datalake_client(args.account_name, args.account_key)
    fs_client = service_client.get_file_system_client(args.file_system)

    upload_directory(args.local_data_dir, fs_client, args.target_prefix)


if __name__ == "__main__":
    main()
