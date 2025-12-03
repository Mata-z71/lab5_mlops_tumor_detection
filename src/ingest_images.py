# src/ingest_images.py
#
# Phase 1 - Bronze Layer: Data Ingestion
# - Read local MRI dataset (yes / no)
# - Upload to ADLS Gen2: raw/tumor_images/yes and raw/tumor_images/no
# - Fully programmatic (no manual portal upload)
# - Idempotent (safe to run multiple times)

import os
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# ========= CONFIG: EDIT THESE THREE =========
ACCOUNT_NAME = "tumorlab560300294"      # e.g. goodreadsreviews60300294
FILE_SYSTEM = "lakehouse"               # e.g. lakehouse
LOCAL_DATA_ROOT = r"..\lab5\brain_tumor_dataset"  # ...\assignment\data\brain_tumor_dataset
ACCOUNT_KEY = "<Account_key>"
# ===========================================

REMOTE_ROOT = "raw/tumor_images"  # as in the PDF: raw/tumor_images/yes, raw/tumor_images/no

def get_service_client():
    """
    Create DataLakeServiceClient using ACCOUNT_KEY.
    This is simplest for local dev / lab.
    """
    return DataLakeServiceClient(
        account_url=f"https://{ACCOUNT_NAME}.dfs.core.windows.net",
        credential=ACCOUNT_KEY
    )


def ensure_directories(fs_client):
    """
    Ensure raw/tumor_images/yes and /no directories exist in the container.
    Safe to call multiple times (idempotent).
    """
    for subdir in ["yes", "no"]:
        dir_path = f"{REMOTE_ROOT}/{subdir}"
        dir_client = fs_client.get_directory_client(dir_path)
        try:
            dir_client.get_directory_properties()
            print(f"[OK] Directory already exists: {dir_path}")
        except:
            print(f"[CREATE] Directory: {dir_path}")
            dir_client.create_directory()

def upload_folder(fs_client, local_folder, label):
    """
    Upload all image files from local_folder into
    raw/tumor_images/<label>/ on ADLS.
    Idempotent: if file already exists with non-zero size, skip.
    """
    dir_path = f"{REMOTE_ROOT}/{label}"
    dir_client = fs_client.get_directory_client(dir_path)

    for root, dirs, files in os.walk(local_folder):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            local_path = os.path.join(root, fname)
            remote_file_name = fname  # just use the file name
            file_client = dir_client.get_file_client(remote_file_name)

            # Check if file already exists and has data → skip (idempotent)
            try:
                props = file_client.get_file_properties()
                if props.size > 0:
                    print(f"[SKIP] {label}/{remote_file_name} already exists (size={props.size})")
                    continue
            except:
                # file does not exist yet → we will upload
                pass

            print(f"[UPLOAD] {local_path} -> {dir_path}/{remote_file_name}")

            with open(local_path, "rb") as f:
                data = f.read()

            # Create or overwrite file
            file_client.upload_data(data, overwrite=True)

def main():
    if not os.path.isdir(LOCAL_DATA_ROOT):
        raise ValueError(f"LOCAL_DATA_ROOT does not exist: {LOCAL_DATA_ROOT}")

    yes_local = os.path.join(LOCAL_DATA_ROOT, "yes")
    no_local = os.path.join(LOCAL_DATA_ROOT, "no")

    if not os.path.isdir(yes_local) or not os.path.isdir(no_local):
        raise ValueError(
            "Expected 'yes' and 'no' folders inside LOCAL_DATA_ROOT: "
            f"{LOCAL_DATA_ROOT}"
        )

    service_client = get_service_client()
    fs_client = service_client.get_file_system_client(FILE_SYSTEM)

    # Make sure directories exist
    ensure_directories(fs_client)

    # Upload
    upload_folder(fs_client, yes_local, "yes")
    upload_folder(fs_client, no_local, "no")

    print("✅ Ingestion finished.")

if __name__ == "__main__":
    main()
