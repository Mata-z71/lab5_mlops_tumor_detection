# Phase 2 – Silver Layer: Image Feature Extraction
# This script will be used as the entry point of the
# extract_features_component in Azure ML.

import argparse
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import mlflow

from utils.image_features import compute_image_features


def _process_one(args):
    """
    Helper function for multiprocessing.
    args: (path, label_str)
    Returns: dict of features + image_id + label
    """
    path, label_str = args
    try:
        feats = compute_image_features(str(path))
        feats["image_id"] = path.name
        # label: yes -> 1, no -> 0
        feats["label"] = 1 if label_str == "yes" else 0
        return feats
    except Exception as e:
        print(f"[ERROR] Failed on {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_data",
        type=str,
        required=True,
        help="Root folder that contains 'yes' and 'no' subfolders."
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Output path for the Silver Parquet file."
    )
    args = parser.parse_args()

    root = Path(args.input_data)

    # Collect all (image_path, label) pairs
    items = []
    for label in ["yes", "no"]:
        folder = root / label
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
            for p in folder.glob(ext):
                items.append((p, label))

    print(f"Found {len(items)} images under {root}")

    # --- Multiprocessing, as required by the lab ---
    start_time = time.time()
    num_workers = max(1, cpu_count() - 1)
    print(f"Using {num_workers} workers")

    records = []
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, items), start=1):
            if result is not None:
                records.append(result)
            if i % 20 == 0:
                print(f"Processed {i}/{len(items)} images...")

    # Build DataFrame
    df = pd.DataFrame(records)

    # Make sure we have at least the required columns
    if "image_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Missing image_id or label in features DataFrame!")

    # Save as Parquet (Silver output)
    out_path = Path(args.output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    # --- Logging metrics required by the lab ---
    elapsed = time.time() - start_time
    num_images = len(df)
    num_features = df.shape[1] - 2  # minus image_id + label
    compute_sku = os.environ.get("AZUREML_COMPUTE", "local")

    print("\n=== Silver Feature Extraction Summary ===")
    print(f"Images       : {num_images}")
    print(f"Num features : {num_features}")
    print(f"Time (sec)   : {elapsed:.2f}")
    print(f"Compute SKU  : {compute_sku}")

    try:
        mlflow.log_metric("num_images", num_images)
        mlflow.log_metric("num_features", num_features)
        mlflow.log_metric("extraction_time_seconds", elapsed)
        mlflow.log_param("compute_sku", compute_sku)
    except Exception as e:
        print(f"[INFO] MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
