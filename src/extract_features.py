import argparse
import os
import time
from pathlib import Path

import pandas as pd
import mlflow

from utils.image_features import compute_image_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root folder containing 'yes' and 'no' subfolders of images.",
    )
    parser.add_argument(
        "--output_parquet",
        type=str,
        required=True,
        help="Where to write the features parquet file.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_dir)

    items = []
    for label in ["yes", "no"]:
        folder = input_root / label
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.bmp"):
            for img_path in folder.glob(ext):
                items.append((img_path, label))

    print(f"Found {len(items)} images under {input_root}")

    start = time.time()
    records = []

    for i, (path, label) in enumerate(items, start=1):
        try:
            feats = compute_image_features(str(path))
            feats["image_id"] = path.name
            feats["label"] = 1 if label == "yes" else 0
            records.append(feats)
        except Exception as e:
            print(f"[ERROR] Failed on {path}: {e}")

        if i % 20 == 0:
            print(f"Processed {i}/{len(items)} images...")

    df = pd.DataFrame(records)
    os.makedirs(Path(args.output_parquet).parent, exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)

    elapsed = time.time() - start
    num_images = len(df)
    num_features = df.shape[1] - 2  # minus image_id + label

    print(f"\n=== Feature extraction summary ===")
    print(f"Images       : {num_images}")
    print(f"Num features : {num_features}")
    print(f"Time (sec)   : {elapsed:.2f}")

    # Try logging metrics if running inside Azure ML
    try:
        mlflow.log_metric("num_images", num_images)
        mlflow.log_metric("num_features", num_features)
        mlflow.log_metric("extraction_time_seconds", elapsed)
    except Exception as e:
        print(f"[INFO] mlflow logging skipped: {e}")


if __name__ == "__main__":
    main()
