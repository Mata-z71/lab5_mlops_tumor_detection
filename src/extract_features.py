# src/extract_features.py
#
# Phase 2 – Silver Layer: Feature Extraction

import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import filters
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor, hessian, prewitt
import mlflow


def compute_glcm_features(gray_img):
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    properties = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

    glcm = graycomatrix(
        gray_img,
        distances=distances,
        angles=angles,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in properties:
        values = graycoprops(glcm, prop)[0]
        for i, ang in enumerate(["0", "45", "90", "135"]):
            feats[f"glcm_{prop}_{ang}"] = float(values[i])
    return feats


def process_one_image(path: Path, label: int):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))

    features = {}

    # Original
    features.update(compute_glcm_features(img))

    # Entropy
    ent = filters.rank.entropy(img, np.ones((9, 9)))
    ent_norm = ent.astype("uint8")
    features.update({f"entropy_{k}": v for k, v in compute_glcm_features(ent_norm).items()})

    # Gaussian
    gauss = cv2.GaussianBlur(img, (5, 5), 0)
    features.update({f"gaussian_{k}": v for k, v in compute_glcm_features(gauss).items()})

    # Hessian
    hes = hessian(img)
    hes_norm = cv2.normalize(hes, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    features.update({f"hessian_{k}": v for k, v in compute_glcm_features(hes_norm).items()})

    # Prewitt
    pre = prewitt(img)
    pre_norm = cv2.normalize(pre, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    features.update({f"prewitt_{k}": v for k, v in compute_glcm_features(pre_norm).items()})

    # Gabor
    gb_real, _ = gabor(img, frequency=0.6)
    gb_norm = cv2.normalize(gb_real, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    features.update({f"gabor_{k}": v for k, v in compute_glcm_features(gb_norm).items()})

    return {
        "image_id": path.name,
        "label": label,
        **features,
    }


def main(input_dir, output_path):
    input_dir = Path(input_dir)

    t0 = time.time()
    rows = []

    for folder_name, label in [("no", 0), ("yes", 1)]:
        folder = input_dir / folder_name
        for file in os.listdir(folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = folder / file
            row = process_one_image(img_path, label)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    elapsed = time.time() - t0

    mlflow.log_metric("num_images", len(df))
    mlflow.log_metric("num_features", df.shape[1] - 2)
    mlflow.log_metric("extraction_time_seconds", elapsed)

    print(f"Done. Extracted {len(df)} images → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--output_features", type=str, required=True)
    args = parser.parse_args()

    main(args.input_data, args.output_features)
