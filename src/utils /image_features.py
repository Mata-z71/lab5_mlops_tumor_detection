# utils/image_features.py

from pathlib import Path

import numpy as np
from scipy import ndimage as nd

from skimage import io
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import sobel, gabor, hessian, prewitt
from skimage.feature import graycomatrix, graycoprops


def _normalize_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize any image to 0–255 uint8 (NumPy 2.0 compatible)."""
    img = image.astype(np.float32)
    # np.ptp(img) = img.max() - img.min()
    denom = np.ptp(img)
    if denom == 0:
        # Avoid division by zero for constant images
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img.min()) / (denom + 1e-8)
    return (img * 255).astype(np.uint8)


def _glcm_features(image: np.ndarray, prefix: str) -> dict:
    """
    Compute GLCM features for a 2D grayscale image.

    - distances: [1]
    - angles: 0, π/4, π/2, 3π/4  → 4 angles
    - properties: contrast, dissimilarity, homogeneity, energy, correlation, ASM

    => 6 properties × 4 angles = 24 features per image per filter.
    """
    img_u8 = _normalize_uint8(image)

    glcm = graycomatrix(
        img_u8,
        [1],
        [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True,
    )

    props = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]

    feats = {}
    for prop in props:
        values = graycoprops(glcm, prop).flatten()  # 4 angles
        for i, v in enumerate(values, start=1):
            feats[f"{prefix}_{prop}_{i}"] = float(v)
    return feats


def _basic_stats(image: np.ndarray, prefix: str) -> dict:
    """Basic statistics (mean, std, min, max) to add extra features."""
    arr = image.astype(float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


def compute_image_features(image_path: str) -> dict:
    """
    Compute a rich set of texture features for one image.

    - Filters: original, entropy, gaussian, sobel, gabor, hessian, prewitt
    - For each: GLCM (24 features) + basic stats (4 features)

    => 7 filters × (24 + 4) = 196 features per image.
    """
    path = Path(image_path)

    # Use skimage.io to read as grayscale (robust on Windows paths)
    img = io.imread(str(path), as_gray=True)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = img.astype(np.float32)

    # Filters
    entropy_img = entropy(_normalize_uint8(img), disk(2))
    gaussian_img = nd.gaussian_filter(img, sigma=1)
    sobel_img = sobel(img)
    gabor_img = gabor(img, frequency=0.9)[1]
    hessian_img = hessian(img, sigmas=range(1, 3, 1))
    if isinstance(hessian_img, (list, tuple)):
        hessian_img = np.mean(np.stack(hessian_img), axis=0)
    prewitt_img = prewitt(img)

    filtered = {
        "original": img,
        "entropy": entropy_img,
        "gaussian": gaussian_img,
        "sobel": sobel_img,
        "gabor": gabor_img,
        "hessian": hessian_img,
        "prewitt": prewitt_img,
    }

    feats = {}
    # GLCM + basic stats for each filtered image
    for name, f_img in filtered.items():
        feats.update(_glcm_features(f_img, name))
        feats.update(_basic_stats(f_img, name))

    return feats
