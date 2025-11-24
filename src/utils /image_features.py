from pathlib import Path
import numpy as np
from skimage import io, color, filters, feature
from skimage.filters import sobel, prewitt, gaussian
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def _stats(prefix: str, arr: np.ndarray, out: dict):
    """Add basic statistics for an array to the out dict."""
    arr = arr.astype(float)
    out[f"{prefix}_mean"] = float(arr.mean())
    out[f"{prefix}_std"] = float(arr.std())
    out[f"{prefix}_min"] = float(arr.min())
    out[f"{prefix}_max"] = float(arr.max())


def _glcm_features(gray_uint8: np.ndarray) -> dict:
    """
    Compute GLCM features for 4 angles.
    gray_uint8 must be 2D uint8 image (0–255).
    """
    # distances = [1]; angles = 0, 45, 90, 135 degrees
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = feature.graycomatrix(
        gray_uint8,
        distances=[1],
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity",
                 "ASM", "energy", "correlation"]:
        vals = feature.graycoprops(glcm, prop)  # shape: [1, len(angles)]
        feats[f"glcm_{prop}_mean"] = float(vals.mean())
        feats[f"glcm_{prop}_std"] = float(vals.std())
    return feats


def compute_image_features(path: str) -> dict:
    """
    Load an image from path and compute:
      - entropy, gaussian, sobel, prewitt, gabor, hessian filters
      - basic stats for each
      - GLCM features with 4 angles
    Returns a dict of feature_name -> value.
    """
    path = Path(path)
    img = io.imread(path)

    # Convert to grayscale [0, 1]
    if img.ndim == 3:
        gray = color.rgb2gray(img)
    else:
        gray = img.astype(float)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    feats = {}

    # --- Filters ---
    ent = entropy((gray * 255).astype(np.uint8), disk(3))
    gauss = gaussian(gray, sigma=1)
    sob = sobel(gray)
    pre = prewitt(gray)

    # Gabor filter (real & imag)
    gabor_real, gabor_imag = filters.gabor(gray, frequency=0.2)

    # Hessian response (largest eigenvalue)
    H_elems = hessian_matrix(gray, sigma=1, order='rc')
    eigvals = hessian_matrix_eigvals(H_elems)
    hess = eigvals[0]  # one response map

    # Stats for each
    _stats("entropy", ent, feats)
    _stats("gaussian", gauss, feats)
    _stats("sobel", sob, feats)
    _stats("prewitt", pre, feats)
    _stats("gabor_real", gabor_real, feats)
    _stats("gabor_imag", gabor_imag, feats)
    _stats("hessian", hess, feats)

    # --- GLCM on 8-bit gray image ---
    gray_uint8 = (gray * 255).astype(np.uint8)
    feats.update(_glcm_features(gray_uint8))

    return feats

