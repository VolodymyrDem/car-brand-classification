"""Hand-crafted image features for classical ML models: HOG (shape) + HSV color histogram."""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog

from src import config as C

FEATURE_IMAGE_SIZE = (192, 128)  # (width, height), close to the median aspect ratio of 1.5
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
COLOR_BINS = 16


def extract_features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(FEATURE_IMAGE_SIZE, Image.BILINEAR)
    gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    hog_vec = hog(gray, feature_vector=True, **HOG_PARAMS)

    hsv = np.asarray(img.convert("HSV"))
    color_vec = np.concatenate([
        np.histogram(hsv[..., ch], bins=COLOR_BINS, range=(0, 256))[0] for ch in range(3)
    ]).astype(np.float32)
    color_vec /= color_vec.sum() / 3

    return np.concatenate([hog_vec, color_vec]).astype(np.float32)


def load_features(meta: pd.DataFrame, cache: Path = C.DATA_PROCESSED / "features_hog_color.npz") -> np.ndarray:
    """Return feature matrix aligned with `meta` rows, computing and caching on first call."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        index = {i: k for k, i in enumerate(data["ids"])}
        if all(i in index for i in meta["id"]):
            return data["X"][[index[i] for i in meta["id"]]]

    all_meta = pd.read_csv(C.METADATA_CSV, dtype={"id": str})
    X = np.stack(Parallel(n_jobs=-1, batch_size=64)(
        delayed(extract_features)(C.IMAGES_DIR / p) for p in all_meta["path"]
    ))
    np.savez_compressed(cache, X=X, ids=all_meta["id"].to_numpy())
    index = {i: k for k, i in enumerate(all_meta["id"])}
    return X[[index[i] for i in meta["id"]]]
