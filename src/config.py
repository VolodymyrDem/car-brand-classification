from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
IMAGES_DIR = DATA_PROCESSED / "images"
METADATA_CSV = DATA_PROCESSED / "metadata.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

HF_DATASET = "tanganke/stanford_cars"
SEED = 42

# Brands with fewer images are dropped (too little data to learn from).
MIN_IMAGES_PER_BRAND = 300
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Images are stored with this maximum side length to speed up loading.
STORED_MAX_SIDE = 384
IMAGE_SIZE = 224

# Multi-word manufacturer names in Stanford Cars class names.
MULTI_WORD_MAKES = ("AM General", "Aston Martin", "Land Rover")


def make_from_class_name(class_name: str) -> str:
    for make in MULTI_WORD_MAKES:
        if class_name.startswith(make + " "):
            return make
    return class_name.split(" ")[0]
