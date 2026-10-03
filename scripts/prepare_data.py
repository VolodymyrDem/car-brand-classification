"""Download Stanford Cars, map model classes to car makes, filter and split.

Output:
    data/processed/images/<split>/<make>/<id>.jpg
    data/processed/metadata.csv  (id, path, make, model, split, width, height)
"""
import io
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402


def class_names() -> list[str]:
    from datasets import load_dataset_builder

    return load_dataset_builder(C.HF_DATASET).info.features["label"].names


def download_parquets() -> list[Path]:
    files = [
        f for f in list_repo_files(C.HF_DATASET, repo_type="dataset")
        if f.startswith(("data/train-", "data/test-")) and f.endswith(".parquet")
    ]
    return [
        Path(hf_hub_download(C.HF_DATASET, f, repo_type="dataset", local_dir=C.DATA_RAW))
        for f in sorted(files)
    ]


def main() -> None:
    names = class_names()
    frames = [pd.read_parquet(p) for p in tqdm(download_parquets(), desc="Reading parquet")]
    df = pd.concat(frames, ignore_index=True)
    df["model"] = df["label"].map(lambda i: names[i])
    df["make"] = df["model"].map(C.make_from_class_name)

    counts = df["make"].value_counts()
    keep = counts[counts >= C.MIN_IMAGES_PER_BRAND].index
    print(f"Makes total: {len(counts)}, kept (>= {C.MIN_IMAGES_PER_BRAND} images): {len(keep)}")
    pd.DataFrame({"make": counts.index, "images": counts.values,
                  "kept": counts.index.isin(keep)}).to_csv(C.TABLES_DIR / "make_counts_all.csv", index=False)
    df = df[df["make"].isin(keep)].reset_index(drop=True)
    df["id"] = [f"{i:05d}" for i in range(len(df))]

    f = C.SPLIT_FRACTIONS
    train_idx, rest_idx = train_test_split(
        df.index, train_size=f["train"], stratify=df["make"], random_state=C.SEED)
    val_idx, test_idx = train_test_split(
        rest_idx, train_size=f["val"] / (f["val"] + f["test"]),
        stratify=df.loc[rest_idx, "make"], random_state=C.SEED)
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="Saving images"):
        img = Image.open(io.BytesIO(r.image["bytes"])).convert("RGB")
        w, h = img.size
        img.thumbnail((C.STORED_MAX_SIDE, C.STORED_MAX_SIDE))
        rel = Path(r.split) / r.make.replace(" ", "_") / f"{r.id}.jpg"
        out = C.IMAGES_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out, quality=95)
        rows.append({"id": r.id, "path": str(rel), "make": r.make, "model": r.model,
                     "split": r.split, "width": w, "height": h})

    meta = pd.DataFrame(rows)
    meta.to_csv(C.METADATA_CSV, index=False)
    print(meta.groupby("split").size().to_string())
    print(f"Classes: {meta['make'].nunique()}  -> {C.METADATA_CSV}")


if __name__ == "__main__":
    main()
