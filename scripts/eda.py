"""Exploratory data analysis: class distribution, sample images, image sizes."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

sns.set_theme(style="whitegrid")


def plot_all_makes() -> None:
    counts = pd.read_csv(C.TABLES_DIR / "make_counts_all.csv")
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = counts["kept"].map({True: "tab:blue", False: "lightgray"})
    ax.barh(counts["make"], counts["images"], color=colors)
    ax.axvline(C.MIN_IMAGES_PER_BRAND, color="red", ls="--",
               label=f"threshold = {C.MIN_IMAGES_PER_BRAND}")
    ax.invert_yaxis()
    ax.set_xlabel("Number of images")
    ax.set_title("Stanford Cars: images per make (blue = used in this work)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "01_all_makes_distribution.png", dpi=150)
    plt.close(fig)


def plot_split_distribution(meta: pd.DataFrame) -> None:
    table = meta.pivot_table(index="make", columns="split", values="id", aggfunc="count")
    table = table[["train", "val", "test"]].sort_values("train", ascending=False)
    table.to_csv(C.TABLES_DIR / "make_counts_by_split.csv")
    ax = table.plot.bar(stacked=True, figsize=(12, 6), color=["tab:blue", "tab:orange", "tab:green"])
    ax.set_ylabel("Number of images")
    ax.set_xlabel("")
    ax.set_title("Images per make by split (stratified 70 / 15 / 15)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "02_split_distribution.png", dpi=150)
    plt.close()


def plot_samples(meta: pd.DataFrame, cols: int = 5) -> None:
    makes = sorted(meta["make"].unique())
    train = meta[meta["split"] == "train"]
    rows = (len(makes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6))
    for ax, make in zip(axes.flat, makes):
        r = train[train["make"] == make].sample(1, random_state=C.SEED).iloc[0]
        ax.imshow(Image.open(C.IMAGES_DIR / r["path"]))
        ax.set_title(make, fontsize=11)
        ax.axis("off")
    for ax in axes.flat[len(makes):]:
        ax.axis("off")
    fig.suptitle("Sample training image for each make", fontsize=14)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "03_sample_images.png", dpi=130)
    plt.close(fig)


def plot_image_sizes(meta: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(meta["width"], meta["height"], s=3, alpha=0.3)
    axes[0].set_xlabel("Width, px")
    axes[0].set_ylabel("Height, px")
    axes[0].set_title("Original image sizes")
    sns.histplot(meta["width"] / meta["height"], bins=50, ax=axes[1])
    axes[1].set_xlabel("Aspect ratio (width / height)")
    axes[1].set_title("Aspect ratio distribution")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "04_image_sizes.png", dpi=150)
    plt.close(fig)


def main() -> None:
    meta = pd.read_csv(C.METADATA_CSV)
    plot_all_makes()
    plot_split_distribution(meta)
    plot_samples(meta)
    plot_image_sizes(meta)
    summary = {
        "images": len(meta),
        "makes": meta["make"].nunique(),
        "models": meta["model"].nunique(),
        **meta["split"].value_counts().to_dict(),
        "imbalance_ratio_max_min": round(meta["make"].value_counts().max() / meta["make"].value_counts().min(), 2),
        "median_width": int(meta["width"].median()),
        "median_height": int(meta["height"].median()),
    }
    pd.Series(summary).to_csv(C.TABLES_DIR / "dataset_summary.csv", header=["value"])
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
