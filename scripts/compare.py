"""Collect metrics of all trained models into a comparison table and charts."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402

ORDER = ["logreg", "random_forest", "resnet18", "vit_small"]
NAMES = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "resnet18": "ResNet18",
    "vit_small": "ViT-Small",
}


def main() -> None:
    rows = [json.loads((C.TABLES_DIR / f"metrics_{m}.json").read_text())
            for m in ORDER if (C.TABLES_DIR / f"metrics_{m}.json").exists()]
    if not rows:
        sys.exit("No metrics_*.json found — train models first.")
    df = pd.DataFrame(rows)
    df["name"] = df["model"].map(NAMES)
    cols = ["level", "name", "accuracy", "top3_accuracy", "macro_f1", "weighted_f1",
            "train_time_s", "inference_ms_per_image", "model_size_mb"]
    table = df[[c for c in cols if c in df]].copy()
    table.to_csv(C.TABLES_DIR / "comparison.csv", index=False)

    md = table.copy()
    for c in ["accuracy", "top3_accuracy", "macro_f1", "weighted_f1"]:
        md[c] = (md[c] * 100).map("{:.1f}%".format)
    (C.TABLES_DIR / "comparison.md").write_text(md.to_markdown(index=False))
    print(md.to_string(index=False))

    long = df.melt(id_vars="name", value_vars=["accuracy", "top3_accuracy", "macro_f1"],
                   var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"accuracy": "Accuracy", "top3_accuracy": "Top-3 accuracy",
                                         "macro_f1": "Macro F1"})
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(long, x="name", y="value", hue="metric", ax=ax)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score on test set")
    ax.set_title("Model comparison (test set)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "comparison_metrics.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["train_time_s"] / 60, df["macro_f1"], s=120)
    for _, r in df.iterrows():
        ax.annotate(r["name"], (r["train_time_s"] / 60, r["macro_f1"]), xytext=(6, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Training time, minutes (log scale)")
    ax.set_ylabel("Macro F1 (test)")
    ax.set_title("Quality vs training cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "comparison_quality_vs_time.png", dpi=150)
    plt.close(fig)

    plot_per_class_f1(df["model"].tolist())


def plot_per_class_f1(models: list[str]) -> None:
    per_class = {}
    for m in models:
        path = C.TABLES_DIR / f"classification_report_{m}.csv"
        if path.exists():
            rep = pd.read_csv(path, index_col=0)
            per_class[NAMES[m]] = rep.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")["f1-score"]
    if not per_class:
        return
    table = pd.DataFrame(per_class)
    table = table.sort_values(table.columns[-1])
    table.round(4).to_csv(C.TABLES_DIR / "per_class_f1.csv")
    ax = table.plot.barh(figsize=(10, 9), width=0.8)
    ax.set_xlabel("F1-score (test)")
    ax.set_xlim(0, 1)
    ax.set_title("Per-make F1-score by model")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(C.FIGURES_DIR / "comparison_per_class_f1.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
