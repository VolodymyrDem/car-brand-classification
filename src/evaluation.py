"""Metrics and plots shared by all models."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             top_k_accuracy_score)

from src import config as C


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, class_names: list[str]) -> dict:
    y_pred = proba.argmax(1)
    labels = np.arange(len(class_names))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "top3_accuracy": top_k_accuracy_score(y_true, proba, k=3, labels=labels),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }


def save_report(name: str, title: str, y_true: np.ndarray, proba: np.ndarray, class_names: list[str],
                extra: dict | None = None) -> dict:
    """Save metrics JSON, per-class report CSV and confusion matrix figure for one model."""
    metrics = {"model": name, **compute_metrics(y_true, proba, class_names), **(extra or {})}
    (C.TABLES_DIR / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2))

    y_pred = proba.argmax(1)
    report = classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(C.TABLES_DIR / f"classification_report_{name}.csv")

    plot_confusion_matrix(y_true, y_pred, class_names, title,
                          C.FIGURES_DIR / f"confusion_matrix_{name}.png")
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, title, out_path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, square=True,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 7},
                cbar_kws={"label": "Recall (row-normalized)"}, ax=ax)
    ax.set_xlabel("Predicted make")
    ax.set_ylabel("True make")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
