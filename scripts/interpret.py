"""Interpretability and error analysis for the deep models.

Produces:
  reports/figures/gradcam_examples.png        Grad-CAM (ResNet18, ViT) + attention rollout (ViT) on test images
  reports/figures/misclassified_<model>.png   most confident wrong predictions
  reports/tables/top_confusions_<model>.csv   most frequent (true -> predicted) error pairs

Requires models/<model>.pt and reports/tables/test_proba_<model>.npy produced by train_deep.py.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import build_transforms, get_device  # noqa: E402

TIMM_NAMES = {
    "resnet18": "resnet18.a1_in1k",
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
}
TITLES = {"resnet18": "ResNet18", "vit_small": "ViT-Small"}


def load_model(name: str, n_classes: int, device):
    model = timm.create_model(TIMM_NAMES[name], pretrained=False, num_classes=n_classes)
    model.load_state_dict(torch.load(C.MODELS_DIR / f"{name}.pt", map_location="cpu"))
    return model.eval().to(device)


def vit_reshape(model):
    def reshape(t):
        t = t[:, model.num_prefix_tokens:, :]
        side = int(t.shape[1] ** 0.5)
        return t.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)
    return reshape


def gradcam(model, name: str, x: torch.Tensor, targets: list[int]) -> np.ndarray:
    if name == "resnet18":
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    else:
        cam = GradCAM(model=model, target_layers=[model.blocks[-1].norm1], reshape_transform=vit_reshape(model))
    return cam(input_tensor=x, targets=[ClassifierOutputTarget(t) for t in targets])


@torch.no_grad()
def attention_rollout(model, x: torch.Tensor) -> np.ndarray:
    """Attention rollout (Abnar & Zuidema, 2020): propagate head-averaged attention through all layers."""
    maps, hooks = [], []
    for blk in model.blocks:
        blk.attn.fused_attn = False
        hooks.append(blk.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(o.mean(1))))
    model(x)
    for h in hooks:
        h.remove()
    n = maps[0].shape[-1]
    rollout = torch.eye(n, device=x.device).expand(x.shape[0], n, n)
    for a in maps:
        a = a + torch.eye(n, device=x.device)
        a = a / a.sum(-1, keepdim=True)
        rollout = a @ rollout
    cls = rollout[:, 0, model.num_prefix_tokens:]
    side = int(cls.shape[-1] ** 0.5)
    cls = cls.reshape(-1, 1, side, side)
    cls = torch.nn.functional.interpolate(cls, size=x.shape[-2:], mode="bilinear")[:, 0]
    cls = cls - cls.amin((1, 2), keepdim=True)
    return (cls / cls.amax((1, 2), keepdim=True)).cpu().numpy()


def overlay(img: np.ndarray, heat: np.ndarray, ax, title: str, color: str = "black") -> None:
    ax.imshow(img)
    ax.imshow(heat, cmap="jet", alpha=0.45)
    ax.set_title(title, fontsize=8, color=color)
    ax.axis("off")


def plot_heatmaps(meta_test, proba, class_names, models, device, n: int, size: int) -> None:
    idx = meta_test.sample(n, random_state=C.SEED).index.to_numpy()
    pos = [meta_test.index.get_loc(i) for i in idx]
    imgs = [Image.open(C.IMAGES_DIR / meta_test.loc[i, "path"]).convert("RGB").resize((size, size)) for i in idx]
    rows = [("Original", None)]
    for name in models:
        rows.append((f"{TITLES[name]} Grad-CAM", name))
    if "vit_small" in models:
        rows.append(("ViT attention rollout", "rollout"))

    fig, axes = plt.subplots(len(rows), n, figsize=(n * 2.2, len(rows) * 2.4))
    for j, img in enumerate(imgs):
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"true: {meta_test.loc[idx[j], 'make']}", fontsize=8)
        axes[0, j].axis("off")

    for r, (label, key) in enumerate(rows[1:], start=1):
        name = "vit_small" if key == "rollout" else key
        model, cfg = models[name]
        _, eval_tf = build_transforms(size, cfg["mean"], cfg["std"])
        x = torch.stack([eval_tf(im) for im in imgs]).to(device)
        preds = proba[name][pos].argmax(1)
        heat = attention_rollout(model, x) if key == "rollout" else gradcam(model, name, x, preds.tolist())
        for j, img in enumerate(imgs):
            p = preds[j]
            ok = class_names[p] == meta_test.loc[idx[j], "make"]
            overlay(np.asarray(img), heat[j], axes[r, j],
                    f"{class_names[p]} ({proba[name][pos[j], p]:.2f})", "green" if ok else "red")
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes, rotation=90,
                        ha="right", va="center", fontsize=9)
    fig.suptitle("Where do the networks look? (green = correct, red = wrong)", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "gradcam_examples.png", dpi=130)
    plt.close(fig)


def error_analysis(name: str, meta_test, proba, class_names, n: int = 12) -> None:
    y_true = meta_test["make"].map({c: i for i, c in enumerate(class_names)}).to_numpy()
    y_pred = proba.argmax(1)
    wrong = np.flatnonzero(y_pred != y_true)

    pairs = pd.DataFrame({"true": np.array(class_names)[y_true[wrong]],
                          "predicted": np.array(class_names)[y_pred[wrong]]})
    pairs.value_counts().rename("count").reset_index().to_csv(
        C.TABLES_DIR / f"top_confusions_{name}.csv", index=False)

    worst = wrong[np.argsort(-proba[wrong, y_pred[wrong]])][:n]
    cols = 4
    rows = int(np.ceil(len(worst) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.8))
    for ax, k in zip(np.ravel(axes), worst):
        r = meta_test.iloc[k]
        ax.imshow(Image.open(C.IMAGES_DIR / r["path"]))
        ax.set_title(f"true: {r['make']}\npred: {class_names[y_pred[k]]} ({proba[k, y_pred[k]]:.2f})", fontsize=9)
        ax.axis("off")
    for ax in np.ravel(axes)[len(worst):]:
        ax.axis("off")
    fig.suptitle(f"{TITLES[name]}: most confident mistakes on the test set", fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / f"misclassified_{name}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="number of test images for heatmaps")
    args = parser.parse_args()

    meta = pd.read_csv(C.METADATA_CSV, dtype={"id": str})
    class_names = sorted(meta["make"].unique())
    meta_test = meta[meta["split"] == "test"]
    device = get_device()

    available = [m for m in TIMM_NAMES
                 if (C.MODELS_DIR / f"{m}.pt").exists() and (C.TABLES_DIR / f"test_proba_{m}.npy").exists()]
    if not available:
        sys.exit("No trained deep models found (models/*.pt + reports/tables/test_proba_*.npy).")

    proba, models = {}, {}
    for name in available:
        proba[name] = np.load(C.TABLES_DIR / f"test_proba_{name}.npy")
        assert len(proba[name]) == len(meta_test), f"{name}: test_proba does not match test split"
        error_analysis(name, meta_test, proba[name], class_names)
        model = load_model(name, len(class_names), device)
        models[name] = (model, timm.data.resolve_model_data_config(model))
        print(f"{name}: error analysis done")

    plot_heatmaps(meta_test, proba, class_names, models, device, args.n, C.IMAGE_SIZE)
    print("Heatmaps saved to", C.FIGURES_DIR / "gradcam_examples.png")


if __name__ == "__main__":
    main()
