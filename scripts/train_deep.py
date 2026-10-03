"""Levels 2 and 3 — fine-tuning pretrained ResNet18 (CNN) and ViT-Small (transformer).

Training scheme (transfer learning):
  1. `--freeze-epochs`: backbone frozen, only the new classification head is trained;
  2. remaining epochs: the whole network is fine-tuned with a lower learning rate
     (linear warmup + cosine decay).
The checkpoint with the best validation macro F1 is kept (early stopping) and evaluated once on test.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.data import build_loaders, get_device  # noqa: E402
from src.evaluation import save_report  # noqa: E402

PRESETS = {
    "resnet18": {
        "timm_name": "resnet18.a1_in1k",
        "title": "ResNet18 (fine-tuned)",
        "level": 2,
        "lr": 1e-3,
        "head_lr": 3e-3,
        "weight_decay": 0.05,
        "epochs": 20,
    },
    "vit_small": {
        "timm_name": "vit_small_patch16_224.augreg_in21k_ft_in1k",
        "title": "ViT-Small/16 (fine-tuned)",
        "level": 3,
        "lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 0.05,
        "epochs": 15,
    },
}


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    head = set(id(p) for p in model.get_classifier().parameters())
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


def run_epoch(model, loader, device, criterion, optimizer=None, scheduler=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss, probs, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * len(y)
            probs.append(logits.float().softmax(1).detach().cpu())
            targets.append(y.cpu())
    probs, targets = torch.cat(probs).numpy(), torch.cat(targets).numpy()
    preds = probs.argmax(1)
    return {
        "loss": total_loss / len(targets),
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }, probs, targets


def plot_history(history: pd.DataFrame, title: str, freeze_epochs: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, label in zip(axes, ["loss", "acc", "macro_f1"], ["Loss", "Accuracy", "Macro F1"]):
        ax.plot(history["epoch"], history[f"train_{metric}"], "o-", label="train")
        ax.plot(history["epoch"], history[f"val_{metric}"], "o-", label="validation")
        if freeze_epochs:
            ax.axvline(freeze_epochs + 0.5, color="gray", ls="--", label="unfreeze backbone")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"Learning curves — {title}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(PRESETS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="use at most N images per split (smoke test)")
    args = parser.parse_args()

    preset = PRESETS[args.model]
    epochs = args.epochs or preset["epochs"]
    lr = args.lr or preset["lr"]
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)

    device = get_device()
    amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {amp}")

    meta = pd.read_csv(C.METADATA_CSV, dtype={"id": str})
    class_names = sorted(meta["make"].unique())

    model = timm.create_model(preset["timm_name"], pretrained=True, num_classes=len(class_names))
    data_cfg = timm.data.resolve_model_data_config(model)
    size = data_cfg["input_size"][-1]
    loaders = build_loaders(meta, class_names, size, data_cfg["mean"], data_cfg["std"],
                            args.batch_size, args.workers, args.limit)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{preset['title']}: {n_params / 1e6:.1f}M params, input {size}px, "
          f"train batches/epoch: {len(loaders['train'])}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    ckpt_path = C.MODELS_DIR / f"{args.model}.pt"
    history, best_f1, best_epoch, bad_epochs = [], -1.0, 0, 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        if epoch == 1 and args.freeze_epochs:
            set_backbone_trainable(model, False)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                          lr=preset["head_lr"], weight_decay=preset["weight_decay"])
            scheduler = None
            phase = "head"
        if epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=preset["weight_decay"])
            steps = (epochs - args.freeze_epochs) * len(loaders["train"])
            warmup = int(args.warmup_epochs * len(loaders["train"]))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
                (s + 1) / max(warmup, 1), 0.5 * (1 + math.cos(math.pi * max(s - warmup, 0) / max(steps - warmup, 1)))))
            phase = "finetune"
            bad_epochs = 0

        t0 = time.time()
        tr, _, _ = run_epoch(model, loaders["train"], device, criterion, optimizer, scheduler, scaler, amp)
        va, _, _ = run_epoch(model, loaders["val"], device, criterion, amp=amp)
        row = {"epoch": epoch, "phase": phase, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()},
               "epoch_time_s": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:02d}/{epochs}] {phase:8s} train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
              f"val loss {va['loss']:.3f} acc {va['acc']:.3f} F1 {va['macro_f1']:.3f} | {row['epoch_time_s']}s",
              flush=True)

        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, bad_epochs = va["macro_f1"], epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        elif phase == "finetune":
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    train_time = time.time() - t_start
    history = pd.DataFrame(history)
    history.to_csv(C.TABLES_DIR / f"history_{args.model}.csv", index=False)
    plot_history(history, preset["title"], args.freeze_epochs, C.FIGURES_DIR / f"learning_curves_{args.model}.png")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    t0 = time.time()
    _, proba, y_test = run_epoch(model, loaders["test"], device, criterion, amp=amp)
    infer_ms = (time.time() - t0) / len(y_test) * 1000
    np.save(C.TABLES_DIR / f"test_proba_{args.model}.npy", proba)

    metrics = save_report(args.model, preset["title"], y_test, proba, class_names, extra={
        "level": preset["level"],
        "timm_name": preset["timm_name"],
        "params_millions": round(n_params / 1e6, 2),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_time_s": round(train_time, 1),
        "inference_ms_per_image": round(infer_ms, 3),
        "model_size_mb": round(ckpt_path.stat().st_size / 2**20, 2),
        "device": str(device),
        "hyperparams": {"epochs": epochs, "freeze_epochs": args.freeze_epochs, "batch_size": args.batch_size,
                        "lr": lr, "head_lr": preset["head_lr"], "weight_decay": preset["weight_decay"],
                        "label_smoothing": args.label_smoothing, "input_size": size},
    })
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
