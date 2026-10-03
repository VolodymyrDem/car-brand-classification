# Car Brand Classification

Coursework for **AI: Principles and Methods** — recognizing a car's make (brand) from a photo
and comparing three models of increasing complexity.

## Task
Multiclass image classification: input — car photo, output — brand (Audi, BMW, Toyota, ...).

## Dataset
[Stanford Cars](https://huggingface.co/datasets/tanganke/stanford_cars) — 16,185 images of 196 car models (49 makes).
Models are aggregated into **makes**; makes with at least 300 images are kept:
**20 makes, 12,681 images**, stratified split 70 / 15 / 15 (train 8,876 / val 1,902 / test 1,903).

## Models
| Level | Model | Approach |
|---|---|---|
| 1 — Simple | Custom CNN trained from scratch (3–4 conv blocks) | Baseline convolutional network |
| 2 — Medium | ResNet18 pretrained on ImageNet, fine-tuned | Deep residual CNN, transfer learning |
| 3 — Complex | Vision Transformer (ViT-Small / DeiT), fine-tuned | Self-attention over image patches, transfer learning |

All models are trained locally with PyTorch (Apple MPS / CUDA / CPU auto-detection).

## Evaluation
Accuracy, top-3 accuracy, macro F1, confusion matrix, learning curves, training time, model size;
Grad-CAM / attention maps for interpretability.

## Project structure
```
data/            raw and processed data (not tracked)
notebooks/       EDA and experiments
src/             data loading, models, training, evaluation
scripts/         CLI entry points
models/          trained weights (not tracked)
reports/         figures and tables for the report
docs/            report
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py   # download, map to makes, split -> data/processed
python scripts/eda.py            # dataset figures -> reports/figures, reports/tables
```

Theory and full work plan (in Ukrainian): [docs/theory.md](docs/theory.md).
