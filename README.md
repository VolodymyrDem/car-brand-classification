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
| 1 — Classical ML | Logistic Regression, Random Forest on HOG + HSV color histogram features | scikit-learn, hyperparameters tuned with stratified 5-fold CV |
| 2 — CNN | ResNet18 pretrained on ImageNet, fine-tuned | Deep residual CNN, transfer learning |
| 3 — Transformer | ViT-Small/16 pretrained on ImageNet-21k, fine-tuned | Self-attention over image patches, transfer learning |

Deep models are trained in Google Colab (NVIDIA T4); the code auto-selects CUDA / Apple MPS / CPU.

## Evaluation
Accuracy, top-3 accuracy, macro F1, confusion matrix, learning curves, training time, model size;
Grad-CAM / attention maps for interpretability.

## Project structure
```
data/            raw and processed data (not tracked)
docs/            theory notes and report
models/          trained weights (not tracked)
notebooks/       Colab training notebook
reports/         figures and tables produced by scripts
scripts/         prepare_data, eda, train_classical, train_deep, compare
src/             config, features, datasets, evaluation helpers
```

## Run in Google Colab
Open [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) in Colab
([direct link](https://colab.research.google.com/github/VolodymyrDem/car-brand-classification/blob/main/notebooks/colab_train.ipynb)),
select *Runtime → Change runtime type → T4 GPU* and run all cells.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/prepare_data.py                     # download, map to makes, split -> data/processed
python scripts/eda.py                              # dataset figures
python scripts/train_classical.py                  # level 1: logistic regression + random forest (5-fold CV)
python scripts/train_deep.py --model resnet18      # level 2
python scripts/train_deep.py --model vit_small     # level 3
python scripts/compare.py                          # comparison table and charts
```

Theory and full work plan (in Ukrainian): [docs/theory.md](docs/theory.md).
