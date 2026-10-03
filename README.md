# Car Brand Classification

Coursework for **AI: Principles and Methods** — recognizing a car's make (brand) from a photo
and comparing three models of increasing complexity.

## Task
Multiclass image classification: input — car photo, output — brand (Audi, BMW, Toyota, ...).

## Dataset
[Stanford Cars](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) — 16,185 images, 196 model classes,
aggregated into car **makes** (brands). Images are cropped by bounding boxes and resized.

## Models
| Level | Model | Approach |
|---|---|---|
| 1 — Simple | Logistic (softmax) regression on HOG + color histogram features | Linear classical ML |
| 2 — Medium | Random Forest / SVM (RBF) on the same features | Non-linear classical ML |
| 3 — Complex | CNN (pretrained EfficientNet / ResNet, fine-tuned) | Deep learning, transfer learning |

## Evaluation
Accuracy, top-3 accuracy, macro F1, confusion matrix, training time, model size; Grad-CAM for the CNN.

## Project structure
```
data/            raw and processed data (not tracked)
notebooks/       EDA and experiments
src/             data loading, features, models, training, evaluation
scripts/         CLI entry points
models/          trained weights (not tracked)
reports/         figures and tables for the report
docs/            report
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
