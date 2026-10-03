"""Level 1 — classical ML: Logistic Regression and Random Forest on HOG + color features.

Hyperparameters are selected with stratified 5-fold cross-validation on train + val
(classical models do not need a separate validation set for early stopping).
The final models are evaluated once on the held-out test set.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C  # noqa: E402
from src.evaluation import save_report  # noqa: E402
from src.features import load_features  # noqa: E402

MODELS = {
    "logreg": {
        "title": "Logistic Regression (HOG + color)",
        "pipeline": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(random_state=C.SEED)),
            ("clf", LogisticRegression(max_iter=3000, random_state=C.SEED)),
        ]),
        "grid": {
            "pca__n_components": [128, 512],
            "clf__C": [0.001, 0.01, 0.1],
            "clf__class_weight": [None, "balanced"],
        },
    },
    "random_forest": {
        "title": "Random Forest (HOG + color)",
        "pipeline": Pipeline([
            ("clf", RandomForestClassifier(n_jobs=-1, random_state=C.SEED)),
        ]),
        "grid": {
            "clf__n_estimators": [300],
            "clf__max_features": ["sqrt", "log2"],
            "clf__class_weight": [None, "balanced_subsample"],
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    meta = pd.read_csv(C.METADATA_CSV, dtype={"id": str})
    class_names = sorted(meta["make"].unique())
    meta["y"] = meta["make"].map({m: i for i, m in enumerate(class_names)})

    t0 = time.time()
    X = load_features(meta)
    print(f"Features: {X.shape} ({time.time() - t0:.0f}s)")

    dev = (meta["split"] != "test").to_numpy()
    X_dev, y_dev = X[dev], meta.loc[dev, "y"].to_numpy()
    X_test, y_test = X[~dev], meta.loc[~dev, "y"].to_numpy()
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=C.SEED)

    for name in args.models:
        spec = MODELS[name]
        print(f"\n=== {spec['title']} ===")
        search = GridSearchCV(spec["pipeline"], spec["grid"], cv=cv, scoring="f1_macro",
                              n_jobs=1 if name == "random_forest" else -1, refit=True,
                              return_train_score=True, verbose=1)
        t0 = time.time()
        search.fit(X_dev, y_dev)
        search_time = time.time() - t0

        cv_table = pd.DataFrame(search.cv_results_)
        keep = [c for c in cv_table if c.startswith("param_")] + [
            "mean_train_score", "mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score"]
        cv_table[keep].sort_values("rank_test_score").to_csv(
            C.TABLES_DIR / f"cv_results_{name}.csv", index=False)

        best = search.best_estimator_
        fit_time = search.refit_time_
        t0 = time.time()
        proba = best.predict_proba(X_test)
        infer_ms = (time.time() - t0) / len(X_test) * 1000

        model_path = C.MODELS_DIR / f"{name}.joblib"
        joblib.dump(best, model_path)
        metrics = save_report(name, spec["title"], y_test, proba, class_names, extra={
            "level": 1,
            "best_params": search.best_params_,
            "cv_macro_f1_mean": search.best_score_,
            "cv_macro_f1_std": cv_table.loc[search.best_index_, "std_test_score"],
            "cv_search_time_s": round(search_time, 1),
            "train_time_s": round(fit_time, 1),
            "inference_ms_per_image": round(infer_ms, 3),
            "model_size_mb": round(model_path.stat().st_size / 2**20, 2),
            "n_features": int(X.shape[1]),
        })
        print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
