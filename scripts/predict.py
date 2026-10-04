"""Predict the car make for one or more photos.

Usage: python scripts/predict.py path/to/car.jpg [more.jpg ...] [--top 3]
"""
import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict import Predictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    predictor = Predictor()
    for path in args.images:
        print(f"\n{path.name}")
        for model, probs in predictor.predict(Image.open(path)).items():
            top = sorted(probs.items(), key=lambda kv: -kv[1])[:args.top]
            print(f"  {model:<10} " + ", ".join(f"{make} {p:.0%}" for make, p in top))


if __name__ == "__main__":
    main()
