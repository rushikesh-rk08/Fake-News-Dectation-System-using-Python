"""Classify news text with the model saved by train.py.

Usage:
    python predict.py "Full text of an article ..."
    python predict.py --file article.txt
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib

warnings.filterwarnings("ignore", category=FutureWarning, message=".*PassiveAggressiveClassifier.*")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="article text; reads standard input if omitted")
    ap.add_argument("--file", type=Path, help="read the article from a text file")
    ap.add_argument("--model", type=Path, default=Path("models/fake_news_model.joblib"))
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Run `python train.py` first.")
    text = args.file.read_text(encoding="utf-8") if args.file else args.text or sys.stdin.read()
    if not text.strip():
        raise SystemExit("No text to classify.")

    model = joblib.load(args.model)
    label = model.predict([text])[0]
    # Signed distance from the decision boundary: negative is FAKE, positive is REAL.
    margin = float(model.decision_function([text])[0])
    print(f"{label}  (margin {margin:+.2f}; values near 0 are uncertain)")


if __name__ == "__main__":
    main()
