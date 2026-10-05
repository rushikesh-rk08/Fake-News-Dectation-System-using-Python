"""Classify news text with the model saved by train.py and show why.

Usage:
    python predict.py "Full text of an article ..."
    python predict.py --file article.txt
    python predict.py --file article.txt --top 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

from fakenews.explain import explain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="article text; reads standard input if omitted")
    ap.add_argument("--file", type=Path, help="read the article from a text file")
    ap.add_argument("--model", type=Path, default=Path("models/fake_news_model.joblib"))
    ap.add_argument("--top", type=int, default=8, help="how many words to list for each side (0 hides them)")
    args = ap.parse_args()

    if not args.model.exists():
        raise SystemExit(f"No model at {args.model}. Run `python train.py` first.")
    text = args.file.read_text(encoding="utf-8") if args.file else args.text or sys.stdin.read()
    if not text.strip():
        raise SystemExit("No text to classify.")

    result = explain(joblib.load(args.model), text, top=max(args.top, 1))
    print(f"{result['label']}  (margin {result['margin']:+.2f}; negative leans FAKE, positive leans REAL, near 0 is uncertain)")
    if result["known_words"] == 0:
        print("None of the words in this text were seen in training, so the label is a default, not a judgement.")
    elif args.top > 0:
        for side, key in (("FAKE", "towards_fake"), ("REAL", "towards_real")):
            words = ", ".join(f"{w} ({c:+.2f})" for w, c in result[key]) or "none"
            print(f"Words pushing towards {side}: {words}")


if __name__ == "__main__":
    main()
