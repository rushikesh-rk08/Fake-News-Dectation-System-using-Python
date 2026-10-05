"""Grid-search the baseline's settings with cross-validation on the training set.

Usage:
    python tune.py --data data/news.csv

Writes results/tuning.json. train.py picks it up and reports the tuned model
next to the default one. The held-out test set is never used here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from fakenews.data import load_news
from fakenews.models import baseline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/news.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    df, _ = load_news(args.data)
    x_train, _, y_train, _ = train_test_split(df["text"], df["label"], test_size=args.test_size,
                                              random_state=args.seed, stratify=df["label"])
    model = baseline(args.seed)
    step = "eta0" if "eta0" in model.named_steps["clf"].get_params() else "C"
    grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__sublinear_tf": [False, True],
        "tfidf__max_df": [0.5, 0.7],
        f"clf__{step}": [0.1, 0.5, 1.0],
    }
    search = GridSearchCV(model, grid, scoring="accuracy", n_jobs=args.jobs,
                          cv=StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed))
    search.fit(x_train, y_train)

    res = search.cv_results_
    rows = []
    for p, m, s in zip(res["params"], res["mean_test_score"], res["std_test_score"]):
        params = {k.split("__")[1]: v for k, v in p.items()}
        params["C"] = params.pop(step)
        rows.append({"params": params, "mean": float(m), "std": float(s)})
    rows.sort(key=lambda r: -r["mean"])
    default = next(r for r in rows if r["params"] == {"ngram_range": (1, 1), "sublinear_tf": False, "max_df": 0.7, "C": 1.0})
    best = rows[0]
    # One-standard-error rule: among settings within one standard deviation of the best
    # score, take the simplest (single words before word pairs), then the highest score.
    close = [r for r in rows if r["mean"] >= best["mean"] - best["std"]]
    selected = min(close, key=lambda r: (r["params"]["ngram_range"][1], -r["mean"]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tuning.json").write_text(json.dumps(
        {"combinations": len(rows), "folds": args.folds, "best": best, "selected": selected,
         "default": {"mean": default["mean"], "std": default["std"]}, "top_10": rows[:10]}, indent=2))
    print(f"Tried {len(rows)} combinations. Default {default['mean']:.4f} ± {default['std']:.4f}; "
          f"best {best['mean']:.4f} ± {best['std']:.4f} with {best['params']}; "
          f"selected {selected['mean']:.4f} ± {selected['std']:.4f} with {selected['params']}")


if __name__ == "__main__":
    main()
