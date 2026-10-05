"""Train and evaluate fake-news classifiers on TF-IDF features.

Usage:
    python train.py --data data/news.csv

Writes metrics, figures and the fitted baseline model to the output folder.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline

# scikit-learn 1.8 marks PassiveAggressiveClassifier for removal in 1.10.
# requirements.txt pins a version that still ships it.
warnings.filterwarnings("ignore", category=FutureWarning, message=".*PassiveAggressiveClassifier.*")

LABELS = ["FAKE", "REAL"]
BASELINE = "Passive-Aggressive"


def load_data(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read the CSV and remove rows that would distort the evaluation."""
    df = pd.read_csv(path)
    missing = {"title", "text", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    raw = len(df)
    df["text"] = df["text"].fillna("").str.strip()
    df = df[df["label"].isin(LABELS)]
    empty = int((df["text"] == "").sum())
    df = df[df["text"] != ""]
    # The same article appears more than once. Left in, copies can land on both
    # sides of the split and inflate the test score.
    duplicates = int(df["text"].duplicated().sum())
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df, {"raw_rows": raw, "empty_text_removed": empty, "duplicate_text_removed": duplicates, "rows_used": len(df)}


def vectorizer() -> TfidfVectorizer:
    # max_df drops words found in more than 70% of articles; they carry no signal.
    return TfidfVectorizer(stop_words="english", max_df=0.7)


def build_models(seed: int) -> dict[str, Pipeline]:
    return {
        BASELINE: make_pipeline(vectorizer(), PassiveAggressiveClassifier(max_iter=50, random_state=seed)),
        "Logistic regression": make_pipeline(vectorizer(), LogisticRegression(C=10, max_iter=1000)),
        "Multinomial naive Bayes": make_pipeline(vectorizer(), MultinomialNB(alpha=0.1)),
        "Neural network (MLP)": make_pipeline(
            TfidfVectorizer(stop_words="english", max_df=0.7, min_df=3, max_features=20000),
            MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=50, random_state=seed),
        ),
    }


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results: dict, path: Path) -> None:
    names = list(results)
    acc = [results[n]["test"]["accuracy"] * 100 for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bars = ax.barh(names[::-1], acc[::-1], color="#2a78d6", height=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_xlim(80, 100)
    ax.set_xlabel("Test accuracy (%), axis starts at 80")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_terms(model: Pipeline, path: Path, n: int = 15) -> dict:
    vec, clf = model.steps[0][1], model.steps[-1][1]
    terms = vec.get_feature_names_out()
    coef = clf.coef_.ravel()  # positive pushes towards REAL (classes are sorted)
    order = np.argsort(coef)
    fake, real = order[:n], order[-n:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, idx, label, color in ((axes[0], fake, "FAKE", "#eb6834"), (axes[1], real, "REAL", "#2a78d6")):
        ax.barh(terms[idx][::-1], np.abs(coef[idx])[::-1], color=color, height=0.6)
        ax.set_title(f"Terms that push towards {label}", fontsize=11)
        ax.set_xlabel("Weight (absolute)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"FAKE": terms[fake].tolist(), "REAL": terms[real].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/news.csv"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds on the training set (0 to skip)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    df, data_info = load_data(args.data)
    data_info["label_counts"] = df["label"].value_counts().to_dict()
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=args.seed, stratify=df["label"])
    print(f"{data_info['rows_used']} articles used ({data_info['duplicate_text_removed']} duplicates and "
          f"{data_info['empty_text_removed']} empty rows removed); {len(x_train)} train / {len(x_test)} test")

    results, fitted = {}, {}
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed) if args.folds else None
    for name, model in build_models(args.seed).items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        pred = model.predict(x_test)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS)
        cm = confusion_matrix(y_test, pred, labels=LABELS)
        entry = {
            "test": {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": dict(zip(LABELS, map(float, prec))),
                "recall": dict(zip(LABELS, map(float, rec))),
                "f1": dict(zip(LABELS, map(float, f1))),
                "confusion_matrix": {"labels": LABELS, "rows_actual_cols_predicted": cm.tolist()},
            },
            "fit_seconds": round(fit_seconds, 2),
        }
        if cv is not None:
            scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
            entry["cv_accuracy"] = {"mean": float(scores.mean()), "std": float(scores.std()), "folds": args.folds}
        results[name], fitted[name] = entry, model
        cv_txt = f"  cv {entry['cv_accuracy']['mean']:.4f} ± {entry['cv_accuracy']['std']:.4f}" if cv is not None else ""
        print(f"{name:<26} test accuracy {entry['test']['accuracy']:.4f}{cv_txt}  ({fit_seconds:.1f}s)")
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_confusion(cm, f"{name}: {entry['test']['accuracy']:.1%} accuracy", args.out / f"confusion_{slug}.png")

    plot_comparison(results, args.out / "model_comparison.png")
    top_terms = plot_top_terms(fitted[BASELINE], args.out / "top_terms.png")
    joblib.dump(fitted[BASELINE], args.models / "fake_news_model.joblib")
    (args.out / "metrics.json").write_text(json.dumps(
        {"data": data_info, "split": {"test_size": args.test_size, "seed": args.seed, "train": len(x_train), "test": len(x_test)},
         "models": results, "top_terms_baseline": top_terms}, indent=2))
    print(f"Saved metrics and figures to {args.out}/ and the baseline model to {args.models}/")


if __name__ == "__main__":
    main()
