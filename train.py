"""Train and evaluate fake-news classifiers on TF-IDF features.

Usage:
    python train.py                       # expects data/news.csv
    python train.py --data data/news.csv --external-dir data/external

What it does:
  1. Cleans the data and makes a stratified train/test split.
  2. Trains the baseline with and without text cleaning, to show what the cleaning costs.
  3. Trains the comparison models on cleaned text and tests each against the baseline.
  4. If the ISOT files (True.csv, Fake.csv) are in --external-dir, scores every model on
     that dataset too, without retraining.
  5. Writes metrics and figures to --out and the chosen model to --models.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from fakenews import LABELS
from fakenews.data import load_isot, load_news
from fakenews.evaluate import mcnemar, score
from fakenews.models import BASELINE, baseline, build_models

BLUE, ORANGE = "#2a78d6", "#eb6834"


def run(model, x_train, y_train, x_test, y_test, cv) -> tuple[dict, np.ndarray]:
    """Fit on the training split, score on the test split, cross-validate on the training split."""
    start = time.perf_counter()
    model.fit(x_train, y_train)
    seconds = time.perf_counter() - start
    pred = model.predict(x_test)
    entry = {"test": score(y_test, pred), "fit_seconds": round(seconds, 2)}
    if cv is not None:
        s = cross_val_score(clone(model), x_train, y_train, cv=cv, scoring="accuracy")
        entry["cv_accuracy"] = {"mean": float(s.mean()), "std": float(s.std()), "folds": cv.get_n_splits()}
    return entry, pred


def describe(name: str, entry: dict) -> str:
    cv = entry.get("cv_accuracy")
    cv_txt = f"  cv {cv['mean']:.4f} ± {cv['std']:.4f}" if cv else ""
    return f"  {name:<34} test {entry['test']['accuracy']:.4f}{cv_txt}"


def plot_confusion(cm, title: str, path: Path) -> None:
    cm = np.asarray(cm)
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


def plot_comparison(rows: list[tuple[str, float, float | None]], path: Path) -> None:
    """Horizontal bars per model: same-dataset test accuracy and, if available, the outside dataset."""
    names = [r[0] for r in rows][::-1]
    same = [r[1] * 100 for r in rows][::-1]
    has_ext = all(r[2] is not None for r in rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.75 * len(rows) + 1.4))
    y = np.arange(len(rows))
    h = 0.36 if has_ext else 0.55
    b1 = ax.barh(y + (h / 2 + 0.02 if has_ext else 0), same, height=h, color=BLUE, label="Same dataset (held-out test set)")
    ax.bar_label(b1, fmt="%.1f%%", padding=4, fontsize=9)
    if has_ext:
        ext = [r[2] * 100 for r in rows][::-1]
        b2 = ax.barh(y - h / 2 - 0.02, ext, height=h, color=ORANGE, label="Outside dataset (ISOT)")
        ax.bar_label(b2, fmt="%.1f%%", padding=4, fontsize=9)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    ax.set_yticks(y, names)
    ax.set_xlim(40, 100)
    ax.axvline(50, color="#888", lw=1, ls=":")
    ax.set_xlabel("Accuracy (%). Axis starts at 40; dotted line is coin-flip level")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_terms(model, path: Path, n: int = 15) -> dict:
    vec, clf = model.steps[0][1], model.steps[-1][1]
    terms = vec.get_feature_names_out()
    coef = clf.coef_.ravel()  # positive pushes towards REAL (classes are sorted)
    order = np.argsort(coef)
    fake, real = order[:n], order[-n:][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, idx, label, color in ((axes[0], fake, "FAKE", ORANGE), (axes[1], real, "REAL", BLUE)):
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
    ap.add_argument("--external-dir", type=Path, default=Path("data/external"),
                    help="folder holding the ISOT files True.csv and Fake.csv (skipped if absent)")
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5, help="cross-validation folds on the training set (0 to skip)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.models.mkdir(parents=True, exist_ok=True)

    df, data_info = load_news(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=args.test_size, random_state=args.seed, stratify=df["label"])
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed) if args.folds else None
    print(f"{data_info['rows_used']} articles used ({data_info['duplicate_text_removed']} duplicates and "
          f"{data_info['empty_text_removed']} empty rows removed); {len(x_train)} train / {len(x_test)} test")

    fitted, preds, results = {}, {}, {}

    # 1. What does cleaning cost on the home dataset?
    print("Baseline, with and without cleaning:")
    raw_name = f"{BASELINE} (no cleaning)"
    raw_model = baseline(args.seed, clean=False)
    results[raw_name], preds[raw_name] = run(raw_model, x_train, y_train, x_test, y_test, cv)
    fitted[raw_name] = raw_model
    print(describe(raw_name, results[raw_name]))

    # 2. All models on cleaned text.
    for name, model in build_models(args.seed, clean=True).items():
        results[name], preds[name] = run(model, x_train, y_train, x_test, y_test, cv)
        fitted[name] = model
        if name == BASELINE:
            print(describe(name, results[name]) + "\nComparison models on cleaned text:")
        else:
            print(describe(name, results[name]))

    # 3. Tuned baseline, if tune.py has been run.
    tuning_path = args.out / "tuning.json"
    tuned_name, chosen = None, BASELINE
    if tuning_path.exists():
        tuning = json.loads(tuning_path.read_text())
        p = dict(tuning["selected"]["params"])
        p["ngram_range"] = tuple(p["ngram_range"])
        tuned_name = f"{BASELINE} (tuned)"
        tuned = baseline(args.seed, clean=True, **p)
        results[tuned_name], preds[tuned_name] = run(tuned, x_train, y_train, x_test, y_test, cv)
        results[tuned_name]["params"] = tuning["selected"]["params"]
        fitted[tuned_name] = tuned
        print(describe(tuned_name, results[tuned_name]))
        # Chosen on cross-validation alone, so the test set stays untouched by the choice.
        if tuning["selected"]["mean"] > tuning["default"]["mean"]:
            chosen = tuned_name

    # 4. Are the differences from the baseline real? Exact McNemar test on the test set.
    for name in results:
        if name != BASELINE:
            results[name]["mcnemar_vs_baseline"] = mcnemar(y_test.to_numpy(), preds[BASELINE], preds[name])

    # 5. Outside dataset: same fitted models, no retraining.
    external = None
    true_csv, fake_csv = args.external_dir / "True.csv", args.external_dir / "Fake.csv"
    if true_csv.exists() and fake_csv.exists():
        ext, ext_info = load_isot(true_csv, fake_csv)
        print(f"Outside dataset (ISOT): {ext_info['rows_used']} articles, models not retrained")
        external = {"dataset": "ISOT Fake News Dataset", "data": ext_info, "models": {}}
        for name, model in fitted.items():
            external["models"][name] = score(ext["label"], model.predict(ext["text"]))
            print(f"  {name:<34} accuracy {external['models'][name]['accuracy']:.4f}")
        # The reverse direction, for the baseline only.
        ex_train, ex_test, ey_train, ey_test = train_test_split(
            ext["text"], ext["label"], test_size=args.test_size, random_state=args.seed, stratify=ext["label"])
        external["reverse"] = {}
        for label, clean in (("no cleaning", False), ("cleaned", True)):
            m = baseline(args.seed, clean=clean).fit(ex_train, ey_train)
            external["reverse"][label] = {
                "isot_test_accuracy": float((m.predict(ex_test) == ey_test).mean()),
                "main_dataset_accuracy": float((m.predict(df["text"]) == df["label"]).mean()),
            }
            r = external["reverse"][label]
            print(f"  trained on ISOT ({label}): ISOT test {r['isot_test_accuracy']:.4f}, main dataset {r['main_dataset_accuracy']:.4f}")
    else:
        print(f"No ISOT files in {args.external_dir}/, skipping the outside-dataset test")

    # 6. Figures, metrics and the saved model.
    order = [raw_name] + [n for n in results if n != raw_name]
    plot_comparison([(n, results[n]["test"]["accuracy"], external["models"][n]["accuracy"] if external else None)
                     for n in order], args.out / "model_comparison.png")
    cm = results[chosen]["test"]["confusion_matrix"]["rows_actual_cols_predicted"]
    plot_confusion(cm, f"{chosen}: {results[chosen]['test']['accuracy']:.1%} accuracy", args.out / "confusion_matrix.png")
    top_terms = plot_top_terms(fitted[chosen], args.out / "top_terms.png")
    joblib.dump(fitted[chosen], args.models / "fake_news_model.joblib")
    (args.out / "metrics.json").write_text(json.dumps({
        "data": data_info,
        "split": {"test_size": args.test_size, "seed": args.seed, "train": len(x_train), "test": len(x_test)},
        "saved_model": chosen, "models": results, "external": external, "top_terms_saved_model": top_terms,
    }, indent=2))
    print(f"Saved '{chosen}' to {args.models}/ and metrics and figures to {args.out}/")


if __name__ == "__main__":
    main()
