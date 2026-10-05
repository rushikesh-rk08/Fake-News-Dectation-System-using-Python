"""Scoring helpers."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from . import LABELS


def score(y_true, y_pred) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": dict(zip(LABELS, map(float, prec))),
        "recall": dict(zip(LABELS, map(float, rec))),
        "f1": dict(zip(LABELS, map(float, f1))),
        "confusion_matrix": {"labels": LABELS,
                             "rows_actual_cols_predicted": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()},
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test: do two models make different errors on the same test set?

    Only the articles where exactly one model is right matter. If the models are
    equally good, each should win about half of those.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_only = int(((pred_a == y_true) & (pred_b != y_true)).sum())
    b_only = int(((pred_b == y_true) & (pred_a != y_true)).sum())
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": p}
