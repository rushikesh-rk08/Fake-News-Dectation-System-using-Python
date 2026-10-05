"""Explain a linear model's prediction by the words that moved it most."""
from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline


def explain(model: Pipeline, text: str, top: int = 8) -> dict:
    """Return the label, the margin and each side's strongest words.

    A word's contribution is its TF-IDF value in this article times the weight
    the classifier learned for it. Positive pushes towards REAL, negative
    towards FAKE. The contributions and the intercept add up to the margin.
    """
    vec, clf = model.steps[0][1], model.steps[-1][1]
    if not hasattr(clf, "coef_"):
        raise TypeError("Explanations need a linear model with a coef_ attribute.")
    row = vec.transform([text])
    coef = clf.coef_.ravel()
    contrib = row.data * coef[row.indices]
    terms = vec.get_feature_names_out()[row.indices]
    order = np.argsort(contrib)
    margin = float(contrib.sum() + clf.intercept_[0])
    pairs = lambda idx: [(str(terms[i]), float(contrib[i])) for i in idx]
    return {
        "label": str(clf.classes_[int(margin > 0)]),
        "margin": margin,
        "known_words": int(row.nnz),
        "towards_fake": pairs([i for i in order[:top] if contrib[i] < 0]),
        "towards_real": pairs([i for i in order[::-1][:top] if contrib[i] > 0]),
    }
