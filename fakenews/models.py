"""Model pipelines. Every model is a single Pipeline, so the vectorizer is refitted per fold."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from .cleaning import STOP_WORDS, TOKEN_PATTERN, clean_text, lowercase

BASELINE = "Passive-Aggressive"


def make_vectorizer(clean: bool = True, **overrides) -> TfidfVectorizer:
    """TF-IDF over single words. max_df drops words found in more than 70% of articles."""
    params = dict(stop_words="english", max_df=0.7, preprocessor=lowercase)
    if clean:
        params.update(stop_words=STOP_WORDS, token_pattern=TOKEN_PATTERN, preprocessor=clean_text)
    params.update(overrides)
    return TfidfVectorizer(**params)


def passive_aggressive(seed: int = 7, C: float = 1.0, max_iter: int = 50):
    """Passive-Aggressive classifier (PA-I): hinge loss with a step size capped at C.

    scikit-learn 1.8 moved this algorithm into SGDClassifier and scheduled the old
    PassiveAggressiveClassifier class for removal, so use the new form when it exists.
    """
    try:
        clf = SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1", eta0=C, max_iter=max_iter, random_state=seed)
        clf._validate_params()
        return clf
    except Exception:  # scikit-learn < 1.8
        from sklearn.linear_model import PassiveAggressiveClassifier
        return PassiveAggressiveClassifier(C=C, max_iter=max_iter, random_state=seed)


def baseline(seed: int = 7, clean: bool = True, C: float = 1.0, **vectorizer_overrides) -> Pipeline:
    return Pipeline([("tfidf", make_vectorizer(clean, **vectorizer_overrides)), ("clf", passive_aggressive(seed, C))])


def build_models(seed: int = 7, clean: bool = True) -> dict[str, Pipeline]:
    return {
        BASELINE: baseline(seed, clean),
        "Logistic regression": Pipeline([("tfidf", make_vectorizer(clean)), ("clf", LogisticRegression(C=10, max_iter=1000))]),
        "Multinomial naive Bayes": Pipeline([("tfidf", make_vectorizer(clean)), ("clf", MultinomialNB(alpha=0.1))]),
        "Neural network (MLP)": Pipeline([
            ("tfidf", make_vectorizer(clean, min_df=3, max_features=20000)),
            ("clf", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=50, random_state=seed)),
        ]),
    }
