import pandas as pd
import pytest

from fakenews.cleaning import clean_text
from fakenews.data import load_isot, load_news
from fakenews.evaluate import mcnemar, score
from fakenews.explain import explain
from fakenews.models import baseline, build_models, make_vectorizer


def tokens(text: str, clean: bool = True) -> list[str]:
    return make_vectorizer(clean).build_analyzer()(text)


def test_cleaning_removes_dateline_links_and_boilerplate():
    out = clean_text("WASHINGTON (Reuters) - Officials met. Featured image via https://example.org/a @someone")
    assert "reuters" not in out and "washington" not in out
    assert "example" not in out and "someone" not in out and "featured image" not in out
    assert "officials met" in out


def test_vectorizer_drops_dates_numbers_and_giveaway_words_only_when_cleaning():
    text = "On Friday October 28 2016 the senator said share this article and print 1,200 copies"
    cleaned = tokens(text)
    assert cleaned == ["senator", "said", "copies"]
    raw = tokens(text, clean=False)
    assert {"friday", "october", "2016", "share", "print"} <= set(raw)


def test_may_is_kept_because_it_is_usually_a_verb():
    assert "may" in clean_text("They May Decide")


def test_load_news_drops_empty_and_duplicate_rows(tmp_path, frame):
    extra = pd.DataFrame([
        {"title": "dup", "text": frame.text[0], "label": "FAKE"},
        {"title": "empty", "text": "   ", "label": "REAL"},
        {"title": "bad label", "text": "something else", "label": "MAYBE"},
    ])
    path = tmp_path / "news.csv"
    pd.concat([frame, extra]).to_csv(path, index=False)
    df, info = load_news(path)
    assert len(df) == len(frame)
    assert info["duplicate_text_removed"] == 1 and info["empty_text_removed"] == 1
    assert set(df.label) == {"FAKE", "REAL"}


def test_load_news_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_news(path)


def test_load_isot_labels_each_file(tmp_path, frame):
    real, fake = frame[frame.label == "REAL"], frame[frame.label == "FAKE"]
    real.drop(columns="label").to_csv(tmp_path / "True.csv", index=False)
    fake.drop(columns="label").to_csv(tmp_path / "Fake.csv", index=False)
    df, info = load_isot(tmp_path / "True.csv", tmp_path / "Fake.csv")
    assert info["label_counts"] == {"REAL": len(real), "FAKE": len(fake)} or info["label_counts"] == {"FAKE": len(fake), "REAL": len(real)}
    assert len(df) == len(frame)


@pytest.mark.parametrize("name", ["Passive-Aggressive", "Logistic regression", "Multinomial naive Bayes"])
def test_models_learn_a_separable_corpus(frame, name):
    train, test = frame.iloc[:60], frame.iloc[60:]
    model = build_models(seed=0)[name].fit(train.text, train.label)
    assert score(test.label, model.predict(test.text))["accuracy"] >= 0.9


def test_explanation_adds_up_to_the_decision(frame):
    model = baseline(seed=0).fit(frame.text, frame.label)
    text = "shocking secret hoax exposed while the senate committee said nothing"
    result = explain(model, text, top=5)
    assert result["label"] == model.predict([text])[0]
    assert result["margin"] == pytest.approx(float(model.decision_function([text])[0]))
    assert all(c < 0 for _, c in result["towards_fake"]) and all(c > 0 for _, c in result["towards_real"])
    assert {w for w, _ in result["towards_fake"]} & {"shocking", "secret", "hoax", "exposed"}
    assert {w for w, _ in result["towards_real"]} & {"senate", "committee", "said"}


def test_explanation_reports_unknown_text(frame):
    model = baseline(seed=0).fit(frame.text, frame.label)
    assert explain(model, "zzzz qqqq")["known_words"] == 0


def test_mcnemar_counts_disagreements():
    y = ["REAL"] * 10
    a = ["REAL"] * 10
    b = ["REAL"] * 4 + ["FAKE"] * 6
    result = mcnemar(y, a, b)
    assert (result["a_only_correct"], result["b_only_correct"]) == (6, 0)
    assert result["p_value"] == pytest.approx(2 * 0.5 ** 6)
    assert mcnemar(y, a, a)["p_value"] == 1.0
