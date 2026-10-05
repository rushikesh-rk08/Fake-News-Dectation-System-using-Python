"""Web demo: paste an article, get a label and the words behind it.

Run with:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from fakenews.explain import explain

MODEL_PATH = Path("models/fake_news_model.joblib")
EXAMPLE = ("The Senate voted on Tuesday to approve the spending bill, sending it to the president, "
           "who said he would sign it, according to a White House spokesman. Lawmakers from both "
           "parties said the measure would keep the government funded through the end of the year.")

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.write("Paste the text of a news article. The model labels it REAL or FAKE from its wording "
         "and shows which words moved the decision.")
st.warning("This model recognises writing style. It does not check facts, and it was trained on "
           "US political news from 2015 to 2016. Treat the label as a signal, not a verdict.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()

text = st.text_area("Article text", value=EXAMPLE, height=220)
top = st.slider("Words to show for each side", 3, 20, 8)

if not text.strip():
    st.info("Enter some text to classify.")
    st.stop()

result = explain(load_model(), text, top=top)
left, right = st.columns(2)
left.metric("Prediction", result["label"])
right.metric("Margin", f"{result['margin']:+.2f}", help="Negative leans FAKE, positive leans REAL. Values near 0 are uncertain.")

if result["known_words"] == 0:
    st.info("None of the words in this text were seen in training, so the label is a default, not a judgement.")
    st.stop()
if abs(result["margin"]) < 0.25:
    st.info("The margin is close to zero, so the model is not confident either way.")

st.subheader("Why")
fake_col, real_col = st.columns(2)
for col, side, key in ((fake_col, "FAKE", "towards_fake"), (real_col, "REAL", "towards_real")):
    col.caption(f"Words pushing towards {side}")
    if result[key]:
        table = pd.DataFrame(result[key], columns=["Word", "Contribution"])
        table["Contribution"] = table["Contribution"].abs().round(3)
        col.dataframe(table, hide_index=True, use_container_width=True)
    else:
        col.write("None")
st.caption("A word's contribution is how often it appears in this text, weighted by rarity, "
           "times the weight the model learned for it.")
