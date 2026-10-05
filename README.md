# Fake News Detection

Classifies news articles as **REAL** or **FAKE** from their text, using TF-IDF features and
four classifiers: a Passive-Aggressive linear model (the baseline), logistic regression,
multinomial naive Bayes and a small neural network.

This is a 2026 rebuild of a class project I first did in January 2021 for a Neural Networks
and Fuzzy Control course during my Bachelor's degree. The original used the TF-IDF and
Passive-Aggressive pipeline. The rebuild keeps that as the baseline and adds data cleaning,
a stratified split, cross-validation, three comparison models, figures and a command-line
predictor.

## Results

Held-out test set of 1,212 articles (20%, stratified, seed 7). Cross-validation is 5-fold on
the training set.

| Model | Test accuracy | CV accuracy (mean ± sd) | F1 FAKE | F1 REAL |
|---|---:|---:|---:|---:|
| Passive-Aggressive (baseline) | 93.1% | 92.9% ± 0.5 | 0.932 | 0.929 |
| Logistic regression | 92.2% | 92.6% ± 0.8 | 0.925 | 0.920 |
| Multinomial naive Bayes | 89.4% | 89.3% ± 0.5 | 0.892 | 0.896 |
| Neural network (MLP, 64 hidden units) | 93.3% | 91.8% ± 0.7 | 0.934 | 0.932 |

The neural network edges the baseline on the test split but is behind it in cross-validation,
and it takes about ten times longer to train. The two are not meaningfully different on this
data; the baseline is the model saved for prediction.

![Model comparison](results/model_comparison.png)

Baseline confusion matrix (rows are actual, columns are predicted):

| | Predicted FAKE | Predicted REAL |
|---|---:|---:|
| **Actual FAKE** | 577 | 37 |
| **Actual REAL** | 47 | 551 |

![Top terms](results/top_terms.png)

Full numbers are in [`results/metrics.json`](results/metrics.json).

## What the model actually learns

The strongest FAKE terms include `2016`, `october`, `november`, `share`, `print` and
`advertisement`. The strongest REAL terms include `said`, `says` and weekday names. So the
classifier is picking up **writing style and where and when the articles were collected**:
attributed quotes and datelines on one side, web-page boilerplate and election-season
vocabulary on the other. It does not check facts.

That has practical consequences:

- Scores describe this dataset (US political news, 2015 to 2016). Expect lower accuracy on
  other topics, countries or years.
- A false story written in a newswire style can be labelled REAL, and a true story in a
  blog style can be labelled FAKE.
- It should not be used on its own to judge whether a specific article is true.

## Data

The public `fake_or_real_news` dataset compiled by George McIntire: 6,335 articles with
`title`, `text` and a `label` of `REAL` or `FAKE`, almost evenly balanced. The file is about
30 MB of third-party article text, so it is not stored in this repository. Download it and
save it as `data/news.csv`.

Cleaning before training:

- 36 rows with empty text are removed.
- 241 articles whose text duplicates an earlier row are removed. Left in, copies of one
  article can fall on both sides of the split and inflate the test score.

That leaves 6,058 articles: 4,846 for training and 1,212 for testing.

## Method

1. **Features.** `TfidfVectorizer` with English stop words removed and `max_df=0.7`, which
   drops words that appear in more than 70% of articles. Fitted on the training set only.
2. **Baseline.** `PassiveAggressiveClassifier(max_iter=50)`, an online linear model that
   leaves its weights alone when a prediction is correct and corrects them just enough when
   it is wrong.
3. **Comparisons.** Logistic regression (`C=10`), multinomial naive Bayes (`alpha=0.1`), and
   an `MLPClassifier` with one hidden layer of 64 units and early stopping, on the 20,000
   most frequent terms.
4. **Evaluation.** Accuracy, per-class precision, recall and F1, a confusion matrix on the
   held-out test set, and 5-fold stratified cross-validation on the training set. Each model
   is a single pipeline, so the vectorizer is refitted inside every fold.

## Run it

```bash
pip install -r requirements.txt
python train.py --data data/news.csv      # a few minutes; writes results/ and models/
python predict.py "Paste the full text of an article here"
python predict.py --file article.txt
```

`predict.py` prints the label and a margin. Negative margins lean FAKE, positive lean REAL,
and values near zero are uncertain.

`train.py` options: `--test-size`, `--seed`, `--folds` (0 skips cross-validation), `--out`,
`--models`.

## Files

| Path | Purpose |
|---|---|
| `train.py` | Cleans the data, trains and evaluates the four models, writes metrics and figures |
| `predict.py` | Classifies new text with the saved baseline model |
| `results/` | `metrics.json`, model comparison, top terms and one confusion matrix per model |
| `requirements.txt` | Python dependencies |

scikit-learn is pinned below 1.10 because `PassiveAggressiveClassifier` is scheduled for
removal in that release.
