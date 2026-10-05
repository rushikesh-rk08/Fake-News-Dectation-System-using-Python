"""Dataset loaders. Both return a frame with `title`, `text` and `label` (FAKE or REAL)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import LABELS


def _tidy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that would distort an evaluation and report what was removed."""
    raw = len(df)
    df = df.assign(text=df["text"].fillna("").astype(str).str.strip())
    df = df[df["label"].isin(LABELS)]
    empty = int((df["text"] == "").sum())
    df = df[df["text"] != ""]
    # The same article can appear more than once. Left in, copies can land on both
    # sides of a split and inflate the test score.
    duplicates = int(df["text"].duplicated().sum())
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    info = {"raw_rows": raw, "empty_text_removed": empty, "duplicate_text_removed": duplicates,
            "rows_used": len(df), "label_counts": df["label"].value_counts().to_dict()}
    return df[["title", "text", "label"]], info


def load_news(path: Path | str) -> tuple[pd.DataFrame, dict]:
    """Main dataset: one CSV with title, text and label columns."""
    df = pd.read_csv(path)
    missing = {"title", "text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return _tidy(df)


def load_isot(true_path: Path | str, fake_path: Path | str) -> tuple[pd.DataFrame, dict]:
    """ISOT dataset: two CSVs, one of real articles and one of fake ones."""
    real = pd.read_csv(true_path).assign(label="REAL")
    fake = pd.read_csv(fake_path).assign(label="FAKE")
    return _tidy(pd.concat([real, fake], ignore_index=True))
