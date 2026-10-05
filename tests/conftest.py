import random

import pandas as pd
import pytest

REAL_WORDS = "senate vote spokesman said committee budget minister statement officials report".split()
FAKE_WORDS = "shocking secret exposed globalist hoax cabal truth sheeple conspiracy leaked".split()
FILLER = "people country government year time news world state public week".split()


def make_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """A small synthetic corpus the models can separate, so tests need no downloads."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        label = "REAL" if i % 2 else "FAKE"
        words = rng.choices(REAL_WORDS if label == "REAL" else FAKE_WORDS, k=12) + rng.choices(FILLER, k=10)
        rng.shuffle(words)
        rows.append({"title": f"Title {i}", "text": " ".join(words) + f" item{i}", "label": label})
    return pd.DataFrame(rows)


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()
