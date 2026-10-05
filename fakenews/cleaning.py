"""Remove text that reveals where or when an article was collected.

A classifier trained on the raw text scores well partly by spotting these cues
(the year 2016, month names, "share", "print", wire-service datelines) rather
than anything about the writing itself. Stripping them gives a lower but more
honest score, and a model that transfers better to articles from other sources.
"""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MONTHS = ("january february march april june july august september october november december "
          "jan feb mar apr jun jul aug sep sept oct nov dec")  # "may" is left in: it is mostly the verb
WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun"
# Left-overs from web pages, photo credits and wire services, not part of the article.
BOILERPLATE = ("share print advertisement advertisements screengrab screenshot getty subscribe newsletter "
               "comments via pic www http https com html source article reuters")
GIVEAWAYS = frozenset((MONTHS + " " + WEEKDAYS + " " + BOILERPLATE).split())

# Words the vectorizer ignores when cleaning is on.
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | GIVEAWAYS)
# Tokens are runs of two or more letters, so years, dates and other numbers are dropped.
TOKEN_PATTERN = r"(?u)\b[^\W\d_][^\W\d_]+\b"

_URL = re.compile(r"https?://\S+|www\.\S+|\.(?:com|org|net)\b|@\w+")
# "WASHINGTON (Reuters) - " and similar openings used by wire services.
_DATELINE = re.compile(r"^[^a-z]{0,60}\((?:Reuters|REUTERS|AP|AFP)\)\s*[-\u2013\u2014]\s*")
PHRASES = ("featured image", "read more", "click here", "image via", "photo by", "sign up", "follow us")


def clean_text(text: str) -> str:
    """Lower-case the text and strip links, wire datelines and page boilerplate phrases.

    Dates, numbers and single giveaway words are removed by the vectorizer through
    STOP_WORDS and TOKEN_PATTERN, which is much faster than doing it here.
    """
    text = _DATELINE.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    for phrase in PHRASES:
        text = text.replace(phrase, " ")
    return text


def lowercase(text: str) -> str:
    """Preprocessor for the uncleaned comparison (what TfidfVectorizer does by default)."""
    return text.lower()
