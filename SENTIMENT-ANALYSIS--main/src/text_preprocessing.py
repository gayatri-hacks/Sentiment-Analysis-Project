"""Shared deterministic preprocessing used for training and Streamlit inference."""
from __future__ import annotations
import re
import string
from functools import lru_cache
import nltk
from nltk.corpus import stopwords

def ensure_nltk_resources() -> None:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        if not nltk.download("stopwords", quiet=True):
            raise RuntimeError("NLTK stopwords are unavailable. Run: python -m nltk.downloader stopwords")

@lru_cache(maxsize=1)
def _resources() -> set[str]:
    ensure_nltk_resources()
    return set(stopwords.words("english"))

def clean_text(text: object) -> str:
    stop_words = _resources()
    value = str(text).lower()
    value = re.sub(r"https?://\S+|www\.\S+", " url ", value)
    value = re.sub(r"@\w+", " user ", value)
    value = re.sub(r"#(\w+)", r"\1", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\d+", " ", value)
    value = re.sub(r"(.)\1{2,}", r"\1", value)
    return " ".join(word for word in value.split() if word not in stop_words and len(word) > 2)
