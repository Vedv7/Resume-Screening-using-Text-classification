"""Text normalization for resume bodies (matches notebook behavior)."""

from __future__ import annotations

import re


def clean_resume(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Mojibake / encoding fixes seen in the corpus
    text = text.replace("naÃ¯ve", "naive").replace("naïve", "naive")
    text = text.replace("â\x80¢", "-").replace("\u2022", "-")
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
