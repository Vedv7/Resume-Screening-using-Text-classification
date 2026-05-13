from pathlib import Path

import pytest

from resume_screening.cleaning import clean_resume


def test_clean_resume_lowercase_and_strips_urls():
    text = "See https://example.com/resume PYTHON 3.11"
    out = clean_resume(text)
    assert "https" not in out
    assert "python" in out
    assert out == out.lower()


def test_clean_resume_digits_removed():
    assert "3" not in clean_resume("version 3.11 and 2024")


def test_clean_resume_collapses_whitespace():
    assert clean_resume("a    b\t\tc") == "a b c"
