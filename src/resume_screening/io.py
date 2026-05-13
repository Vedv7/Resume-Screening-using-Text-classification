"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_resume_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if not {"Category", "Resume"}.issubset(df.columns):
        raise ValueError(f"Expected columns Category and Resume; got {list(df.columns)}")
    return df
