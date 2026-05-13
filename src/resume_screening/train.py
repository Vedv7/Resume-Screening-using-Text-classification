"""Training and persistence for the tuned LinearSVC pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from resume_screening.cleaning import clean_resume
from resume_screening.io import load_resume_csv


def _param_grid(*, quick: bool) -> dict[str, list[Any]]:
    if quick:
        return {
            "tfidf__max_features": [500, 1000],
            "tfidf__ngram_range": [(1, 1)],
            "classifier__C": [0.1, 1.0],
        }
    return {
        "tfidf__max_features": [1000, 2000, 3000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.1, 1, 10],
    }


def prepare_xy(df: pd.DataFrame) -> tuple[pd.Series, np.ndarray, LabelEncoder]:
    work = df.copy()
    work["Resume"] = work["Resume"].astype(str).map(clean_resume)
    le = LabelEncoder()
    y = le.fit_transform(work["Category"].astype(str))
    return work["Resume"], y, le


def _stratify_labels(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return None
    return y


def train_linear_svc_bundle(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    quick: bool = False,
    cv: int | None = None,
) -> dict[str, Any]:
    """Fit GridSearchCV(Tfidf + LinearSVC); return bundle + test accuracy."""
    X, y, le = prepare_xy(df)
    cv_folds = 3 if quick else (cv or 5)
    strat = _stratify_labels(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LinearSVC(random_state=random_state)),
        ]
    )
    grid = GridSearchCV(
        pipe,
        _param_grid(quick=quick),
        cv=cv_folds,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    return {
        "pipeline": grid.best_estimator_,
        "label_encoder": le,
        "grid_search": grid,
        "test_accuracy": acc,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": bundle["pipeline"],
        "label_encoder": bundle["label_encoder"],
        "meta": {
            "test_accuracy": bundle["test_accuracy"],
            "best_params": bundle["best_params"],
            "best_cv_score": bundle["best_cv_score"],
        },
    }
    joblib.dump(payload, path)


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def train_from_csv(
    csv_path: str | Path,
    out_path: str | Path,
    *,
    quick: bool = False,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_resume_csv(csv_path)
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()
    bundle = train_linear_svc_bundle(df, random_state=random_state, quick=quick)
    save_bundle(bundle, out_path)
    bundle.pop("grid_search", None)
    return bundle


def predict_category(bundle: dict[str, Any], resume_text: str) -> str:
    cleaned = clean_resume(resume_text)
    code = bundle["pipeline"].predict([cleaned])[0]
    return str(bundle["label_encoder"].inverse_transform([code])[0])
