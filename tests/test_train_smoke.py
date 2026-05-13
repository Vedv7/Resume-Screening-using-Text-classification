from pathlib import Path

import pytest

from resume_screening.io import load_resume_csv
from resume_screening.train import predict_category, train_from_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "ResumeDataSet.csv"


@pytest.mark.skipif(not DATA.is_file(), reason="ResumeDataSet.csv not present")
def test_train_smoke_quick(tmp_path):
    out = tmp_path / "bundle.joblib"
    bundle = train_from_csv(DATA, out, quick=True, max_rows=400, random_state=0)
    assert out.is_file()
    assert 0 <= bundle["test_accuracy"] <= 1
    assert "pipeline" in bundle
    assert "label_encoder" in bundle
    cat = predict_category(bundle, "Python pandas scikit-learn machine learning NLP")
    assert isinstance(cat, str)
    assert len(cat) > 0


@pytest.mark.skipif(not DATA.is_file(), reason="ResumeDataSet.csv not present")
def test_load_dataset_columns():
    df = load_resume_csv(DATA)
    assert set(df.columns) >= {"Category", "Resume"}
    assert len(df) >= 900
