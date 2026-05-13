"""Command-line entry points for training and inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from resume_screening.train import load_bundle, predict_category, train_from_csv


def main_train(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train TF-IDF + LinearSVC and save a joblib bundle.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("ResumeDataSet.csv"),
        help="CSV with Category and Resume columns.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("models/resume_linear_svc.joblib"),
        help="Output joblib path.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid and fewer CV folds (for CI / smoke runs).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows read (debug / fast iteration).",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    if not args.data.is_file():
        print(f"Data file not found: {args.data}", file=sys.stderr)
        return 1

    bundle = train_from_csv(
        args.data,
        args.out,
        quick=args.quick,
        max_rows=args.max_rows,
        random_state=args.seed,
    )
    print(f"Wrote {args.out}")
    print(f"Held-out accuracy: {bundle['test_accuracy']:.4f}")
    print(f"Best CV score: {bundle['best_cv_score']:.4f}")
    print(f"Best params: {bundle['best_params']}")
    return 0


def main_predict(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Predict job category from resume text.")
    p.add_argument("--model", type=Path, required=True, help="Joblib bundle from resume-train.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Raw resume text.")
    g.add_argument("--file", type=Path, help="UTF-8 text file with resume body.")
    args = p.parse_args(argv)

    if not args.model.is_file():
        print(f"Model file not found: {args.model}", file=sys.stderr)
        return 1

    body = args.text
    if args.file is not None:
        body = args.file.read_text(encoding="utf-8")

    bundle = load_bundle(args.model)
    label = predict_category(bundle, body)
    print(label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_train())
