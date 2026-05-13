"""Resume text → job category classification (TF–IDF + sklearn)."""

__version__ = "0.1.0"

from resume_screening.cleaning import clean_resume

__all__ = ["__version__", "clean_resume"]
