# Resume Intelligence

NLP-powered resume classification system for automatically categorizing candidate resumes across multiple job domains.

Built using Python, Scikit-learn, and traditional machine learning pipelines to reduce manual recruiter screening effort at scale.

---

## Overview

Recruiters often spend significant time manually reviewing resumes and matching candidates to relevant job roles.

Resume Intelligence automates this process by:

- Reading raw resume text
- Cleaning and normalizing candidate data
- Extracting meaningful textual features
- Classifying resumes into predefined job categories

The system is designed to improve recruiter efficiency, reduce manual screening effort, and enable scalable hiring workflows.

---

## Problem Statement

Traditional resume screening suffers from:

| Challenge | Impact |
|-----------|--------|
| Manual review | Slow hiring cycles |
| High resume volume | Operational bottlenecks |
| Human inconsistency | Potential bias |
| Poor scalability | Increased recruiting costs |

Resume Intelligence solves this using machine learning-driven text classification.

---

## Dataset

The model was trained on:

| Metric | Value |
|--------|-------|
| Total Resumes | 962 |
| Job Categories | 25 |
| Input Format | Raw Text |

Example categories include:

- Data Science
- Python Developer
- Java Developer
- DevOps Engineer
- Testing
- Web Designing
- Business Analyst

---

## Pipeline

```text
Raw Resume Text
       ↓
Text Cleaning
       ↓
Token Normalization
       ↓
TF-IDF Vectorization
       ↓
Model Training
       ↓
Resume Classification
       ↓
Predicted Job Category
```

---

## Preprocessing

The following NLP preprocessing steps were applied:

- Lowercasing
- URL removal
- Special character removal
- Number removal
- Whitespace normalization
- Label encoding

---

## Feature Engineering

### TF-IDF Vectorization

Text features were generated using:

- Unigrams
- Bigrams
- Feature dimensionality tuning

This helps the model identify role-specific keywords while reducing noise.

---

## Models Evaluated

### Logistic Regression
Accuracy: **99%**

### Linear Support Vector Classifier (LinearSVC)
Accuracy: **99%**

### K-Nearest Neighbors
Accuracy: **98%**

### Naive Bayes
Accuracy: **99%**

---

## Best Model

LinearSVC delivered the most stable performance across all 25 job categories.

### Best Hyperparameters

```python
{
    "classifier__C": 0.1,
    "tfidf__max_features": 3000,
    "tfidf__ngram_range": (1,1)
}
```

Hyperparameter tuning was performed using GridSearchCV.

---

## Repository Structure

```bash
resume-intelligence/
├── data/
├── notebooks/
├── models/
├── src/
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/resume-intelligence.git
cd resume-intelligence
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Training

```bash
python train.py
```

---

## Business Impact

This system can help recruiting teams:

- Reduce manual screening effort
- Improve hiring speed
- Standardize candidate classification
- Scale to thousands of applications

---

## Future Improvements

- BERT / Transformer-based embeddings
- Multi-language resume support
- Real-time API deployment
- Skill extraction and candidate ranking

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Author

**Veda Swaroop**  
AI / ML Engineer | Applied NLP | Recruitment Intelligence
