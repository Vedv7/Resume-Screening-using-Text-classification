# Contributing

Thanks for your interest in improving this project.

## How to contribute

1. **Fork** [https://github.com/Vedv7/Resume-Screening-using-Text-classification](https://github.com/Vedv7/Resume-Screening-using-Text-classification) and create a branch from `main`.
2. **Set up** a virtual environment and install the package in editable mode (includes tests):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -e ".[dev,notebook]"
   ```

   For a lighter environment: `pip install -e ".[dev]"` (omit Jupyter extras). `requirements.txt` remains a minimal flat list for quick installs.

3. **Make changes** (package, notebook, or docs). Keep commits focused and messages clear.
4. **Run tests** — `python -m pytest` from the repo root (requires `pip install -e ".[dev]"`).
5. **Open a pull request** describing what changed and why.

## Guidelines

- Prefer small, reviewable changes over large mixed diffs.
- If you change the modeling pipeline, note any impact on metrics or reproducibility in the PR.
- Run `python -m pytest` before opening a PR when Python code under `src/` or `tests/` changes.
- Do not commit secrets, API keys, or personal resume data.

## Questions

Open a [GitHub issue](https://github.com/Vedv7/Resume-Screening-using-Text-classification/issues) for bugs, ideas, or discussion.
