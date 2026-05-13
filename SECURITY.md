# Security policy

## Supported versions

This is a small educational / portfolio repository (**[Vedv7/Resume-Screening-using-Text-classification](https://github.com/Vedv7/Resume-Screening-using-Text-classification)**). Security fixes apply to the **default branch** (`main`) only.

## Reporting a vulnerability

If you discover a security issue (for example unsafe deserialization patterns in a future saved model format, or dependency advisories):

1. **Do not** open a public issue with exploit details.  
2. Email the repository owner via the contact options on their [GitHub profile](https://github.com/Vedv7), or use **GitHub private vulnerability reporting** if enabled on the repository.

Include steps to reproduce and, if possible, suggested mitigation.

## Dependency hygiene

- Install from `requirements.txt` in a dedicated virtual environment.  
- Run `pip audit` (or your org’s equivalent) before deploying any fork to production.
