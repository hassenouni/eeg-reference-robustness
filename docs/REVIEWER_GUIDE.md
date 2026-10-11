# Reviewer guide

This repository is organized for inspection of the methods accompanying **Reference Robustness and Cross-Patient Generalization of Preictal Scalp-EEG Connectivity** (Hassen Ouni and Lotfi Saidi).

## Suggested reading order

1. Read `config/study.json` for the numerical protocol.
2. Read `manifest.py` for annotation validation, patient grouping, cohort ranking, and window selection.
3. Read `signals.py` and `ica.py` for sample loading, filtering, ICA, and montage construction.
4. Read `features.py` for spectral, connectivity, and weighted graph calculations.
5. Read `workers.py` for window processing, caching, and the matched CAR/Cz comparison.
6. Read `evaluation.py` for training-only scaling, patient separation, metrics, and multiplicity correction.
7. Read `sensitivity.py` for patient omission on previously saved scores without retraining.

All implementation files are in `src/eegref/`. [PAPER_TO_CODE.md](PAPER_TO_CODE.md) links the individual functions.

## Inspect without study execution

```bash
python -m pytest -q
python scripts/synthetic_demo.py
python -m eegref --help
```

The demonstration uses generated EEG-like signals, does not access EDF files, does not fit a classifier, and writes only a technical feature summary. The tests use small synthetic arrays and known numerical identities.

## Inspect data selection without training

```bash
python -m eegref audit --data-root /path/to/data --output outputs/audit
```

Review `cohort_ranking.csv`, `cohort.csv`, `audit_issues.csv`, and `windows.csv`. The `extract` stage adds ICA audit records and feature tables without training. Evaluation is a separate explicit command.

## Scope

This is a methods implementation package. It includes neither raw EEG nor precomputed manuscript performance tables. Source EEG stays read-only. The patient is the inference unit; overlapping epochs only contribute to within-window feature estimates. Fixed unsupervised ICA is fitted locally within each window, including held-out windows. Scalers and classifiers are fitted using training patients or source data exclusively.
