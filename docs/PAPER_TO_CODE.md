# Manuscript-to-code traceability

Manuscript: **Reference Robustness and Cross-Patient Generalization of Preictal Scalp-EEG Connectivity**. Source document identity is recorded in `source_provenance.json`.

| Manuscript section | File and functions | Responsibility |
|---|---|---|
| 2.1, Algorithm 1 | [cli.py](../src/eegref/cli.py): `main`, `extract_study` | Separate audit, extraction, evaluation, sensitivity stages |
| 2.2 Cohort selection | [manifest.py](../src/eegref/manifest.py): `header`, `read_annotations`, `build` | EDF validation, technical ranking, 18/10 selection, chb01/chb21 identity |
| 2.3 Preprocessing | [signals.py](../src/eegref/signals.py): `load_window` | Volts, polyphase resampling, filters, prior-only context, montage |
| 2.3 ICA | [ica.py](../src/eegref/ica.py): `clean_ica`, `component_scores` | Rank-aware FastICA, automatic criteria, residual-preserving subtraction |
| 2.4 Window labels | [manifest.py](../src/eegref/manifest.py): `build`, `overlaps` | Preictal timing and seizure exclusion margins |
| 2.5.1 Spectral baseline | [features.py](../src/eegref/features.py): `epochs`, `spectral` | 119 epochs; log absolute/relative powers, 180 inputs |
| 2.5.2-2.5.3 CC/corCC | [features.py](../src/eegref/features.py): `connectivity` | Energy-normalized CC, positive-lag asymmetry |
| 2.5.4-2.5.5 COH/wPLI | [features.py](../src/eegref/features.py): `connectivity` | Nine Hann subsegments, frequency-wise estimates, 1-45 Hz average |
| 2.5.6 Weighted graphs | [features.py](../src/eegref/features.py): `distances`, `graph_features`, `extract` | Strength, clustering, path length, global/local efficiency |
| 2.6 CAR/Cz robustness | [signals.py](../src/eegref/signals.py): `load_window`; [workers.py](../src/eegref/workers.py): `worker`; [evaluation.py](../src/eegref/evaluation.py): `reference_robustness` | Same cleaned samples, 153 edge ranks, relative strength, patient aggregation |
| 2.7 Preictal differences | [evaluation.py](../src/eegref/evaluation.py): `evaluate`, `inference` | Mean paired strength differences and patient-level intervals |
| 2.8 LOPO | [evaluation.py](../src/eegref/evaluation.py): `model`, `evaluate` | Held-out patient separation, 180 versus 185 features, AUROC |
| 2.9 External transfer | [evaluation.py](../src/eegref/evaluation.py): `evaluate` | Source-only SVM/scaler, per-target-patient AP, family contrasts |
| 2.10 Statistics | [evaluation.py](../src/eegref/evaluation.py): `bh`, `inference` | Nominal t intervals, two-sided t/Wilcoxon, stated BH families |
| 2.11 Traceability | [cli.py](../src/eegref/cli.py), [workers.py](../src/eegref/workers.py) | Code/environment/config signatures, sample digests, ICA audit/cache |
| 3.6 Sensitivity method | [sensitivity.py](../src/eegref/sensitivity.py): `sensitivity_analysis` | Patient omission using stored scores, no retraining |

The pipeline uses weighted networks without binary thresholding. No model selection or threshold calibration is performed using held-out or target labels.
