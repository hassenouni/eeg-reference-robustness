# EEG processing and analysis pipeline

This document follows the methods in **Reference Robustness and Cross-Patient Generalization of Preictal Scalp-EEG Connectivity**, by Hassen Ouni and Lotfi Saidi. Numerical definitions are detailed in [METHODS.md](METHODS.md); individual functions are mapped in [PAPER_TO_CODE.md](PAPER_TO_CODE.md).

## 1. Source audit and patient selection

The audit reads EDF headers and original seizure annotations. It validates duration, channels, source integrity, and seizure intervals before feature extraction. Technical availability determines inclusion; connectivity estimates and classification scores play no role in selection. CHB-MIT sessions chb01 and chb21 form one independent patient.

**Interface:** original recordings and annotations -> inventory, cohort ranking, selected patients, and window manifest. Implementation: `manifest.build`.

## 2. Annotated five-minute windows

Preictal windows occupy the 300 seconds immediately before a seizure onset and exclude ictal samples. Interictal candidates are nonoverlapping 300-second windows outside a one-hour margin before onset and after termination. Up to four controls are sampled per preictal window using the fixed seed. Adjacent recording annotations are considered on the recording-case timeline. A window never crosses an EDF boundary.

**Interface:** source-relative path, patient and recording case, start/stop/context seconds, label, and reference-experiment flag. Implementation: `manifest.build`.

## 3. Sample processing and ICA

Load physical EEG samples in volts, harmonize channel labels, and resample to 256 Hz. Apply the 1-45 Hz Butterworth band-pass and 50 Hz notch in forward and reverse directions. Include at most ten seconds of preceding non-ictal context, then remove it before ICA. Source samples beyond the window endpoint are never loaded.

Fit rank-aware FastICA on a 128 Hz copy of each window. Fixed ocular and muscle criteria select components for subtraction from the 256 Hz signal. Preserve the channel mean and PCA residual. Non-convergence and invalid windows are explicitly excluded. Each window has an ICA audit record.

CHB-MIT ICA operates on 18 native bipolar derivations. Siena ICA operates on 19 electrodes before constructing the prediction montage and the two reference conditions.

**Interface:** cleaned 300-second arrays and ICA diagnostics. Implementation: `signals.load_window`, `ica.clean_ica`, `ica.component_scores`.

## 4. Shared prediction montage and epochs

Prediction uses the same ordered set of 18 bipolar derivations in both cohorts. Divide each window into 119 epochs of five seconds with a 2.5-second step. All epoch measurements remain grouped within their original window and patient.

**Interface:** epoch array with shape `(119, 18, 1280)`. Implementation: `features.epochs`.

## 5. Spectral and connectivity representations

The spectral baseline comprises five frequency bands, log10 absolute power and relative power, for each of 18 signals: 180 features. Apply the logarithm before averaging across epochs.

Compute CC, corCC, COH, and classical unsquared wPLI on the same epoch signals. CC uses the maximum absolute normalized cross-correlation over +/-0.5 seconds. corCC uses positive-lag asymmetry. COH and wPLI use nine overlapping one-second Hann subsegments, estimate values per frequency, then average over 1-45 Hz.

Each connectivity estimator produces a symmetric weighted 18-node network with 153 unique edges. Extract mean strength, weighted clustering, characteristic path length, global efficiency, and local efficiency. Compute graph features per epoch and then average over the 119 epochs.

**Interface:** one row per successful window: 180 spectral features and four separate sets of five graph features. Each classification comparison uses 180 or 185 inputs. Implementation: `features.spectral`, `features.connectivity`, `features.graph_features`, `features.extract`.

## 6. Controlled reference robustness

From the same cleaned Siena window, construct common-average reference over 19 electrodes and Cz reference. Remove Cz in both conditions. Compare corresponding off-diagonal edge ranks using Spearman similarity and compare mean strength using symmetric relative change. Summarize eight matched windows per patient, then aggregate patient summaries.

**Interface:** window metrics, patient means, and cohort mean/sample SD. Implementation: `workers.worker`, `evaluation.reference_robustness`.

## 7. Preictal-interictal differences

For each connectivity measure, compute each patient's mean preictal strength minus mean interictal strength. Give every patient equal weight. Report mean differences, nominal Student t intervals, and multiplicity-adjusted tests separately for each cohort.

**Interface:** patient differences and method-level summaries. Implementation: `evaluation.evaluate`, `evaluation.inference`, `evaluation.bh`.

## 8. Patient-independent and external evaluation

Use a linear SVM with C=1 and a StandardScaler fitted on training data only. Compare the spectral baseline with four models that each add one measure's five graph features.

In LOPO, every window belonging to the held-out patient stays outside scaler and classifier fitting. Score each held-out patient using AUROC. In external transfer, fit the source cohort once and apply the scaler and model unchanged to the target cohort, in both directions. Evaluate target patients with average precision. No target-label adaptation is performed.

**Interface:** held-out window decisions, patient scores, and paired gains over the same spectral baseline. Implementation: `evaluation.model`, `evaluation.evaluate`.

## 9. Patient-level inference, sensitivity, and exports

Use two-sided paired t-tests for LOPO gains and Wilcoxon signed-rank tests for transfer gains. Apply BH-FDR within the stated method/dataset/direction families. Compare corCC/wPLI with CC/COH through a paired family contrast. Confidence intervals describe mean patient differences.

Sensitivity excludes one previously scored patient at a time without retraining. Reporting functions produce the tables and figures from computed artifacts. Source data, model fitting, statistical analysis, and figure export remain separate stages.

**Interface:** patient-level contrasts, omission summaries, CSV tables, and PNG/SVG figures. Implementation: `evaluation.py`, `sensitivity.py`, `reporting.py`.

## Inspect or execute by stage

```bash
python -m eegref audit --data-root /path/to/data --output outputs/study
python -m eegref extract --data-root /path/to/data --output outputs/study --workers 2
python -m eegref evaluate --output outputs/study
python -m eegref report --output outputs/study
```

The first two commands perform no classifier training. The last command reads saved artifacts and performs no training. For exact output schemas and cache behavior, see [WORKFLOW.md](WORKFLOW.md).
