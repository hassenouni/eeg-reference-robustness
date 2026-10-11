# Numerical methods and implementation conventions

## Samples, montage, and filtering

MNE reads EDF physical samples in volts. Historical electrode names T3/T4/T5/T6 map to T7/T8/P7/P8. Prediction uses these ordered derivations:

```text
FP1-F7 F7-T7 T7-P7 P7-O1 FP1-F3 F3-C3 C3-P3 P3-O1
FP2-F4 F4-C4 C4-P4 P4-O2 FP2-F8 F8-T8 T8-P8 P8-O2 FZ-CZ CZ-PZ
```

CHB-MIT uses native bipolar signals, reversing sign if only the reverse derivation exists. Siena uses 19 electrode signals, cleaned before subtracting electrodes to construct the same bipolar montage. No referential potentials are reconstructed from CHB-MIT bipolar signals.

Polyphase resampling produces 256 Hz data. `scipy.signal.butter(N=2, [1,45], btype='bandpass', output='sos')` produces four poles per forward pass; `sosfiltfilt` applies it forward and backward. The 50 Hz IIR notch has Q=30 and also uses forward-backward filtering. At most 10 s of preceding non-ictal context is included and then cropped. No source samples after the window endpoint are loaded. Edge reflection is the SciPy default. This is an offline zero-phase method.

## Mandatory window-wise ICA

Fit FastICA on an anti-aliased 128 Hz copy, in microvolt numerical units. Apply component subtraction to the filtered 256 Hz signal. Retain covariance eigenvalues greater than `1e-8` times the largest eigenvalue to determine rank. Reject rank below two.

FastICA uses `parallel`, `logcosh`, `unit-variance` whitening, `eigh`, seed 20260930, tolerance 0.001, and maximum 2000 iterations.

Welch component powers use 2 s Hann segments and 1 s overlap, normalized by 1-45 Hz power. An ocular component must satisfy all three thresholds: 1-4 Hz power fraction >=0.60, frontopolar mixing-energy fraction >=0.60, and maximum absolute low-band correlation with a frontopolar EEG channel >=0.60. A muscle component must satisfy both thresholds: 30-45 Hz power fraction >=0.60 and maximum single-channel mixing-energy fraction >=0.35. Frequency endpoints are included in these artifact bands. The frontopolar channels are EEG proxies.

Subtract only rejected-component reconstructions, preserving the channel mean and PCA residual. Reject non-convergence, flat or nonfinite inputs, insufficient rank, or rejection of every component. There is no bypass without ICA. Audit files record scores, criteria, rank, iterations, component indices, and correction-energy fraction. The cache stores mixing, unmixing, and ICA mean arrays.

## Epochs and spectral features

A 300 s window contains `floor((300-5)/2.5)+1 = 119` complete epochs. The Hann periodogram on 1280 samples has 0.2 Hz frequency resolution. Bands are `[1,4)`, `[4,8)`, `[8,13)`, `[13,30)`, and `[30,45]` Hz.

Band power is the sum of periodogram density bins multiplied by 0.2 Hz. Relative powers divide by total power across these five bands. The log10 transform is applied to each epoch's absolute band power **before** epoch averaging. Numerical zeros use the smallest positive float as the log/division floor.

Feature ordering is channel-major: five log-power bands followed by five relative-power bands for each of the 18 derivations. `spectral_000` through `spectral_179` follow the montage order above.

## Connectivity

For each pair of epoch signals, remove the temporal mean and normalize by full-epoch Euclidean energy. Let `r_ij(lag)` be the resulting normalized linear cross-correlation.

- **CC:** maximum absolute `r_ij(lag)` for lags -128 through +128 samples, including zero.
- **corCC:** maximum absolute `r_ij(lag) - r_ij(-lag)` for positive lags 1 through 128. No factor of one-half and no per-graph maximum normalization are applied. The edge is stored symmetrically and is not interpreted as directed coupling.

For frequency estimators, each 5 s epoch contains nine 1 s segments, with 0.5 s step. Subtract each segment's mean, multiply by a periodic Hann window, and apply a 256-point real FFT. Frequency bins are spaced by 1 Hz. Let `Z_k(f)=conj(X_i,k(f))*X_j,k(f)` and average over the nine segments:

- **COH:** `abs(mean(Z_k))**2 / (mean(abs(X_i,k)**2)*mean(abs(X_j,k)**2))`.
- **wPLI:** `abs(mean(imag(Z_k))) / mean(abs(imag(Z_k)))`.

Use zero when a denominator is zero. wPLI is classical, unsquared, and not debiased. Compute each estimator per frequency, then take the arithmetic mean over 45 bins from 1 through 45 Hz. The sign convention for the cross-spectrum does not change either magnitude-based estimator.

## Five weighted graph features

Networks are symmetric with zero diagonal. No thresholding is applied. Feature order is strength, clustering, characteristic path length, global efficiency, local efficiency. Compute graph features for each epoch, then average across epochs. In general this differs from computing features once on the average adjacency matrix.

For an n-node network W, let binary adjacency A mark positive weights and let `k_i=sum_j A_ij`.

1. **Mean strength:** `mean_i(sum_j W_ij)`. This is node strength, not mean edge weight.
2. **Weighted clustering:** for each node, sum the ordered neighbor triangles `(W_ij*W_jk*W_ki)**(1/3)` and divide by `k_i*(k_i-1)`, then average over nodes. Nodes of degree below two contribute zero. Weights remain on their native scale.
3. **Characteristic path length:** edge lengths are `1/W_ij`; absent edges have infinite length. Floyd-Warshall finds shortest paths. Average only reachable off-diagonal ordered pairs; an empty graph returns zero.
4. **Global efficiency:** average inverse shortest-path lengths over all `n*(n-1)` ordered pairs, with disconnected pairs contributing zero.
5. **Local efficiency:** use the induced graph of each node's neighbors. In that subgraph, edge lengths are `1/W_jk**(1/3)`. Multiply each inverse shortest path by `(W_ij*W_ik)**(1/3)`, sum ordered neighbor pairs, and divide by `k_i*(k_i-1)`. Average over nodes. This is the modified weighted neighborhood convention used in this implementation.

The manuscript names the graph measures without uniquely specifying all disconnected-path and local-efficiency conventions; the definitions above make the implementation explicit and inspectable.

## Controlled reference experiment

In Siena, compute CAR over all 19 cleaned electrodes. Construct Cz-referenced signals from the **same** cleaned window. Exclude Cz in both conditions, yielding 18 nodes and 153 unique pairs. Use eight selected windows per patient.

For each window, average adjacency edges across epochs. Compute Spearman correlation of the 153 matched edges. Compute symmetric relative strength change as `abs(S_CAR-S_Cz)/(0.5*(S_CAR+S_Cz))`. Undefined constant-edge correlations or zero-strength comparisons are rejected. Average window metrics within patient, then report mean and sample SD across patients. A patient needs all eight successful robustness windows.

## Classification and patient-level inference

The baseline contains 180 spectral features. Each connectivity model adds its own five graph features, yielding 185 inputs. The classifier is `Pipeline(StandardScaler(), SVC(kernel='linear', C=1, probability=False, class_weight=None))`. No hyperparameter search is performed.

LOPO excludes every window of the held-out patient from scaler and SVM fitting. External transfer fits the source cohort only. Evaluate continuous SVM decision scores separately for each target patient. AUROC is primary for LOPO and average precision is primary for transfer. Target labels are used only for evaluation. ICA remains fixed-rule, local, and unsupervised.

Mean paired preictal strength differences give every patient equal weight. Patient-wise gains compare each connectivity model with the same patient's spectral baseline. The family contrast is `mean(delta_corCC, delta_wPLI) - mean(delta_CC, delta_COH)` per patient.

Mean differences use nominal 95% Student t intervals with `n-1` degrees of freedom and sample SD. LOPO and preictal differences use two-sided one-sample t-tests of paired differences. Transfer gains and family contrasts use two-sided Wilcoxon signed-rank tests (`zero_method='wilcox'`, SciPy `method='auto'`). An all-zero difference vector has p=1.

BH-FDR families are: four methods within each dataset for preictal differences, four within each dataset for LOPO, four within each direction for transfer, and the two directions for the family contrast. Intervals are not adjusted for multiplicity. Sensitivity omits one saved patient score at a time, without retraining; its p-values are nominal and exploratory.
