# Technical validation

The test suite exercises the numerical implementation with generated signals and known reference identities. It does not run a full EDF cohort or establish clinical performance.

```bash
python -m pytest -q
```

Checks cover full-window epoch counts, flat-signal rejection, CC against SciPy's direct correlation, corCC lag asymmetry, coherence against SciPy's independent estimator, wPLI against an explicit segment calculation, known weighted graph identities, train-only scaling, patient separation, statistics, annotation guards, montage construction, ICA criteria and reconstruction, and sensitivity without retraining.

The CI workflow is configured for Windows/Linux and Python 3.10/3.11. A local test run is distinct from a successful remote GitHub CI run. No CI badge is asserted before publication.

`requirements.txt` records a pinned reference numerical environment. Package version numbers, configuration, implementation SHA-256, and input sample fingerprints are recorded by actual extraction runs. Small floating-point differences can depend on numerical libraries and hardware.

## Local verification record

Verified on 2026-10-11 with Windows and Python 3.10.11 using the numerical versions pinned in `requirements.txt`:

- Editable package installation completed.
- `python -m pytest -q`: **20 passed**, no warnings.
- Installed command-line entry point and `python -m eegref --help` completed.
- Synthetic demonstration returned 119 epochs, 180 spectral features, and five graph features for each of the four connectivity measures.
- No full source-EEG cohort extraction or study classifier training was executed for this package.
