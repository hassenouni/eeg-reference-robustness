# Data, annotation handling, and cohort selection

Obtain the original datasets and annotations from their providers:

- [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/), DOI 10.13026/C2K01R.
- [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/), DOI 10.13026/5d4a-j060.

These are dataset references supplied with the manuscript. Downloading, access conditions, and dataset licensing remain the user's responsibility. EEG recordings are not redistributed here.

## Input layout

```text
data/
  chb01/
    chb01-summary.txt
    chb01_01.edf
    ...
  chb21/
    chb21-summary.txt
    ...
  PN01/
    Seizures-list-PN01.txt
    PN01-1.edf
    ...
```

Additional nesting above patient folders is accepted. Each EDF must be directly under its patient/case folder, and each annotation file must be in the same case folder. Use original annotation filenames. Recognized dataset folders are `chbXX` and `PNXX`. Unrelated EDF recordings must not be mixed into this root.

## Selection

EDF headers establish channel compatibility, recording duration, acquisition clock, sample rates, and file-size integrity. `chb01` and `chb21` form one independent patient for cohort selection, evaluation, and inference, while recording-case identifiers remain available. Cross-file seizure guards use the recording-case timeline; sessions remain separate acquisition timelines.

Each preictal window is `[onset-300, onset)` and must fit entirely inside its EDF without intersecting another seizure. Interictal candidates are nonoverlapping 300 s windows on a file-start grid. Exclude any candidate intersecting `[onset-3600, termination+3600)` for a known seizure, including neighboring recordings in the same case. Missing or invalid annotations are unknown, never assumed seizure-free. Their entire recording interval is blocked when evaluating neighboring windows.

Eligibility requires at least two selectable preictal and four selectable interictal windows. Siena also requires eight selectable analysis windows. Rank eligible patients by usable preictal/control sets, compatible-file fraction, available controls, and patient identifier, in that order. Retain 18 CHB-MIT and 10 Siena. Selection fails rather than silently reducing requested cohort sizes.

Controls are sampled without replacement with NumPy's seeded generator; up to four controls are selected per preictal window. Eight Siena windows are then selected without replacement for robustness. Random seed: 20260930. Sorted source traversal and patient iteration make this deterministic for the same source inventory.

## Annotation parsing

CHB-MIT summaries require matching start/end counts and the declared seizure count. Siena clock intervals are converted to seconds relative to EDF start, with midnight rollover. An explicitly marked electrical onset has priority. Known historical filename aliases are recorded in `audit_issues.csv`; ambiguous or out-of-range times lead to exclusion rather than guessed clock correction.

`annotation_overrides` in the configuration may map a relative EDF path to explicit `[onset_s, termination_s]` intervals. Any override should be accompanied by a documented reason in the user's analysis record. Overrides are included in the configuration signature.

## Generated selection records

The audit generates `cohort.csv`, `cohort_ranking.csv`, and `windows.csv` from the user's actual EDF inventory. These locally generated records are not distributed in the public source repository. The implementation never forces an archived selection onto another collection of recordings.

## Privacy and source integrity

Source recordings are opened read-only. Generated inventory files may contain original acquisition timestamps; review generated outputs before publishing them. Header hashes and file metadata identify inputs, and cache entries include SHA-256 of loaded physical samples. No absolute local path is embedded in the distributed study configuration.
