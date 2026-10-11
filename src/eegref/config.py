"""Validate the fixed manuscript protocol before any source data are read."""
import json
from pathlib import Path


def load_config(path, data_root=None):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1, "seed": 20260930, "fs": 256, "window_s": 300,
        "epoch_s": 5, "step_s": 2.5, "band_hz": [1, 45], "butter_order": 2,
        "notch_hz": 50, "notch_q": 30, "filter_context_s": 10,
        "max_lag_s": 0.5, "welch_segment_s": 1, "welch_overlap": 0.5,
        "interictal_guard_s": 3600, "interictal_per_preictal": 4,
        "robustness_windows": 8, "ica_mode": "fastica_required",
        "cohort_sizes": {"CHB-MIT": 18, "Siena": 10},
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"Manuscript protocol requires {key}={expected!r}")
    expected_ica = {
        "fit_fs": 128, "rank_relative_eigenvalue": 1e-8, "max_iter": 2000,
        "tol": 0.001, "ocular_low_fraction": 0.6, "ocular_frontal_fraction": 0.6,
        "ocular_proxy_correlation": 0.6, "muscle_high_fraction": 0.6,
        "muscle_focal_fraction": 0.35,
    }
    if config.get("ica") != expected_ica:
        raise ValueError("ICA settings differ from the manuscript protocol")
    if not isinstance(config.get("annotation_overrides", {}), dict):
        raise ValueError("annotation_overrides must map relative EDF paths to intervals")
    root = Path(data_root if data_root is not None else config["data_root"])
    # Relative roots are relative to the invocation directory, not the config file.
    config["data_root"] = str(root.resolve())
    return config
