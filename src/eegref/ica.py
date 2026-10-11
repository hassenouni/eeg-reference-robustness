"""Mandatory window-wise ICA with explicit, conservative artifact heuristics.

These heuristics flag candidates; they are not ICLabel or a clinical adjudication.
No class label or prediction score is used to fit or select components.
"""
import warnings

import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def component_scores(sources, mixing, data, names, fs, settings):
    freq, power = signal.welch(sources, fs=fs, window='hann', nperseg=2*fs,
                               noverlap=fs, axis=-1)
    total = power[:, (freq >= 1) & (freq <= 45)].sum(axis=-1)
    low = power[:, (freq >= 1) & (freq <= 4)].sum(axis=-1)/np.maximum(total, 1e-30)
    high = power[:, (freq >= 30) & (freq <= 45)].sum(axis=-1)/np.maximum(total, 1e-30)
    frontal = [i for i, name in enumerate(names) if name.startswith(('FP1', 'FP2'))]
    energy = mixing**2
    normalized = energy/np.maximum(energy.sum(axis=0), 1e-30)
    frontal_fraction = normalized[frontal].sum(axis=0)
    focal_fraction = normalized.max(axis=0)
    # Frontal EEG proxies, not recorded EOG; require spectral and spatial agreement.
    band = signal.butter(2, [1, 4], btype='bandpass', fs=fs, output='sos')
    source_low = signal.sosfiltfilt(band, sources, axis=-1)
    proxy = signal.sosfiltfilt(band, data[frontal], axis=-1)
    def normalize(x):
        x = x-x.mean(axis=-1, keepdims=True)
        return x/np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-30)
    correlation = np.abs(normalize(source_low) @ normalize(proxy).T).max(axis=-1)
    rows = []
    for k in range(len(sources)):
        ocular = (low[k] >= settings['ocular_low_fraction'] and
                  frontal_fraction[k] >= settings['ocular_frontal_fraction'] and
                  correlation[k] >= settings['ocular_proxy_correlation'])
        muscle = (high[k] >= settings['muscle_high_fraction'] and
                  focal_fraction[k] >= settings['muscle_focal_fraction'])
        rows.append(dict(component=k, low_fraction=float(low[k]), high_fraction=float(high[k]),
                         frontal_fraction=float(frontal_fraction[k]), focal_fraction=float(focal_fraction[k]),
                         frontal_proxy_correlation=float(correlation[k]),
                         excluded=bool(ocular or muscle),
                         reason=';'.join(name for flag, name in [(ocular, 'ocular_candidate'), (muscle, 'muscle_candidate')] if flag)))
    return rows


def clean_ica(values, names, config):
    if config.get('ica_mode') != 'fastica_required':
        raise ValueError('ICA is mandatory: use ica_mode=fastica_required')
    settings = config['ica']
    if not np.isfinite(values).all() or np.any(values.std(axis=-1) < 1e-12):
        raise ValueError('ICA input contains nonfinite or flat channels')
    if config['fs'] != 256 or settings['fit_fs'] != 128:
        raise ValueError('This ICA protocol requires input 256 Hz and fit 128 Hz')
    # Anti-aliased fitting copy; removal is applied to the original 256-Hz window.
    fit_data = signal.resample_poly(values, 1, 2, axis=-1)*1e6
    eigenvalues = np.linalg.eigvalsh(np.cov(fit_data))
    rank = int(np.sum(eigenvalues > eigenvalues[-1]*settings['rank_relative_eigenvalue']))
    if rank < 2:
        raise ValueError('Insufficient rank for ICA')
    ica = FastICA(n_components=rank, algorithm='parallel', whiten='unit-variance',
                  whiten_solver='eigh', fun='logcosh', random_state=config['seed'],
                  max_iter=settings['max_iter'], tol=settings['tol'])
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        source_fit = ica.fit_transform(fit_data.T).T
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise ValueError(f'ICA did not converge within {settings["max_iter"]} iterations; no uncleaned fallback')
    scores = component_scores(source_fit, ica.mixing_, fit_data, names, settings['fit_fs'], settings)
    excluded = [r['component'] for r in scores if r['excluded']]
    if len(excluded) >= rank:
        raise ValueError('All ICA components flagged; window rejected')
    uv = values.T*1e6
    cleaned = uv.copy()
    if excluded:
        # Subtraction retains any PCA residual and the original channel mean.
        sources = ica.transform(uv)
        cleaned -= sources[:, excluded] @ ica.mixing_[:, excluded].T
    cleaned = cleaned.T/1e6
    if not np.isfinite(cleaned).all():
        raise ValueError('Nonfinite ICA reconstruction')
    info = dict(method='FastICA', fit_fs=settings['fit_fs'], apply_fs=config['fs'],
                fit_scope='single_300s_window', rank=rank, n_iter=int(ica.n_iter_),
                excluded=excluded, component_scores=scores, channel_names=names,
                removed_energy_fraction=float(np.sum((values-cleaned)**2)/np.sum(values**2)),
                parameters=settings, seed=config['seed'],
                selection='automatic_heuristic_not_manually_reviewed')
    arrays = dict(ica_mixing=ica.mixing_, ica_unmixing=ica.components_, ica_mean=ica.mean_)
    return cleaned, info, arrays
