"""Epoch estimators and vectorized graph statistics; no fitted transformations."""
import numpy as np
from scipy import fft, signal

METHODS = ('CC', 'COH', 'corCC', 'wPLI')
GRAPH_NAMES = ('strength', 'clustering', 'path_length', 'global_efficiency', 'local_efficiency')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45.00001))


def epochs(data, config):
    size = round(config['epoch_s']*config['fs'])
    step = round(config['step_s']*config['fs'])
    expected = round(config['window_s']*config['fs'])
    if data.ndim != 2 or data.shape[1] != expected or not np.isfinite(data).all():
        raise ValueError('Expected a finite, complete analysis window')
    x = np.lib.stride_tricks.sliding_window_view(data, size, axis=-1)[:, ::step, :].transpose(1, 0, 2).copy()
    if np.any(x.std(axis=-1) < 1e-12):
        raise ValueError('Flat channel in an epoch; connectivity is undefined')
    return x


def spectral(x, fs):
    f, psd = signal.periodogram(x, fs=fs, window='hann', detrend='constant', axis=-1)
    powers = np.stack([psd[..., (f >= a) & (f < b)].sum(axis=-1)*(f[1]-f[0]) for a, b in BANDS], axis=-1)
    relative = powers / np.maximum(powers.sum(axis=-1, keepdims=True), np.finfo(float).tiny)
    # Preserve channel identities in the shared 18-derivation montage.
    return np.concatenate([np.log10(np.maximum(powers, np.finfo(float).tiny)), relative], axis=-1).mean(axis=0).ravel()


def connectivity(x, config):
    e, n, samples = x.shape
    pairs = np.triu_indices(n, 1)
    centered = x-x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1)
    z = centered/norm[..., None]
    lag = round(config['max_lag_s']*config['fs'])
    if not 0 < lag < samples:
        raise ValueError('Invalid correlation lag')
    # Full-epoch energy normalization, with no per-lag normalization.
    size = fft.next_fast_len(2*samples-1)
    transform = fft.rfft(z, size, axis=-1)
    cc, corrected = np.empty((e, len(pairs[0]))), np.empty((e, len(pairs[0])))
    # Pair chunks bound peak memory for 119 epochs.
    for first in range(0, len(pairs[0]), 16):
        i, j = pairs[0][first:first+16], pairs[1][first:first+16]
        corr = fft.irfft(transform[:, i]*transform[:, j].conj(), size, axis=-1)
        pos, neg = corr[..., 1:lag+1], corr[..., -1:-lag-1:-1]
        cc[:, first:first+16] = np.maximum(np.abs(corr[..., :lag+1]).max(axis=-1), np.abs(neg).max(axis=-1))
        corrected[:, first:first+16] = np.abs(pos-neg).max(axis=-1)
    seg = round(config['welch_segment_s']*config['fs'])
    step = round(seg*(1-config['welch_overlap']))
    if seg >= samples or step < 1:
        raise ValueError('Welch requires multiple subsegments per epoch')
    sub = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::step, :]
    sub = (sub-sub.mean(axis=-1, keepdims=True))*signal.windows.hann(seg, sym=False)
    freq = fft.rfftfreq(seg, 1/config['fs'])
    keep = (freq >= config['band_hz'][0]) & (freq <= config['band_hz'][1])
    ft = fft.rfft(sub, axis=-1)[..., keep]
    auto = np.mean(np.abs(ft)**2, axis=-2)
    coh, wpli = np.empty_like(cc), np.empty_like(cc)
    for first in range(0, len(pairs[0]), 16):
        i, j = pairs[0][first:first+16], pairs[1][first:first+16]
        cross = ft[:, i].conj()*ft[:, j]
        denom = auto[:, i]*auto[:, j]
        c = np.divide(np.abs(cross.mean(axis=-2))**2, denom, out=np.zeros_like(denom), where=denom > 0)
        coh[:, first:first+16] = np.clip(c, 0, 1).mean(axis=-1)
        # Evaluate Im(conj(X)*Y) explicitly: complex multiplication can leave
        # spurious imaginary roundoff for identical signals on SIMD/FMA builds.
        # Equal real products subtract exactly, preserving the zero-lag null.
        imag = ft[:, i].real*ft[:, j].imag - ft[:, i].imag*ft[:, j].real
        denom = np.abs(imag).mean(axis=-2)
        # Mean over Welch subsegments per frequency, then broadband mean.
        p = np.divide(np.abs(imag.mean(axis=-2)), denom, out=np.zeros_like(denom), where=denom > 0)
        wpli[:, first:first+16] = np.clip(p, 0, 1).mean(axis=-1)
    matrices = {}
    for name, values in zip(METHODS, (cc, coh, corrected, wpli)):
        matrix = np.zeros((e, n, n))
        matrix[:, pairs[0], pairs[1]] = values
        matrix[:, pairs[1], pairs[0]] = values
        matrices[name] = matrix
    return matrices


def distances(weights, power=1):
    d = np.full_like(weights, np.inf)
    np.divide(1, weights**power, out=d, where=weights > 0)
    ii = np.arange(weights.shape[-1])
    d[..., ii, ii] = 0
    for k in ii:
        d = np.minimum(d, d[..., :, k, None]+d[..., None, k, :])
    return d


def graph_features(weights):
    """Strength, Onnela clustering, finite-pair path mean, efficiencies.

    Local efficiency uses the weighted neighborhood definition with cube-root
    lengths (BCT modified variant). Disconnected pairs contribute zero efficiency.
    Path length is over reachable off-diagonal pairs, with zero for an empty graph.
    corCC remains in its original scale (up to 2); no per-graph max normalization.
    """
    w = np.asarray(weights, dtype=float)
    n = w.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    strength = w.sum(axis=-1).mean(axis=-1)
    adj = w > 0
    degree = adj.sum(axis=-1)
    denominator = degree*(degree-1)
    cube = np.cbrt(w)
    triangles = np.einsum('...ij,...jk,...ki->...i', cube, cube, cube, optimize=True)
    clustering = np.divide(triangles, denominator, out=np.zeros_like(triangles), where=denominator > 0).mean(axis=-1)
    d = distances(w)
    finite = np.isfinite(d) & mask
    lengths = np.where(finite, d, 0).sum(axis=(-1, -2))/np.maximum(finite.sum(axis=(-1, -2)), 1)
    inv = np.divide(1, d, out=np.zeros_like(d), where=(d > 0) & np.isfinite(d))
    global_eff = inv.sum(axis=(-1, -2))/(n*(n-1))
    local = np.zeros(w.shape[:-1])
    for u in range(n):
        # Removing all non-neighbors prevents paths leaving the induced subgraph.
        neighbors = adj[..., u, :]
        subgraph = w * (neighbors[..., :, None] & neighbors[..., None, :])
        nd = distances(subgraph, power=1/3)
        inverse = np.divide(1, nd, out=np.zeros_like(nd), where=(nd > 0) & np.isfinite(nd))
        numerator = (cube[..., u, :, None]*cube[..., u, None, :]*inverse).sum(axis=(-1, -2))
        local[..., u] = np.divide(numerator, denominator[..., u], out=np.zeros_like(numerator), where=denominator[..., u] > 0)
    return np.stack([strength, clustering, lengths, global_eff, local.mean(axis=-1)], axis=-1)


def extract(data, config):
    x = epochs(data, config)
    matrices = connectivity(x, config)
    result = {'spectral': spectral(x, config['fs']), 'n_epochs': np.array(len(x))}
    for method, matrix in matrices.items():
        result[method] = graph_features(matrix).mean(axis=0)
        result[method+'_edges'] = matrix.mean(axis=0)[np.triu_indices(matrix.shape[-1], 1)]
    return result
