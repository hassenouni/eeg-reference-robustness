import json
from pathlib import Path

import numpy as np
import pytest

from eegref.ica import clean_ica, component_scores


def test_ica_mandatory_and_flat_rejected():
    config = json.loads((Path(__file__).parents[1] / 'config' / 'study.json').read_text())
    with pytest.raises(ValueError, match='mandatory'):
        clean_ica(np.ones((3, 512)), ['FP1', 'FP2', 'CZ'], dict(config, ica_mode='none_documented'))
    with pytest.raises(ValueError, match='flat'):
        clean_ica(np.ones((3, 512)), ['FP1', 'FP2', 'CZ'], config)


def test_known_artifact_is_removed_and_audited():
    config = json.loads((Path(__file__).parents[1] / 'config' / 'study.json').read_text())
    rng = np.random.default_rng(18)
    t = np.arange(256*32)/256
    blink = sum(np.exp(-((t-p)/.12)**2) for p in [3, 7, 13, 22, 29])
    muscle = np.sin(2*np.pi*36*t)*(1+.3*np.sin(2*np.pi*.5*t))
    brain = rng.laplace(size=len(t))
    sources = np.stack([blink*8, muscle, brain])
    mix = np.array([[2., .03, .1], [1.8, .02, -.1], [.01, 1, .3]])
    values = mix@sources*1e-6
    cleaned, info, arrays = clean_ica(values, ['FP1', 'FP2', 'CZ'], config)
    assert info['rank'] == 3
    assert info['excluded']
    assert info['removed_energy_fraction'] > .1
    assert np.isfinite(cleaned).all()
    assert cleaned.shape == values.shape
    assert arrays['ica_unmixing'].shape == (3, 3)
    assert all(r['reason'] for r in info['component_scores'] if r['excluded'])
    repeated, repeat_info, _ = clean_ica(values, ['FP1', 'FP2', 'CZ'], config)
    np.testing.assert_allclose(repeated, cleaned)
    assert repeat_info['excluded'] == info['excluded']


def test_worker_has_importable_module_for_windows():
    from eegref.workers import worker
    import pickle
    assert worker.__module__ == 'eegref.workers'
    assert pickle.loads(pickle.dumps(worker)) is worker
