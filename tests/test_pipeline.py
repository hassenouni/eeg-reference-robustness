import json
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from eegref.evaluation import bh, inference, model
from eegref.features import connectivity, epochs, graph_features
from eegref.manifest import canonical, clock_value, overlaps


@pytest.fixture
def config():
    return json.loads((Path(__file__).parents[1] / 'config' / 'study.json').read_text())


def test_epoch_count_and_flat_rejection(config):
    x = np.random.default_rng(12).normal(size=(3, 76800))
    assert epochs(x, config).shape == (119, 3, 1280)
    x[1] = 0
    with pytest.raises(ValueError, match='Flat'):
        epochs(x, config)


def test_cc_and_coherence_against_scipy(config):
    x = np.random.default_rng(4).normal(size=(2, 3, 1280))
    actual = connectivity(x, config)
    for epoch in range(2):
        a, b = x[epoch, :2]-x[epoch, :2].mean(axis=-1, keepdims=True)
        c = signal.correlate(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        mid = len(a)-1
        assert actual['CC'][epoch, 0, 1] == pytest.approx(np.abs(c[mid-128:mid+129]).max())
        assert actual['corCC'][epoch, 0, 1] == pytest.approx(np.abs(c[mid+1:mid+129]-c[mid-1:mid-129:-1]).max())
        f, coh = signal.coherence(a, b, fs=256, window='hann', nperseg=256, noverlap=128)
        assert actual['COH'][epoch, 0, 1] == pytest.approx(coh[(f >= 1) & (f <= 45)].mean())
    for matrix in actual.values():
        np.testing.assert_allclose(matrix, matrix.transpose(0, 2, 1))
        assert np.all(np.diagonal(matrix, axis1=1, axis2=2) == 0)
    # Multiple observations prevent the erroneous coherence == 1 of a single FFT.
    assert actual['COH'][0, 0, 1] < .5


def test_identical_channels_have_zero_corrected_correlation(config):
    x = np.random.default_rng(42).normal(size=1280)
    m = connectivity(np.stack([x, x])[None], config)
    assert m['CC'][0, 0, 1] == pytest.approx(1)
    assert m['corCC'][0, 0, 1] < 1e-12
    assert m['COH'][0, 0, 1] == pytest.approx(1)
    assert m['wPLI'][0, 0, 1] == 0


def test_complete_and_disconnected_graphs():
    complete = (np.ones((4, 4))-np.eye(4))[None]
    np.testing.assert_allclose(graph_features(complete)[0], [3, 1, 1, 1, 1])
    np.testing.assert_allclose(graph_features(complete*.125)[0], [.375, .125, 8, .125, .125])
    np.testing.assert_allclose(graph_features(np.zeros_like(complete)), 0)
    line = np.zeros((1, 3, 3))
    line[0, 0, 1] = line[0, 1, 0] = line[0, 1, 2] = line[0, 2, 1] = 1
    np.testing.assert_allclose(graph_features(line)[0], [4/3, 0, 4/3, 5/6, 0])


def test_train_only_scaler():
    x, y = np.array([[0.], [1.], [2.], [3.]]), [0, 0, 1, 1]
    fit = model().fit(x, y)
    before = fit[0].mean_.copy()
    fit.decision_function([[1e8]])
    np.testing.assert_equal(fit[0].mean_, before)
    assert before[0] == 1.5


def test_statistics():
    np.testing.assert_allclose(bh([.01, .04, .03, .2]), [.04, .053333333333, .053333333333, .2])
    assert inference([0, 0, 0], 'wilcoxon')['p'] == 1
    assert inference([.2, .3, .4])['ci_low'] > 0


def test_annotation_utilities():
    assert canonical('EEG T3-T5') == 'T7-P7'
    assert clock_value('07:53.17') == 28397
    with pytest.raises(ValueError, match='ambiguous'):
        clock_value('11.41.04 opure 11.40.43')
    assert not overlaps(0, 300, 300, 400)
    assert overlaps(0, 300, 299, 400)
