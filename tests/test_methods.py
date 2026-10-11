import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import signal, stats
from eegref.config import load_config
from eegref.features import METHODS, GRAPH_NAMES, connectivity, spectral
from eegref.evaluation import evaluate, inference, validate_feature_frame
from eegref.manifest import recording_seconds
from eegref.sensitivity import sensitivity_analysis

CONFIG = Path(__file__).parents[1]/'config/study.json'


def test_fixed_config_and_relative_data_root(tmp_path):
    settings = load_config(CONFIG, tmp_path)
    assert settings['data_root'] == str(tmp_path.resolve())
    changed = json.loads(CONFIG.read_text())
    changed['fs'] = 128
    path = tmp_path/'changed.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='fs'):
        load_config(path)


def test_wpli_independent_segment_formula():
    config = json.loads(CONFIG.read_text())
    rng = np.random.default_rng(28)
    x = rng.normal(size=(1, 2, 1280))
    segments = []
    for start in range(0, 1025, 128):
        segment = x[0, :, start:start+256]
        segment = (segment-segment.mean(axis=-1, keepdims=True))*signal.windows.hann(256, sym=False)
        segments.append(np.fft.rfft(segment, axis=-1))
    fft = np.stack(segments)
    imaginary = (fft[:, 0].conj()*fft[:, 1]).imag
    denominator = abs(imaginary).mean(axis=0)
    expected = np.divide(abs(imaginary.mean(axis=0)), denominator,
                         out=np.zeros_like(denominator), where=denominator > 0)[1:46].mean()
    assert connectivity(x, config)['wPLI'][0, 0, 1] == pytest.approx(expected)


def test_spectral_log_precedes_epoch_average():
    rng = np.random.default_rng(21)
    first = rng.normal(size=(18, 1280))*1e-6
    x = np.stack([first, 10*first])
    features = spectral(x, 256).reshape(18, 10)
    base = spectral(first[None], 256).reshape(18, 10)
    # 100-fold power gives +2 log10 in one epoch, hence +1 after averaging.
    np.testing.assert_allclose(features[:, :5], base[:, :5]+1)
    np.testing.assert_allclose(features[:, 5:], base[:, 5:])
    np.testing.assert_allclose(features[:, 5:].sum(axis=1), 1)


def test_patient_interval_against_scipy():
    values = np.array([.1, .2, -.04, .3])
    actual = inference(values)
    ci = stats.t.interval(.95, len(values)-1, loc=values.mean(), scale=stats.sem(values))
    np.testing.assert_allclose([actual['ci_low'], actual['ci_high']], ci)
    assert actual['p'] == pytest.approx(stats.ttest_1samp(values, 0).pvalue)


def test_recording_timeline_is_not_host_timezone_dependent():
    assert recording_seconds('1970-01-01T00:00:00') == 0
    assert recording_seconds('2020-01-02T00:00:00')-recording_seconds('2020-01-01T23:59:59') == 1


def feature_frame():
    rows = []
    for dataset in ['CHB-MIT', 'Siena']:
        for patient in range(3):
            for label in [0, 1]:
                row = dict(dataset=dataset, patient=f'{dataset}_{patient}',
                           window_id=f'{dataset}_{patient}_{label}', label=label, n_epochs=119)
                # Deterministic numerical columns, not EEG study observations.
                row.update({f'spectral_{i:03d}': float(patient*10+label+i*.001) for i in range(180)})
                row.update({method+'_'+name: float(patient+label*(patient+1)+.01)
                            for method in METHODS for name in GRAPH_NAMES})
                rows.append(row)
    return pd.DataFrame(rows)


def test_feature_validation_rejects_duplicates_and_nonfinite():
    frame = feature_frame()
    validate_feature_frame(frame)
    with pytest.raises(ValueError, match='Duplicate'):
        validate_feature_frame(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, 'spectral_000'] = np.inf
    with pytest.raises(ValueError, match='finite'):
        validate_feature_frame(frame)


def test_validation_partitions_and_dimensions_without_svm_training(tmp_path, monkeypatch):
    import eegref.evaluation as module
    frame = feature_frame()
    calls = []
    class SpyModel:
        def fit(self, x, y):
            self.train_markers = set(np.floor(x[:, 0]).astype(int))
            self.dimension = x.shape[1]
            return self
        def decision_function(self, x):
            calls.append((self.dimension, self.train_markers, set(np.floor(x[:, 0]).astype(int))))
            return x[:, 0]
    monkeypatch.setattr(module, 'model', SpyModel)
    monkeypatch.setattr(module, 'figures', lambda *args: None)
    evaluate(frame, tmp_path)
    assert len(calls) == 40  # (6 LOPO folds + 2 transfer directions) * 5 models
    assert {dimension for dimension, _, _ in calls} == {180, 185}
    # First 30 calls are LOPO; feature markers identify patient-specific ranges.
    for _, train, test in calls[:30]:
        assert train.isdisjoint(test)
    scores = pd.read_csv(tmp_path/'patient_scores.csv')
    assert len(scores) == 60
    assert scores.groupby(['experiment', 'direction', 'patient']).size().eq(5).all()
    sensitivity_analysis(tmp_path)
    sensitivity = pd.read_csv(tmp_path/'sensitivity_summary.csv')
    assert not sensitivity.retrained.any()


def test_siena_montages_share_one_cleaned_input(tmp_path, monkeypatch):
    import eegref.signals as module
    from eegref.manifest import ELECTRODES, BIPOLAR
    config = json.loads(CONFIG.read_text())
    config['data_root'] = str(tmp_path)
    cleaned = np.arange(19, dtype=float)[:, None] + np.sin(np.arange(76800)[None, :]/20)
    class Raw:
        ch_names = ELECTRODES
        info = {'sfreq': 256}
        n_times = 76800
        def get_data(self, picks, start, stop):
            return cleaned[picks, start:stop].copy()
        def close(self):
            pass
    monkeypatch.setattr(module.mne.io, 'read_raw_edf', lambda *a, **k: Raw())
    monkeypatch.setattr(module.signal, 'resample_poly', lambda x, *a, **k: x)
    monkeypatch.setattr(module.signal, 'sosfiltfilt', lambda sos, x, **k: x)
    monkeypatch.setattr(module.signal, 'filtfilt', lambda b, a, x, **k: x)
    calls = []
    def clean(values, names, settings):
        calls.append(names)
        return values, {}, {}
    monkeypatch.setattr(module, 'clean_ica', clean)
    row = dict(path='PN01/test.edf', dataset='Siena', start_s=0, stop_s=300, context_start_s=0)
    data, _, _, _ = module.load_window(row, config)
    assert len(calls) == 1
    keep = [i for i, name in enumerate(ELECTRODES) if name != 'CZ']
    np.testing.assert_allclose(data['CAR'], (cleaned-cleaned.mean(axis=0))[keep])
    np.testing.assert_allclose(data['Cz'], (cleaned-cleaned[ELECTRODES.index('CZ')])[keep])
    for index, pair in enumerate(BIPOLAR):
        a, b = pair.split('-')
        np.testing.assert_allclose(data['bipolar'][index], cleaned[ELECTRODES.index(a)]-cleaned[ELECTRODES.index(b)])
    with pytest.raises(ValueError, match='within data_root'):
        module.load_window(dict(row, path='../outside.edf'), config)


def test_report_exports_from_synthetic_scores_without_training(tmp_path, monkeypatch):
    import eegref.evaluation as module
    from eegref.reporting import report
    frame = feature_frame()
    class DecisionOnly:
        def fit(self, x, y):
            return self
        def decision_function(self, x):
            return x[:, 0]
    monkeypatch.setattr(module, 'model', DecisionOnly)
    monkeypatch.setattr(module, 'figures', lambda *args: None)
    evaluate(frame, tmp_path/'evaluation')
    sensitivity_analysis(tmp_path/'evaluation')
    window_rows, inventory = [], []
    for dataset in ['CHB-MIT', 'Siena']:
        for patient in range(3):
            path = f'{dataset}_{patient}/record.edf'
            inventory.append(dict(path=path, events=[[500, 520]], duration_s=1000))
            window_rows.append(dict(dataset=dataset, patient=f'{dataset}_{patient}', path=path, label=1))
    (tmp_path/'inventory.json').write_text(json.dumps(inventory))
    pd.DataFrame(window_rows).to_csv(tmp_path/'windows.csv', index=False)
    report(tmp_path)
    for filename in ['table1_cohort.csv', 'table3_preictal_differences.csv',
                     'table4_lopo.csv', 'table5_transfer.csv', 'table6_sensitivity.csv',
                     'figure3_preictal.svg', 'figure4_patient_heatmaps.png',
                     'figure5_family_contrasts.svg']:
        assert (tmp_path/'report'/filename).stat().st_size > 0


def test_outcome_blind_selection_and_window_guards(tmp_path, monkeypatch):
    import eegref.manifest as module
    settings = json.loads(CONFIG.read_text())
    settings.update(data_root=str(tmp_path/'data'), cohort_sizes={'CHB-MIT': 2, 'Siena': 2})
    records = []
    for case, dataset in [('chb01', 'CHB-MIT'), ('chb03', 'CHB-MIT'), ('PN01', 'Siena'), ('PN03', 'Siena')]:
        path = tmp_path/'data'/case/'record.edf'
        path.parent.mkdir(parents=True)
        path.touch()
        records.append(dict(path=f'{case}/record.edf', case=case, patient=case, dataset=dataset,
                            timestamp='2020-01-01T00:00:00', duration_s=14400,
                            size_ok=True, canonical_labels=module.BIPOLAR if dataset == 'CHB-MIT' else module.ELECTRODES))
    lookup = {record['path']: record for record in records}
    monkeypatch.setattr(module, 'header', lambda path, root: dict(lookup[path.relative_to(root).as_posix()]))
    events = {(record['case'], 'record.edf'): [(1200., 1220.), (2500., 2520.)] for record in records}
    monkeypatch.setattr(module, 'read_annotations', lambda *args: (events, []))
    first = module.build(settings, tmp_path/'audit1')
    second = module.build(settings, tmp_path/'audit2')
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby('dataset').patient.nunique().to_dict() == settings['cohort_sizes']
    assert first.loc[first.dataset == 'Siena'].groupby('patient').robustness.sum().eq(8).all()
    assert first.loc[first.label == 1, 'start_s'].isin([900., 2200.]).all()
    assert (first.stop_s-first.start_s).eq(300).all()
    for row in first[first.label == 0].itertuples():
        assert not any(module.overlaps(row.start_s, row.stop_s, onset-3600, end+3600)
                       for onset, end in events[(row.case, 'record.edf')])
