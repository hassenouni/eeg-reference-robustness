"""Read only required EDF samples; never write into source recordings."""
from fractions import Fraction
from pathlib import Path
import hashlib

import mne
import numpy as np
from scipy import signal

from .manifest import BIPOLAR, ELECTRODES, canonical
from .ica import clean_ica


def load_window(row, config):
    relative = row['path']
    root = Path(config['data_root']).resolve()
    source = (root/relative).resolve()
    if not source.is_relative_to(root):
        raise ValueError('EDF path must stay within data_root')
    if config['ica_mode'] != 'fastica_required' or config.get('reviewed_cleaned_files'):
        raise ValueError('Mandatory automatic ICA must be applied to original EDFs')
    raw = mne.io.read_raw_edf(source, preload=False, verbose='ERROR')
    try:
        # MNE adds -0/-1 to duplicate EDF labels; restore names for deterministic first occurrence.
        names = {}
        for i, name in enumerate(raw.ch_names):
            label = canonical(name)
            if label.rsplit('-', 1)[-1].isdigit():
                label = label.rsplit('-', 1)[0]
            names.setdefault(label, i)
        operations = []
        if row['dataset'] == 'CHB-MIT':
            for pair in BIPOLAR:
                reverse = '-'.join(pair.split('-')[::-1])
                if pair in names:
                    operations.append((names[pair], 1))
                elif reverse in names:
                    operations.append((names[reverse], -1))
                else:
                    raise ValueError('Missing derivation '+pair)
            picks = [i for i, _ in operations]
        else:
            picks = [names[x] for x in ELECTRODES]
        fs = float(raw.info['sfreq'])
        # Past-only padding: future samples never cross the seizure onset.
        # Each state window is independently zero-phase filtered with reflected edges.
        start, stop = row['start_s'], row['stop_s']
        if not (0 <= row['context_start_s'] <= start < stop <= raw.n_times/fs):
            raise ValueError('Invalid window bounds')
        if not np.isclose(stop-start, config['window_s']):
            raise ValueError('Expected a 300-second window')
        if not (0 <= row['context_start_s'] <= start < stop <= raw.n_times/fs):
            raise ValueError('Invalid window or context bounds')
        if not np.isclose(stop-start, config['window_s']):
            raise ValueError('Analysis window must contain exactly 300 seconds')
        first = round(row['context_start_s']*fs)
        last = round(stop*fs)
        values = raw.get_data(picks=picks, start=first, stop=last)
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        if row['dataset'] == 'CHB-MIT':
            values *= np.array([sign for _, sign in operations])[:, None]
        ratio = Fraction(config['fs']/fs).limit_denominator(10000)
        values = signal.resample_poly(values, ratio.numerator, ratio.denominator, axis=-1)
        sos = signal.butter(config['butter_order'], config['band_hz'], btype='bandpass', fs=config['fs'], output='sos')
        values = signal.sosfiltfilt(sos, values, axis=-1)
        b, a = signal.iirnotch(config['notch_hz'], config['notch_q'], fs=config['fs'])
        values = signal.filtfilt(b, a, values, axis=-1)
        offset = round((start-first/fs)*config['fs'])
        values = values[:, offset:offset+round(config['window_s']*config['fs'])]
        if values.shape[1] != round(config['window_s']*config['fs']) or not np.isfinite(values).all():
            raise ValueError('Incomplete or nonfinite processed window')
        values, ica_info, ica_arrays = clean_ica(values, BIPOLAR if row['dataset'] == 'CHB-MIT' else ELECTRODES, config)
        if row['dataset'] == 'CHB-MIT':
            return {'bipolar': values}, digest, ica_info, ica_arrays
        lookup = {name: i for i, name in enumerate(ELECTRODES)}
        bipolar = np.stack([values[lookup[a]]-values[lookup[b]] for a, b in (pair.split('-') for pair in BIPOLAR)])
        keep = [i for i, name in enumerate(ELECTRODES) if name != 'CZ']
        car = (values-values.mean(axis=0, keepdims=True))[keep]
        cz = (values-values[lookup['CZ']])[keep]
        return {'bipolar': bipolar, 'CAR': car, 'Cz': cz}, digest, ica_info, ica_arrays
    finally:
        raw.close()
