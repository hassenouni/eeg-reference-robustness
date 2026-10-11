"""Importable worker functions for Windows multiprocessing spawn."""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
from scipy.stats import spearmanr
from .features import METHODS, GRAPH_NAMES, extract
from .signals import load_window


def worker(task):
    row, config, cache_dir, signature, inventory = task
    try:
        source = Path(config['data_root'])/row['path']
        fingerprint = f"{signature}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{inventory['header_sha256']}"
        key = hashlib.sha256((fingerprint+json.dumps(row, sort_keys=True)).encode()).hexdigest()
        cache = Path(cache_dir)/(key+'.npz')
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                result = {k: f[k] for k in f.files}
        else:
            data, digest, ica_info, ica_arrays = load_window(row, config)
            result = extract(data['bipolar'], config)
            result.update(ica_arrays)
            result['ica_json'] = np.array(json.dumps(ica_info))
            result['source_samples_sha256'] = np.array(digest)
            result['window_id'] = np.array(row['window_id'])
            result['patient'] = np.array(row['patient'])
            if row['robustness']:
                car, cz = extract(data['CAR'], config), extract(data['Cz'], config)
                for method in METHODS:
                    result['CAR_'+method+'_edges'] = car[method+'_edges']
                    result['Cz_'+method+'_edges'] = cz[method+'_edges']
                    rho = spearmanr(car[method+'_edges'], cz[method+'_edges']).statistic
                    a, b = car[method][0], cz[method][0]
                    result['robust_'+method] = np.array([rho, abs(a-b)/(.5*(a+b)) if a+b else np.nan])
                    if not np.isfinite(result['robust_'+method]).all():
                        raise ValueError('Undefined reference robustness: '+method)
            temporary = cache.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, **result)
            temporary.replace(cache)
        features = dict(row)
        ica_info = json.loads(str(result['ica_json']))
        features['ica_rank'] = ica_info['rank']
        features['ica_iterations'] = ica_info['n_iter']
        features['ica_excluded_count'] = len(ica_info['excluded'])
        features['ica_removed_energy_fraction'] = ica_info['removed_energy_fraction']
        audit_dir = Path(cache_dir).parent/'ica_audit'
        audit_dir.mkdir(exist_ok=True)
        (audit_dir/(row['window_id']+'.json')).write_text(json.dumps(dict(path=row['path'], window_id=row['window_id'], **ica_info), indent=2), encoding='utf-8')
        features['source_samples_sha256'] = str(result['source_samples_sha256'])
        features['n_epochs'] = int(result['n_epochs'])
        features.update({f'spectral_{i:03d}': v for i, v in enumerate(result['spectral'])})
        robust = []
        for method in METHODS:
            features.update({method+'_'+name: float(v) for name, v in zip(GRAPH_NAMES, result[method])})
            if row['robustness']:
                rho, relative = result['robust_'+method]
                robust.append(dict(patient=row['patient'], window_id=row['window_id'], method=method, rho=rho, relative_strength=relative))
        return features, robust, None
    except Exception:
        return None, [], dict(window_id=row['window_id'], path=row['path'], error=traceback.format_exc())
