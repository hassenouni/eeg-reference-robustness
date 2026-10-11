"""Stages: audit, extract (no classifier), evaluate, sensitivity, run."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pandas as pd
from .config import load_config
from .evaluation import evaluate, reference_robustness
from .manifest import build
from .sensitivity import sensitivity_analysis
from .workers import worker


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def extract_study(config, output, workers):
    code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))
    code_hash = hashlib.sha256(code).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in
                ['numpy', 'scipy', 'pandas', 'scikit-learn', 'mne', 'matplotlib', 'threadpoolctl']}
    signature = hashlib.sha256((json.dumps(config, sort_keys=True) + code_hash +
                                json.dumps(versions, sort_keys=True)).encode()).hexdigest()
    status_path = output / 'run.json'
    previous = None
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding='utf-8'))
        if previous.get('signature') != signature:
            raise ValueError('Configuration, environment or code changed; use a new output directory')
    windows = build(config, output)
    inventory = {row['path']: row for row in json.loads(
        (output / 'inventory.json').read_text(encoding='utf-8'))}
    manifest_hash = hashlib.sha256((output / 'windows.csv').read_bytes()).hexdigest()
    if previous and previous.get('windows_sha256') != manifest_hash:
        raise ValueError('Window selection changed; use a new output directory')
    status = dict(status='extracting', signature=signature, config=config,
                  code_sha256=code_hash, windows_sha256=manifest_hash, versions=versions,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  intended_windows=len(windows), classifier_fitted=False)
    write_json(status_path, status)
    cache = output / 'cache'
    cache.mkdir(exist_ok=True)
    tasks = [(row, config, str(cache), signature, inventory[row['path']])
             for row in windows.to_dict('records')]
    features, robust, errors = [], [], []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for index, (feature, reference, error) in enumerate(pool.map(worker, tasks), 1):
                if error:
                    errors.append(error)
                else:
                    features.append(feature)
                    robust.extend(reference)
                if index % 25 == 0 or index == len(tasks):
                    print(f'Windows: {index}/{len(tasks)}; excluded: {len(errors)}', flush=True)
        pd.DataFrame(errors, columns=['window_id', 'path', 'error']).to_csv(
            output / 'extraction_errors.csv', index=False)
        if not features:
            raise RuntimeError('All windows failed; inspect extraction_errors.csv')
        frame = pd.DataFrame(features)
        frame.to_csv(output / 'features.csv', index=False)
        reference_robustness(robust, output, required=config['robustness_windows'])
        eligible = frame.groupby(['dataset', 'patient']).label.nunique()
        counts = eligible[eligible == 2].groupby(level=0).size().to_dict()
        if counts != config['cohort_sizes']:
            raise RuntimeError(f'Incomplete two-state cohort after extraction: {counts}')
        status.update(status='extracted_with_exclusions' if errors else 'extracted',
                      successful_windows=len(features), failed_windows=len(errors))
    except Exception as error:
        status.update(status='failed', error=str(error))
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(status_path, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'extract', 'evaluate', 'sensitivity', 'report', 'run'])
    parser.add_argument('--config', type=Path, default=Path('config/study.json'))
    parser.add_argument('--data-root', type=Path, help='Root containing chbXX and PNXX folders')
    parser.add_argument('--output', type=Path, default=Path('outputs/study'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--demographics', type=Path, help='Optional dataset,patient,age_years,sex CSV for report')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command in ['audit', 'extract', 'run']:
        config = load_config(args.config, args.data_root)
        if args.command == 'audit':
            windows = build(config, output)
            print(f'Selected {len(windows)} windows; no EEG preprocessing or training performed')
            return
        extract_study(config, output, args.workers)
    if args.command in ['evaluate', 'run']:
        frame = pd.read_csv(output / 'features.csv')
        evaluate(frame, output / 'evaluation')
        sensitivity_analysis(output / 'evaluation')
    elif args.command == 'report':
        from .reporting import report
        report(output, args.demographics)
    elif args.command == 'sensitivity':
        sensitivity_analysis(output / 'evaluation')


if __name__ == '__main__':
    main()
