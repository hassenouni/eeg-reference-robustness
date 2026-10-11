"""Read-only EDF/annotation audit and deterministic, outcome-blind selection."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd

ELECTRODES = 'FP1 FP2 F7 F3 FZ F4 F8 T7 C3 CZ C4 T8 P7 P3 PZ P4 P8 O1 O2'.split()
BIPOLAR = 'FP1-F7 F7-T7 T7-P7 P7-O1 FP1-F3 F3-C3 C3-P3 P3-O1 FP2-F4 F4-C4 C4-P4 P4-O2 FP2-F8 F8-T8 T8-P8 P8-O2 FZ-CZ CZ-PZ'.split()


def canonical(label):
    return '-'.join({'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}.get(x, x)
                    for x in label.upper().strip().removeprefix('EEG ').split('-'))


def header(path, root):
    with path.open('rb') as f:
        h = f.read(256)
        n = int(h[252:256])
        b = f.read(n * 256)
    def field(offset, width):
        return [b[offset*n+i*width:offset*n+(i+1)*width].decode('latin1').strip() for i in range(n)]
    duration, records = float(h[244:252]), int(h[236:244])
    labels = field(0, 16)
    samples = list(map(int, field(216, 8)))
    timestamp = datetime.strptime(h[168:176].decode()+' '+h[176:184].decode(), '%d.%m.%y %H.%M.%S')
    return dict(path=path.relative_to(root).as_posix(), case=path.parent.name,
                patient='chb01' if path.parent.name == 'chb21' else path.parent.name,
                dataset='CHB-MIT' if path.parent.name.startswith('chb') else 'Siena',
                duration_s=records*duration, labels=labels, canonical_labels=list(map(canonical, labels)),
                timestamp=timestamp.isoformat(), clock_s=timestamp.hour*3600+timestamp.minute*60+timestamp.second,
                sample_rates=[s/duration for s in samples], size_bytes=path.stat().st_size,
                mtime_ns=path.stat().st_mtime_ns,
                size_ok=path.stat().st_size == int(h[184:192])+records*sum(samples)*2,
                header_sha256=hashlib.sha256(h+b).hexdigest())


def clock_value(text):
    matches = re.findall(r'(\d{1,2})[.:](\d{2})[.:](\d{2})', text)
    if len(matches) != 1:
        raise ValueError('ambiguous or invalid clock: '+text)
    h, m, s = map(int, matches[0])
    if h > 23 or m > 59 or s > 59:
        raise ValueError('invalid clock: '+text)
    return h*3600+m*60+s


def read_annotations(root, records):
    events, issues = {}, []
    for txt in sorted(root.rglob('*summary.txt')):
        for block in re.split(r'File Name:\s*', txt.read_text(errors='replace'))[1:]:
            name = block.splitlines()[0].strip()
            start = re.findall(r'Seizure(?:\s+\d+)? Start Time:\s*(\d+) seconds', block)
            end = re.findall(r'Seizure(?:\s+\d+)? End Time:\s*(\d+) seconds', block)
            count = re.search(r'Number of Seizures in File:\s*(\d+)', block)
            key = (txt.parent.name, name)
            events[key] = list(zip(map(float, start), map(float, end))) if count and len(start) == len(end) == int(count[1]) else None
    lookup = {(r['case'], Path(r['path']).name): r for r in records}
    for txt in sorted(root.rglob('Seizures-list-*.txt')):
        current, onset = None, None
        for line in txt.read_text(errors='replace').splitlines():
            if re.match(r'File name:', line, re.I):
                name = line.split(':', 1)[1].strip()
                fixed = name.replace('PNO6', 'PN06').replace('PN11-.edf', 'PN11-1.edf').replace('PN01.edf', 'PN01-1.edf')
                if (txt.parent.name, fixed) not in lookup and txt.parent.name == 'PN11':
                    candidates = [k[1] for k in lookup if k[0] == 'PN11']
                    if len(candidates) == 1:
                        fixed = candidates[0]
                current = (txt.parent.name, fixed)
                if fixed != name:
                    issues.append(dict(case=current[0], file=name, issue='filename_alias', detail=fixed))
                if current not in lookup:
                    issues.append(dict(case=current[0], file=fixed, issue='annotation_without_edf', detail=''))
                events.setdefault(current, [])
                onset = None
            elif current and re.match(r'(Seizure )?Start time:', line, re.I):
                try:
                    # Explicit electrical onset has priority when both are supplied.
                    text = line.split(':', 1)[1]
                    if 'ELECTRIC ONSET' in text.upper():
                        text = re.search(r'(\d{1,2}[.:]\d{2}[.:]\d{2})\s*\(ELECTRIC ONSET', text, re.I)[1]
                        issues.append(dict(case=current[0], file=current[1], issue='electrical_onset_selected', detail=text))
                    onset = clock_value(text)
                except ValueError as e:
                    events[current] = None
                    issues.append(dict(case=current[0], file=current[1], issue='invalid_annotation', detail=str(e)))
            elif current and re.match(r'(Seizure )?End time:', line, re.I):
                try:
                    end = clock_value(line.split(':', 1)[1])
                    if current not in lookup or onset is None:
                        raise ValueError('missing EDF or onset')
                    a = (onset-lookup[current]['clock_s']) % 86400
                    b = a+(end-onset) % 86400
                    if events[current] is not None:
                        events[current].append((float(a), float(b)))
                except ValueError as e:
                    events[current] = None
                    issues.append(dict(case=current[0], file=current[1], issue='invalid_annotation', detail=str(e)))
                onset = None
    return events, issues


def overlaps(a, b, c, d):
    return a < d and c < b


def build(config, out):
    root = Path(config['data_root'])
    if not root.is_dir():
        raise FileNotFoundError(root)
    records, issues = [], []
    for p in sorted(root.rglob('*.edf')):
        try:
            records.append(header(p, root))
        except (ValueError, OSError) as e:
            issues.append(dict(case=p.parent.name, file=p.name, issue='header_error', detail=str(e)))
    events, annotation_issues = read_annotations(root, records)
    issues += annotation_issues
    all_events = defaultdict(list)
    for r in records:
        r['events'] = config['annotation_overrides'].get(r['path'], events.get((r['case'], Path(r['path']).name)))
        reason = []
        if not r['size_ok']:
            reason.append('invalid_file_size')
        labels = set(r['canonical_labels'])
        missing = [x for x in (BIPOLAR if r['dataset'] == 'CHB-MIT' else ELECTRODES)
                   if x not in labels and not (r['dataset'] == 'CHB-MIT' and '-'.join(x.split('-')[::-1]) in labels)]
        if missing:
            reason.append('missing_channels:'+','.join(missing))
        if r['events'] is None or any(not (0 <= a < b <= r['duration_s']) for a, b in r['events']):
            reason.append('missing_or_invalid_annotations')
            issues.append(dict(case=r['case'], file=Path(r['path']).name, issue='excluded_annotations', detail=str(r['events'])))
            # An unknown interval cannot be used as a seizure-free neighboring record.
            ts = recording_seconds(r['timestamp'])
            all_events[r['case']].append((ts, ts+r['duration_s']))
        else:
            ts = recording_seconds(r['timestamp'])
            all_events[r['case']].extend((ts+a, ts+b) for a, b in r['events'])
        if config['patients'] and r['case'] not in config['patients']:
            reason.append('not_selected')
        r['exclusion'] = ';'.join(reason)
    candidates = defaultdict(lambda: {0: [], 1: []})
    w, guard = config['window_s'], config['interictal_guard_s']
    for r in records:
        if r['exclusion']:
            continue
        ts = recording_seconds(r['timestamp'])
        def row(start, label):
            context = max(0., start-config['filter_context_s'])
            for _, event_end in all_events[r['case']]:
                if ts+context < event_end <= ts+start:
                    context = event_end-ts
            return dict(window_id=hashlib.sha256(f"{r['path']}:{start}:{label}".encode()).hexdigest()[:20],
                        path=r['path'], dataset=r['dataset'], case=r['case'], patient=r['patient'],
                        start_s=float(start), stop_s=float(start+w), context_start_s=float(context), label=label,
                        robustness=False)
        for a, b in r['events']:
            if a >= w and not any(overlaps(ts+a-w, ts+a, c, d) for c, d in all_events[r['case']]):
                candidates[(r['dataset'], r['patient'])][1].append(row(a-w, 1))
        for start in np.arange(0, r['duration_s']-w+0.1, w):
            if not any(overlaps(ts+start, ts+start+w, c-guard, d+guard) for c, d in all_events[r['case']]):
                candidates[(r['dataset'], r['patient'])][0].append(row(start, 0))
    rng = np.random.default_rng(config['seed'])
    # Freeze cohort before feature extraction and any model evaluation.
    ranking = []
    for (dataset, patient), states in candidates.items():
        n0, n1 = len(states[0]), len(states[1])
        rs = [r for r in records if r['dataset'] == dataset and r['patient'] == patient]
        quality = sum(not r['exclusion'] for r in rs)/len(rs)
        usable = n1+min(n0, n1*config['interictal_per_preictal'])
        eligible = n1 >= 2 and n0 >= 4 and (dataset != 'Siena' or usable >= config['robustness_windows'])
        ranking.append(dict(dataset=dataset, patient=patient, valid_preictal=n1,
                            available_interictal=n0, compatible_file_fraction=quality,
                            eligible=eligible, selected=False))
    for dataset, target in config['cohort_sizes'].items():
        ranked = sorted([r for r in ranking if r['dataset'] == dataset and r['eligible']],
                        key=lambda r: (-min(r['valid_preictal'], r['available_interictal']/config['interictal_per_preictal']),
                                       -r['compatible_file_fraction'], -r['available_interictal'], r['patient']))
        if len(ranked) < target:
            raise ValueError(f'{dataset}: only {len(ranked)} eligible patients; requested {target}')
        for rank, r in enumerate(ranked, 1):
            r['rank'] = rank
            r['selected'] = rank <= target
    retained = {(r['dataset'], r['patient']) for r in ranking if r['selected']}
    selected, summary = [], []
    for (dataset, patient), states in sorted(candidates.items()):
        n0, n1 = len(states[0]), len(states[1])
        count = min(n0, n1*config['interictal_per_preictal'])
        rows = states[1] + [states[0][int(i)] for i in sorted(rng.choice(n0, count, replace=False))] if (dataset, patient) in retained else []
        if dataset == 'Siena' and len(rows) >= config['robustness_windows']:
            for i in rng.choice(len(rows), config['robustness_windows'], replace=False):
                rows[int(i)]['robustness'] = True
        summary.append(dict(dataset=dataset, patient=patient, available_preictal=n1,
                            available_interictal=n0, selected_preictal=n1 if rows else 0,
                            selected_interictal=count if rows else 0,
                            robustness_windows=sum(r['robustness'] for r in rows),
                            status='included' if rows else 'not_selected_by_prespecified_ranking'))
        selected.extend(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out/'inventory.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    pd.DataFrame(issues, columns=['case', 'file', 'issue', 'detail']).to_csv(out/'audit_issues.csv', index=False)
    pd.DataFrame(summary).to_csv(out/'cohort.csv', index=False)
    pd.DataFrame(ranking).to_csv(out/'cohort_ranking.csv', index=False)
    frame = pd.DataFrame(selected)
    if frame.empty:
        raise ValueError('No patients have both admissible states; inspect audit_issues.csv')
    frame.to_csv(out/'windows.csv', index=False)
    return frame


def recording_seconds(timestamp):
    """Deterministic naive-recording timeline, independent of machine timezone."""
    return (datetime.fromisoformat(timestamp) - datetime(1970, 1, 1)).total_seconds()
