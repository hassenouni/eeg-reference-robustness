"""Patient omission sensitivity on existing scores; no classifier retraining."""
from pathlib import Path
import pandas as pd
from .evaluation import inference


def sensitivity_analysis(directory):
    directory = Path(directory)
    deltas = pd.read_csv(directory / 'patient_deltas.csv')
    families = pd.read_csv(directory / 'family_patient_deltas.csv')
    groups = list(deltas.groupby(['experiment', 'direction', 'method']))
    if not families.empty:
        groups += [(('transfer', direction, 'robust_minus_conventional'),
                    group.rename(columns={'difference': 'delta'}))
                   for direction, group in families.groupby('direction')]
    omissions, summaries = [], []
    for (experiment, direction, method), group in groups:
        if len(group) < 3:
            continue
        test = 'wilcoxon' if experiment == 'transfer' else 't'
        rows = []
        for patient in group.patient:
            result = inference(group.loc[group.patient != patient, 'delta'], test)
            rows.append(dict(experiment=experiment, direction=direction, method=method,
                             excluded_patient=patient, **result))
        omissions.extend(rows)
        summaries.append(dict(experiment=experiment, direction=direction, method=method,
                              original_gain=float(group.delta.mean()),
                              min_gain=min(row['mean'] for row in rows),
                              max_gain=max(row['mean'] for row in rows),
                              nominal_significant=sum(row['p'] < .05 for row in rows),
                              exclusions=len(rows), retrained=False,
                              multiplicity_adjusted=False))
    pd.DataFrame(omissions).to_csv(directory / 'sensitivity_exclusions.csv', index=False)
    pd.DataFrame(summaries).to_csv(directory / 'sensitivity_summary.csv', index=False)
