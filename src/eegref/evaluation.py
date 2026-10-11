"""Patient-level inference and validation with train-only standardization."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import METHODS, GRAPH_NAMES


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1])
    return result


def inference(values, test='t'):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return dict(n=n, mean=float(x.mean()) if n else np.nan, sd=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, improved=int((x > 0).sum()))
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    margin = stats.t.ppf(.975, n-1)*sd/np.sqrt(n)
    p = 1.0 if np.all(x == 0) else (stats.wilcoxon(x, alternative='two-sided', method='auto').pvalue if test == 'wilcoxon' else stats.ttest_1samp(x, 0).pvalue)
    return dict(n=n, mean=mean, sd=sd, ci_low=mean-margin, ci_high=mean+margin, p=float(p), improved=int((x > 0).sum()))


def model():
    return make_pipeline(StandardScaler(), SVC(kernel='linear', C=1, probability=False, class_weight=None))


def evaluate(frame, out):
    out.mkdir(parents=True, exist_ok=True)
    validate_feature_frame(frame)
    spec = [f'spectral_{i:03d}' for i in range(180)]
    sets = {'spectral': spec}
    sets.update({m: spec+[m+'_'+g for g in GRAPH_NAMES] for m in METHODS})
    scores, predictions, omissions = [], [], []
    # Remove unscorable patients before fitting any comparison so all models use matched cohorts.
    keep = frame.groupby(['dataset', 'patient']).label.transform('nunique') == 2
    for (ds, patient), _ in frame[~keep].groupby(['dataset', 'patient']):
        omissions.append(dict(dataset=ds, patient=patient, reason='only_one_state_after_extraction'))
    frame = frame[keep].reset_index(drop=True)
    counts = frame.groupby('dataset').patient.nunique()
    if set(counts.index) != {'CHB-MIT', 'Siena'} or (counts < 2).any():
        raise ValueError('Evaluation requires both datasets and at least two two-state patients per dataset')
    def run(train, test, experiment, direction):
        if train.empty or train.label.nunique() != 2:
            return
        for name, columns in sets.items():
            fitted = model().fit(train[columns].to_numpy(), train.label.to_numpy())
            decision = fitted.decision_function(test[columns].to_numpy())
            for patient in test.patient.unique():
                mask = test.patient.to_numpy() == patient
                y, s = test.label.to_numpy()[mask], decision[mask]
                scores.append(dict(experiment=experiment, direction=direction, patient=patient, method=name,
                                   auroc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                                   prevalence=float(y.mean()), n_windows=len(y), n_train=len(train)))
            for row, value in zip(test.to_dict('records'), decision):
                predictions.append(dict(experiment=experiment, direction=direction, patient=row['patient'],
                                        window_id=row['window_id'], method=name, label=row['label'], score=float(value)))
    for ds, data in frame.groupby('dataset'):
        for patient in data.patient.unique():
            run(data[data.patient != patient], data[data.patient == patient], 'LOPO', ds)
    for source, target in [('CHB-MIT', 'Siena'), ('Siena', 'CHB-MIT')]:
        train, test = frame[frame.dataset == source], frame[frame.dataset == target]
        if not test.empty:
            run(train, test, 'transfer', source+' -> '+target)
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(out/'patient_scores.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv', index=False)
    pd.DataFrame(omissions, columns=['dataset', 'patient', 'reason']).to_csv(out/'evaluation_exclusions.csv', index=False)
    comparisons, differences, families, family_values = [], [], [], []
    if not score_frame.empty:
        for (experiment, direction), sub in score_frame.groupby(['experiment', 'direction']):
            metric = 'auroc' if experiment == 'LOPO' else 'ap'
            pivot = sub.pivot(index='patient', columns='method', values=metric)
            delta = pivot[list(METHODS)].subtract(pivot.spectral, axis=0)
            rows = []
            for method in METHODS:
                for patient, value in delta[method].items():
                    differences.append(dict(experiment=experiment, direction=direction, patient=patient, method=method, delta=value))
                rows.append(dict(experiment=experiment, direction=direction, method=method, metric=metric,
                                 **inference(delta[method], 't' if experiment == 'LOPO' else 'wilcoxon')))
            for row, q in zip(rows, bh([r['p'] for r in rows])):
                row['q'] = q
            comparisons += rows
            if experiment == 'transfer':
                d = delta[['corCC', 'wPLI']].mean(axis=1)-delta[['CC', 'COH']].mean(axis=1)
                families.append(dict(direction=direction, **inference(d, 'wilcoxon')))
                family_values += [dict(direction=direction, patient=p, difference=v) for p, v in d.items()]
    for row, q in zip(families, bh([r['p'] for r in families])):
        row['q'] = q
    for name, rows in [('comparisons', comparisons), ('patient_deltas', differences), ('family_contrasts', families), ('family_patient_deltas', family_values)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'), index=False)
    effects, patient_effects = [], []
    for ds, sub in frame.groupby('dataset'):
        rows = []
        for method in METHODS:
            column = method+'_strength'
            pivot = sub.groupby(['patient', 'label'])[column].mean().unstack()
            d = pivot[1]-pivot[0]
            rows.append(dict(dataset=ds, method=method, feature='mean_node_strength', **inference(d)))
            patient_effects += [dict(dataset=ds, patient=p, method=method, difference=v) for p, v in d.items()]
        for row, q in zip(rows, bh([r['p'] for r in rows])):
            row['q'] = q
        effects += rows
    pd.DataFrame(effects).to_csv(out/'preictal_effects.csv', index=False)
    pd.DataFrame(patient_effects).to_csv(out/'preictal_patient_effects.csv', index=False)
    if not score_frame.empty:
        figures(pd.DataFrame(differences), pd.DataFrame(effects), out)
    return dict(patients=frame.groupby('dataset').patient.nunique().to_dict(), windows=len(frame), folds=len(scores))


def reference_robustness(window_rows, out, required=8):
    frame = pd.DataFrame(window_rows)
    if frame.empty:
        return
    frame.to_csv(out/'robustness_windows.csv', index=False)
    counts = frame.groupby(['patient', 'method']).size()
    eligible = counts[counts == required].index
    means = frame.groupby(['patient', 'method'])[['rho', 'relative_strength']].mean()
    means = means.loc[means.index.isin(eligible)].reset_index()
    means.to_csv(out/'robustness_patients.csv', index=False)
    means.groupby('method')[['rho', 'relative_strength']].agg(['count', 'mean', 'std']).to_csv(out/'robustness_summary.csv')


def figures(deltas, effects, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(deltas.groupby(['experiment', 'direction']))
    fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 4), squeeze=False)
    colors = ['#9b6a38', '#b34b4b', '#267c8f', '#447c40']
    for ax, ((experiment, direction), sub) in zip(axes[0], groups):
        for j, method in enumerate(METHODS):
            values = sub[sub.method == method].delta.to_numpy()
            summary = inference(values)
            ax.scatter(np.full(len(values), j), values, color=colors[j], alpha=.5, s=18)
            if len(values) > 1:
                ax.errorbar(j+.12, summary['mean'], yerr=summary['ci_high']-summary['mean'], color='black', fmt='s', capsize=4)
        ax.axhline(0, color='gray', lw=1)
        ax.set_xticks(range(4), METHODS)
        ax.set_ylabel('Gain AUROC' if experiment == 'LOPO' else 'Gain AP')
        ax.set_title(experiment+'\n'+direction)
    fig.suptitle('Patient-level connectivity gains')
    fig.tight_layout()
    fig.savefig(out/'performance.png', dpi=160)
    plt.close(fig)


def validate_feature_frame(frame):
    """Reject malformed exported features before training any estimator."""
    spectral = [f'spectral_{i:03d}' for i in range(180)]
    graph = [method+'_'+name for method in METHODS for name in GRAPH_NAMES]
    required = ['dataset', 'patient', 'window_id', 'label'] + spectral + graph
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError('Missing feature columns: '+', '.join(sorted(missing)))
    if frame.empty or frame[required].isna().any().any():
        raise ValueError('Empty or missing-valued feature table')
    if not frame.label.isin([0, 1]).all():
        raise ValueError('Labels must be binary 0/1')
    if set(frame.dataset) - {'CHB-MIT', 'Siena'}:
        raise ValueError('Unrecognized dataset')
    if frame.window_id.duplicated().any():
        raise ValueError('Duplicate window IDs would count observations repeatedly')
    if not np.isfinite(frame[spectral+graph].to_numpy(dtype=float)).all():
        raise ValueError('Features must be finite')
    if 'n_epochs' in frame and not (frame.n_epochs == 119).all():
        raise ValueError('Every feature row must aggregate 119 epochs')
