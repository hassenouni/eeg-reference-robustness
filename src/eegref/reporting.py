"""Generate tables and figures from computed artifacts; no fitted estimators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .features import METHODS


def export_cohort_table(directory, demographics=None):
    """Describe selected source EDFs; preserve the manuscript's source-file scope."""
    directory = Path(directory)
    inventory = json.loads((directory/'inventory.json').read_text(encoding='utf-8'))
    windows = pd.read_csv(directory/'windows.csv')
    metadata = pd.read_csv(demographics) if demographics else None
    if metadata is not None:
        required = {'dataset', 'patient', 'age_years', 'sex'}
        if not required.issubset(metadata.columns):
            raise ValueError('Demographics require dataset, patient, age_years, sex')
        if metadata.duplicated(['dataset', 'patient']).any():
            raise ValueError('Demographics must contain one row per independent patient')
    records = []
    for dataset, group in windows.groupby('dataset'):
        paths = set(group.path)
        sources = [row for row in inventory if row['path'] in paths]
        row = dict(dataset=dataset, patients=group.patient.nunique(),
                   source_seizures=sum(len(item['events']) for item in sources),
                   source_duration_hours=sum(item['duration_s'] for item in sources)/3600,
                   prediction_signals=18,
                   selected_preictal_windows=int((group.label == 1).sum()),
                   selected_interictal_windows=int((group.label == 0).sum()))
        if metadata is not None:
            matched = group[['dataset', 'patient']].drop_duplicates().merge(
                metadata, on=['dataset', 'patient'], how='left', validate='one_to_one')
            if matched[['age_years', 'sex']].isna().any().any():
                raise ValueError('Demographic metadata are incomplete for selected patients')
            if not matched.sex.isin(['M', 'F']).all():
                raise ValueError('Sex metadata must use M/F')
            row.update(age_mean=float(matched.age_years.mean()),
                       age_sd=float(matched.age_years.std(ddof=1)),
                       male_n=int((matched.sex == 'M').sum()),
                       male_percent=float(100*(matched.sex == 'M').mean()))
        records.append(row)
    pd.DataFrame(records).to_csv(directory/'report/table1_cohort.csv', index=False)


def export_tables(directory):
    directory = Path(directory)
    report = directory/'report'
    report.mkdir(exist_ok=True)
    evaluation = directory/'evaluation'
    if (directory/'robustness_summary.csv').exists():
        pd.read_csv(directory/'robustness_summary.csv', header=[0, 1], index_col=0).to_csv(
            report/'table2_reference_robustness.csv')
    effects = pd.read_csv(evaluation/'preictal_effects.csv')
    effects.to_csv(report/'table3_preictal_differences.csv', index=False)
    comparisons = pd.read_csv(evaluation/'comparisons.csv')
    scores = pd.read_csv(evaluation/'patient_scores.csv')
    absolute = scores.groupby(['experiment', 'direction', 'method']).agg(
        auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
        ap_mean=('ap', 'mean'), ap_sd=('ap', 'std')).reset_index()
    absolute.to_csv(report/'absolute_performance.csv', index=False)
    lopo = comparisons[comparisons.experiment == 'LOPO'].merge(
        absolute[absolute.experiment == 'LOPO'], on=['experiment', 'direction', 'method'])
    # Include baseline rows separately; no synthetic gain is assigned to them.
    baseline = absolute[(absolute.experiment == 'LOPO') & (absolute.method == 'spectral')]
    pd.concat([baseline, lopo], ignore_index=True).to_csv(report/'table4_lopo.csv', index=False)
    comparisons[comparisons.experiment == 'transfer'].to_csv(report/'table5_transfer.csv', index=False)
    pd.read_csv(evaluation/'family_contrasts.csv').to_csv(report/'table5_family_contrasts.csv', index=False)
    if (evaluation/'sensitivity_summary.csv').exists():
        pd.read_csv(evaluation/'sensitivity_summary.csv').to_csv(report/'table6_sensitivity.csv', index=False)


def _save(figure, directory, stem):
    import matplotlib.pyplot as plt
    figure.tight_layout()
    for extension in ['png', 'svg']:
        figure.savefig(directory/(stem+'.'+extension), dpi=300, bbox_inches='tight')
    plt.close(figure)


def _forest(table, group_column, directory, stem, xlabel):
    import matplotlib.pyplot as plt
    groups = list(table.groupby(group_column, sort=True))
    figure, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 3.8), squeeze=False)
    for axis, (name, group) in zip(axes[0], groups):
        for position, method in enumerate(METHODS):
            row = group[group.method == method].iloc[0]
            axis.errorbar(row['mean'], position,
                          xerr=[[max(0, row['mean']-row.ci_low)],
                                [max(0, row.ci_high-row['mean'])]],
                          fmt='o', color='#176b87', capsize=4)
            axis.annotate(f"q={row.q:.3g}", (row['mean'], position),
                          xytext=(4, 8), textcoords='offset points', fontsize=8)
        axis.axvline(0, color='0.6', linewidth=1)
        axis.set_yticks(range(len(METHODS)), METHODS)
        axis.invert_yaxis()
        axis.set(title=str(name), xlabel=xlabel)
    _save(figure, directory, stem)


def export_figures(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    report = directory/'report'
    report.mkdir(exist_ok=True)
    evaluation = directory/'evaluation'
    robustness_path = directory/'robustness_patients.csv'
    if robustness_path.exists():
        robustness = pd.read_csv(robustness_path)
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for axis, metric, label in zip(axes, ['rho', 'relative_strength'],
                                       ['CAR-Cz Spearman similarity', 'Relative strength change']):
            for position, method in enumerate(METHODS):
                values = robustness.loc[robustness.method == method, metric]
                axis.scatter(np.full(len(values), position), values, color='0.65', s=18)
                axis.errorbar(position+.1, values.mean(), yerr=values.std(ddof=1),
                              fmt='o', color='black', capsize=4)
            axis.set_xticks(range(4), METHODS)
            axis.set_ylabel(label)
        _save(figure, report, 'figure2_reference_metrics')
    # Each cache stores epoch-averaged adjacency edges for matched cleaned windows.
    matched = []
    for path in sorted((directory/'cache').glob('*.npz')):
        with np.load(path, allow_pickle=False) as cache:
            if 'CAR_CC_edges' not in cache:
                continue
            matched.append((str(cache['patient']),
                            {reference+'_'+method: cache[reference+'_'+method+'_edges'].copy()
                             for reference in ['CAR', 'Cz'] for method in METHODS}))
    if matched:
        patients = sorted({patient for patient, _ in matched})
        # Equal patient weighting; only patients with eight paired successful windows.
        patients = [p for p in patients if sum(patient == p for patient, _ in matched) == 8]
        if patients:
            figure, axes = plt.subplots(3, 4, figsize=(12, 9))
            for column, method in enumerate(METHODS):
                means = {}
                for reference in ['CAR', 'Cz']:
                    edges = np.mean([np.mean([values[reference+'_'+method]
                                              for patient, values in matched if patient == p], axis=0)
                                     for p in patients], axis=0)
                    matrix = np.zeros((18, 18))
                    indices = np.triu_indices(18, 1)
                    matrix[indices] = edges
                    means[reference] = matrix + matrix.T
                vmax = max(np.max(means['CAR']), np.max(means['Cz']))
                for row, reference in enumerate(['CAR', 'Cz', 'absolute difference']):
                    matrix = means[reference] if row < 2 else abs(means['CAR']-means['Cz'])
                    matrix = np.ma.array(matrix, mask=np.eye(18, dtype=bool))
                    artist = axes[row, column].imshow(matrix, vmin=0, vmax=vmax, cmap='viridis')
                    axes[row, column].set_title(method+' / '+reference)
                    figure.colorbar(artist, ax=axes[row, column], shrink=.65)
            _save(figure, report, 'figure2_reference_matrices')
    effects = pd.read_csv(evaluation/'preictal_effects.csv')
    _forest(effects, 'dataset', report, 'figure3_preictal', 'Preictal - interictal strength')
    comparisons = pd.read_csv(evaluation/'comparisons.csv')
    _forest(comparisons[comparisons.experiment == 'LOPO'], 'direction', report,
            'figure4_lopo_gains', 'Change in AUROC')
    _forest(comparisons[comparisons.experiment == 'transfer'], 'direction', report,
            'figure5_transfer', 'Change in average precision')
    deltas = pd.read_csv(evaluation/'patient_deltas.csv')
    groups = list(deltas[deltas.experiment == 'LOPO'].groupby('direction'))
    figure, axes = plt.subplots(1, len(groups), figsize=(6*len(groups), 6), squeeze=False)
    limit = max(abs(deltas.delta.min()), abs(deltas.delta.max()), .001)
    for axis, (direction, group) in zip(axes[0], groups):
        matrix = group.pivot(index='patient', columns='method', values='delta').reindex(columns=METHODS)
        artist = axis.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        axis.set_xticks(range(4), METHODS)
        axis.set_yticks(range(len(matrix)), matrix.index)
        axis.set_title(direction)
        figure.colorbar(artist, ax=axis, label='Change in AUROC')
    _save(figure, report, 'figure4_patient_heatmaps')
    families = pd.read_csv(evaluation/'family_contrasts.csv')
    figure, axis = plt.subplots(figsize=(7, 3))
    for position, row in enumerate(families.itertuples()):
        axis.errorbar(row.mean, position,
                      xerr=[[max(0, row.mean-row.ci_low)], [max(0, row.ci_high-row.mean)]],
                      fmt='o', capsize=4, color='#176b87')
        axis.annotate(f'q={row.q:.3g}', (row.mean, position), xytext=(4, 8),
                      textcoords='offset points', fontsize=8)
    axis.set_yticks(range(len(families)), families.direction)
    axis.axvline(0, color='0.6', linewidth=1)
    axis.set_xlabel('Robust - conventional family: change in AP')
    _save(figure, report, 'figure5_family_contrasts')


def report(directory, demographics=None):
    directory = Path(directory)
    export_tables(directory)
    export_cohort_table(directory, demographics)
    export_figures(directory)
