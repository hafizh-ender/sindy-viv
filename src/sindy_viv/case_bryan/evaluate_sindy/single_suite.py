"""Evaluate single-cylinder or tandem experiments once per window and summarize predictions.

Training and evaluation JSONs have separate roles. Full-window metrics exclude
partial IVPs; their existing prefix reports and both time-series views are kept.
Error and coefficient heatmaps omit titles; frequency/amplitude plots retain descriptive titles.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps

from sindy_viv.case_bryan.analysis.data_windows import channel_token, data_selection, save_figure
from sindy_viv.case_bryan.analysis.frequency import compute_frequency_analysis
from sindy_viv.case_bryan.analysis.error_heatmaps import (
    TANDEM_IVP_HEATMAP_SCALE, save_nrmse_heatmaps, style_heatmap, heatmap_colorbar, fit_heatmap_annotations,
)
from sindy_viv.case_bryan.model_storage.paths import model_base_directory, resolve_model_directory, training_directory
from sindy_viv.case_bryan.plot import create_subplots, fft_figure_size, LINE_WIDTH
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.feature_library.functions import prepare_library
from sindy_viv.utils import add_dot_text
from .frequency_functions import compare_frequency_analysis
from .functions import evaluate_sindy
from .model_evaluation import validate_max_state_factor


def _write_json(path, payload):
    """Create the report directory and write an indented JSON payload, rejecting NaN/infinity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def _finite(value):
    """Return a finite float or None for missing/nonfinite diagnostic values."""
    return None if value is None or not np.isfinite(value) else float(value)


def _phase_pair(reference, prediction, cylinder=0):
    """CL-minus-h phase at one shared reference-h frequency, wrapped into [-pi, pi]."""
    names = reference.variable_names
    displacement, lift = f'h_{cylinder}', f'C_{{L{cylinder}}}'
    if displacement not in names or lift not in names:
        return None
    h, cl = names.index(displacement), names.index(lift)
    peak = int(reference.peak_indices[h])
    if peak < 0:
        return dict(frequency=None, reference=None, prediction=None)
    phases = []
    for spectrum in (reference, prediction):
        difference = spectrum.phases[peak, cl]-spectrum.phases[peak, h]
        phases.append(_finite(np.angle(np.exp(1j*difference))))
    return dict(frequency=float(reference.frequencies[peak]), reference=phases[0], prediction=phases[1])


def _spectral_report(record, folder, kind, *, phase_comparison=False):
    """Analyze the already computed prediction, avoiding a second IVP solve."""
    names = record['variable_names']
    spectra = [compute_frequency_analysis(record['time_star'], record[key], names,
                                          fft_window='rectangular') for key in ('reference', 'prediction')]
    reference, prediction = spectra
    rows = compare_frequency_analysis(reference, prediction)
    for column, name in enumerate(names):
        rows[name].update(reference_peak_to_peak=float(np.ptp(record['reference'][:, column])/2),
                          prediction_peak_to_peak=float(np.ptp(record['prediction'][:, column])/2))
    phases = {str(cylinder): _phase_pair(reference, prediction, cylinder) for cylinder in (0, 1)
              if f'h_{cylinder}' in names} if phase_comparison else {}
    payload = dict(kind=kind, integration_complete=True, fft_window='rectangular', peak_to_peak_definition='(max(x)-min(x))/2',
                   model=prediction.summary(), reference=reference.summary(), comparison=rows,
                   phase_CL_minus_h=phases.get('0'), phase_CL_minus_h_by_cylinder=phases,
                   metrics_path=str(record['metrics_path']))
    _write_json(folder/f'{kind}_frequency_amplitude.json', payload)
    pd.DataFrame.from_dict(rows, orient='index').rename_axis('variable').to_csv(
        folder/f'{kind}_frequency_amplitude.csv')
    figure, axes = create_subplots(len(names), 1, figsize=fft_figure_size(len(names)), squeeze=False)
    try:
        for column, name in enumerate(names):
            ax = axes[column, 0]
            for spectrum, label in ((reference, 'Reference'), (prediction, 'SINDy')):
                visible = (spectrum.frequencies >= 0) & (spectrum.frequencies <= 2)
                ax.plot(spectrum.frequencies[visible], spectrum.amplitudes[visible, column], label=label,
                        linewidth=LINE_WIDTH, linestyle='--' if label == 'SINDy' else '-')
            ax.set(xlim=(0, 2), ylabel=rf'$|{name}|$')
            ax.set_box_aspect(1)
            ax.legend()
        axes[-1, 0].set_xlabel(r'$f^*=fD/U_\infty$')
        save_figure(figure, folder/f'{kind}_fft_comparison.png')
    finally:
        plt.close(figure)
    return payload


def _missing_spectral_report(folder, kind, names, row):
    """Replace stale spectra from an earlier successful run with the current status."""
    _write_json(folder/f'{kind}_frequency_amplitude.json', dict(
        kind=kind, integration_complete=row['complete'], model=None, reference=None,
        comparison=None, phase_CL_minus_h=None, error=row.get('error'), solver=row.get('solver'),
        tstar_region=row.get('tstar_region'), requested_tstar_region=row['requested_tstar_region']))
    pd.DataFrame(index=names, columns=['prediction_frequency', 'prediction_peak_to_peak',
                                     'prediction_dominant_amplitude']).rename_axis('variable').to_csv(
        folder/f'{kind}_frequency_amplitude.csv', na_rep='N/A')
    (folder/f'{kind}_fft_comparison.png').unlink(missing_ok=True)


def _nrmse_heatmap(rows, names, vrs, path, *, title=None, weights=False, xlabel=r'$V_R$', row_labels=None):
    """Plot variable/term rows against VR, model or L/D columns, including N/A cells.

    ``vrs`` supplies the column labels; ``xlabel`` describes their meaning.
    ``row_labels`` optionally simplifies display labels without changing CSV keys.
    Coefficient maps retain signed values and use a symmetric color scale.
    """
    matrix = np.array([[np.nan if row['values'].get(name) is None else row['values'][name]
                        for row in rows] for name in names], dtype=float)
    figure, ax = create_subplots(1, 1, figsize=(max(6, len(vrs)*.9), max(3, len(names)*.55)), squeeze=True)
    try:
        cmap = plt.get_cmap('coolwarm' if weights else 'viridis').copy()
        cmap.set_bad('lightgray')
        options = {}
        if weights:
            finite = np.abs(matrix[np.isfinite(matrix)])
            limit = max(float(finite.max()) if len(finite) else 0., 1e-12)
            options = dict(vmin=-limit, vmax=limit)
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect='equal', cmap=cmap, **options)
        ax.set_xticks(range(len(vrs)), [str(vr) for vr in vrs])
        ax.set_yticks(range(len(names)), [rf'${name}$' for name in (names if row_labels is None else row_labels)])
        ax.set_xlabel(xlabel)
        style_heatmap(ax, image, matrix, weights=weights)
        if title:
            ax.set_title(title)
        heatmap_colorbar(ax, image, 'Coefficient' if weights else 'NRMSE')
        fit_heatmap_annotations(figure, ax)
        save_figure(figure, path)
    finally:
        plt.close(figure)
    pd.DataFrame(matrix, index=names, columns=vrs).rename_axis('variable').to_csv(path.with_suffix('.csv'), na_rep='N/A')


def _coefficient_table(directory, layout, parameterization, vr, lpd=None, *, substitute_parameters=True,
                       configurations=None):
    """Read validated coefficients, optionally substituting the selected parameters.

    With ``substitute_parameters=False``, preserve the complete tensor-feature
    names (including VR powers) and fitted coefficients in library order.
    The default contracts the parameter library to effective response coefficients.
    Explicit ``configurations`` supply response/parameter library descriptors
    for manually entered CSVs that have no training.json.
    """
    # Read the feature index explicitly, preserving numeric-looking names such as 1.
    table = pd.read_csv(directory/'model.csv', index_col=0, converters={0: str})
    if parameterization == 'individual':
        return table
    metadata_path = directory/'training.json'
    if metadata_path.is_file():
        configs = json.loads(metadata_path.read_text(encoding='utf-8'))['configurations']
    elif configurations is not None:
        configs = configurations
    else:
        raise FileNotFoundError(f'Provide library configurations for manual coefficients: {directory}')
    response_names = list(layout.response_names)
    parameter_names = {'vr':['V_R'], 'lpd':['L/D'], 'vr_lpd':['L/D','V_R']}[parameterization]
    parameter_values = {'V_R':vr, 'L/D':lpd}
    if any(parameter_values[name] is None for name in parameter_names):
        raise ValueError('Effective coefficient heatmaps require every selected parameter value.')
    response = prepare_library(configs['state_feature_library'])
    parameters = prepare_library(configs['parameter_feature_library'])
    library = ps.ParameterizedLibrary(feature_library=response, parameter_library=parameters,
                                      num_features=len(response_names), num_parameters=len(parameter_names))
    library.fit(np.ones((1,len(response_names)+len(parameter_names))))
    features = library.get_feature_names(response_names+parameter_names)
    response_features = response.get_feature_names(response_names)
    parameter_features = parameters.get_feature_names(parameter_names)
    targets = add_dot_text(list(layout.state_names))
    if (table.index.has_duplicates or table.columns.has_duplicates or set(features) != set(table.index)
            or set(targets) != set(table.columns)):
        raise ValueError(f'Saved coefficient names do not match the parameterized library: {directory}')
    coefficients = table.loc[features,targets].to_numpy(dtype=float)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError('Fitted coefficients must be finite.')
    if not substitute_parameters:
        return pd.DataFrame(coefficients,index=features,columns=targets)
    parameter_sample = np.array([[parameter_values[name] for name in parameter_names]], dtype=float)
    parameter_weights = np.asarray(parameters.transform(parameter_sample))[0]
    tensor = coefficients.reshape(len(parameter_features),len(response_features),len(targets))
    effective = np.tensordot(parameter_weights,tensor,axes=(0,0))
    if not np.all(np.isfinite(effective)):
        raise ValueError('Effective coefficients must be finite.')
    return pd.DataFrame(effective,index=response_features,columns=targets)


def _tensor_coefficient_heatmap(directories, selection, layout, folder, *, configurations=None):
    """Save the fitted VR-tensor coefficients without substituting VR values.

    Single-cylinder selections have one model column. Tandem selections have
    one column per L/D, with a separate heatmap for each acceleration equation.
    PNG/CSV/JSON companions preserve the full feature names, signed values and
    source model paths. Every VR at a given L/D must resolve to the same model.
    ``configurations`` supports manual models without saved training metadata.
    """
    tables, sources = {}, {}
    for lpd in selection.lpds:
        paths = {directories[vr if lpd is None else (lpd,vr)].resolve()
                 for vr in selection.vrs}
        if len(paths) != 1:
            raise ValueError('A tensor coefficient column requires one VR-parameterized model per L/D.')
        directory = paths.pop()
        tables[lpd] = _coefficient_table(directory,layout,'vr',selection.vrs[0],lpd,
                                         substitute_parameters=False,configurations=configurations)
        sources['model' if lpd is None else str(lpd)] = str(directory/'model.csv')
    terms = list(next(iter(tables.values())).index)
    if any(set(table.index) != set(terms) for table in tables.values()):
        raise ValueError('Tensor coefficient heatmap requires the same features across L/D.')
    targets = [rf'\ddot{{h}}_{cylinder}' for cylinder in layout.cylinder_ids]
    columns = ['SINDy'] if selection.cylinder_type=='single' else list(selection.lpds)
    folder.mkdir(parents=True,exist_ok=True)
    display_labels = [' '.join(factor for factor in term.split() if factor!='1') or '1' for term in terms]
    for cylinder,target in zip(layout.cylinder_ids,targets):
        rows = [dict(**({'model':'SINDy'} if lpd is None else {'LpD':lpd}),
                     values={term:float(table.loc[term,target]) for term in terms})
                for lpd,table in tables.items()]
        stem = 'hddot_tensor_coefficients' if selection.cylinder_type=='single' else f'hddot{cylinder}_tensor_coefficients'
        _nrmse_heatmap(rows,terms,columns,folder/f'{stem}_heatmap.png',weights=True,
                       xlabel='Model' if selection.cylinder_type=='single' else r'$L/D$',
                       row_labels=display_labels)
        _write_json(folder/f'{stem}_summary.json',dict(
            target=target,targets=[target],parameterization='vr',features=terms,records=rows,
            coefficient_definition='fitted tensor coefficients with VR terms retained; no parameter substitution',
            column_axis='model' if selection.cylinder_type=='single' else 'LpD',
            source_models=sources))


def _weight_heatmaps(directories, selection, options, layout):
    """Save effective coefficient maps and, for VR models, the full tensor map.

    Effective maps have one column per selected VR and one map per target/L/D.
    Additional tensor maps keep VR terms and aggregate L/D, one map per target.
    """
    folder = _summary_folder(directories, selection, options)/'coefficients'
    mode = options['parameterization']
    configurations = None
    if mode != 'individual' and any(not (directory/'training.json').is_file()
                                     for directory in directories.values()):
        configurations = dict(
            state_feature_library=json.loads(Path(options['state_feature_library_case']).read_text(encoding='utf-8')),
            parameter_feature_library=json.loads(Path(options['input_feature_library_case']).read_text(encoding='utf-8')))
    for lpd in selection.lpds:
        tables = {vr:_coefficient_table(directories[vr if lpd is None else (lpd,vr)],layout,mode,vr,lpd,
                                        configurations=configurations)
                  for vr in selection.vrs}
        terms = list(tables[selection.vrs[0]].index)
        if any(set(terms) != set(table.index) for table in tables.values()):
            raise ValueError('Coefficient heatmap requires the same response terms across VR.')
        group_folder = folder if lpd is None else folder/f'LpD = {lpd}'
        group_folder.mkdir(parents=True,exist_ok=True)
        targets = [rf'\ddot{{h}}_{cylinder}' for cylinder in layout.cylinder_ids]
        for cylinder,target in zip(layout.cylinder_ids,targets):
            labels = terms
            rows = []
            for vr,table in tables.items():
                values = {term:float(table.loc[term,target]) for term in terms}
                rows.append(dict(VR=vr,**({'LpD':lpd} if lpd is not None else {}),values=values))
            stem = 'hddot_coefficients' if selection.cylinder_type=='single' else f'hddot{cylinder}_coefficients'
            _nrmse_heatmap(rows,labels,list(selection.vrs),group_folder/f'{stem}_heatmap.png',weights=True)
            _write_json(group_folder/f'{stem}_summary.json',dict(
                target=target,targets=[target],parameterization=mode,records=rows,
                coefficient_definition='response coefficients evaluated at each selected VR/L/D' if mode!='individual'
                                       else 'individual fitted coefficients',
                source_models={str(vr):str(directories[vr if lpd is None else (lpd,vr)]/'model.csv')
                               for vr in selection.vrs},
                ))
    if mode == 'vr':
        _tensor_coefficient_heatmap(directories,selection,layout,folder,configurations=configurations)


def _trend(rows, names, folder, kind, *, phase_comparison=False, cylinder_ids=(0,)):
    """Save reference/model parameter curves; NaN predictions leave visible gaps."""
    quantities = [
        ('frequency', 'reference_frequency', 'prediction_frequency', 'Dominant Frequency'),
        ('amplitude_peak_to_peak', 'reference_peak_to_peak', 'prediction_peak_to_peak', 'Amplitude, Peak-to-Peak'),
        ('amplitude_fft', 'reference_dominant_amplitude', 'prediction_dominant_amplitude', 'Amplitude, Dominant FFT'),
    ]
    table = []
    for name in names:
        for quantity, ref_key, pred_key, description in quantities:
            figure, ax = create_subplots(1, 1, figsize=(6, 4), squeeze=True)
            try:
                for key, label in ((ref_key, 'Reference'), (pred_key, 'SINDy')):
                    values = [((row.get('spectral') or {}).get('comparison', {}).get(name, {}).get(key)) for row in rows]
                    ax.plot([row['VR'] for row in rows], [np.nan if x is None else x for x in values], 'o', label=label,
                            linewidth=LINE_WIDTH, markersize=3, linestyle='--' if label == 'SINDy' else '-')
                ax.set(xlabel=r'$V_R$', ylabel=rf'${name}$')
                title = rf'{description} (${name}$)' if quantity == 'frequency' else description
                ax.set_title(title, fontsize=14)
                ax.set_xticks([row['VR'] for row in rows])
                ax.legend()
                save_figure(figure, folder/f'{kind}_{quantity}_{channel_token(name)}_vs_VR.png')
            finally:
                plt.close(figure)
        for row in rows:
            values = (row.get('spectral') or {}).get('comparison', {}).get(name, {})
            table.append(dict(VR=row['VR'], **({'LpD':row['LpD']} if 'LpD' in row else {}),
                              variable=name, complete=row['complete'], **values))
    pd.DataFrame(table).to_csv(folder/f'{kind}_frequency_amplitude_summary.csv', index=False, na_rep='N/A')
    if not phase_comparison:
        return
    for cylinder in cylinder_ids:
        figure, ax = create_subplots(1, 1, figsize=(6, 4), squeeze=True)
        try:
            phases = [(row.get('spectral') or {}).get('phase_CL_minus_h_by_cylinder', {}).get(str(cylinder))
                      for row in rows]
            phase_table = []
            for key, label in (('reference', 'Reference'), ('prediction', 'SINDy')):
                values = [(phase or {}).get(key) for phase in phases]
                ax.plot([row['VR'] for row in rows], [np.nan if v is None else v/np.pi for v in values], 'o', label=label,
                        linewidth=LINE_WIDTH, markersize=3, linestyle='--' if label == 'SINDy' else '-')
            for row, phase in zip(rows, phases):
                phase_table.append(dict(VR=row['VR'], **({'LpD':row['LpD']} if 'LpD' in row else {}), **(phase or {})))
            ax.set(xlabel=r'$V_R$', ylabel=rf'$(\phi_{{C_{{L{cylinder}}}}}-\phi_{{h_{cylinder}}})/\pi$', ylim=(-1.05, 1.05))
            ax.set_xticks([row['VR'] for row in rows])
            ax.legend()
            suffix = 'phase_CL_minus_h' if len(cylinder_ids) == 1 else f'phase_CL{cylinder}_minus_h{cylinder}'
            save_figure(figure, folder/f'{kind}_{suffix}_vs_VR.png')
            pd.DataFrame(phase_table).to_csv(folder/f'{kind}_{suffix}.csv', index=False, na_rep='N/A')
        finally:
            plt.close(figure)


def _model_directories(options, selection):
    """Resolve one saved-model directory per selected VR/L/D trajectory.

    Use training selectors, library/filter identities and fit-window metadata;
    missing or ambiguous matches raise instead of selecting an arbitrary model.
    """
    directories = {}
    for lpd in selection.lpds:
        for vr in selection.vrs:
            base = model_base_directory(options['results_root'], selection.case_name, options['dynamics_case'],
                Path(options['differentiator_case']).stem,
                'individual' if options['parameterization'] == 'individual' else 'parametric_'+options['parameterization'],
                selection.cylinder_type, options['parameterization'], options['state_feature_library_case'],
                options['input_feature_library_case'], options['optimizer_case'], selection.re, selection.mstar, vr, lpd,
                filtering_case=options.get('filtering_case'))
            key = vr if lpd is None else (lpd, vr)
            directories[key] = resolve_model_directory(base, selection.cylinder_type, options['parameterization'], vr, lpd,
                training_vr_list=options['training_vr_list'], training_lpd_list=options['training_lpd_list'],
                fit_window_case=options['fit_window_case'])
    return directories


def _summary_folder(directories, selection, options):
    """Build the common summary path identifying model, reference and evaluation windows.

    The returned Path includes the selected VR/L/D set and requested time bounds;
    this helper constructs the path without creating directories.
    """
    folder = Path(os.path.commonpath([str(path.resolve()) for path in directories.values()]))/'evaluation'/'summary'
    folder /= 'reference='+Path(options['differentiator_case']).stem
    folder /= 'evaluation_window='+selection.window_name
    trajectories = [dict(VR=vr, LpD=lpd, requested_tstar0=selection.bounds[(lpd, vr)][0],
                         requested_tstarf=selection.bounds[(lpd, vr)][1])
                    for lpd in selection.lpds for vr in selection.vrs]
    return training_directory(folder, 'vr' if selection.cylinder_type == 'single' else 'vr_lpd',
                              trajectories, options['fit_window_case'])


def evaluate_suite(
    dynamics_case, differentiator_case, state_feature_library_case, optimizer_case,
    fit_window_case, derivative_window_cases, integral_window_case, spectral_window_cases, *,
    cylinder_type='single', parameterization='individual', vr_list=None, lpd_list=None,
    training_vr_list=None, training_lpd_list=None,
    input_feature_library_case=None, phase_comparison=False, weight_heatmap=False,
    method='RK45', max_state_factor=10., rtol=1e-6, atol=1e-9, results_root='results',
    filtering_case=None,
    model_directories=None,
):
    """Run the checklist's derivative, IVP, spectral, amplitude and coefficient diagnostics.

    Each VR/L/D trajectory is loaded, predicted, analyzed and released before proceeding.
    Failed trajectories are recorded as N/A without blocking the next trajectory.
    Case 4/5 automatically receive an additional coupled-motion IVP.

    ``model_directories`` optionally maps every selected VR (single) or
    (L/D, VR) pair (tandem) to an explicit saved-model folder. For manual CSVs
    without training.json, training selectors are omitted from model lookup;
    evaluation windows and explicit library/name validation still apply. The
    fit-window argument then labels reports without asserting the model was fitted.
    """
    validate_max_state_factor(max_state_factor)
    if weight_heatmap and dynamics_case != 3:
        raise ValueError('Coefficient heatmap requires Case 3 models.')
    if phase_comparison and dynamics_case not in (1, 2):
        raise ValueError('h/CL phase comparison requires full-state Case 1 or 2.')
    if parameterization not in ('individual', 'vr', 'lpd', 'vr_lpd'):
        raise ValueError('Unknown parameterization.')
    if cylinder_type == 'single' and (parameterization in ('lpd', 'vr_lpd') or training_lpd_list is not None):
        raise ValueError('L/D parameterization and training_lpd_list require tandem data.')
    if (parameterization == 'individual') != (input_feature_library_case is None):
        raise ValueError('Only parametric models require input_feature_library_case.')
    selection = data_selection(cylinder_type, vr_list, lpd_list)
    if parameterization in ('vr', 'vr_lpd') and training_vr_list is None:
        training_vr_list = list(selection.vrs)
    if parameterization in ('lpd', 'vr_lpd') and training_lpd_list is None:
        training_lpd_list = list(selection.lpds)
    options = dict(dynamics_case=dynamics_case, differentiator_case=differentiator_case,
        state_feature_library_case=state_feature_library_case, input_feature_library_case=input_feature_library_case,
        optimizer_case=optimizer_case, fit_window_case=fit_window_case, parameterization=parameterization,
        training_vr_list=training_vr_list, training_lpd_list=training_lpd_list, results_root=results_root,
        filtering_case=filtering_case)
    lpds = None if cylinder_type == 'single' else selection.lpds
    windows = [data_selection(cylinder_type, selection.vrs, lpds, fit_window_case=path) for path in derivative_window_cases]
    integral_selection = data_selection(cylinder_type, selection.vrs, lpds, fit_window_case=integral_window_case)
    spectral_paths = {Path(path).resolve() for path in spectral_window_cases}
    if not spectral_paths.issubset({Path(path).resolve() for path in derivative_window_cases}):
        raise ValueError('Spectral windows must also be derivative evaluation windows.')
    if model_directories is None:
        directories = _model_directories(options, selection)
    else:
        keys = {vr if lpd is None else (lpd,vr) for lpd,vr in selection.bounds}
        if set(model_directories) != keys:
            raise ValueError('Explicit model directories must cover exactly the selected trajectories.')
        directories = {key:Path(path).resolve() for key,path in model_directories.items()}
        for path in directories.values():
            if not (path/'model.csv').is_file():
                raise FileNotFoundError(f'No model.csv in explicit model directory: {path}')
    failures, summaries = [], []
    tasks = [('derivative', window, Path(path).resolve() in spectral_paths)
             for path, window in zip(derivative_window_cases, windows)]
    tasks.append(('integral', integral_selection, True))
    if dynamics_case in (4, 5):
        tasks.append(('coupled_integral', integral_selection, True))
    layout = get_case_layout(dynamics_case, cylinder_type)
    for kind, window, spectral in tasks:
        names = list(layout.state_names)
        if kind == 'derivative':
            names = add_dot_text(names)
        elif kind == 'coupled_integral':
            names = [name for cylinder in layout.cylinder_ids for name in
                     (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', f'C_{{L{cylinder}}}', rf'\dot{{C}}_{{L{cylinder}}}')]
        folder = _summary_folder(directories, window, options)
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for lpd, vr in window.bounds:
            key = vr if lpd is None else (lpd, vr)
            start, stop = window.bounds[(lpd, vr)]
            row = dict(VR=vr, complete=False, values={}, requested_tstar_region=[start, stop])
            report_folder = directories[key]/'evaluation'
            if lpd is not None:
                row['LpD'] = lpd
                report_folder /= f'LpD = {lpd}'
            individual = training_directory(report_folder/f'VR = {vr}'/'analysis',
                'individual', [dict(VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop)])
            individual /= 'evaluation_window='+window.window_name
            print(f'{kind}: VR={vr}'+('' if lpd is None else f', L/D={lpd}')+
                  f', {window.window_name}, t*={start}..{stop}', flush=True)
            record = None
            try:
                evaluation_options = options
                if model_directories is not None:
                    evaluation_options = dict(options,model_dir=directories[key])
                    if not (directories[key]/'training.json').is_file():
                        evaluation_options.update(fit_window_case=None,training_vr_list=None,training_lpd_list=None)
                record = evaluate_sindy(**evaluation_options, cylinder_type=cylinder_type, vr_list=[vr],
                    lpd_list=None if lpd is None else [lpd],
                    evaluation_type='derivative' if kind == 'derivative' else 'integral',
                    coupled_motion=kind == 'coupled_integral', tstar0=start, tstarf=stop,
                    method=method, max_state_factor=max_state_factor, rtol=rtol, atol=atol,
                    save_plot=True, return_data=True)[key]
                if record['variable_names'] != names:
                    raise ValueError('Evaluation returned unexpected state order.')
                row.update(complete=record.get('integration_complete', True),
                           metrics_path=str(record['metrics_path']), solver=record['solver'],
                           tstar_region=record['tstar_region'])
                if row['complete']:
                    row['values'] = {name: _finite(record['metrics'][name]['NRMSE']) for name in names}
                    if spectral:
                        row['spectral'] = _spectral_report(record, individual, kind,
                                                         phase_comparison=phase_comparison and kind == 'integral')
            except (ValueError, RuntimeError, OSError, FloatingPointError, OverflowError) as error:
                row['error'] = str(error)
                failures.append(dict(kind=kind, window=window.window_name, VR=vr,
                                     **({'LpD':lpd} if lpd is not None else {}), error=str(error)))
                print(f'ERROR: {kind}, dataset={key}: {error}; continuing.', flush=True)
            finally:
                # A spectral/plotting failure must not retain a large trajectory into the next VR.
                record = None
            if spectral and 'spectral' not in row:
                _missing_spectral_report(individual, kind, names, row)
            rows.append(row)
        for lpd in window.lpds:
            group = rows if lpd is None else [row for row in rows if row['LpD'] == lpd]
            group_folder = folder if lpd is None else folder/f'LpD = {lpd}'
            group_folder.mkdir(parents=True, exist_ok=True)
            _nrmse_heatmap(group, names, list(window.vrs), group_folder/f'{kind}_nrmse_heatmap.png')
            if spectral:
                _trend(group, names, group_folder, kind, phase_comparison=phase_comparison and kind == 'integral',
                       cylinder_ids=layout.cylinder_ids)
        if cylinder_type == 'two_tandem':
            frame = pd.DataFrame([dict(VR=row['VR'], LpD=row['LpD'], **row['values']) for row in rows],
                                 columns=['VR', 'LpD', *names])
            frame.to_csv(folder/f'{kind}_nrmse_summary.csv', index=False, na_rep='N/A')
            save_nrmse_heatmaps(frame, names, folder, kind+'_nrmse', cylinder_type=cylinder_type,
                               vrs=window.vrs, lpds=window.lpds,
                               figure_scale=TANDEM_IVP_HEATMAP_SCALE
                               if kind in ('integral', 'coupled_integral') else 1.)
        path = folder/f'{kind}_suite_summary.json'
        _write_json(path, dict(kind=kind, cylinder_type=cylinder_type, variables=names, evaluation_window=window.window_name,
                              records=rows, failures=[row for row in failures if row['kind'] == kind and row['window'] == window.window_name]))
        summaries.append(str(path))
    if weight_heatmap:
        _weight_heatmaps(directories,selection,options,layout)
    return dict(summaries=summaries, failures=failures)


def evaluate_single_suite(*args, **options):
    """Retain the existing single-cylinder Python entry point."""
    return evaluate_suite(*args, cylinder_type='single', **options)


def evaluate_two_tandem_suite(*args, **options):
    """Evaluate both tandem cylinders across VR and L/D selections."""
    return evaluate_suite(*args, cylinder_type='two_tandem', **options)
