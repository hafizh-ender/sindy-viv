"""Summarize saved numerical-derivative IVP reconstruction errors."""

import json
from pathlib import Path

import pandas as pd

from .data_windows import data_selection
from .error_heatmaps import save_nrmse_heatmaps
from .paths import analysis_directory
from sindy_viv.case_bryan.evaluate_sindy.error_plots import _matching_report
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.plot import select_plot_variables
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout


def plot_derivative_ivp_errors(
    dynamics_case, differentiator_case, *, cylinder_type='single', vr_list=None, lpd_list=None,
    fit_window_case=None, tstar0=None, tstarf=None, integration_method=None,
    plot_variables=None, plot_cylinders=None, allow_missing=False,
    results_root='results', output_dir=None, show=False,
    filtering_case=None,
):
    """Read assess_derivative.py reports and save title-free heatmaps and tables.

    The fit-window JSON supplies evaluation bounds, independently for each VR
    and L/D. It does not identify a fitted SINDy model. No differentiation or
    integration is repeated. Matching reports must already cover the requested
    interval. Modern time folders and migrated reports directly below the
    differentiator folder are both supported; ambiguous matches are rejected.
    filtering_case selects filtered reports and gives them a separate summary folder.
    None uses filtering=None for reports and summaries; legacy unfiltered report
    paths remain readable when no report exists in the explicit None folder.
    """
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    layout = get_case_layout(dynamics_case, cylinder_type)
    names = select_plot_variables(layout.state_names, plot_variables, plot_cylinders)
    differentiator = Path(differentiator_case).stem
    rows, sources, missing, methods = [], [], [], set()
    for lpd in sorted(selection.lpds, key=lambda value: -1 if value is None else value):
        for vr in sorted(selection.vrs):
            start, stop = selection.bounds[(lpd, vr)]
            base = (analysis_directory(results_root, selection.case_name, cylinder_type,
                                      selection.re, selection.mstar, lpd=lpd, vr=vr)
                    /'derivative_assessment'/f'dynamics_case_{dynamics_case}')
            legacy_base = base/differentiator
            base /= 'filtering='+(Path(filtering_case).stem if filtering_case is not None else 'None')
            base /= differentiator
            preferred = training_directory(base, 'individual', [dict(
                VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop)], fit_window_case)
            searches = [(preferred, False), (base, True)]
            if filtering_case is None:
                searches.append((legacy_base, True))
            for directory, recursive in searches:
                try:
                    path, payload = _matching_report(directory, 'derivative_ivp', differentiator, start, stop,
                                                     recursive=recursive, integration_method=integration_method)
                    break
                except FileNotFoundError as error:
                    last_error = error
            else:
                if not allow_missing:
                    raise FileNotFoundError(f'{last_error} Run script/assess_derivative.py for these bounds first.') from last_error
                missing.append(dict(VR=vr, LpD=lpd, reason=str(last_error)))
                continue
            metrics = payload['metrics']
            if any(name not in metrics or 'NRMSE' not in metrics[name] for name in names):
                raise ValueError(f'Assessment report does not contain the requested state metrics: {path}')
            rows.append(dict(VR=vr, LpD=lpd, **{name: metrics[name]['NRMSE'] for name in names}))
            method = (payload.get('solver') or {}).get('method', 'unknown')
            methods.add(method)
            sources.append(dict(VR=vr, LpD=lpd, path=str(path), tstar_region=payload['tstar_region'],
                                requested_tstar0=start, requested_tstarf=stop, integration_method=method))
    if not rows:
        raise FileNotFoundError('No matching derivative IVP reports. Run script/assess_derivative.py first.')
    if len(methods) != 1:
        raise ValueError('Saved IVP reports use different integration methods; select --integration_method.')
    frame = pd.DataFrame(rows)
    for entry in missing:
        frame.loc[len(frame)] = dict(VR=entry['VR'], LpD=entry['LpD'])
    frame = frame.sort_values(['LpD', 'VR'], na_position='first')
    folder = Path(output_dir) if output_dir is not None else (
        analysis_directory(results_root, selection.case_name, cylinder_type, selection.re, selection.mstar)
        /'derivative_assessment'/f'dynamics_case_{dynamics_case}')
    folder /= 'filtering='+(Path(filtering_case).stem if filtering_case is not None else 'None')
    if output_dir is None:
        folder = folder/differentiator/'summary'
    folder /= 'integration_method='+str(next(iter(methods)))
    trajectories = [dict(VR=vr, LpD=lpd, requested_tstar0=selection.bounds[(lpd, vr)][0],
                         requested_tstarf=selection.bounds[(lpd, vr)][1])
                    for lpd in selection.lpds for vr in selection.vrs]
    folder = training_directory(folder, 'vr' if cylinder_type == 'single' else 'vr_lpd',
                                trajectories, fit_window_case)
    folder.mkdir(parents=True, exist_ok=True)
    stem = 'derivative_ivp_nrmse'
    frame.to_csv(folder/(stem+'_summary.csv'), index=False)
    paths = save_nrmse_heatmaps(frame, names, folder, stem, cylinder_type=cylinder_type,
                               vrs=selection.vrs, lpds=selection.lpds, show=show,
                               figure_scale=1.25)
    payload = dict(evaluation='derivative_ivp', dynamics_case=dynamics_case, differentiator=differentiator,
                   variables=names, source_reports=sources, missing=missing,
                   fit_window_case=None if fit_window_case is None else str(fit_window_case),
                   filtering_case=None if filtering_case is None else str(filtering_case),
                   requested_tstar0=tstar0, requested_tstarf=tstarf,
                   integration_method=next(iter(methods)), figures=list(map(str, paths)))
    (folder/(stem+'_summary.json')).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return paths
