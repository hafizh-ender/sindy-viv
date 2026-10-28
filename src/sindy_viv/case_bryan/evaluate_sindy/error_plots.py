"""Aggregate saved evaluation metrics without rerunning SINDy or simulation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sindy_viv.case_bryan.analysis.data_windows import data_selection, channel_token, save_figure
from sindy_viv.case_bryan.analysis.error_heatmaps import (
    TANDEM_IVP_HEATMAP_SCALE, save_nrmse_heatmaps, style_heatmap_grid,
)
from sindy_viv.case_bryan.model_storage import model_base_directory, resolve_model_directory
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.plot import select_plot_variables
from .model_evaluation import model_differentiator_name


def _matching_report(report_dir, kind, reference_name, start, stop, *, recursive=False, integration_method=None):
    """
    Select one report by kind, differentiator and requested t* sample bounds.

    Recorded endpoints may differ from requested limits by at most one sampling
    step. Ambiguous windows are rejected; files are never chosen by modification
    time. Missing files raise FileNotFoundError and are optionally plotted as gaps.
    """
    matches = []
    candidates = report_dir.rglob(kind+'_error_metrics_*.json') if recursive else report_dir.glob(kind+'_error_metrics_*.json')
    for path in candidates:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('evaluation') != kind:
            continue
        if payload.get('integration_complete') is False:
            continue  # A truncated trajectory is not a complete-window NRMSE result.
        if reference_name is not None and payload.get('reference_differentiator') != reference_name:
            continue
        if integration_method is not None and (payload.get('solver') or {}).get('method') != integration_method:
            continue
        first, last = payload['tstar_region']
        step = (last-first)/max(payload['sample_count']-1, 1)
        tolerance = max(abs(first), abs(last), 1)*1e-12
        if start is not None and not -tolerance <= first-start <= step+tolerance:
            continue
        if stop is not None and not -tolerance <= stop-last <= step+tolerance:
            continue
        matches.append((path, payload))
        
    if not matches:
        raise FileNotFoundError(f'No {kind} report matching the evaluation bounds/differentiator in {report_dir}. Run evaluation first.')
    
    if len(matches) != 1:
        raise ValueError(f'Multiple {kind} reports match in {report_dir}; specify tstar0/tstarf and evaluation_differentiator_case.')
    
    return matches[0]


def plot_saved_errors(
    dynamics_case, optimizer_case, *, evaluation_type='derivative', cylinder_type='single',
    parameterization='individual', differentiator_case=None, state_feature_library_case=None,
    input_feature_library_case=None, vr_list=None, lpd_list=None, weak=False,
    weak_feature_library_case=None, fit_window_case=None, training_vr_list=None,
    training_lpd_list=None, model_dir=None, evaluation_differentiator_case=None,
    tstar0=None, tstarf=None, coupled_motion=False, results_root='results',
    output_dir=None, plot_variables=None, plot_cylinders=None, allow_missing=False, show=False,
    evaluation_window_case=None, heatmap_only=False,
) -> list[Path]:
    """Read saved NRMSE reports; plot VR curves and tandem L/D-by-VR heatmaps.

    Model/training selectors match evaluate_sindy; fit_window_case identifies
    the trained model, while tstar0/tstarf select existing evaluation reports.
    All channels retain their saved NRMSE normalization (a ratio, not percent).
    Matching summary files overwrite. A CSV/JSON companion records source files,
    actual time windows, missing cases and the plotted numerical values.
    Readable dataset, training-window, reference and evaluation-time folders
    replace hash suffixes, including beneath an explicit output_dir.
    evaluation_window_case supplies per-trajectory scoring bounds and is
    mutually exclusive with tstar0/tstarf. It is independent of fit_window_case,
    which identifies training. heatmap_only saves title-free heatmaps for both
    cylinder types, with a single row across VR for single-cylinder data.
    """
    if evaluation_type not in ('derivative', 'integral'):
        raise ValueError('evaluation_type must be derivative or integral.')
    if coupled_motion and (evaluation_type != 'integral' or dynamics_case not in (4, 5)):
        raise ValueError('coupled_motion supports only integral Cases 4/5.')
    if parameterization not in ('individual', 'vr', 'lpd', 'vr_lpd'):
        raise ValueError('Unknown parameterization.')
    if parameterization in ('lpd', 'vr_lpd') and cylinder_type == 'single':
        raise ValueError('L/D parameterization requires tandem data.')
    if state_feature_library_case is None or (parameterization != 'individual' and input_feature_library_case is None):
        raise ValueError('Provide the state library and, for parametric models, the input library.')
    if parameterization == 'individual' and input_feature_library_case is not None:
        raise ValueError('Individual models do not use an input feature-library config.')
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case=evaluation_window_case,
                               tstar0=tstar0, tstarf=tstarf)
    diff_name = model_differentiator_name(dynamics_case, differentiator_case, weak)
    reference_case = evaluation_differentiator_case or differentiator_case
    reference_name = Path(reference_case).stem if reference_case is not None else None
    if evaluation_type == 'integral' and dynamics_case in (1, 3):
        reference_name = 'no_differentiator'
    mode = 'individual' if parameterization == 'individual' else 'parametric_'+parameterization
    if weak:
        mode = 'weak_'+mode
    kind = 'coupled_integral' if coupled_motion else evaluation_type
    rows, sources, missing, directories = [], [], [], []
    for lpd in selection.lpds:
        for vr in selection.vrs:
            base = model_base_directory(results_root, selection.case_name, dynamics_case, diff_name, mode,
                cylinder_type, parameterization, state_feature_library_case, input_feature_library_case,
                optimizer_case, selection.re, selection.mstar, vr, lpd,
                weak_feature_library_case=weak_feature_library_case)
            try:
                directory = resolve_model_directory(base, cylinder_type, parameterization, vr, lpd,
                    model_dir=model_dir, training_vr_list=training_vr_list, training_lpd_list=training_lpd_list,
                    fit_window_case=fit_window_case)
                report_dir = directory/'evaluation'
                if lpd is not None:
                    report_dir /= f'LpD = {lpd}'
                report_dir /= f'VR = {vr}'
                start, stop = selection.bounds[(lpd, vr)]
                path, payload = _matching_report(report_dir, kind, reference_name, start, stop)
            except FileNotFoundError as error:
                if not allow_missing:
                    raise
                missing.append(dict(VR=vr, LpD=lpd, reason=str(error)))
                continue
            directories.append(directory.resolve())
            sources.append(dict(VR=vr, LpD=lpd, path=str(path), tstar_region=payload['tstar_region'],
                                requested_tstar0=start, requested_tstarf=stop))
            metrics = payload['metrics']
            rows.append(dict(VR=vr, LpD=lpd, **{name: value['NRMSE'] for name, value in metrics.items()}))
    if not rows:
        raise FileNotFoundError('No matching evaluation reports were found for any selected trajectory.')
    available = [name for name in rows[0] if name not in ('VR', 'LpD')]
    if any(set(row)-{'VR', 'LpD'} != set(available) for row in rows):
        raise ValueError('Saved reports have inconsistent modeled channels; select a single dynamics case/model family.')
    names = select_plot_variables(available, plot_variables, plot_cylinders)
    frame = pd.DataFrame(rows)
    for row in missing:
        frame.loc[len(frame)] = dict(VR=row['VR'], LpD=row['LpD'])
    frame = frame.sort_values(['LpD', 'VR'], na_position='first')
    folder = Path(output_dir) if output_dir is not None else Path(os.path.commonpath(directories))/'evaluation'/'summary'
    if reference_name is not None:
        folder /= f'reference={reference_name}'
    if evaluation_window_case is not None:
        folder /= 'evaluation_window='+Path(evaluation_window_case).stem
    trajectories = [dict(VR=vr, LpD=lpd, requested_tstar0=selection.bounds[(lpd, vr)][0],
                         requested_tstarf=selection.bounds[(lpd, vr)][1])
                    for lpd in selection.lpds for vr in selection.vrs]
    folder = training_directory(folder, 'vr' if cylinder_type == 'single' else 'vr_lpd',
                                trajectories, fit_window_case)
    folder.mkdir(parents=True, exist_ok=True)
    stem = kind+'_nrmse'
    frame[['VR', 'LpD', *names]].to_csv(folder/(stem+'_summary.csv'), index=False)
    paths = []
    if heatmap_only:
        paths = save_nrmse_heatmaps(frame, names, folder, stem, cylinder_type=cylinder_type,
                                   vrs=selection.vrs, lpds=selection.lpds, show=show,
                                   figure_scale=TANDEM_IVP_HEATMAP_SCALE
                                   if cylinder_type == 'two_tandem' and evaluation_type == 'integral' else 1.)
    for name in ([] if heatmap_only else names):
        fig, ax = plt.subplots(figsize=(6, 4))
        for lpd in selection.lpds:
            group = frame if lpd is None else frame[frame.LpD == lpd]
            ax.plot(group.VR, group[name], 'o-', label=None if lpd is None else f'$L/D={lpd:g}$')
        ax.set(xlabel=r'$V_R$', ylabel='NRMSE', title=f'{kind}: ${name}$')
        ax.grid(alpha=.3)
        if cylinder_type == 'two_tandem':
            ax.legend()
        fig.tight_layout()
        paths.append(save_figure(fig, folder/f'{stem}_{channel_token(name)}_vs_VR.png', show))
        if cylinder_type == 'single':
            continue
        grid = frame.pivot(index='LpD', columns='VR', values=name).reindex(
            index=sorted(selection.lpds), columns=sorted(selection.vrs))
        fig, ax = plt.subplots(figsize=(max(6, len(grid.columns)*.6), max(4, len(grid.index)*.6)))
        cmap = plt.get_cmap('viridis').copy()
        cmap.set_bad('lightgray')
        image = ax.imshow(grid.to_numpy(dtype=float), origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(grid.columns)), [f'{vr:g}' for vr in grid.columns])
        ax.set_yticks(range(len(grid.index)), [f'{lpd:g}' for lpd in grid.index])
        ax.set(xlabel=r'$V_R$', ylabel=r'$L/D$', title=f'{kind}: ${name}$')
        style_heatmap_grid(ax, grid.shape)
        fig.colorbar(image, ax=ax, label='NRMSE')
        fig.tight_layout()
        paths.append(save_figure(fig, folder/f'{stem}_{channel_token(name)}_heatmap.png', show))
    (folder/(stem+'_summary.json')).write_text(json.dumps(dict(evaluation=kind, source_reports=sources,
        requested_tstar0=tstar0, requested_tstarf=tstarf, reference_differentiator=reference_name,
        fit_window_case=None if fit_window_case is None else str(fit_window_case), variables=names,
        evaluation_window_case=None if evaluation_window_case is None else str(evaluation_window_case),
        missing=missing, figures=list(map(str, paths))), indent=2, allow_nan=False), encoding='utf-8')
    return paths
