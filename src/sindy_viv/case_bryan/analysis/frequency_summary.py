"""Read saved raw-data FFT summaries and plot dominant frequency trends."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_windows import channel_token, data_selection, save_figure, analysis_directory
from .frequency import FFT_WINDOWS
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.plot import (
    LINE_WIDTH, create_subplots, select_plot_variables, set_axis_fonts,
)


def _report_bounds(path: Path, payload: dict) -> tuple:
    """Read requested bounds, including older reports that stored them in filenames."""
    window = payload.get('fit_window') or {}
    if 'requested_tstar0' in window and 'requested_tstarf' in window:
        return window['requested_tstar0'], window['requested_tstarf']
    label = path.stem.removeprefix('frequency_analysis_')
    if label.startswith('cylinder_'):
        label = label.split('_', 2)[2]
    if label.startswith('full_range'):
        return None, None
    try:
        start, stop = label.split('_')[:2]
        return tuple(None if value == 'None' else float(value) for value in (start, stop))
    except (ValueError, TypeError):
        raise ValueError(f'Cannot identify the requested FFT interval in {path}. Rerun frequency analysis.') from None


def _load_channels(folder, bounds, fit_window_case, fft_window, names):
    """Merge compatible channel subsets; reject conflicting duplicate FFT results."""
    expected_window = None if fit_window_case is None else Path(fit_window_case).stem
    matches = {name: [] for name in names}
    for path in sorted(folder.glob('frequency_analysis_*.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as error:
            raise ValueError(f'Cannot read saved frequency analysis {path}: {error}') from error
        if payload.get('fft_window', 'rectangular') != fft_window:
            continue
        if (payload.get('fit_window') or {}).get('name') != expected_window:
            continue
        if _report_bounds(path, payload) != bounds:
            continue
        if payload.get('frequency_coordinate') != 'f*=cycles/t*':
            raise ValueError(f'Expected frequencies in f*=cycles/t*, as produced by frequency analysis: {path}')
        channels = payload.get('channels')
        if not isinstance(channels, dict):
            raise ValueError(f'No dominant-frequency channels in {path}. Rerun frequency analysis.')
        for name in names:
            if name in channels:
                matches[name].append((path, payload, channels[name]))
    missing = [name for name, candidates in matches.items() if not candidates]
    if missing:
        raise FileNotFoundError(
            f'Run frequency analysis first for {folder}, channels {missing}, '
            f'fft_window={fft_window}, fit_window={expected_window}, t*={bounds}. '
            'Use matching --fit_window_case or --tstar0/--tstarf and --fft_window.')
    result = {}
    for name, candidates in matches.items():
        first_path, first_payload, first_channel = candidates[0]
        if not isinstance(first_channel, dict) or not {'frequency', 'amplitude', 'phase_rad'} <= first_channel.keys():
            raise ValueError(f'Invalid dominant-frequency summary for {name} in {first_path}. Rerun frequency analysis.')
        # Display limits and channel subsets can produce equivalent duplicate reports.
        # Compare the numerical summary and sample interval rather than choosing a newest file.
        signature = (first_channel, first_payload.get('tstar_region'),
                     first_payload.get('sample_count'), first_payload.get('frequency_resolution'))
        if any((channel, payload.get('tstar_region'), payload.get('sample_count'),
                payload.get('frequency_resolution')) != signature for _, payload, channel in candidates[1:]):
            paths = ', '.join(str(path) for path, _, _ in candidates)
            raise ValueError(f'Conflicting saved FFT results for {name}: {paths}. '
                             'Keep consistent reports for these settings or use a separate --results_root.')
        frequency = first_channel.get('frequency')
        if frequency is not None and (isinstance(frequency, bool) or not isinstance(frequency, (int, float))
                                      or not np.isfinite(frequency) or frequency <= 0):
            raise ValueError(f'Invalid dominant frequency for {name} in {first_path}.')
        result[name] = dict(**first_channel,
            source_files=[str(path) for path, _, _ in candidates],
            tstar_region=first_payload.get('tstar_region'),
            frequency_resolution=first_payload.get('frequency_resolution'),
            sample_count=first_payload.get('sample_count'))
    return result


def plot_frequency_vr_summary(
    *, cylinder_type='single', vr_list=None, lpd_list=None, fit_window_case=None,
    tstar0=None, tstarf=None, fft_window='rectangular', plot_variables=None,
    plot_cylinders=None, include_natural_frequency=False, surface=False,
    results_root='results', output_dir=None, show=False,
) -> list[Path]:
    """Plot saved dominant f* versus VR, with L/D curves and an optional surface.

    Input JSONs are written by frequency_analysis_single.py/two_tandem.py.
    Window/taper arguments identify previously analyzed records. No raw data is
    loaded and no FFT is computed. Missing reports fail before output is written;
    unresolved (constant) channels remain null/NaN gaps rather than zero peaks.
    Matching settings overwrite readable PNG, CSV and JSON filenames. The CSV
    includes amplitudes, phases, resolution and source files for each data point.
    """
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS}.')
    if surface and cylinder_type != 'two_tandem':
        raise ValueError('The L/D-by-VR surface requires two_tandem data.')
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    names = select_plot_variables(selection.names, plot_variables, plot_cylinders)
    rows, trajectories = [], []
    for lpd in sorted(selection.lpds, key=lambda value: -1 if value is None else value):
        for vr in sorted(selection.vrs):
            source = analysis_directory(results_root, selection.case_name, cylinder_type,
                                        selection.re, selection.mstar, lpd=lpd, vr=vr)/'frequency_analysis'
            start, stop = selection.bounds[(lpd, vr)]
            channels = _load_channels(source, (start, stop), fit_window_case, fft_window, names)
            trajectories.append(dict(VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop))
            for name, channel in channels.items():
                rows.append(dict(VR=vr, LpD=lpd, variable=name, natural_frequency=1/vr,
                                 requested_tstar0=start, requested_tstarf=stop, **channel))
    frame = pd.DataFrame(rows)
    folder = Path(output_dir) if output_dir is not None else analysis_directory(
        results_root, selection.case_name, cylinder_type, selection.re, selection.mstar)/'frequency_vr_summary'
    folder /= f'fft_window={fft_window}'
    folder = training_directory(folder, 'vr' if cylinder_type == 'single' else 'vr_lpd',
                                trajectories, fit_window_case)
    folder /= 'natural_frequency='+('on' if include_natural_frequency else 'off')
    paths = []
    for name in names:
        values = frame[frame.variable == name]
        stem = f'frequency_{channel_token(name)}_vs_VR'
        fig, ax = create_subplots(1, figsize=(6, 4))
        for index, lpd in enumerate(sorted(selection.lpds, key=lambda value: -1 if value is None else value)):
            group = values if lpd is None else values[values.LpD == lpd]
            ax.plot(group.VR, group.frequency.to_numpy(dtype=float),
                    marker=('o', 's', '^', 'D', 'v')[index % 5], markersize=3,
                    linewidth=LINE_WIDTH, label=f'${name}$' if lpd is None else f'$L/D={lpd:g}$')
        if include_natural_frequency:
            reference_vrs = np.linspace(min(selection.vrs), max(selection.vrs), 200)
            ax.plot(reference_vrs, 1/reference_vrs, color='0.4', linestyle='--',
                    linewidth=LINE_WIDTH, label=r'$f_N^*=1/V_R$')
        ax.set(xlabel=r'$V_R$', ylabel=r'$f^*=fD/U_\infty$')
        ax.set_xticks(sorted(selection.vrs))
        ax.set_title(f'Dominant Frequency (${name}$)', fontsize=14)
        if cylinder_type == 'two_tandem' or include_natural_frequency:
            ax.legend(frameon=False, fontsize=10)
        paths.append(save_figure(fig, folder/(stem+'.png'), show))
        if surface:
            grid = values.pivot(index='VR', columns='LpD', values='frequency').sort_index().sort_index(axis=1)
            xx, yy = np.meshgrid(grid.columns.to_numpy(), grid.index.to_numpy())
            zz = grid.to_numpy(dtype=float)
            fig = plt.figure(figsize=(8, 6))
            fig.subplots_adjust(left=.02, right=.82, bottom=.08, top=.90)
            ax = fig.add_subplot(projection='3d')
            if min(grid.shape) >= 2 and np.isfinite(zz).all():
                ax.plot_surface(xx, yy, zz, cmap='viridis')
            else:
                finite = np.isfinite(zz)
                ax.scatter(xx[finite], yy[finite], zz[finite], c=zz[finite], cmap='viridis')
            if include_natural_frequency:
                for index, lpd in enumerate(grid.columns):
                    ax.plot(np.full(len(grid.index), lpd), grid.index, 1/grid.index.to_numpy(),
                            color='0.4', linestyle='--', linewidth=LINE_WIDTH,
                            label=r'$f_N^*=1/V_R$' if index == 0 else None)
                ax.legend(frameon=False, fontsize=10)
            ax.set(xlabel=r'$L/D$', ylabel=r'$V_R$', zlabel=r'$f^*=fD/U_\infty$')
            ax.set_xticks(grid.columns)
            ax.set_yticks(grid.index)
            ax.set_title(f'Dominant Frequency (${name}$)', fontsize=14)
            set_axis_fonts(ax)
            paths.append(save_figure(fig, folder/f'frequency_{channel_token(name)}_surface.png', show, tight=False))
        export = values.copy()
        for column in ('source_files', 'tstar_region'):
            export[column] = export[column].map(json.dumps)
        export.to_csv(folder/(stem+'.csv'), index=False)
        records = values.astype(object).where(values.notna(), None).to_dict(orient='records')
        payload = dict(frequency_coordinate='f*=cycles/t*', fft_window=fft_window,
                       fit_window=None if fit_window_case is None else str(fit_window_case),
                       variable=name, include_natural_frequency=include_natural_frequency,
                       records=records)
        (folder/(stem+'.json')).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return paths
