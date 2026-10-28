"""Compare raw/filtered time histories and FFTs without differentiation or fitting."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .data_windows import analysis_directory, channel_token, data_selection, load_trajectory, save_figure
from .frequency import FFT_WINDOWS, FrequencyAnalysis, compute_frequency_analysis
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.paths import number_text, training_directory
from sindy_viv.case_bryan.plot import LINE_WIDTH, create_subplots, fft_figure_size, format_parameter_title, plot_compare, select_plot_variables
from sindy_viv.utils import initiate_object, load_config


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write finite JSON summaries, preserving undefined frequencies as null."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')


def _plot_time_series(
    tstar: np.ndarray, raw: np.ndarray, filtered: np.ndarray, names: Sequence[str],
    path: Path, title: str, show: bool,
) -> Path:
    """Overlay measurements with the standard time-series layout, sizing and axis styling."""
    figure, _ = plot_compare(
        tstar, raw, filtered, names, {}, xlim=(float(tstar[0]), float(tstar[-1])),
        ylim_mean_time_start=float(tstar[0]), ylim_mean_time_end=float(tstar[-1]),
        comparison_labels=('Raw', 'Filtered'),
    )
    try:
        figure.suptitle(title, fontsize=14)
        return save_figure(figure, path, show, tight=False)
    finally:
        plt.close(figure)


def _plot_spectra(
    raw: FrequencyAnalysis, filtered: FrequencyAnalysis, path: Path,
    title: str, xlim: Sequence[float] | None, show: bool,
) -> Path:
    """Overlay the same frequency bins for every selected raw/filtered channel."""
    figure, axes = create_subplots(len(raw.variable_names), squeeze=False,
                                  figsize=fft_figure_size(len(raw.variable_names)))
    try:
        for column, (name, ax) in enumerate(zip(raw.variable_names, axes[:, 0])):
            ax.plot(raw.frequencies, raw.amplitudes[:, column], label='Raw', color='black', linewidth=LINE_WIDTH)
            ax.plot(filtered.frequencies, filtered.amplitudes[:, column], label='Filtered',
                    color='red', linewidth=LINE_WIDTH, linestyle='--')
            ax.set(ylabel=rf'$|{name}|$', xlim=xlim or (0., float(raw.frequencies[-1])))
            ax.set_box_aspect(1)
            if column == 0:
                ax.legend(frameon=False, fontsize=10)
        axes[-1, 0].set_xlabel(r'$f^*=fD/U_\infty$')
        figure.suptitle(title, fontsize=14)
        return save_figure(figure, path, show)
    finally:
        plt.close(figure)


def _plot_aggregate(
    records: list[dict[str, Any]], variable: str, axis: str, folder: Path, show: bool,
) -> Path:
    """Plot dominant frequency and its amplitude against VR or L/D, with gaps."""
    horizontal, grouping = ('VR', 'LpD') if axis == 'vr' else ('LpD', 'VR')
    rows = [row for row in records if row['variable'] == variable]
    groups = sorted({row[grouping] for row in rows}, key=lambda value: -1 if value is None else value)
    figure, axes = create_subplots(2, figsize=(6, 10.5))
    try:
        for ax in axes:
            ax.set_box_aspect(1)
        for index, value in enumerate(groups):
            group = sorted((row for row in rows if row[grouping] == value), key=lambda row: row[horizontal])
            x = [row[horizontal] for row in group]
            group_label = None if value is None else (f'$L/D={value:g}$' if grouping == 'LpD' else f'$V_R={value:g}$')
            marker = ('o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*')[index % 10]
            for kind, style, color in (('raw', ':', 'black'), ('filtered', '-', 'red')):
                for ax, metric in zip(axes, ('frequency', 'amplitude')):
                    y = [np.nan if row[f'{kind}_{metric}'] is None else row[f'{kind}_{metric}'] for row in group]
                    ax.plot(x, y, color=color, linestyle=style, marker=marker, markersize=3,
                            linewidth=LINE_WIDTH, label=kind.capitalize() if group_label is None else
                            group_label if kind == 'filtered' else '_nolegend_')
        axes[0].set(ylabel=r'Dominant $f^*$', title=f'Filtering Frequency Comparison (${variable}$)')
        axes[1].set(ylabel='Dominant Bin Amplitude', xlabel=r'$V_R$' if axis == 'vr' else r'$L/D$')
        axes[1].set_xticks(sorted({row[horizontal] for row in rows}))
        if groups == [None]:
            axes[0].legend(frameon=False, fontsize=10)
        else:
            axes[0].legend(frameon=False, fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1.))
            axes[1].legend(handles=[Line2D([], [], color=color, linestyle=style,
                                           markersize=3, linewidth=LINE_WIDTH, label=label)
                                    for label, style, color in (('Raw', ':', 'black'), ('Filtered', '-', 'red'))],
                           frameon=False, fontsize=10, ncol=2)
        return save_figure(figure, folder/f'filtering_{channel_token(variable)}_vs_{horizontal}.png', show)
    finally:
        plt.close(figure)


def assess_filtering(
    filtering_case: str | Path, *, cylinder_type: str = 'single',
    vr_list: Sequence[float] | None = None, lpd_list: Sequence[float] | None = None,
    fit_window_case: str | Path | None = None, tstar0: float | None = None, tstarf: float | None = None,
    fft_window: str = 'rectangular', amplitude_floor: float = 0.,
    plots: Literal['individual', 'aggregate', 'both'] = 'both',
    aggregate_axis: Literal['vr', 'lpd', 'both'] = 'both',
    plot_variables: Sequence[str] | None = None, plot_cylinders: Sequence[int] | None = None,
    xlim: Sequence[float] | None = None, save_spectra: bool = False,
    time_series: bool = True,
    results_root: str | Path = 'results', output_dir: str | Path | None = None, show: bool = False,
) -> dict[str, Any]:
    """Diagnose smoothing with time histories, FFTs and 2D parameter summaries.

    Filter full loaded measurements in tau, then crop raw and filtered arrays
    to identical t* windows and compute FFTs in f*=cycles/t*. No derivative,
    model, or prior frequency report is required. Load/release one trajectory
    at a time; only channel summaries are retained across VR/L/D selections.

    plots selects individual time-series/spectrum figures, aggregate figures,
    or both. time_series=False omits time-history figures and their JSON files.
    Time histories use recorded t* on the horizontal axis and one raw/filtered
    overlay per selected channel, with no timestamp shifts or resampling.
    Aggregates show dominant frequency and its amplitude versus VR (one curve
    pair per L/D), and versus L/D (one pair per VR) for tandem data. Single
    data use VR when aggregate_axis='both'; requesting 'lpd' alone is invalid.
    Constant/unresolved frequencies remain null in JSON and gaps in plots.

    filtering_case uses the fitting workflow's module/class_name/kwargs config.
    fit_window_case or explicit tstar0/tstarf select reporting bounds; xlim
    affects only the individual spectrum display, never dominant-bin selection.
    plot_variables/plot_cylinders select measured channels. fft_window and
    amplitude_floor follow compute_frequency_analysis. save_spectra additionally
    exports full raw/filtered FFT bins in NPZ files. show displays saved figures.

    results_root uses the existing physical analysis directories; output_dir
    overrides the root, with separate cylinder/individual/aggregate folders.
    Config, channel, FFT, floor, dataset and time folders identify outputs.
    Matching settings overwrite PNG/CSV/JSON/NPZ files. Always save per-record
    JSON and aggregate CSV/JSON, regardless of the chosen figure mode. Return
    paths and small summaries, without retaining signals or full FFT arrays.
    """
    if plots not in ('individual', 'aggregate', 'both'):
        raise ValueError('plots must be individual, aggregate, or both.')
    if aggregate_axis not in ('vr', 'lpd', 'both'):
        raise ValueError('aggregate_axis must be vr, lpd, or both.')
    if cylinder_type == 'single' and aggregate_axis == 'lpd':
        raise ValueError('L/D aggregate plots require two_tandem data.')
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS}.')
    if not np.isscalar(amplitude_floor) or not np.isfinite(amplitude_floor) or amplitude_floor < 0:
        raise ValueError('amplitude_floor must be finite and nonnegative.')
    if xlim is not None:
        xlim = tuple(xlim)
        if len(xlim) != 2 or not np.all(np.isfinite(xlim)) or not 0 <= xlim[0] < xlim[1]:
            raise ValueError('xlim must contain increasing nonnegative frequency limits.')
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    names = select_plot_variables(selection.names, plot_variables, plot_cylinders)
    config = load_config(filtering_case)
    filtering = initiate_object(**config)
    if not callable(filtering):
        raise ValueError('The filtering config must construct a callable filter.')
    settings = (Path('filtering='+Path(filtering_case).stem)/f'fft_window={fft_window}'
                /('amplitude_floor='+number_text(amplitude_floor))/('channels='+'_'.join(map(channel_token, names))))
    snapshot = dict(filtering=dict(name=Path(filtering_case).stem, source=str(filtering_case), config=config),
                    processing_order='load in tau; filter full record; crop both signals; FFT in t*',
                    frequency_coordinate='f*=cycles/t*', filtering_coordinate='tau=t*/VR',
                    fft_window=fft_window, amplitude_floor=float(amplitude_floor), variables=names,
                    spectrum_display_xlim=None if xlim is None else list(xlim),
                    fit_window=None if fit_window_case is None else str(fit_window_case),
                    peak_definition='independent dominant positive-frequency bin for raw and filtered signals')
    trajectories, records, paths, reports, time_series_paths = [], [], [], [], []
    for lpd in selection.lpds:
        for vr in selection.vrs:
            time, data = load_trajectory(selection, vr, lpd, names)
            filtered_data = np.asarray(filtering(data, time))
            if filtered_data.shape != data.shape or not np.all(np.isfinite(filtered_data)):
                raise ValueError('Filtering must return finite measurements with the same shape as its input.')
            start, stop = selection.bounds[(lpd, vr)]
            region = select_time_region(time, vr, start, stop)
            tstar = time[region].copy()*vr
            raw_values, filtered_values = data[region].copy(), filtered_data[region].copy()
            total_samples = len(time)
            del time, data, filtered_data
            raw = compute_frequency_analysis(tstar, raw_values, names,
                                             fft_window=fft_window, amplitude_floor=amplitude_floor)
            filtered = compute_frequency_analysis(tstar, filtered_values, names,
                                                  fft_window=fft_window, amplitude_floor=amplitude_floor)
            metadata = dict(VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop,
                            tstar_region=list(map(float, tstar[[0, -1]])), sample_count=len(tstar),
                            total_samples=total_samples, frequency_resolution=float(raw.frequencies[1]))
            trajectories.append(metadata)
            raw_summary, filtered_summary = raw.summary(), filtered.summary()
            for column, name in enumerate(names):
                before, after = raw_summary['channels'][name], filtered_summary['channels'][name]
                raw_peak = int(raw.peak_indices[column])
                amplitude_at_raw_peak = None if raw_peak < 0 else float(filtered.amplitudes[raw_peak, column])
                records.append(dict(**metadata, variable=name,
                    raw_frequency=before['frequency'], filtered_frequency=after['frequency'],
                    frequency_shift=None if before['frequency'] is None or after['frequency'] is None
                                    else after['frequency']-before['frequency'],
                    raw_amplitude=before['amplitude'], filtered_amplitude=after['amplitude'],
                    filtered_amplitude_at_raw_peak=amplitude_at_raw_peak,
                    amplitude_ratio_at_raw_peak=None if raw_peak < 0 else amplitude_at_raw_peak/before['amplitude']))
            if output_dir is None:
                folder = analysis_directory(results_root, selection.case_name, cylinder_type,
                                            selection.re, selection.mstar, lpd=lpd, vr=vr)/'filtering_diagnostics'
            else:
                folder = Path(output_dir)/cylinder_type/'individual'
                if lpd is not None:
                    folder /= f'LpD = {lpd}'
                folder /= f'VR = {vr}'
            folder = training_directory(folder/settings, 'individual', [metadata], fit_window_case)
            report = folder/'filtering_frequency.json'
            _write_json(report, dict(**snapshot, **metadata, raw=raw_summary, filtered=filtered_summary))
            reports.append(report)
            if save_spectra:
                np.savez_compressed(folder/'filtering_spectra.npz', frequencies=raw.frequencies,
                                    raw_amplitudes=raw.amplitudes, filtered_amplitudes=filtered.amplitudes,
                                    variable_names=np.asarray(names))
            if plots in ('individual', 'both'):
                parameters = {'V_R': vr}
                if lpd is not None:
                    parameters['L/D'] = lpd
                case_label = format_parameter_title(parameters)
                if time_series:
                    time_path = _plot_time_series(tstar, raw_values, filtered_values, names,
                        folder/'filtering_time_series.png', case_label, show)
                    paths.append(time_path)
                    time_series_paths.append(time_path)
                    _write_json(time_path.with_suffix('.json'), dict(**snapshot, **metadata,
                        time_coordinate='t*', plot_type='raw_filtered_time_series', figure=time_path.name))
                display = 'full' if xlim is None else '..'.join(map(number_text, xlim))
                title = f'Filtering Spectra\n{case_label}'
                paths.append(_plot_spectra(raw, filtered, folder/f'display_frequency={display}'/'filtering_spectra.png',
                                           title, xlim, show))
            del tstar, raw, filtered, raw_values, filtered_values
    base = (analysis_directory(results_root, selection.case_name, cylinder_type, selection.re, selection.mstar)
            /'filtering_diagnostics' if output_dir is None else Path(output_dir)/cylinder_type/'aggregate')
    folder = training_directory(base/settings, 'vr' if cylinder_type == 'single' else 'vr_lpd', trajectories, fit_window_case)
    folder.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda row: (-1 if row['LpD'] is None else row['LpD'], row['VR'], names.index(row['variable'])))
    csv_path, json_path = folder/'filtering_frequency_summary.csv', folder/'filtering_frequency_summary.json'
    frame = pd.DataFrame(records)
    frame['tstar_region'] = frame['tstar_region'].map(json.dumps)
    frame.to_csv(csv_path, index=False, na_rep='N/A')
    _write_json(json_path, dict(**snapshot, records=records))
    if plots in ('aggregate', 'both'):
        axes = ('vr',) if cylinder_type == 'single' else ('vr', 'lpd') if aggregate_axis == 'both' else (aggregate_axis,)
        for axis in axes:
            for name in names:
                paths.append(_plot_aggregate(records, name, axis, folder, show))
    return dict(figure_paths=paths, individual_reports=reports, summary_csv=csv_path,
                summary_json=json_path, records=records, time_series_paths=time_series_paths)
