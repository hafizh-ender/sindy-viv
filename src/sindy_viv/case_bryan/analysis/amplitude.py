"""Compute and persist oscillation amplitudes, then plot saved parameter trends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from sindy_viv.case_bryan.analysis.data_windows import (
    data_selection, iter_raw_windows, channel_token, save_figure, analysis_directory,
)
from sindy_viv.case_bryan.analysis.frequency import compute_frequency_analysis, FFT_WINDOWS
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.plot import LINE_WIDTH, create_subplots, select_plot_variables, set_axis_fonts

DEFAULT_AMPLITUDE_METHOD = 'peak_to_peak'
AMPLITUDE_METHODS = ('peak_to_peak', 'hilbert', 'rms', 'fft')
AMPLITUDE_LABELS = {
    'peak_to_peak': 'half peak-to-peak',
    'hilbert': 'mean Hilbert envelope',
    'rms': r'$\sqrt{2}$ times centered RMS',
    'fft': 'dominant FFT component',
}
AMPLITUDE_TITLES = {
    'peak_to_peak': 'Time-Series Amplitude',
    'hilbert': 'Hilbert Envelope Amplitude',
    'rms': 'RMS Amplitude',
    'fft': 'FFT Amplitude',
}
AMPLITUDE_DEFINITIONS = {
    'peak_to_peak': '(max(x) - min(x))/2 over the selected time window',
    'hilbert': 'mean(abs(hilbert(x - mean(x)))) over the selected time window',
    'rms': 'sqrt(2*mean((x - mean(x))**2)) over the selected time window',
    'fft': 'coherent-gain-corrected amplitude at the dominant positive-frequency FFT bin',
}


def compute_signal_amplitudes(
    tstar: np.ndarray, data: np.ndarray, variable_names: Sequence[str], *,
    method: str = DEFAULT_AMPLITUDE_METHOD, fft_window: str = 'rectangular',
) -> dict[str, float]:
    """Return oscillation amplitudes after removing each channel's mean.

    The default peak_to_peak is half the range of the selected time series.
    It uses window-wide extrema, rather than averaging amplitudes over cycles.
    fft uses the dominant positive-frequency bin and coherent-gain-corrected
    amplitude from analysis.frequency. hilbert is the mean analytic-envelope
    magnitude; rms is sqrt(2)*RMS of the centered signal; peak_to_peak is half
    its range. The latter three are whole-signal measures, not Fourier peaks.
    Hilbert edge effects are retained. Input samples must be uniform in t*.
    """
    if method not in AMPLITUDE_METHODS:
        raise ValueError(f'method must be one of {AMPLITUDE_METHODS}.')
    if method == 'fft':
        summary = compute_frequency_analysis(tstar, data, variable_names, fft_window=fft_window).summary()
        return {name: row['amplitude'] for name, row in summary['channels'].items()}
    time, values = np.asarray(tstar, dtype=float), np.asarray(data)
    names = list(variable_names)
    if values.ndim == 1:
        values = values[:, None]
    if (time.ndim != 1 or len(time) < 2 or values.shape != (len(time), len(names))
            or not names or len(set(names)) != len(names) or values.dtype.kind not in 'biuf'
            or not np.all(np.isfinite(time)) or not np.all(np.isfinite(values))):
        raise ValueError('Amplitude data must have finite, aligned time and uniquely named real channels.')
    dt = np.diff(time)
    if dt[0] <= 0 or not np.allclose(dt, dt[0], rtol=1e-5, atol=dt[0]*1e-4):
        raise ValueError('Amplitude analysis requires uniform increasing t* samples.')
    centered = values.astype(float)-values.mean(axis=0)
    if method == 'hilbert':
        amplitudes = np.abs(hilbert(centered, axis=0)).mean(axis=0)
    elif method == 'rms':
        amplitudes = np.sqrt(2*np.mean(centered**2, axis=0))
    else:
        amplitudes = np.ptp(centered, axis=0)/2
    return dict(zip(names, map(float, amplitudes)))


def _method_directory(base, selection, method, fft_window) -> Path:
    """Group analysis settings and physical dataset labels in readable folders."""
    if method not in AMPLITUDE_METHODS:
        raise ValueError(f'method must be one of {AMPLITUDE_METHODS}.')
    folder = Path(base)/f'method={method}'
    if method == 'fft':
        if fft_window not in FFT_WINDOWS:
            raise ValueError(f'fft_window must be one of {FFT_WINDOWS}.')
        folder /= f'fft_window={fft_window}'
    return folder


def _amplitude_path(selection, results_root, method, fft_window, vr, lpd, *, fit_window_case=None) -> Path:
    """Use individual-model dataset/time folders and a stable amplitude filename."""
    base = analysis_directory(results_root, selection.case_name, selection.cylinder_type,
                              selection.re, selection.mstar)/'amplitude'/'data'
    folder = _method_directory(base, selection, method, fft_window)
    if selection.cylinder_type == 'two_tandem':
        folder /= f'LpD = {lpd}'
    folder /= f'VR = {vr}'
    start, stop = selection.bounds[(lpd, vr)]
    trajectories = [dict(VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop)]
    return training_directory(folder, 'individual', trajectories, fit_window_case)/'amplitude.json'


def compute_amplitude(
    vr_list: Sequence[float] | None = None, *, cylinder_type: str = 'single',
    lpd_list: Sequence[float] | None = None, fit_window_case: str | Path | None = None,
    tstar0: float | None = None, tstarf: float | None = None,
    variables: Sequence[str] | None = None, plot_cylinders: Sequence[int] | None = None,
    method: str = DEFAULT_AMPLITUDE_METHOD, fft_window: str = 'rectangular', results_root: str | Path = 'results',
) -> dict:
    """Load raw CSVs, select per-case t* windows, and save amplitude JSON/CSV.

    Returns {VR: record} for single cylinders or {(LpD, VR): record} for tandem.
    Each record contains amplitudes, processing metadata and its saved path.
    variables accepts the existing families (h, hdot, CL, ...) or exact names.
    The default amplitude is half the peak-to-peak range in the selected window.
    Matching files overwrite prior results, consistent with other analyses.
    """
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    names = select_plot_variables(selection.names, variables, plot_cylinders)
    # Validate settings before loading the first full trajectory.
    _amplitude_path(selection, results_root, method, fft_window, selection.vrs[0], selection.lpds[0],
                    fit_window_case=fit_window_case)
    records = {}
    for vr, lpd, time, values, metadata in iter_raw_windows(selection, names):
        amplitudes = compute_signal_amplitudes(time, values, names, method=method, fft_window=fft_window)
        path = _amplitude_path(selection, results_root, method, fft_window, vr, lpd,
                               fit_window_case=fit_window_case)
        payload = dict(**metadata, method=method, fft_window=fft_window if method == 'fft' else None,
                       amplitude_definition=AMPLITUDE_DEFINITIONS[method],
                       mean_removed=True, amplitudes=amplitudes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        pd.Series(amplitudes, name='amplitude').rename_axis('variable').to_csv(path.with_suffix('.csv'))
        records[vr if lpd is None else (lpd, vr)] = dict(**payload, path=path)
        del time, values
    return records


def _plot_amplitudes(
    cylinder_type: str, vr_list=None, *, lpd_list=None, fit_window_case=None,
    tstar0=None, tstarf=None, variables=None, plot_cylinders=None,
    method=DEFAULT_AMPLITUDE_METHOD, fft_window='rectangular', results_root='results', output_dir=None, show=False,
) -> list[Path]:
    """Read exact saved analysis settings; plot individual channels without recomputation."""
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    names = select_plot_variables(selection.names, variables, plot_cylinders)
    rows, sources, trajectories = [], [], []
    for lpd in selection.lpds:
        for vr in selection.vrs:
            path = _amplitude_path(selection, results_root, method, fft_window, vr, lpd,
                                   fit_window_case=fit_window_case)
            payload = json.loads(path.read_text(encoding='utf-8'))
            if any(name not in payload['amplitudes'] for name in names):
                raise ValueError(f'Compute all requested amplitude channels first: {path}')
            rows.append(dict(VR=vr, LpD=lpd, **{name: payload['amplitudes'][name] for name in names}))
            sources.append(str(path))
            trajectories.append(payload)
    frame = pd.DataFrame(rows)
    folder = Path(output_dir) if output_dir is not None else analysis_directory(
        results_root, selection.case_name, cylinder_type, selection.re, selection.mstar)/'amplitude'
    folder = _method_directory(folder, selection, method, fft_window)
    folder = training_directory(folder, 'vr' if cylinder_type == 'single' else 'vr_lpd',
                                trajectories, fit_window_case)
    folder.mkdir(parents=True, exist_ok=True)
    stem = 'amplitude_summary'
    frame.to_csv(folder/(stem+'.csv'), index=False)
    paths = []
    for name in names:
        token = channel_token(name)
        fig, ax = create_subplots(1, figsize=(6, 4))
        for index, lpd in enumerate(selection.lpds):
            group = frame if lpd is None else frame[frame.LpD == lpd]
            group = group.sort_values('VR')
            ax.plot(group.VR, group[name], marker=('o', 's', '^', 'D', 'v')[index % 5],
                    linewidth=LINE_WIDTH, markersize=3,
                    label=None if lpd is None else f'$L/D={lpd:g}$')
        ax.set(xlabel=r'$V_R$', ylabel=f'${name}$')
        ax.set_title(AMPLITUDE_TITLES[method], fontsize=14)
        if cylinder_type == 'two_tandem':
            ax.legend(frameon=False, fontsize=10)
        paths.append(save_figure(fig, folder/f'amplitude_{token}_vs_VR.png', show))
        if cylinder_type == 'single':
            continue
        fig, ax = create_subplots(1, figsize=(6, 4))
        for index, (vr, group) in enumerate(frame.groupby('VR', sort=True)):
            group = group.sort_values('LpD')
            ax.plot(group.LpD, group[name], marker=('o', 's', '^', 'D', 'v')[index % 5],
                    linewidth=LINE_WIDTH, markersize=3, label=f'$V_R={vr:g}$')
        ax.set(xlabel=r'$L/D$', ylabel=f'${name}$')
        ax.set_title(AMPLITUDE_TITLES[method], fontsize=14)
        ax.legend(frameon=False, fontsize=10)
        paths.append(save_figure(fig, folder/f'amplitude_{token}_vs_LpD.png', show))
        grid = frame.pivot(index='LpD', columns='VR', values=name).sort_index().sort_index(axis=1)
        xx, yy = np.meshgrid(grid.columns.to_numpy(), grid.index.to_numpy())
        fig = plt.figure(figsize=(8, 6))
        # Keep space for the projected z label: Matplotlib's tight bounding box
        # can crop 3D labels even when the plotted surface itself fits.
        fig.subplots_adjust(left=.02, right=.82, bottom=.08, top=.90)
        ax = fig.add_subplot(projection='3d')
        if min(grid.shape) >= 2:
            ax.plot_surface(xx, yy, grid.to_numpy(), cmap='viridis')
        else:
            ax.scatter(xx.ravel(), yy.ravel(), grid.to_numpy().ravel(), c=grid.to_numpy().ravel())
        ax.set(xlabel=r'$V_R$', ylabel=r'$L/D$', zlabel=f'${name}$')
        ax.set_title(AMPLITUDE_TITLES[method], fontsize=14)
        set_axis_fonts(ax)
        paths.append(save_figure(fig, folder/f'amplitude_{token}_surface.png', show, tight=False))
    (folder/(stem+'.json')).write_text(json.dumps(dict(method=method, source_files=sources,
        amplitude_definition=AMPLITUDE_DEFINITIONS[method],
        fft_window=fft_window if method == 'fft' else None, variables=names,
        figures=list(map(str, paths))), indent=2), encoding='utf-8')
    return paths


def plot_amplitude_single(vr_list=None, **kwargs) -> list[Path]:
    """Load saved amplitudes and save one VR-versus-amplitude figure per channel.

    kwargs match compute_amplitude selectors, with output_dir/show for figures.
    No raw measurements or amplitudes are recomputed.
    """
    return _plot_amplitudes('single', vr_list, **kwargs)


def plot_amplitude_two_tandem(vr_list=None, **kwargs) -> list[Path]:
    """Plot saved amplitudes versus VR, versus L/D, and on their joint surface."""
    return _plot_amplitudes('two_tandem', vr_list, **kwargs)
