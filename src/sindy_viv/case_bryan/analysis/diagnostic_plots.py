"""Raw derivative histories and stacked FFT spectra across physical cases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.analysis.data_windows import (
    data_selection, load_trajectory, iter_raw_windows, channel_token, save_figure, analysis_directory,
)
from sindy_viv.case_bryan.analysis.frequency import compute_frequency_analysis, FFT_WINDOWS
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.paths import number_text, training_directory
from sindy_viv.case_bryan.plot import plot_data, select_plot_variables, set_axis_fonts
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
from sindy_viv.utils import add_dot_text, initiate_object, load_config


def plot_derivatives(
    dynamics_case: int, differentiator_case: str | Path, *, cylinder_type='single',
    vr_list=None, lpd_list=None, fit_window_case=None, tstar0=None, tstarf=None,
    plot_variables=None, plot_cylinders=None, results_root='results', output_dir=None, show=False,
    filtering_case=None,
) -> list[Path]:
    """Differentiate full records in tau, crop in t*, and plot only computed Xdot.

    Case layouts are identical to training/assessment; Cases 2/4/5 also include
    the second lift derivative. No raw derivative comparison or IVP is run.
    Figures use t* horizontally; differentiated values remain rates in tau.
    JSON companions record names, differentiator settings and actual intervals.
    Readable dataset/window folders follow individual fitted-model conventions;
    filenames identify the plotted channels and matching selections overwrite.
    Unfiltered plots and JSON companions use filtering=None between the
    dynamics-case folder and differentiator, alongside named filter folders.
    """
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    layout = get_case_layout(dynamics_case, cylinder_type)
    config = load_config(differentiator_case)
    filtering_config = load_config(filtering_case) if filtering_case is not None else None
    filtering = initiate_object(**filtering_config) if filtering_case is not None else None
    names = add_dot_text(layout.state_names)
    selected = select_plot_variables(names, plot_variables, plot_cylinders)
    stem = 'derivatives_'+'_'.join(channel_token(name) for name in selected)
    paths = []
    for lpd in selection.lpds:
        for vr in selection.vrs:
            time, data = load_trajectory(selection, vr, lpd, layout.measured_names)
            trajectory = prepare_trajectory(time, data, list(layout.measured_names),
                initiate_object(**config), dynamics_case, cylinder_type, filtering=filtering)
            region = select_time_region(time, vr, *selection.bounds[(lpd, vr)])
            tstar = time[region].copy()*vr
            values = trajectory.derivatives[region][:, [names.index(name) for name in selected]].copy()
            del time, data, trajectory
            folder = (Path(output_dir) if output_dir is not None else analysis_directory(
                results_root, selection.case_name, cylinder_type, selection.re, selection.mstar,
                lpd=lpd, vr=vr)/'derivative_plot')
            if output_dir is not None:
                if lpd is not None:
                    folder /= f'LpD = {lpd}'
                folder /= f'VR = {vr}'
            folder /= f'dynamics_case_{dynamics_case}'
            folder /= 'filtering=' + (Path(filtering_case).stem if filtering_case is not None else 'None')
            folder /= Path(differentiator_case).stem
            start, stop = selection.bounds[(lpd, vr)]
            trajectories = [dict(VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop)]
            folder = training_directory(folder, 'individual', trajectories, fit_window_case)
            metadata = dict(VR=vr, LpD=lpd, dynamics_case=dynamics_case, differentiator=config,
                differentiation_scope='full trajectory before cropping', derivative_coordinate='tau=t*/VR',
                time_coordinate='t*', tstar_region=list(map(float, tstar[[0, -1]])),
                requested_bounds=selection.bounds[(lpd, vr)], sample_count=len(tstar),
                fit_window=selection.window_name, variables=selected)
            if filtering_case is not None:
                metadata['filtering'] = dict(name=Path(filtering_case).stem, source=str(filtering_case), config=filtering_config)
            figure, _ = plot_data(tstar, values, selected, {'V_R': vr} if lpd is None else {'V_R': vr, 'L/D': lpd},
                                  ylim_mean_time_start=tstar[0], ylim_mean_time_end=tstar[-1])
            paths.append(save_figure(figure, folder/(stem+'.png'), show, tight=False))
            (folder/(stem+'.json')).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
            del tstar, values
    return paths


def _trace_bins(amplitudes: np.ndarray, maximum: int) -> np.ndarray:
    """Keep native bins; if capped, retain each frequency block's largest peak."""
    count = len(amplitudes)
    if count <= maximum:
        return np.arange(count)
    boundaries = np.linspace(0, count, maximum+1, dtype=int)
    return np.array([first+np.argmax(amplitudes[first:last])
                     for first, last in zip(boundaries[:-1], boundaries[1:])])


def plot_frequency_waterfalls(
    *, cylinder_type='single', vr_list=None, lpd_list=None, fit_window_case=None,
    tstar0=None, tstarf=None, fft_window='rectangular', plot_variables=None, plot_cylinders=None,
    xlim=None, max_frequency_points=2000, results_root='results', output_dir=None, show=False,
) -> list[Path]:
    """Stack independent f*-amplitude curves at their discrete VR coordinates.

    Each VR retains its own native FFT grid, including different-duration fit
    windows. There is no interpolation between spectra and no connecting surface.
    The point limit applies per curve: frequency blocks retain their largest
    native-amplitude bin, preserving peaks without interpolating amplitudes.
    JSON keeps original dominant-bin summaries. NPZ frequencies/amplitudes are
    concatenated vectors; trace_offsets delimit each VR's displayed curve.
    Files have stable channel names beneath FFT, dataset, window and display
    setting folders. Dataset/time folders follow the fitted-model conventions.
    """
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS}.')
    if isinstance(max_frequency_points, bool) or not isinstance(max_frequency_points, int) or max_frequency_points < 2:
        raise ValueError('max_frequency_points must be an integer of at least two.')
    limits = (0., 1.) if xlim is None else tuple(xlim)
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or not 0 <= limits[0] < limits[1]:
        raise ValueError('xlim must contain increasing nonnegative frequency limits.')
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    names = select_plot_variables(selection.names, plot_variables, plot_cylinders)
    paths = []
    for lpd in selection.lpds:
        # Retain only visible FFT amplitudes, never full histories or phase matrices.
        spectra, metadata = [], []
        local = data_selection(cylinder_type, selection.vrs, None if lpd is None else [lpd],
                               fit_window_case, tstar0, tstarf)
        for vr, _, time, values, record in iter_raw_windows(local, names):
            analysis = compute_frequency_analysis(time, values, names, fft_window=fft_window)
            stop = min(limits[1], float(analysis.frequencies[-1]))
            if limits[0] >= stop:
                raise ValueError(f'Frequency limits lie outside Nyquist for VR={vr}, L/D={lpd}.')
            first = int(np.searchsorted(analysis.frequencies, limits[0], side='left'))
            last = int(np.searchsorted(analysis.frequencies, stop, side='right'))
            if first == last:
                raise ValueError(f'The selected band contains no FFT bins for VR={vr}, L/D={lpd}; widen xlim.')
            spectra.append((vr, analysis.frequencies[first:last].copy(), analysis.amplitudes[first:last].copy()))
            metadata.append(dict(**record, frequency_analysis=analysis.summary()))
            del time, values, analysis
        spectra.sort(key=lambda item: item[0])
        metadata.sort(key=lambda item: item['VR'])
        vrs = np.array([row[0] for row in spectra])
        base = (Path(output_dir) if output_dir is not None else analysis_directory(
            results_root, selection.case_name, cylinder_type, selection.re, selection.mstar,
            lpd=lpd)/'frequency_waterfalls')
        if output_dir is not None and lpd is not None:
            base /= f'LpD = {lpd}'
        base /= f'fft_window={fft_window}'
        # L/D already belongs to the parent physical-case folder.
        path_metadata = [dict(record, LpD=None) for record in metadata]
        folder = training_directory(base, 'vr', path_metadata, fit_window_case)
        folder = (folder/f'frequency={number_text(limits[0])}..{number_text(limits[1])}'
                  /f'max_points={max_frequency_points}')
        for j, name in enumerate(names):
            stem = f'frequency_waterfall_{channel_token(name)}'
            fig = plt.figure(figsize=(9, 6))
            fig.subplots_adjust(left=.02, right=.82, bottom=.08, top=.90)
            ax = fig.add_subplot(projection='3d')
            trace_frequencies, trace_amplitudes = [], []
            colors = plt.get_cmap('tab10')
            for index, (vr, frequency, amplitude) in enumerate(spectra):
                bins = _trace_bins(amplitude[:, j], max_frequency_points)
                xx, zz = frequency[bins], amplitude[bins, j]
                ax.plot(xx, np.full(len(xx), vr), zz, color=colors(index % 10), linewidth=1,
                        marker='.' if len(xx) == 1 else None, label=f'$V_R={vr:g}$')
                trace_frequencies.append(xx)
                trace_amplitudes.append(zz)
            ax.set_yticks(vrs, [f'{vr:g}' for vr in vrs])
            margin = .5 if len(vrs) == 1 else .08*(vrs[-1]-vrs[0])
            ax.set_ylim(vrs[0]-margin, vrs[-1]+margin)
            ax.set_xlim(*limits)
            ax.set(xlabel=r'$f^*=fD/U_\infty$', ylabel=r'$V_R$', zlabel=f'$|{name}|$')
            ax.set_title('Frequency Waterfall'+('' if lpd is None else f' ($L/D = {lpd:g}$)'), fontsize=14)
            set_axis_fonts(ax)
            path = folder/(stem+'.png')
            paths.append(save_figure(fig, path, show, tight=False))
            point_counts = np.array([len(values) for values in trace_frequencies])
            np.savez_compressed(folder/(stem+'.npz'), vr=vrs, point_counts=point_counts,
                trace_offsets=np.r_[0, np.cumsum(point_counts)],
                frequencies=np.concatenate(trace_frequencies), amplitudes=np.concatenate(trace_amplitudes))
            payload = dict(variable=name, LpD=lpd, fft_window=fft_window, trajectories=metadata,
                cylinder_type=cylinder_type, Re=selection.re, mstar=selection.mstar,
                fit_window_case=None if fit_window_case is None else str(fit_window_case),
                frequency_limits=list(map(float, limits)),
                frequency_coordinate='f*=cycles/t*', visualization='discrete VR spectra',
                interpolate_between_vr=False, frequency_grid='native per trajectory',
                max_frequency_points_per_trace=max_frequency_points,
                point_selection='native bins; largest native peak per frequency block when capped',
                trace_data_layout='concatenated frequencies/amplitudes; trace_offsets delimit VR curves')
            (folder/(stem+'.json')).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return paths


def plot_frequency_surfaces(**kwargs) -> list[Path]:
    """Compatibility entry point: plot discrete stacked FFT curves over VR."""
    return plot_frequency_waterfalls(**kwargs)
