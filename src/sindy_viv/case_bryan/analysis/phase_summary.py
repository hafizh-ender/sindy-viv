"""Compute raw-data relative phases and plot their trends against VR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_windows import analysis_directory, data_selection, iter_raw_windows, save_figure
from .frequency import FFT_WINDOWS, compute_frequency_analysis, compute_phase_difference
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.plot import LINE_WIDTH, create_subplots


def plot_phase_vr_summary(
    *, cylinder_type='single', vr_list=None, lpd_list=None, fit_window_case=None,
    tstar0=None, tstarf=None, fft_window='rectangular', results_root='results',
    output_dir=None, show=False,
) -> list[Path]:
    """Plot (phi_signal-phi_reference)/pi using raw signals in selected t* windows.

    Single: CL0-h0. Tandem: CL0-h0, CL1-h1 and h1-h0. Each pair is evaluated
    at the reference displacement's dominant frequency. Load one trajectory
    at a time; saved dominant-only FFT JSONs cannot supply common-bin phases.
    PNG/CSV/JSON filenames overwrite for matching settings. No fitting or
    differentiation is needed, and prior frequency analysis is not required.
    """
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS}.')
    selection = data_selection(cylinder_type, vr_list, lpd_list, fit_window_case, tstar0, tstarf)
    pairs = [('CL0_minus_h0', 'C_{L0}', 'h_0')]
    if cylinder_type == 'two_tandem':
        pairs += [('CL1_minus_h1', 'C_{L1}', 'h_1'), ('h1_minus_h0', 'h_1', 'h_0')]
    names = list(dict.fromkeys(name for _, signal, reference in pairs for name in (signal, reference)))
    records, trajectories = [], []
    for vr, lpd, tstar, values, metadata in iter_raw_windows(selection, names):
        spectrum = compute_frequency_analysis(tstar, values, names, fft_window=fft_window)
        trajectories.append(metadata)
        for pair, signal, reference in pairs:
            records.append(dict(pair=pair, **metadata, frequency_resolution=float(spectrum.frequencies[1]),
                                phase_origin_tstar=float(tstar[0]),
                                **compute_phase_difference(spectrum, signal, reference)))
        del spectrum
    folder = Path(output_dir) if output_dir is not None else analysis_directory(
        results_root, selection.case_name, cylinder_type, selection.re, selection.mstar)/'phase_vr_summary'
    folder /= f'fft_window={fft_window}'
    folder = training_directory(folder, 'vr' if cylinder_type == 'single' else 'vr_lpd',
                                trajectories, fit_window_case)
    paths = []
    for pair, signal, reference in pairs:
        selected = sorted((row for row in records if row['pair'] == pair),
                          key=lambda row: (-1 if row['LpD'] is None else row['LpD'], row['VR']))
        fig, ax = create_subplots(1, figsize=(6, 4))
        for index, lpd in enumerate(sorted(selection.lpds, key=lambda value: -1 if value is None else value)):
            group = [row for row in selected if row['LpD'] == lpd]
            ax.plot([row['VR'] for row in group],
                    [np.nan if row['phase_difference_over_pi'] is None else row['phase_difference_over_pi']
                     for row in group],
                    marker=('o', 's', '^', 'D', 'v')[index % 5], markersize=3,
                    linewidth=LINE_WIDTH, label=None if lpd is None else f'$L/D={lpd:g}$')
        ax.set(xlabel=r'$V_R$', ylabel=rf'$(\phi_{{{signal}}}-\phi_{{{reference}}})/\pi$',
               ylim=(-1.05, 1.05))
        ax.set_xticks(sorted(selection.vrs))
        ax.set_title('Phase Difference', fontsize=14)
        if cylinder_type == 'two_tandem':
            ax.legend(frameon=False, fontsize=10)
        stem = f'phase_{pair}_vs_VR'
        paths.append(save_figure(fig, folder/(stem+'.png'), show))
        table = pd.DataFrame(selected)
        table['tstar_region'] = table['tstar_region'].map(json.dumps)
        table.to_csv(folder/(stem+'.csv'), index=False, na_rep='N/A')
        payload = dict(frequency_coordinate='f*=cycles/t*', fft_window=fft_window,
                       definition='wrapped (phi_signal - phi_reference) / pi',
                       phase_frequency='dominant positive-frequency bin of reference displacement',
                       phase_convention='cosine; positive means signal leads reference',
                       fit_window=None if fit_window_case is None else str(fit_window_case),
                       records=selected)
        (folder/(stem+'.json')).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return paths
