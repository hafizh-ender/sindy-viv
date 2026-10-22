"""Experimental FFT magnitude and phase plots for single-cylinder trajectories."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME,
    SINGLE_CYLINDER_MSTAR,
    SINGLE_CYLINDER_RE,
    SINGLE_CYLINDER_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder
from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS, dominant_frequency_analysis
from sindy_viv.case_bryan.frequency_plot import plot_frequency_analysis
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from sindy_viv.case_bryan.plot import select_plot_variables, plot_selection_suffix

def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(description="Plot FFT magnitude and phase for Bryan's single-cylinder data.")

    # Physical case and dataset selection
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to analyze."
    )

    # Analysis interval
    argument_parser.add_argument(
        '--fit_window_case',
        help='Per-VR t* intervals from a fitting-window JSON; use instead of --tstar0/--tstarf.'
    )
    argument_parser.add_argument(
        "--tstar0",
        type=float,
        default=None,
        help="Start time for the analysis in t*."
    )
    argument_parser.add_argument(
        "--tstarf",
        type=float,
        default=None,
        help="End time for the analysis in t*. If None, use the last time point."
    )

    # Frequency analysis settings
    argument_parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
        help="FFT taper applied after mean removal; amplitudes are corrected for the window gain."
    )

    # Plot settings
    argument_parser.add_argument(
        '--plot_variables',
        nargs='+',
        default=None,
        help="Variables to plot: h, hdot, hddot, CL, CD, or exact channel names; default: all."
    )
    argument_parser.add_argument(
        '--plot_cylinders',
        type=int,
        nargs='+',
        choices=(0,),
        default=None,
        help="Cylinder IDs to plot; the single-cylinder dataset contains only 0."
    )
    argument_parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="Frequency-axis limits in f* = f D / U_inf; defaults to 0 through 1 (up to Nyquist)."
    )
    args = argument_parser.parse_args()
    if args.fit_window_case is not None and (args.tstar0 is not None or args.tstarf is not None):
        argument_parser.error('Use either --fit_window_case or --tstar0/--tstarf, not both.')
    if args.xlim is not None:
        if not np.all(np.isfinite(args.xlim)) or not 0 <= args.xlim[0] < args.xlim[1]:
            argument_parser.error("--xlim requires finite increasing nonnegative frequency limits.")

    return args

def main() -> None:
    """Save and display one FFT figure per selected single-cylinder trajectory."""
    args = parse_args()

    re = SINGLE_CYLINDER_RE
    mstar = SINGLE_CYLINDER_MSTAR
    variable_names = select_plot_variables(SINGLE_CYLINDER_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    suffix = plot_selection_suffix(SINGLE_CYLINDER_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    if args.fft_window != 'rectangular':
        suffix += f"_fft_window={args.fft_window}"

    pairs = [(None, vr) for vr in args.vr_list]
    bounds = dict.fromkeys(pairs, (args.tstar0, args.tstarf))
    if args.fit_window_case is not None:
        config = load_fit_window_config(args.fit_window_case, 'single')
        bounds = prepare_fit_window_bounds(config, pairs, 'single')

    # Analyze each VR separately so FFT arrays and figures are released between cases.
    for vr in args.vr_list:
        start, stop = bounds[(None, vr)]
        time, data = load_data_single_cylinder(re, mstar, vr, variables=variable_names)
        region = select_time_region(time, vr, start, stop)

        # Detach the selected window so FFT buffers do not retain complete source trajectories.
        time = time[region].copy()
        data = np.array(data[region, :], dtype=np.float64, copy=True)

        # The loader returns tau; using t* here gives frequencies f* = f D / U_inf.
        tstar = time * vr
        results_path = (
            Path("results") / SINGLE_CYLINDER_CASE_NAME / "figures"
            / f"Re = {re}" / f"Mstar = {mstar}" / f"VR = {vr}"
        )
        results_path.mkdir(parents=True, exist_ok=True)

        if start is None and stop is None:
            window = "full_range"
        else:
            window = f"{start}_{stop}"
        if args.fit_window_case is not None:
            window += f"_fit_window={Path(args.fit_window_case).stem}"

        frequency_filename = f"frequency_analysis_{window}.png"
        if args.xlim is not None:
            frequency_filename = f"frequency_analysis_{window}_frequency_{args.xlim[0]}_{args.xlim[1]}.png"
        frequency_filename = frequency_filename.removesuffix('.png') + suffix + '.png'

        title = (
            rf"$Re={re},\quad m^*={mstar},\quad V_R={vr}$"
            f"\n$t^* \\in [{tstar[0]:.3f}, {tstar[-1]:.3f}]$"
        )

        fig, axs = plot_frequency_analysis(
            tstar, data, variable_names, xlim=args.xlim, title=title, fft_window=args.fft_window,
        )

        try:
            output_path = results_path / frequency_filename
            fig.savefig(output_path, dpi=300)
            summary = dominant_frequency_analysis(tstar, data, variable_names, fft_window=args.fft_window)
            if args.fit_window_case is not None:
                summary['fit_window'] = dict(
                    name=Path(args.fit_window_case).stem, source=args.fit_window_case,
                    requested_tstar0=start, requested_tstarf=stop,
                )
            output_path.with_suffix(".json").write_text(
                json.dumps(summary, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(f"Saved VR={vr} figure to {output_path}")
            plt.show()
        finally:
            plt.close(fig)

            # Release plotted line buffers before loading the next full trajectory.
            del fig, axs, time, data, tstar

if __name__ == "__main__":
    main()
