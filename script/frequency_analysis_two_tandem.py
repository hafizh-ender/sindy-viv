"""Experimental FFT magnitude and phase plots for two tandem cylinders."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.constants import (
    TWO_TANDEM_CYLINDERS_CASE_NAME,
    TWO_TANDEM_CYLINDERS_MSTAR,
    TWO_TANDEM_CYLINDERS_RE,
    TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
)
from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS, dominant_frequency_analysis
from sindy_viv.case_bryan.frequency_plot import plot_frequency_analysis
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from sindy_viv.case_bryan.input import load_data_two_tandem_cylinders
from sindy_viv.case_bryan.plot import select_plot_variables, plot_selection_suffix


def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(description="Plot FFT magnitude and phase for Bryan's two-tandem-cylinder data.")

    # Physical case and dataset selection
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="List of VR cases to analyze."
    )
    argument_parser.add_argument(
        "--lpd_list",
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help="List of L/D cases to analyze."
    )

    # Analysis interval
    argument_parser.add_argument(
        '--fit_window_case',
        help='Per-L/D and per-VR t* intervals from a fitting-window JSON; use instead of --tstar0/--tstarf.'
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
        choices=(0, 1),
        default=None,
        help="Cylinder IDs to plot: 0 (upstream), 1 (downstream); default: both."
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
    """Save and display one FFT figure per cylinder for each selected L/D and VR."""
    args = parse_args()
    re = TWO_TANDEM_CYLINDERS_RE
    mstar = TWO_TANDEM_CYLINDERS_MSTAR
    selected_names = select_plot_variables(TWO_TANDEM_CYLINDERS_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    suffix = plot_selection_suffix(TWO_TANDEM_CYLINDERS_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    if args.fft_window != 'rectangular':
        suffix += f"_fft_window={args.fft_window}"
    pairs = [(lpd, vr) for lpd in args.lpd_list for vr in args.vr_list]
    bounds = dict.fromkeys(pairs, (args.tstar0, args.tstarf))
    if args.fit_window_case is not None:
        config = load_fit_window_config(args.fit_window_case, 'two_tandem')
        bounds = prepare_fit_window_bounds(config, pairs, 'two_tandem')
    for lpd in args.lpd_list:
        for vr in args.vr_list:
            start, stop = bounds[(lpd, vr)]
            time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, variables=selected_names)
            region = select_time_region(time, vr, start, stop)

            # Detach the selected window before allocating any FFT buffers.
            time = time[region].copy()
            data = np.array(data[region, :], dtype=np.float64, copy=True)

            # The loader returns tau; convert back to t* exactly once for fluid frequencies.
            tstar = time * vr
            results_path = (
                Path("results") / TWO_TANDEM_CYLINDERS_CASE_NAME / "figures"
                / f"LpD = {lpd}" / f"VR = {vr}"
            )
            results_path.mkdir(parents=True, exist_ok=True)
            window = "full_range" if start is None and stop is None else f"{start}_{stop}"
            if args.fit_window_case is not None:
                window += f"_fit_window={Path(args.fit_window_case).stem}"
            for cylinder in (0, 1):

                # Exact-name selections may contain channels from only one cylinder.
                cylinder_names = select_plot_variables(TWO_TANDEM_CYLINDERS_STATE_VARIABLES, plot_cylinders=[cylinder])
                variable_names = [name for name in selected_names if name in cylinder_names]
                if not variable_names:
                    continue

                # Resolve channels by name so the plot does not depend on loader column positions.
                columns = [selected_names.index(name) for name in variable_names]
                title = (
                    rf"$Re={re},\quad m^*={mstar},\quad L/D={lpd},\quad V_R={vr}$"
                    f"\nCylinder {cylinder}, $t^* \\in [{tstar[0]:.3f}, {tstar[-1]:.3f}]$"
                )
                filename = f"frequency_analysis_cylinder_{cylinder}_{window}"
                if args.xlim is not None:
                    filename += f"_frequency_{args.xlim[0]}_{args.xlim[1]}"
                filename += suffix
                fig, axs = plot_frequency_analysis(
                    tstar, data[:, columns], variable_names, xlim=args.xlim, title=title, fft_window=args.fft_window,
                )
                try:
                    output_path = results_path / f"{filename}.png"
                    fig.savefig(output_path, dpi=300)
                    summary = dominant_frequency_analysis(tstar, data[:, columns], variable_names,
                                                          fft_window=args.fft_window)
                    if args.fit_window_case is not None:
                        summary['fit_window'] = dict(
                            name=Path(args.fit_window_case).stem, source=args.fit_window_case,
                            requested_tstar0=start, requested_tstarf=stop,
                        )
                    output_path.with_suffix(".json").write_text(
                        json.dumps(summary, indent=2, allow_nan=False),
                        encoding="utf-8",
                    )
                    print(f"Saved L/D={lpd}, VR={vr}, cylinder={cylinder} figure to {output_path}")
                    # plt.show()
                finally:
                    plt.close(fig)

                    # Release this cylinder's plotted buffers before preparing the next figure.
                    del fig, axs
            del time, data, tstar


if __name__ == "__main__":
    main()
