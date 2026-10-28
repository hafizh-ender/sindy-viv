import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt


from sindy_viv.case_bryan.constants import (
    TWO_TANDEM_CYLINDERS_CASE_NAME,
    TWO_TANDEM_CYLINDERS_MSTAR,
    TWO_TANDEM_CYLINDERS_RE,
    TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_two_tandem_cylinders
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from sindy_viv.case_bryan.plot import plot_data, select_plot_variables, plot_selection_suffix

def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(
        description=(
            'Plot all state variables for two tandem oscillating cylinders for all VR and L/D '
            'cases.'
        ),
    )

    # Physical case and dataset selection
    argument_parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help='List of VR cases to plot.',
    )
    argument_parser.add_argument(
        '--lpd_list',
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help='List of L/D cases to plot.',
    )

    # Plot settings
    argument_parser.add_argument(
        '--plot_variables',
        nargs='+',
        default=None,
        help='Variables to plot: h, hdot, hddot, CL, CD, or exact channel names; default: all.',
    )
    argument_parser.add_argument(
        '--plot_cylinders',
        type=int,
        nargs='+',
        choices=(0, 1),
        default=None,
        help='Cylinder IDs to plot: 0 (upstream), 1 (downstream); default: both.',
    )
    argument_parser.add_argument(
        '--xlim',
        type=float,
        nargs=2,
        default=None,
        help='X-axis limits for the plots, in t*.',
    )
    argument_parser.add_argument(
        '--ylim_mean_time',
        type=float,
        nargs=2,
        default=[0, None],
        help='Time range to compute the mean value for setting Y-axis limits.',
    )
    argument_parser.add_argument(
        '--fit_window_case',
        help='Per-L/D and per-VR recorded-time windows from JSON; use instead of --xlim.',
    )
    args = argument_parser.parse_args()
    if args.fit_window_case is not None and args.xlim is not None:
        argument_parser.error('Use either --fit_window_case or --xlim, not both.')
    return args

def main() -> None:
    """Save selected tandem time histories for each requested L/D and VR pair.

    Use per-trajectory fit windows and canonical channel names; figures and
    large loaded arrays are released before processing the next dataset.
    """
    args = parse_args()

    re = TWO_TANDEM_CYLINDERS_RE
    mstar = TWO_TANDEM_CYLINDERS_MSTAR
    variable_names = select_plot_variables(TWO_TANDEM_CYLINDERS_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    suffix = plot_selection_suffix(TWO_TANDEM_CYLINDERS_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    window_case = getattr(args, 'fit_window_case', None)
    bounds = None
    if window_case is not None:
        bounds = prepare_fit_window_bounds(load_fit_window_config(window_case, 'two_tandem'),
                                          [(lpd, vr) for lpd in args.lpd_list for vr in args.vr_list],
                                          'two_tandem')

    # Loop through each VR and L/D case and plot the variables
    for lpd in args.lpd_list:
        for vr in args.vr_list:
            # Load data for current VR
            time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, variables=variable_names)
            limits = args.xlim
            if bounds is not None:
                start, stop = bounds[(lpd, vr)]
                region = select_time_region(time, vr, start, stop)
                time, data = time[region].copy(), data[region].copy()
                limits = (float(time[0]*vr) if start is None else start,
                          float(time[-1]*vr) if stop is None else stop)

            # Return time to t* from tau
            time = time * vr
            fig, axs = plot_data(
                time=time,
                data=data,
                state_variable_names=variable_names,
                parameter_values_dict={
                    "L/D": lpd,
                    "V_R": vr
                },
                xlim=limits,
                ylim_mean_time_start=args.ylim_mean_time[0],
                ylim_mean_time_end=args.ylim_mean_time[1],
            )

            try:
                # Save the plot
                filename = (f"plot_{args.xlim[0]}_{args.xlim[1]}{suffix}.png" if args.xlim is not None
                            else f"plot_full_range{suffix}.png")
                if window_case is not None:
                    filename = f"plot_fit_window={Path(window_case).stem}_{limits[0]:g}_{limits[1]:g}{suffix}.png"
                results_path = os.path.join(
                    "results",
                    TWO_TANDEM_CYLINDERS_CASE_NAME,
                    "analysis",
                    f"LpD = {lpd}",
                    f"VR = {vr}",
                    filename,
                )

                # Create the results directory if it doesn't exist
                os.makedirs(os.path.dirname(results_path), exist_ok=True)

                # Save the figure
                fig.savefig(results_path, dpi=300)
            finally:
                # Release line buffers even if saving a batch figure fails.
                plt.close(fig)

                # Release local references before loading the next full trajectory.
                del fig, axs, time, data


if __name__ == "__main__":
    main()
