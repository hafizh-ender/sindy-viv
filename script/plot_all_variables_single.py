import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt


from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME,
    SINGLE_CYLINDER_MSTAR,
    SINGLE_CYLINDER_RE,
    SINGLE_CYLINDER_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from sindy_viv.case_bryan.plot import plot_data, select_plot_variables, plot_selection_suffix

def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(
        description=(
            "Plot all available variables of Bryan's LBM data of a single oscillating cylinder "
            'for all VR cases.'
        ),
    )

    # Physical case and dataset selection
    argument_parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help='List of VR cases to plot.',
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
        choices=(0,),
        default=None,
        help='Cylinder IDs to plot; the single-cylinder dataset contains only 0.',
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
        help='Per-VR recorded-time windows from JSON; use instead of --xlim.',
    )
    args = argument_parser.parse_args()
    if args.fit_window_case is not None and args.xlim is not None:
        argument_parser.error('Use either --fit_window_case or --xlim, not both.')
    return args

def main() -> None:
    """Parse single-cylinder time-series options and save one plot per selected VR.

    Resolve named fit windows or explicit display limits, select channels by
    name, and close each figure before loading another full trajectory.
    """
    args = parse_args()

    re = SINGLE_CYLINDER_RE
    mstar = SINGLE_CYLINDER_MSTAR
    variable_names = select_plot_variables(SINGLE_CYLINDER_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    suffix = plot_selection_suffix(SINGLE_CYLINDER_STATE_VARIABLES, args.plot_variables, args.plot_cylinders)
    window_case = getattr(args, 'fit_window_case', None)
    bounds = None
    if window_case is not None:
        bounds = prepare_fit_window_bounds(load_fit_window_config(window_case, 'single'),
                                          [(None, vr) for vr in args.vr_list], 'single')

    # Loop through each VR case and plot the data
    for vr in args.vr_list:
        # Load data for current VR
        time, data = load_data_single_cylinder(re, mstar, vr, variables=variable_names)
        limits = args.xlim
        if bounds is not None:
            start, stop = bounds[(None, vr)]
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
                "Re": re,
                "m^*": mstar,
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
                SINGLE_CYLINDER_CASE_NAME,
                "analysis",
                f"Re = {re}",
                f"Mstar = {mstar}",
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
