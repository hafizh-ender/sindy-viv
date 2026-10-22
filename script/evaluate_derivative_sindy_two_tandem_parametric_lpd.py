"""Command-line entry point for derivative evaluation of two_tandem_parametric_lpd."""

from sindy_viv._typing import EvaluationRecord

import argparse

from sindy_viv.case_bryan.evaluate_sindy.functions import evaluate_sindy


def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    parser = argparse.ArgumentParser(description="Evaluate derivative performance for two_tandem_parametric_lpd.")

    # Physical case and dataset selection
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
        help="Physical state case."
    )
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="Select only these VR datasets."
    )
    parser.add_argument(
        '--lpd_list',
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help="Select only these L/D datasets."
    )

    # SINDy fitting configuration
    parser.add_argument(
        '--differentiator_case',
        default=None,
        help="Training differentiator config; optional for weak case 1."
    )
    parser.add_argument(
        '--state_feature_library_case',
        required=True,
        help="State feature-library config."
    )
    parser.add_argument(
        '--input_feature_library_case',
        required=True,
        help="Parameter feature-library config."
    )
    parser.add_argument(
        '--optimizer_case',
        required=True,
        help="Optimizer config used to locate the saved model."
    )
    parser.add_argument(
        '--weak',
        action='store_true',
        help="Read a weakly fitted model."
    )

    # Training dataset identification
    # Training selectors choose a saved fit; vr_list/lpd_list and tstar0/tstarf choose evaluation data.
    parser.add_argument(
        '--training_vr_list',
        type=float,
        nargs='+',
        default=None,
        help="VR values used to fit the saved model."
    )
    parser.add_argument(
        '--training_lpd_list',
        type=float,
        nargs='+',
        default=None,
        help="L/D values used to fit the saved model."
    )
    parser.add_argument(
        '--fit_window_case',
        default=None,
        help="JSON config used for fitting windows; evaluation tstar0/tstarf remain scoring bounds."
    )
    parser.add_argument(
        '--model_dir',
        default=None,
        help="Exact folder containing model.csv; resolves ambiguous fits."
    )

    # Evaluation settings
    # Time bounds select scored samples; xlim only changes the displayed plot interval.
    parser.add_argument(
        '--tstar0',
        type=float,
        default=None,
        help="First scored time in recorded t* units."
    )
    parser.add_argument(
        '--tstarf',
        type=float,
        default=None,
        help="Last scored time in recorded t* units."
    )
    parser.add_argument(
        '--evaluation_differentiator_case',
        default=None,
        help="Optional reference differentiator; required for weak case-1 derivative evaluation."
    )

    # Plot settings
    parser.add_argument(
        '--plot_variables',
        nargs='+',
        default=None,
        help="Output variables to plot: h, hdot, hddot, CL, CLdot, CLddot, or exact names; derivative plots use derivative names. Default: all."
    )
    parser.add_argument(
        '--plot_cylinders',
        type=int,
        nargs='+',
        choices=(0, 1),
        default=None,
        help="Cylinder IDs to plot; metrics retain every modeled cylinder. Default: all."
    )
    parser.add_argument(
        '--no_plot',
        dest='save_plot',
        action='store_false',
        help="Save metrics without a comparison figure."
    )
    parser.add_argument(
        '--xlim',
        type=float,
        nargs=2,
        default=None,
        help="Plot limits in t*; does not change scored samples."
    )
    parser.add_argument(
        '--ylim_source',
        choices=('reference', 'prediction', 'both'),
        default='reference',
        help='Data used for y-axis limits; does not change metrics.'
    )
    parser.add_argument(
        '--ylim_mean_time',
        type=float,
        nargs=2,
        default=(None, None),
        help="Y-axis statistics interval in t*."
    )

    # Output settings
    parser.add_argument(
        '--results_root',
        default='results',
        help="Root containing the saved computation results."
    )
    parser.add_argument(
        '--return_data',
        action='store_true',
        help="Return full trajectories as well as saved reports; defaults to summary records."
    )

    # Computational settings
    parser.add_argument(
        '--block_size',
        type=int,
        default=10000,
        help="Maximum rows per feature/metric block."
    )
    return parser.parse_args()


def main() -> dict[float | tuple[float, float], EvaluationRecord]:
    """Delegate this explicit cylinder/parameterization command to the package API."""
    return evaluate_sindy(
        evaluation_type="derivative", cylinder_type="two_tandem",
        parameterization="lpd", **vars(parse_args()),
    )


if __name__ == '__main__':
    main()
