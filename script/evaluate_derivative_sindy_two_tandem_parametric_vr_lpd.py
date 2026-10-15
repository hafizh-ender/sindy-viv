"""Command-line entry point for derivative evaluation of two_tandem_parametric_vr_lpd."""

import argparse

from sindy_viv.case_bryan.evaluate_sindy.functions import evaluate_sindy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate derivative performance for two_tandem_parametric_vr_lpd.")
    parser.add_argument('--dynamics_case', type=int, choices=(1, 2), required=True, help="Physical state case.")
    parser.add_argument('--differentiator_case', default=None, help="Training differentiator config; optional for weak case 1.")
    parser.add_argument('--state_feature_library_case', required=True, help="State feature-library config.")
    parser.add_argument('--input_feature_library_case', required=True, help="Parameter feature-library config.")
    parser.add_argument('--optimizer_case', required=True, help="Optimizer config used to locate the saved model.")
    parser.add_argument('--vr_list', type=int, nargs='+', default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14], help="Select only these VR datasets.")
    parser.add_argument('--lpd_list', type=float, nargs='+', default=[1.5, 2.0, 3.0, 3.5, 4.0], help="Select only these L/D datasets.")
    # Time bounds select scored samples; xlim only changes the displayed plot interval.
    parser.add_argument('--tstar0', type=float, default=None, help="First scored time in recorded t* units.")
    parser.add_argument('--tstarf', type=float, default=None, help="Last scored time in recorded t* units.")
    parser.add_argument('--evaluation_differentiator_case', default=None, help="Optional reference differentiator; required for weak case-1 derivative evaluation.")
    parser.add_argument('--weak', action='store_true', help="Read a weakly fitted model.")
    parser.add_argument('--no_plot', dest='save_plot', action='store_false', help="Save metrics without a comparison figure.")
    parser.add_argument('--xlim', type=float, nargs=2, default=None, help="Plot limits in t*; does not change scored samples.")
    parser.add_argument('--ylim_mean_time', type=float, nargs=2, default=(None, None), help="Y-axis statistics interval in t*.")
    parser.add_argument('--block_size', type=int, default=10000, help="Maximum rows per feature/metric block.")
    parser.add_argument('--results_root', default='results', help="Root containing the saved computation results.")
    return parser.parse_args()


def main():
    return evaluate_sindy(
        evaluation_type="derivative", cylinder_type="two_tandem",
        parameterization="vr_lpd", **vars(parse_args()),
    )


if __name__ == '__main__':
    main()
