"""Assess full-record numerical derivatives through IVP and recorded-channel checks."""

import argparse
from sindy_viv.case_bryan.evaluate_sindy import assess_derivatives


def parse_args() -> argparse.Namespace:
    """Parse raw-data selection, differentiator and reporting interval."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )
    parser.add_argument(
        '--differentiator_case',
        required=True,
    )
    parser.add_argument(
        '--cylinder_type',
        choices=('single', 'two_tandem'),
        default='single',
    )
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
    )
    parser.add_argument(
        '--lpd_list',
        type=float,
        nargs='+',
    )
    parser.add_argument(
        '--tstar0',
        type=float,
    )
    parser.add_argument(
        '--tstarf',
        type=float,
    )
    parser.add_argument(
        '--fit_window_case',
        help='Per-trajectory reporting windows in JSON; use instead of --tstar0/--tstarf.',
    )
    parser.add_argument(
        '--results_root',
        default='results',
    )
    parser.add_argument(
        '--method',
        default='trapezoid',
        help='trapezoid (exact piecewise-linear IVP), RK45, DOP853, etc.',
    )
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e-07,
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-09,
    )
    parser.add_argument(
        '--no_plot',
        dest='save_plot',
        action='store_false',
    )
    parser.add_argument(
        '--filtering_case',
        default=None,
        help=(
            'Optional filtering/smoothing JSON applied before differentiation; selects the '
            'filtering config folder.'
        ),
    )
    return parser.parse_args()


def main() -> dict:
    """Save derivative reconstruction and recorded-derivative assessment reports."""
    return assess_derivatives(**vars(parse_args()))


if __name__ == '__main__':
    main()
