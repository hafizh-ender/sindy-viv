"""Assess model derivative or integral frequencies using the saved-model selectors."""

import argparse
from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS
from sindy_viv.case_bryan.evaluate_sindy import evaluate_frequency_derivative, evaluate_frequency_integral


def parse_args() -> argparse.Namespace:
    """Parse model selection, frequency reporting and optional coupled integration."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        '--evaluation_type',
        choices=('derivative', 'integral'),
        required=True,
    )
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )
    parser.add_argument(
        '--cylinder_type',
        choices=('single', 'two_tandem'),
        default='single',
    )
    parser.add_argument(
        '--parameterization',
        choices=('individual', 'vr', 'lpd', 'vr_lpd'),
        default='individual',
    )
    parser.add_argument(
        '--state_feature_library_case',
        required=True,
    )
    parser.add_argument(
        '--input_feature_library_case',
    )
    parser.add_argument(
        '--optimizer_case',
        required=True,
    )
    parser.add_argument(
        '--differentiator_case',
    )
    parser.add_argument(
        '--evaluation_differentiator_case',
    )
    parser.add_argument(
        '--weak',
        action='store_true',
    )
    parser.add_argument(
        '--weak_feature_library_case',
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
        '--training_vr_list',
        type=float,
        nargs='+',
    )
    parser.add_argument(
        '--training_lpd_list',
        type=float,
        nargs='+',
    )
    parser.add_argument(
        '--fit_window_case',
    )
    parser.add_argument(
        '--model_dir',
    )
    parser.add_argument(
        '--results_root',
        default='results',
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
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
    )
    parser.add_argument(
        '--amplitude_floor',
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--model_only',
        dest='compare_reference',
        action='store_false',
        help='Save model frequencies without computing a reference derivative or comparison.',
    )
    parser.add_argument(
        '--save_spectra',
        action='store_true',
        help='Also save full spectra in an NPZ file.',
    )
    parser.add_argument(
        '--save_plot',
        action='store_true',
    )
    parser.add_argument(
        '--coupled_motion',
        action='store_true',
        help='Case 4/5 integral evaluation with evolving motion.',
    )
    parser.add_argument(
        '--structural_mstar',
        type=float,
    )
    parser.add_argument(
        '--max_state_factor',
        type=float,
        default=10.0,
        help=(
            'Integral only: stop at N times each state reference-window maximum absolute value (N'
            ' > 1; default 10).'
        ),
    )
    parser.add_argument(
        '--method',
        default='RK45',
    )
    parser.add_argument(
        '--rtol',
        type=float,
        default=1e-06,
    )
    parser.add_argument(
        '--atol',
        type=float,
        default=1e-09,
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
    """Write frequency reports for the selected derivative or integral model assessment."""
    args = vars(parse_args())
    kind = args.pop('evaluation_type')
    evaluate = evaluate_frequency_derivative if kind == 'derivative' else evaluate_frequency_integral
    return evaluate(**args)


if __name__ == '__main__':
    main()
