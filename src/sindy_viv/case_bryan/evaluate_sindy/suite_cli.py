"""Shared command-line selectors for the single and tandem evaluation suites."""

import argparse


def suite_parser(cylinder_type):
    """Build the single/tandem evaluation-suite argument parser without running it.

    Cylinder type determines valid parameterization modes and default VR/L/D
    selections; separate options identify training and diagnostic windows.
    """
    single = cylinder_type == 'single'
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate one saved single-cylinder experiment and aggregate its diagnostics.'
            if single
            else 'Evaluate both tandem cylinders and aggregate diagnostics across VR and L/D.'
        ),
    )
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )
    parser.add_argument(
        '--parameterization',
        choices=('individual', 'vr') if single else ('individual', 'vr', 'lpd', 'vr_lpd'),
        default='individual',
    )
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=list(range(3, 9) if single else range(5, 15)),
    )
    parser.add_argument(
        '--training_vr_list',
        type=int,
        nargs='+',
    )
    if not single:
        parser.add_argument(
            '--lpd_list',
            type=float,
            nargs='+',
            default=[1.5, 2.0, 3.0, 3.5, 4.0],
        )
        parser.add_argument(
            '--training_lpd_list',
            type=float,
            nargs='+',
        )
    parser.add_argument(
        '--differentiator_case',
        required=True,
    )
    parser.add_argument(
        '--filtering_case',
        help=(
            'Optional smoothing JSON applied before differentiation; also selects the saved model'
            ' folder.'
        ),
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
        '--fit_window_case',
        required=True,
        help='Training-window JSON.',
    )
    parser.add_argument(
        '--derivative_window_cases',
        nargs='+',
        required=True,
        help='Evaluation-window JSONs for derivative reconstruction.',
    )
    parser.add_argument(
        '--integral_window_case',
        required=True,
        help='Evaluation-window JSON for standard and Case 4/5 coupled IVPs.',
    )
    parser.add_argument(
        '--spectral_window_cases',
        nargs='+',
        required=True,
        help='Derivative windows receiving frequency and amplitude comparisons.',
    )
    parser.add_argument(
        '--phase_comparison',
        action='store_true',
        help="Compare CL-minus-h phase at each cylinder's reference displacement peak for IVPs.",
    )
    parser.add_argument(
        '--weight_heatmap',
        action='store_true',
        help=(
            'Case 3 hddot coefficient heatmaps, including effective coefficients at selected VR/L/D.'
            ' VR models also retain tensor terms in one single-cylinder model column or a tandem L/D aggregate.'
        ),
    )
    parser.add_argument(
        '--method',
        default='RK45',
    )
    parser.add_argument(
        '--max_state_factor',
        type=float,
        default=10.0,
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
        '--results_root',
        default='results',
    )
    return parser
