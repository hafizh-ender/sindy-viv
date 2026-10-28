"""Command-line selectors for Bryan feature-matrix correlation and conditioning."""

import argparse

from .assessment import assess_bryan_feature_library


def assessment_parser():
    """Return the feature-assessment parser with data, library and numerical selectors.

    Constructing the parser does not load trajectories or create report files;
    main resolves the cylinder-count alias after parsing.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Assess feature-library correlation and raw condition numbers without fitting a SINDy'
            ' model.'
        ),
    )
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )
    cylinders = parser.add_mutually_exclusive_group()
    cylinders.add_argument(
        '--cylinder_type',
        choices=('single', 'two_tandem'),
    )
    cylinders.add_argument(
        '--n_cylinders',
        type=int,
        choices=(1, 2),
        help='Alternative to cylinder_type; default is 1.',
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
        '--parameterization',
        choices=('individual', 'vr', 'lpd', 'vr_lpd'),
        default='individual',
    )
    parser.add_argument(
        '--state_feature_library_case',
        '--feature_library_case',
        required=True,
        help='JSON for the state/control response library.',
    )
    parser.add_argument(
        '--input_feature_library_case',
        '--parameter_feature_library_case',
        help='VR/L/D parameter library JSON, required for parameterized tensor features.',
    )
    parser.add_argument(
        '--fit_window_case',
        help='Training-window JSON; defaults to the cylinder-specific overall.json.',
    )
    parser.add_argument(
        '--tstar0',
        type=float,
        help='Explicit window start, used instead of a fit-window JSON.',
    )
    parser.add_argument(
        '--tstarf',
        type=float,
        help='Explicit window end, used instead of a fit-window JSON.',
    )
    parser.add_argument(
        '--differentiator_case',
        default='configs/differentiator/finite_difference_basic.json',
        help=(
            'Used only to construct lift-rate states for Cases 2/4/5; full trajectory before '
            'cropping.'
        ),
    )
    parser.add_argument(
        '--filtering_case',
        help='Optional smoothing of full measurements before preparation.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8192,
        help='Feature transform/statistics batch rows; no subsampling.',
    )
    parser.add_argument(
        '--rank_rtol',
        type=float,
        help='Relative SVD rank tolerance; default eps*max(N,P).',
    )
    parser.add_argument(
        '--results_root',
        default='results',
    )
    parser.add_argument(
        '--output_dir',
        help='Override the assessment root; a case-name subfolder is retained.',
    )
    parser.add_argument(
        '--normalized_only',
        action='store_true',
        help='Write only normalized condition reports/heatmaps; preserve existing raw and correlation outputs.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
    )
    return parser


def main(argv=None):
    """Parse optional argv, resolve cylinder selectors and run the Bryan assessment.

    Return the report-path/summary mapping and print its output directory.
    When argv is None, argparse reads the process command line.
    """
    parser = assessment_parser()
    options = vars(parser.parse_args(argv))
    count = options.pop('n_cylinders')
    options['cylinder_type'] = options['cylinder_type'] or ('two_tandem' if count == 2 else 'single')
    result = assess_bryan_feature_library(**options)
    print('Feature-library assessment saved to', result['summary_json'].parent)
    return result
