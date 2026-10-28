"""Shared CLI options for the explicit single/tandem plotting entry points."""

import argparse

from .frequency import FFT_WINDOWS


def raw_plot_parser(cylinder_type: str | None, description: str, *, variable_selection=True) -> argparse.ArgumentParser:
    """Create raw-data/window/figure selectors, retaining explicit cylinder scripts."""
    parser = argparse.ArgumentParser(
        description=description,
    )
    if cylinder_type is None:
        parser.add_argument(
            '--cylinder_type',
            choices=('single', 'two_tandem'),
            default='single',
        )
    else:
        parser.set_defaults(cylinder_type=cylinder_type)
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
    )
    if cylinder_type != 'single':
        parser.add_argument(
            '--lpd_list',
            type=float,
            nargs='+',
        )
    parser.add_argument(
        '--fit_window_case',
        help='Per-trajectory analysis windows in recorded t*.',
    )
    parser.add_argument(
        '--tstar0',
        type=float,
    )
    parser.add_argument(
        '--tstarf',
        type=float,
    )
    if variable_selection:
        parser.add_argument(
            '--plot_variables',
            nargs='+',
            help='Variable families or exact channel names.',
        )
        parser.add_argument(
            '--plot_cylinders',
            type=int,
            nargs='+',
            choices=(0,) if cylinder_type == 'single' else (0, 1),
        )
    parser.add_argument(
        '--results_root',
        default='results',
    )
    parser.add_argument(
        '--output_dir',
        help='Override the figure/summary output directory.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Also display figures; default saves and closes them.',
    )
    return parser


def _run(parser, function):
    """Convert selection errors into readable CLI errors and list saved figures."""
    options = vars(parser.parse_args())
    try:
        paths = function(**options)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    for path in paths:
        print(f'Saved {path}')
    return paths


def derivative_main(cylinder_type: str):
    """Run full-record differentiation and selected-window derivative plotting."""
    from .diagnostic_plots import plot_derivatives
    parser = raw_plot_parser(cylinder_type, 'Plot computed derivatives without raw-derivative comparison or integration.')
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=1,
    )
    parser.add_argument(
        '--differentiator_case',
        required=True,
    )
    parser.add_argument(
        '--filtering_case',
        help='Optional smoothing JSON applied to full signals before differentiation.',
    )
    return _run(parser, plot_derivatives)


def filtering_main(cylinder_type: str | None = None):
    """Compare raw/filtered time histories, FFTs and aggregate frequency changes."""
    from .filtering_diagnostics import assess_filtering
    parser = raw_plot_parser(cylinder_type, 'Compare raw and filtered time series and frequency spectra without differentiating or fitting.')
    parser.add_argument(
        '--filtering_case',
        required=True,
        help='Filtering/smoothing JSON used by the fitting workflow.',
    )
    parser.add_argument(
        '--plots',
        choices=('individual', 'aggregate', 'both'),
        default='both',
        help=(
            'Individual time-series/spectrum figures, aggregate frequency plots, or both; '
            'numerical summaries are always saved.'
        ),
    )
    parser.add_argument(
        '--no_time_series',
        dest='time_series',
        action='store_false',
        help='Skip time-series comparisons in individual diagnostics; enabled by default.',
    )
    parser.add_argument(
        '--aggregate_axis',
        choices=('vr', 'lpd', 'both'),
        default='both',
        help='2D summaries versus VR and/or L/D; single-cylinder data use VR.',
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
        '--xlim',
        type=float,
        nargs=2,
        help='Display f* limits for individual spectra; summaries use all bins.',
    )
    parser.add_argument(
        '--save_spectra',
        action='store_true',
        help='Also save full raw/filtered FFT amplitudes as NPZ.',
    )
    try:
        result = assess_filtering(**vars(parser.parse_args()))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    for path in (*result['figure_paths'], result['summary_csv'], result['summary_json']):
        print(f'Saved {path}')
    return result


def frequency_surface_main(cylinder_type: str):
    """Run discrete stacked spectra; retain the original script entry-point name."""
    from .diagnostic_plots import plot_frequency_waterfalls
    parser = raw_plot_parser(cylinder_type, 'Stack independent FFT curves at discrete VR values in 3D.')
    parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
    )
    parser.add_argument(
        '--xlim',
        type=float,
        nargs=2,
        help='Displayed f* band; default 0 to 1, limited by Nyquist.',
    )
    parser.add_argument(
        '--max_frequency_points',
        type=int,
        default=2000,
        help='Maximum native bins per VR curve; peak-preserving selection when capped.',
    )
    return _run(parser, plot_frequency_waterfalls)


def frequency_summary_main(cylinder_type: str):
    """Plot existing FFT dominant-bin summaries without loading raw signals."""
    from .frequency_summary import plot_frequency_vr_summary
    parser = raw_plot_parser(cylinder_type,
        'Plot dominant frequency versus VR from saved frequency_analysis JSONs; run frequency analysis first.')
    parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
        help='FFT taper used by the saved frequency analysis; does not recompute it.',
    )
    parser.add_argument(
        '--include_natural_frequency',
        action='store_true',
        help='Include the structural natural frequency f_N*=1/VR.',
    )
    if cylinder_type == 'two_tandem':
        parser.add_argument(
            '--surface',
            action='store_true',
            help='Also plot dominant frequency on the L/D-by-VR surface.',
        )
    parser._option_string_actions['--fit_window_case'].help = (
        'Window JSON used by the saved frequency analysis; identifies results without cropping or recomputation.')
    parser._option_string_actions['--tstar0'].help = 'Requested start t* of the saved FFT analysis.'
    parser._option_string_actions['--tstarf'].help = 'Requested end t* of the saved FFT analysis.'
    return _run(parser, plot_frequency_vr_summary)


def derivative_ivp_summary_main():
    """Summarize saved differentiator IVP errors with evaluation-window selectors."""
    from .derivative_ivp_summary import plot_derivative_ivp_errors
    parser = raw_plot_parser(None, 'Plot title-free heatmaps of saved numerical-derivative IVP NRMSE.')
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=2,
    )
    parser.add_argument(
        '--differentiator_case',
        required=True,
    )
    parser.add_argument(
        '--filtering_case',
        help='Filtering JSON identifying the saved derivative assessments.',
    )
    parser.add_argument(
        '--integration_method',
        help='Select the integration method used by the saved assessment.',
    )
    parser.add_argument(
        '--allow_missing',
        action='store_true',
        help='Show missing reports as gray cells.',
    )
    return _run(parser, plot_derivative_ivp_errors)


def phase_summary_main(cylinder_type: str):
    """Compute raw-data phase differences, normalized by pi, and plot versus VR."""
    from .phase_summary import plot_phase_vr_summary
    parser = raw_plot_parser(cylinder_type,
        'Plot raw CL-minus-h phases (and tandem h1-minus-h0) versus VR; no prior analysis required.',
        variable_selection=False)
    parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
    )
    return _run(parser, plot_phase_vr_summary)


def error_plot_main(cylinder_type: str | None, evaluation_type: str, *, heatmap_summary=False):
    """Read existing SINDy reports; training-window identity does not crop metrics."""
    from sindy_viv.case_bryan.evaluate_sindy.error_plots import plot_saved_errors
    parser = raw_plot_parser(cylinder_type, 'Plot saved SINDy NRMSE versus VR and tandem L/D; no model is recomputed.')
    parser.set_defaults(evaluation_type=evaluation_type)
    parser.add_argument(
        '--dynamics_case',
        type=int,
        choices=(1, 2, 3, 4, 5),
        required=True,
    )
    parser.add_argument(
        '--differentiator_case',
    )
    parser.add_argument(
        '--evaluation_differentiator_case',
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--parameterization',
        choices=('individual', 'vr', 'lpd', 'vr_lpd'),
        default='individual',
    )
    mode.add_argument(
        '--individual',
        dest='parameterization',
        action='store_const',
        const='individual',
    )
    mode.add_argument(
        '--parameterized',
        dest='parameterization',
        action='store_const',
        const='vr',
        help='Alias for --parameterization vr; use vr_lpd or lpd explicitly when needed.',
    )
    parser.add_argument(
        '--weak',
        action='store_true',
    )
    parser.add_argument(
        '--weak_feature_library_case',
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
        '--model_dir',
    )
    parser.add_argument(
        '--allow_missing',
        action='store_true',
        help='Keep missing reports as gaps/gray cells and record their reasons.',
    )
    if evaluation_type == 'integral':
        parser.add_argument(
            '--coupled_motion',
            action='store_true',
        )
    if heatmap_summary:
        comparison = 'IVP reconstruction' if evaluation_type == 'integral' else 'derivative reconstruction'
        parser.description = f'Plot title-free heatmaps of saved SINDy {comparison} NRMSE.'
        parser.add_argument(
            '--training_fit_window_case',
            help='Training-window JSON identifying the saved fitted model.',
        )
        parser._option_string_actions['--fit_window_case'].help = (
            'Per-trajectory evaluation windows; use instead of tstar0/tstarf. Does not select training windows.')

        def summarize(**options):
            """Translate CLI training/evaluation window names and render saved-error heatmaps.

            The closure passes the remaining model/data selectors to plot_saved_errors;
            it reads saved reports without repeating fitting or IVP integration.
            """
            options['evaluation_window_case'] = options.pop('fit_window_case')
            options['fit_window_case'] = options.pop('training_fit_window_case')
            return plot_saved_errors(heatmap_only=True, **options)

        return _run(parser, summarize)
    parser.add_argument(
        '--evaluation_window_case',
        help='Per-trajectory evaluation-window JSON; use instead of tstar0/tstarf.',
    )
    parser._option_string_actions['--fit_window_case'].help = (
        'Training-window JSON used to identify the saved fit; tstar0/tstarf select scored reports.')
    return _run(parser, plot_saved_errors)
