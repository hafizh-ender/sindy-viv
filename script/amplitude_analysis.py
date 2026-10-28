"""Compute raw-data amplitudes and/or plot saved amplitudes over VR and L/D."""

from sindy_viv.case_bryan.analysis.amplitude import (
    AMPLITUDE_METHODS, DEFAULT_AMPLITUDE_METHOD,
    compute_amplitude, plot_amplitude_single, plot_amplitude_two_tandem,
)
from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS
from sindy_viv.case_bryan.analysis.plot_cli import raw_plot_parser


def main():
    """Parse amplitude selectors and compute records, plot saved records, or run both.

    Return the selected operation result and print written paths. User data
    and selection errors are converted to argparse errors with usage guidance.
    """
    parser = raw_plot_parser(None, __doc__)
    parser.add_argument(
        '--action',
        choices=('compute', 'plot', 'both'),
        default='both',
    )
    parser.add_argument(
        '--method',
        choices=AMPLITUDE_METHODS,
        default=DEFAULT_AMPLITUDE_METHOD,
        help='Amplitude measure; default: half peak-to-peak range of the selected time series.',
    )
    parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
        help='FFT taper, used only with --method fft.',
    )
    args = vars(parser.parse_args())
    action = args.pop('action')
    args['variables'] = args.pop('plot_variables')
    output, show = args.pop('output_dir'), args.pop('show')
    try:
        result = None
        if action in ('compute', 'both'):
            result = compute_amplitude(**args)
            for record in result.values():
                print(f'Saved {record["path"]}')
        if action in ('plot', 'both'):
            plot = plot_amplitude_single if args.pop('cylinder_type') == 'single' else plot_amplitude_two_tandem
            result = plot(**args, output_dir=output, show=show)
            for path in result:
                print(f'Saved {path}')
        return result
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
