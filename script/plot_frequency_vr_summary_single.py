"""Plot dominant frequency versus VR from saved single-cylinder FFT analyses."""

from sindy_viv.case_bryan.analysis.plot_cli import frequency_summary_main


def main():
    """Read existing frequency_analysis_single.py results and save summaries."""
    return frequency_summary_main('single')


if __name__ == '__main__':
    main()
