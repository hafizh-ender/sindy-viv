"""Plot dominant frequency versus VR/LpD from saved tandem FFT analyses."""

from sindy_viv.case_bryan.analysis.plot_cli import frequency_summary_main


def main():
    """Read existing frequency_analysis_two_tandem.py results and save summaries."""
    return frequency_summary_main('two_tandem')


if __name__ == '__main__':
    main()
