"""Raw/filtered time series and FFTs per L/D/VR, plus 2D frequency summaries."""

from sindy_viv.case_bryan.analysis.plot_cli import filtering_main


def main():
    """Run the two-tandem filtering diagnostic workflow."""
    return filtering_main('two_tandem')


if __name__ == '__main__':
    main()
