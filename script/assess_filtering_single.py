"""Raw/filtered time series and FFTs per VR, plus 2D frequency summaries."""

from sindy_viv.case_bryan.analysis.plot_cli import filtering_main


def main():
    """Run the single-cylinder filtering diagnostic workflow."""
    return filtering_main('single')


if __name__ == '__main__':
    main()
