"""Raw/filtered time-series and frequency diagnostics for single or tandem cylinders."""

from sindy_viv.case_bryan.analysis.plot_cli import filtering_main


def main():
    """Generate selected per-trajectory spectra and 2D frequency summaries."""
    return filtering_main()


if __name__ == '__main__':
    main()
