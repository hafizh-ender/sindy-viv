"""Plot saved SINDy NRMSE over VR and tandem L/D."""

from sindy_viv.case_bryan.analysis.plot_cli import error_plot_main


def main():
    """Run this explicit cylinder workflow through the shared CLI."""
    return error_plot_main('single', 'derivative')


if __name__ == "__main__":
    main()
