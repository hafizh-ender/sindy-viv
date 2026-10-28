"""Aggregate saved SINDy derivative-reconstruction errors without figure titles."""

from sindy_viv.case_bryan.analysis.plot_cli import error_plot_main


def main():
    """Parse saved SINDy derivative-error selectors and return aggregate heatmap paths."""
    return error_plot_main(None, 'derivative', heatmap_summary=True)


if __name__ == '__main__':
    main()
