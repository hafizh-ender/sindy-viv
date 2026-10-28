"""Aggregate saved SINDy IVP errors with separate training/evaluation windows."""

from sindy_viv.case_bryan.analysis.plot_cli import error_plot_main


def main():
    """Parse saved SINDy IVP-error selectors and return aggregate heatmap paths."""
    return error_plot_main(None, 'integral', heatmap_summary=True)


if __name__ == '__main__':
    main()
