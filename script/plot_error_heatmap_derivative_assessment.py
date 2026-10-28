"""Aggregate saved numerical-derivative IVP errors into title-free heatmaps."""

from sindy_viv.case_bryan.analysis.plot_cli import derivative_ivp_summary_main


def main():
    """Parse numerical-derivative report selectors and return saved IVP-error heatmap paths."""
    return derivative_ivp_summary_main()


if __name__ == '__main__':
    main()
