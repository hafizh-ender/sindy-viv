"""Plot the raw single-cylinder lift-minus-displacement phase versus VR."""

from sindy_viv.case_bryan.analysis.plot_cli import phase_summary_main


def main():
    """Plot saved single-cylinder lift-minus-displacement phase summaries across VR."""
    return phase_summary_main('single')


if __name__ == '__main__':
    main()
