"""Plot raw tandem lift/motion and inter-cylinder displacement phases versus VR."""

from sindy_viv.case_bryan.analysis.plot_cli import phase_summary_main


def main():
    """Plot tandem lift/motion and inter-cylinder phase summaries across selected VR/L/D."""
    return phase_summary_main('two_tandem')


if __name__ == '__main__':
    main()
