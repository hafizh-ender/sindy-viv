"""Stack tandem FFT curves over discrete VR, separately for each L/D/channel."""

from sindy_viv.case_bryan.analysis.plot_cli import frequency_surface_main


def main():
    """Run this explicit cylinder workflow through the shared CLI."""
    return frequency_surface_main('two_tandem')


if __name__ == "__main__":
    main()
