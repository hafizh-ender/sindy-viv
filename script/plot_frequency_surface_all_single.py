"""Stack single-cylinder FFT curves in 3D with one curve per discrete VR."""

from sindy_viv.case_bryan.analysis.plot_cli import frequency_surface_main


def main():
    """Run this explicit cylinder workflow through the shared CLI."""
    return frequency_surface_main('single')


if __name__ == "__main__":
    main()
