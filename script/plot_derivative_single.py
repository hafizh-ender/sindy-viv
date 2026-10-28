"""Plot computed derivatives on selected recorded-time windows."""

from sindy_viv.case_bryan.analysis.plot_cli import derivative_main


def main():
    """Run this explicit cylinder workflow through the shared CLI."""
    return derivative_main('single')


if __name__ == "__main__":
    main()
