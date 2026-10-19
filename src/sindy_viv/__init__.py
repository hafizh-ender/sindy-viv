
from __future__ import annotations

def main() -> int:
    """Show package entry points without starting a computation implicitly."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sindy-viv",
        description="SINDy models and evaluation tools for cylinder VIV data.",
        epilog=("From the repository, run python script/perform_sindy_single_individual.py --help "
                "or python script/evaluate_integral_sindy_single_individual.py --help. "
                "Python API: sindy_viv.case_bryan.compute_sindy.compute_sindy and "
                "sindy_viv.case_bryan.evaluate_sindy.evaluate_sindy."),
    )
    parser.parse_args()
    parser.print_help()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
