"""Evaluate one saved single-cylinder experiment and aggregate its diagnostics."""

from sindy_viv.case_bryan.evaluate_sindy.single_suite import evaluate_single_suite
from sindy_viv.case_bryan.evaluate_sindy.suite_cli import suite_parser


def parse_args():
    """Parse single-cylinder suite selectors using the shared suite parser."""
    return suite_parser('single').parse_args()


def main():
    """Run the selected single-cylinder evaluation suite and return its summary.

    Exit with status 1 if the suite recorded failures, allowing batch runners
    to continue later experiments while reporting the failed stage.
    """
    result = evaluate_single_suite(**vars(parse_args()))
    if result['failures']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()
