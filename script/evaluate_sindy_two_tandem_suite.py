"""Evaluate one saved tandem experiment across VR and L/D for both cylinders."""

from sindy_viv.case_bryan.evaluate_sindy.single_suite import evaluate_two_tandem_suite
from sindy_viv.case_bryan.evaluate_sindy.suite_cli import suite_parser


def parse_args():
    """Parse tandem suite selectors, including L/D and training parameter sets."""
    return suite_parser('two_tandem').parse_args()


def main():
    """Run diagnostics for both tandem cylinders and return the suite summary.

    Exit with status 1 if any stage failed so the batch runner can record the
    failure without discarding successful trajectory reports.
    """
    result = evaluate_two_tandem_suite(**vars(parse_args()))
    if result['failures']:
        raise SystemExit(1)
    return result


if __name__ == '__main__':
    main()
