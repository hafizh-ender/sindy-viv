"""Command-line entry point for the weak Bryan-case computation."""

from typing import NoReturn

import argparse

from sindy_viv.case_bryan.compute_sindy.functions import compute_weak_sindy


def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(description="Perform WeakSINDy on Bryan's LBM data of a single oscillating cylinder for all VR cases, individually.")
    argument_parser.add_argument(
        "--dynamics_case",
        type=int,
        required=True,
        help="Dynamics case to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--differentiator_case",
        type=str,
        required=False,
        help="Reserved differentiator config for lift-rate states in cases 2 and 4."
    )
    argument_parser.add_argument(
        "--state_feature_library_case",
        type=str,
        required=True,
        help="Feature library case to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--optimizer_case",
        type=str,
        required=True,
        help="Optimizer case to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="Flag to save the SINDy model as a pickle file."
    )

    # Reserved integration bounds use CSV fluid time t*; WeakSINDy fitting is not implemented.
    argument_parser.add_argument("--tweak0", type=float, default=None, help="Inclusive integration start in t*; defaults to the trajectory start.")
    argument_parser.add_argument("--tweakf", type=float, default=None, help="Inclusive integration end in t*; defaults to the trajectory end.")
    argument_parser.add_argument("--K", type=int, default=100, help="Number of weak integration windows.")
    argument_parser.add_argument("--H_xt", type=float, default=None, help="Time half-window in tau; defaults to one twentieth of the selected trajectory span.")
    argument_parser.add_argument("--p", type=int, default=4, help="Weak test-function polynomial degree.")
    argument_parser.add_argument("--random_state", type=int, default=0)
    return argument_parser.parse_args()


def main() -> NoReturn:
    """Delegate this explicit cylinder/parameterization command to the package API."""
    return compute_weak_sindy(
        cylinder_type="single", parameterization="individual", **vars(parse_args()),
    )


if __name__ == "__main__":
    main()
