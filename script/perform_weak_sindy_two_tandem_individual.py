"""Command-line entry point for the weak Bryan-case computation."""

import pysindy as ps

import argparse

from sindy_viv.case_bryan.compute_sindy.weak_functions import compute_weak_sindy


def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(description="Perform WeakSINDy on Bryan's LBM data of two tandem oscillating cylinders for all VR and L/D cases, individually.")

    # Physical case and dataset selection
    argument_parser.add_argument(
        "--dynamics_case",
        type=int,
        required=True,
        help="Dynamics case to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="List of VR cases to perform WeakSINDy on."
    )
    argument_parser.add_argument(
        "--lpd_list",
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help="List of L/D cases to perform WeakSINDy on."
    )

    # SINDy fitting configuration
    argument_parser.add_argument(
        "--differentiator_case",
        type=str,
        required=False,
        help="Differentiator config for lift-rate states in cases 2, 4 and 5."
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
        "--weak_feature_library_case", type=str, default=None,
        help="WeakPDELibrary settings JSON; defaults to configs/weak_feature_library/default.json."
    )

    # Integration bounds use CSV fluid time t*.
    # Window count, half width and test degree come from the weak-library JSON.
    argument_parser.add_argument(
        "--tweak0",
        type=float,
        default=None,
        help="Inclusive integration start in t*; defaults to the trajectory start."
    )
    argument_parser.add_argument(
        "--tweakf",
        type=float,
        default=None,
        help="Inclusive integration end in t*; defaults to the trajectory end."
    )
    argument_parser.add_argument(
        "--random_state",
        type=int,
        default=0
    )

    argument_parser.add_argument(
        "--fit_window_case", type=str, default=None,
        help="Per-trajectory t* window JSON; use instead of --tweak0/--tweakf."
    )

    # Output settings
    argument_parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="Flag to save the SINDy model as a pickle file."
    )
    return argument_parser.parse_args()


def main() -> ps.SINDy | dict[float | tuple[float, float], ps.SINDy]:
    """Delegate this explicit cylinder/parameterization command to the package API."""
    return compute_weak_sindy(
        cylinder_type="two_tandem", parameterization="individual", **vars(parse_args()),
    )


if __name__ == "__main__":
    main()
