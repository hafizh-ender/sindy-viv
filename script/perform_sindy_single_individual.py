"""Command-line entry point for the importable Bryan-case computation."""

import argparse

from sindy_viv.case_bryan.compute_sindy.functions import compute_sindy


def parse_args():
    argument_parser = argparse.ArgumentParser(description="Perform SINDy on Bryan's LBM data of a single oscillating cylinder for all VR cases, individually.")
    argument_parser.add_argument(
        "--dynamics_case",
        type=int,
        required=True,
        help="Dynamics case to perform SINDy on."
    )
    argument_parser.add_argument(
        "--differentiator_case",
        type=str,
        required=True,
        help="Differentiator case to perform SINDy on."
    )
    argument_parser.add_argument(
        "--state_feature_library_case",
        type=str,
        required=True,
        help="Feature library case to perform SINDy on."
    )
    argument_parser.add_argument(
        "--optimizer_case",
        type=str,
        required=True,
        help="Optimizer case to perform SINDy on."
    )
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to perform SINDy on."
    )
    argument_parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="Flag to save the SINDy model as a pickle file."
    )

    return argument_parser.parse_args()


def main():
    return compute_sindy(
        cylinder_type="single", parameterization="individual", **vars(parse_args()),
    )


if __name__ == "__main__":
    main()
