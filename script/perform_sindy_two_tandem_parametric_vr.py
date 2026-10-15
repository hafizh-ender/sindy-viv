"""Command-line entry point for the importable Bryan-case computation."""

import argparse

from sindy_viv.case_bryan.compute_sindy.functions import compute_sindy


def parse_args():
    argument_parser = argparse.ArgumentParser(description="Perform SINDy on Bryan's LBM data of two tandem oscillating cylinders for all VR and L/D cases, parametrically on VR for each L/D.")
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
        help="Feature library case to perform SINDy on for state variables."
    )
    argument_parser.add_argument(
        "--input_feature_library_case",
        type=str,
        required=True,
        help="Parameter feature-library config."
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
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="List of VR cases to perform SINDy on."
    )
    argument_parser.add_argument(
        "--lpd_list",
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help="List of L/D cases to perform SINDy on."
    )
    argument_parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="Flag to save the SINDy model as a pickle file."
    )

    return argument_parser.parse_args()


def main():
    return compute_sindy(
        cylinder_type="two_tandem", parameterization="vr", **vars(parse_args()),
    )


if __name__ == "__main__":
    main()
