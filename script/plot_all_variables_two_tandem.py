import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps

import sindy_viv
from sindy_viv.case_bryan.constants import (
    TWO_TANDEM_CYLINDERS_CASE_NAME,
    TWO_TANDEM_CYLINDERS_MSTAR,
    TWO_TANDEM_CYLINDERS_RE,
    TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_two_tandem_cylinders
from sindy_viv.case_bryan.plot import plot_data

def parse_args():
    argument_parser = argparse.ArgumentParser(description="Plot all state variables for two tandem oscillating cylinders for all VR and L/D cases.")
    argument_parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="X-axis limits for the plots."
    )
    argument_parser.add_argument(
        "--ylim_mean_time",
        type=float,
        nargs=2,
        default=[0, None],
        help="Time range to compute the mean value for setting Y-axis limits."
    )
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        help="List of VR cases to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--lpd_list",
        type=float,
        nargs='+',
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help="List of L/D cases to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--save_pkl",
        action="store_true",
        help="Flag to save the SINDy model as a pickle file."
    )
    
    return argument_parser.parse_args()

def main():
    # Load the arguments
    args = parse_args()
    
    re = TWO_TANDEM_CYLINDERS_RE
    mstar = TWO_TANDEM_CYLINDERS_MSTAR
    
    # Loop through each VR and L/D case and evaluate SINDy
    for lpd in args.lpd_list:
        for vr in args.vr_list:
            # Load data for current VR
            time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd)
    
            # Return time to its scaled version
            time = time * vr
            
            fig, axs = plot_data(
                time=time,
                data=data,
                state_variable_names=TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
                parameter_values_dict={
                    "L/D": lpd,
                    "V_R": vr
                },
                xlim=args.xlim,
                ylim_mean_time_start=args.ylim_mean_time[0],
                ylim_mean_time_end=args.ylim_mean_time[1],
            )
        
            # Save the plot
            results_path = os.path.join(
                "results",
                TWO_TANDEM_CYLINDERS_CASE_NAME,
                f"figures",
                f"LpD = {lpd}",
                f"VR = {vr}",
                f"plot_{args.xlim[0]}_{args.xlim[1]}.png" if args.xlim is not None else f"plot_full_range.png"
            )
            
            # Create the results directory if it doesn't exist
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            
            # Save the figure
            fig.savefig(results_path, dpi=300)
                    

if __name__ == "__main__":
    main()