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
    SINGLE_CYLINDER_CASE_NAME,
    SINGLE_CYLINDER_MSTAR,
    SINGLE_CYLINDER_RE,
    SINGLE_CYLINDER_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder
from sindy_viv.case_bryan.plot import plot_data

def parse_args():
    argument_parser = argparse.ArgumentParser(description="Plot all available variables of Bryan's LBM data of a single oscillating cylinder for all VR cases.")
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to plot."
    )
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
    return argument_parser.parse_args()

def main():
    # Load the arguments
    args = parse_args()
    
    re = SINGLE_CYLINDER_RE
    mstar = SINGLE_CYLINDER_MSTAR
    
    # Loop through each VR case and plot the data
    for vr in args.vr_list:
        # Load data for current VR
        time, data = load_data_single_cylinder(re, mstar, vr)
        
        # Return time to its scaled version
        time = time * vr
        
        fig, axs = plot_data(
            time=time,
            data=data,
            state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
            parameter_values_dict={
                "Re": re,
                "m^*": mstar,
                "V_R": vr
            },
            xlim=args.xlim,
            ylim_mean_time_start=args.ylim_mean_time[0],
            ylim_mean_time_end=args.ylim_mean_time[1],
        )
        
        # Save the plot
        results_path = os.path.join(
            "results",
            SINGLE_CYLINDER_CASE_NAME,
            f"figures",
            f"Re = {re}",
            f"Mstar = {mstar}",
            f"VR = {vr}",
            f"plot_{args.xlim[0]}_{args.xlim[1]}.png" if args.xlim is not None else f"plot_full_range.png"
        )
        
        # Create the results directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        # Save the figure
        fig.savefig(results_path, dpi=300)

if __name__ == "__main__":
    main()