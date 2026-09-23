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
from sindy_viv.case_bryan.prepare_sindy.input import prepare_input_single_individual
from sindy_viv.sindy.functions import perform_sindy
from sindy_viv.utils import add_dot_text, initiate_object, load_config

def parse_args():
    argument_parser = argparse.ArgumentParser(description="Evaluate SINDy on Bryan's LBM data of a single oscillating cylinder for all VR cases, individually.")
    argument_parser.add_argument(
        "--dynamics_case",
        type=int,
        required=True,
        help="Dynamics case to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--differentiator_case",
        type=str,
        required=True,
        help="Differentiator case to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--feature_library_case",
        type=str,
        required=True,
        help="Feature library case to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--optimizer_case",
        type=str,
        required=True,
        help="Optimizer case to evaluate SINDy on."
    )
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to evaluate SINDy on."
    )
    
    return argument_parser.parse_args()

def main():
    # Load the arguments
    args = parse_args()
    
    # Load the differentiator, feature library, and optimizer configurations
    differentiator_config = load_config(args.differentiator_case)
    feature_library_config = load_config(args.feature_library_case)
    optimizer_config = load_config(args.optimizer_case)
    
    # Extract base names for file outputs
    differentiator_name = os.path.basename(args.differentiator_case).split('.')[0]
    feature_library_name = os.path.basename(args.feature_library_case).split('.')[0]
    optimizer_name = os.path.basename(args.optimizer_case).split('.')[0]
    dynamics_case_name = f"dynamics_case_{args.dynamics_case}"
    
    re = SINGLE_CYLINDER_RE
    mstar = SINGLE_CYLINDER_MSTAR
    
    # Loop through each VR case and evaluate SINDy
    for vr in args.vr_list:
        # Load data for current VR
        time, data = load_data_single_cylinder(re, mstar, vr)
        
        # Instantiate differentiator, feature library, and optimizer
        differentiator = initiate_object(**differentiator_config)
        feature_library = initiate_object(**feature_library_config)
        optimizer = initiate_object(**optimizer_config)
        
        # Prepare the input for SINDy
        t, X, X_dot, chosen_state_variables = prepare_input_single_individual(
            time=time,
            full_state_variable_data=data,
            full_state_variable_names=SINGLE_CYLINDER_STATE_VARIABLES,
            differentiator=differentiator,
            case=args.dynamics_case,
        )
        derived_state_variable_names = add_dot_text(chosen_state_variables)
        
        # Fit the SINDy model
        model = perform_sindy(
            time=t,
            state_variable_data=X,
            feature_library=feature_library, 
            optimizer=optimizer,
            state_variable_differentiated_data=X_dot,
            differentiation_method=None,
            feature_names=chosen_state_variables,
            input_variable_data=None,
        )
        
        # Save the results
        results_dir = os.path.join(
            "results",
            SINGLE_CYLINDER_CASE_NAME,
            dynamics_case_name,
            differentiator_name,
            "individual",
            feature_library_name,
            optimizer_name,
            f"Re = {re}",
            f"Mstar = {mstar}",
            f"VR = {vr}"
        )
        
        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        coefficients = model.coefficients()
        feature_names = model.get_feature_names()
        coefficients_df = pd.DataFrame(
            coefficients.T,
            columns=derived_state_variable_names,
            index=feature_names
        )
        coefficients_csv_path = os.path.join(results_dir, "model.csv")
        coefficients_df.to_csv(coefficients_csv_path)
        
        model_pickle_path = os.path.join(results_dir, "model.pkl")
        with open(model_pickle_path, "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    main()