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
from sindy_viv.case_bryan.prepare_sindy.input import prepare_input_two_tandem_parametric_vr_lpd
from sindy_viv.sindy.functions import perform_sindy
from sindy_viv.utils import add_dot_text, initiate_object, load_config

def parse_args():
    argument_parser = argparse.ArgumentParser(description="Evaluate SINDy on Bryan's LBM data of two tandem oscillating cylinders for all VR and L/D cases, parametrically on both VR and L/D.")
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
        "--state_feature_library_case",
        type=str,
        required=True,
        help="Feature library case to evaluate SINDy on for state variables."
    )
    argument_parser.add_argument(
        "--input_feature_library_case",
        type=str,
        required=True,
        help="Feature library case to evaluate SINDy on for input variables."
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
    
    # Load the differentiator, feature library, and optimizer configurations
    differentiator_config = load_config(args.differentiator_case)
    state_feature_library_config = load_config(args.state_feature_library_case)
    input_feature_library_config = load_config(args.input_feature_library_case)
    optimizer_config = load_config(args.optimizer_case)
    
    # Extract base names for file outputs
    differentiator_name = os.path.basename(args.differentiator_case).split('.')[0]
    state_feature_library_name = os.path.basename(args.state_feature_library_case).split('.')[0]
    input_feature_library_name = os.path.basename(args.input_feature_library_case).split('.')[0]
    optimizer_name = os.path.basename(args.optimizer_case).split('.')[0]
    dynamics_case_name = f"dynamics_case_{args.dynamics_case}"
    
    re = TWO_TANDEM_CYLINDERS_RE
    mstar = TWO_TANDEM_CYLINDERS_MSTAR
    
    # Loop through each VR and L/D case and evaluate SINDy
    time_all_dict = {}
    data_all_dict = {}
    
    for lpd in args.lpd_list:
        time_dict = {}
        data_dict = {}
        
        for vr in args.vr_list:
            # Load data for current VR
            time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd)
            
            # Store the time and data in dictionaries for parametric evaluation
            time_dict[vr] = time
            data_dict[vr] = data
            
        time_all_dict[lpd] = time_dict
        data_all_dict[lpd] = data_dict
            
    # Instantiate differentiator, feature library, and optimizer
    differentiator = initiate_object(**differentiator_config)
    state_feature_library = initiate_object(**state_feature_library_config)
    input_feature_library = initiate_object(**input_feature_library_config)
    optimizer = initiate_object(**optimizer_config)
    
    # Prepare the input for SINDy
    t_s, X_s, X_dot_s, chosen_state_variables, U_s, chosen_input_variables = prepare_input_two_tandem_parametric_vr_lpd(
        times=time_all_dict,
        full_state_variables=data_all_dict,
        full_state_variable_names=TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
        input_variables=[(lpd, vr) for lpd in args.lpd_list for vr in args.vr_list],
        differentiators=differentiator,
        case=args.dynamics_case,
    )
    derived_state_variable_names = add_dot_text(chosen_state_variables)
    
    # For parametric SINDy, use the parametric feature library for the state variables and the input variables
    overall_feature_library = ps.ParameterizedLibrary(
        feature_library=state_feature_library,
        parameter_library=input_feature_library,
        num_features=len(chosen_state_variables),
        num_parameters=len(chosen_input_variables),
    )
        
    # Fit the SINDy model
    model = perform_sindy(
        time=t_s,
        state_variable_data=X_s,
        feature_library=overall_feature_library, 
        optimizer=optimizer,
        state_variable_differentiated_data=X_dot_s,
        differentiation_method=None,
        feature_names=chosen_state_variables + chosen_input_variables,
        input_variable_data=U_s,
    )
    
    # Save the results
    results_dir = os.path.join(
        "results",
        TWO_TANDEM_CYLINDERS_CASE_NAME,
        dynamics_case_name,
        differentiator_name,
        "parametric_vr_lpd",
        f"state_{state_feature_library_name}",
        f"input_{input_feature_library_name}",
        optimizer_name,
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
    
    if args.save_pkl:
        model_pickle_path = os.path.join(results_dir, "model.pkl")
        with open(model_pickle_path, "wb") as f:
            pickle.dump(model, f)

if __name__ == "__main__":
    main()