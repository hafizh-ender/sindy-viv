"""Cylinder SINDy computations selected by explicit case branches."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pysindy as ps

from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME,
    SINGLE_CYLINDER_RE,
    SINGLE_CYLINDER_MSTAR,
    SINGLE_CYLINDER_MSTAR_TRUE,
    
    TWO_TANDEM_CYLINDERS_CASE_NAME,
    TWO_TANDEM_CYLINDERS_RE,
    TWO_TANDEM_CYLINDERS_MSTAR,
    TWO_TANDEM_CYLINDERS_MSTAR_TRUE,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.prepare_sindy.input import (
    prepare_input_single_individual, prepare_input_single_parametric_vr,
    prepare_input_two_tandem_individual, prepare_input_two_tandem_parametric_vr,
    prepare_input_two_tandem_parametric_vr_lpd,
)
from sindy_viv.case_bryan.prepare_sindy.optimizer import prepare_optimizer
from sindy_viv.feature_library import prepare_library
from sindy_viv.sindy.functions import perform_sindy
from sindy_viv.utils import add_dot_text, initiate_object, load_config

def compute_sindy(
    dynamics_case,
    differentiator_case,
    optimizer_case,
    *,
    cylinder_type="single",
    parameterization="individual",
    state_feature_library_case=None,
    input_feature_library_case=None,
    vr_list=None,
    lpd_list=None,
    save_pkl=False,
    results_root="results",
    re=None,
    mstar=None,
    dtype=np.float64,
    constraint_mstar=None,
):
    """
    Fit individual, VR-parametric, or joint L/D-VR SINDy models.

    Parameters:
        dynamics_case (int): Case number for the dynamics to be modeled (1 or 2)
        differentiator_case (str): Path to the differentiator configuration file
        optimizer_case (str): Path to the optimizer configuration file
        cylinder_type (str): Type of cylinder configuration ('single' or 'two_tandem')
        parameterization (str): Explicit parameterization ('individual', 'vr', or 'vr_lpd')
        state_feature_library_case (str): Path to the state feature library configuration file for every fit
        input_feature_library_case (str or None): Parameter library, required for 'vr' and 'vr_lpd'
        vr_list (list): List of reduced velocities to consider for parametric fits
        lpd_list (list): List of L/D ratios to consider for parametric fits (only for two_tandem)
        save_pkl (bool): Whether to save the fitted model as a pickle file
        results_root (str): Root directory for saving results
        re (float): Reynolds number (optional, defaults to case-specific value)
        mstar (float): Mass ratio (optional, defaults to case-specific value)
        dtype (numpy.dtype): Data type for the loaded data (default: np.float64)
        constraint_mstar (float): Mass ratio for physical constraints in the optimizer (optional, defaults to case-specific value)
    """
    if dynamics_case not in (1, 2):
        raise ValueError("dynamics_case must be 1 or 2.")
    
    if cylinder_type == "single":
        case_name = SINGLE_CYLINDER_CASE_NAME
        re = SINGLE_CYLINDER_RE if re is None else re
        mstar = SINGLE_CYLINDER_MSTAR if mstar is None else mstar
        constraint_mstar = SINGLE_CYLINDER_MSTAR_TRUE if constraint_mstar is None else constraint_mstar
        vr_list = (3, 4, 5, 6, 7, 8) if vr_list is None else vr_list
        variable_names = ['h_0', r'\dot{h}_0', 'C_{L0}']
        lpd_list = (None,)
    elif cylinder_type == "two_tandem":
        case_name = TWO_TANDEM_CYLINDERS_CASE_NAME
        re = TWO_TANDEM_CYLINDERS_RE if re is None else re
        mstar = TWO_TANDEM_CYLINDERS_MSTAR if mstar is None else mstar
        constraint_mstar = TWO_TANDEM_CYLINDERS_MSTAR_TRUE if constraint_mstar is None else constraint_mstar
        vr_list = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14) if vr_list is None else vr_list
        lpd_list = (1.5, 2.0, 3.0, 3.5, 4.0) if lpd_list is None else lpd_list
        variable_names = ['h_0', r'\dot{h}_0', 'C_{L0}', 'h_1', r'\dot{h}_1', 'C_{L1}']
    else:
        raise ValueError("cylinder_type must be 'single' or 'two_tandem'.")
    
    vr_list, lpd_list = tuple(vr_list), tuple(lpd_list)
    if not vr_list or not lpd_list:
        raise ValueError("At least one VR and, for tandem cases, one L/D value is required.")

    if parameterization not in ("individual", "vr", "vr_lpd"):
        raise ValueError("parameterization must be 'individual', 'vr', or 'vr_lpd'.")
    
    if parameterization == "vr_lpd" and cylinder_type != "two_tandem":
        raise ValueError("parameterization='vr_lpd' requires cylinder_type='two_tandem'.")
    
    # Validate the library arguments against the explicitly selected fit mode.
    if parameterization == "individual" and input_feature_library_case is not None:
        raise ValueError("Individual fits cannot use input_feature_library_case; omit it or select a parametric fit.")
    
    if parameterization != "individual" and input_feature_library_case is None:
        raise ValueError("Parametric fits require input_feature_library_case.")
    
    if state_feature_library_case is None:
        raise ValueError("All fits require state_feature_library_case.")
    
    state_library_config = load_config(state_feature_library_case)

    # Each group corresponds to one fitted model; joint fits use the complete selected grid.
    if parameterization == "individual":
        trajectory_groups = [(lpd, vr) for lpd in lpd_list for vr in vr_list]
        fit_mode = "individual"
    elif parameterization == "vr" or parameterization == "vr_lpd":
        input_library_config = load_config(input_feature_library_case)
        
        if parameterization == "vr":
            trajectory_groups = [(lpd, None) for lpd in lpd_list]
            fit_mode = "parametric_vr"
        else:
            trajectory_groups = [(None, None)]
            fit_mode = "parametric_vr_lpd"
            
    differentiator_config = load_config(differentiator_case)
    optimizer_config = load_config(optimizer_case)
    
    models = {}
    for lpd, vr in trajectory_groups:
        differentiator = initiate_object(**differentiator_config)
        
        if parameterization == "individual":
            if cylinder_type == "single":
                time, data = load_data_single_cylinder(re, mstar, vr, dtype=dtype, variables=variable_names)
                t, X, X_dot, state_names = prepare_input_single_individual(
                    time, data, variable_names, differentiator, case=dynamics_case,
                )
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=dtype, variables=variable_names)
                t, X, X_dot, state_names = prepare_input_two_tandem_individual(
                    time, data, variable_names, differentiator, case=dynamics_case,
                )
                
            del time, data
            
            U, parameter_names = None, []
        elif parameterization == "vr":
            times, states = {}, {}
            for selected_vr in vr_list:
                if cylinder_type == "single":
                    times[selected_vr], states[selected_vr] = load_data_single_cylinder(
                        re, mstar, selected_vr, dtype=dtype, variables=variable_names,
                    )
                else:
                    times[selected_vr], states[selected_vr] = load_data_two_tandem_cylinders(
                        re, mstar, selected_vr, lpd, dtype=dtype, variables=variable_names,
                    )
            if cylinder_type == "single":
                t, X, X_dot, state_names, U, parameter_names = prepare_input_single_parametric_vr(
                    times, states, variable_names, list(vr_list), differentiator, case=dynamics_case,
                )
            else:
                t, X, X_dot, state_names, U, parameter_names = prepare_input_two_tandem_parametric_vr(
                    times, states, variable_names, list(vr_list), differentiator, case=dynamics_case,
                )
                
            # Release raw dictionaries before building the regression feature matrix.
            del times, states
        else:
            times, states = {}, {}
            
            for selected_lpd in lpd_list:
                times[selected_lpd], states[selected_lpd] = {}, {}
                for selected_vr in vr_list:
                    times[selected_lpd][selected_vr], states[selected_lpd][selected_vr] = load_data_two_tandem_cylinders(
                        re, mstar, selected_vr, selected_lpd, dtype=dtype, variables=variable_names,
                    )
                    
            selection = [(selected_lpd, selected_vr) for selected_lpd in lpd_list for selected_vr in vr_list]
            
            t, X, X_dot, state_names, U, parameter_names = prepare_input_two_tandem_parametric_vr_lpd(
                times, states, variable_names, selection, differentiator, case=dynamics_case,
            )
            
            del times, states

        if parameterization == "individual":
            library = prepare_library(state_library_config)
        else:
            library = ps.ParameterizedLibrary(
                feature_library=prepare_library(state_library_config),
                parameter_library=prepare_library(input_library_config),
                num_features=len(state_names), num_parameters=len(parameter_names),
            )
            
        # Resolve physical constraints only after the states and parameter features are known.
        optimizer = prepare_optimizer(
            optimizer_config, library, X, state_names,
            case=dynamics_case, cylinder_type=cylinder_type, mstar=constraint_mstar,
            vr=vr, input_data=U, parameter_names=parameter_names,
        )
        
        model = perform_sindy(
            time=t, state_variable_data=X, feature_library=library, optimizer=optimizer,
            state_variable_differentiated_data=X_dot,
            feature_names=state_names + parameter_names, input_variable_data=U,
        )

        results_dir = Path(results_root) / case_name / f"dynamics_case_{dynamics_case}" / Path(differentiator_case).stem / fit_mode
        if parameterization == "individual":
            results_dir /= Path(state_feature_library_case).stem
        else:
            results_dir = results_dir / f"state_{Path(state_feature_library_case).stem}" / f"input_{Path(input_feature_library_case).stem}"
        results_dir /= Path(optimizer_case).stem
        if cylinder_type == "single":
            results_dir = results_dir / f"Re = {re}" / f"Mstar = {mstar}"
            if parameterization == "individual":
                results_dir /= f"VR = {vr}"
        elif parameterization == "individual":
            results_dir = results_dir / f"LpD = {lpd}" / f"VR = {vr}"
        elif parameterization == "vr":
            results_dir /= f"LpD = {lpd}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        pd.DataFrame(
            model.coefficients().T, columns=add_dot_text(state_names), index=model.get_feature_names(),
        ).to_csv(results_dir / "model.csv")
        
        if save_pkl:
            with (results_dir / "model.pkl").open("wb") as stream:
                pickle.dump(model, stream)
        if parameterization == "individual":
            key = vr if cylinder_type == "single" else (lpd, vr)
            models[key] = model
        elif cylinder_type == "two_tandem" and parameterization == "vr":
            models[lpd] = model
        else:
            return model
        
    return models


def compute_weak_sindy(
    dynamics_case,
    optimizer_case,
    *,
    cylinder_type="single",
    parameterization="individual",
    state_feature_library_case=None,
    input_feature_library_case=None,
    vr_list=None,
    lpd_list=None,
    save_pkl=False,
    differentiator_case=None,
    tweak0=None,
    tweakf=None,
    K=100,
    H_xt=None,
    p=4,
    random_state=0,
    results_root="results",
    re=None,
    mstar=None,
    dtype=np.float64,
    constraint_mstar=None,
):
    """Placeholder retaining the WeakSINDy configuration and integration-window inputs."""
    raise NotImplementedError("WeakSINDy is not implemented.")
