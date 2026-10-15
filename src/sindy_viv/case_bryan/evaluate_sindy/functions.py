"""Derivative and integral evaluation selected by explicit case branches."""

from pathlib import Path

import numpy as np
import pysindy as ps

from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME, SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR,
    TWO_TANDEM_CYLINDERS_CASE_NAME, TWO_TANDEM_CYLINDERS_RE, TWO_TANDEM_CYLINDERS_MSTAR,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.prepare_sindy.input import prepare_input_single_individual, prepare_input_two_tandem_individual
from sindy_viv.sindy.functions import prepare_parameter_library
from sindy_viv.utils import add_dot_text, initiate_object, load_config
from ._evaluation import (
    load_coefficients, model_differentiator_name, predict_derivative,
    prepare_integral_states, report_evaluation, select_time_region, simulate, validate_selection,
)


def evaluate_sindy(
    dynamics_case,
    optimizer_case,
    *,
    evaluation_type="derivative",
    cylinder_type="single",
    parameterization="individual",
    differentiator_case=None,
    state_feature_library_case=None,
    input_feature_library_case=None,
    vr_list=None,
    lpd_list=None,
    tstar0=None,
    tstarf=None,
    evaluation_differentiator_case=None,
    weak=False,
    save_plot=True,
    xlim=None,
    ylim_mean_time=(None, None),
    method="RK45",
    rtol=1e-6,
    atol=1e-9,
    solver_options=None,
    block_size=10_000,
    results_root="results",
    re=None,
    mstar=None,
    dtype=np.float64,
):
    """Evaluate saved models by derivative reconstruction or initial-value simulation.

    Select ``evaluation_type='derivative'`` or ``'integral'``; cylinder and
    parameterization options match compute_sindy. Returns {vr: evaluation}
    for single cylinders or {(lpd, vr): evaluation} for tandem cylinders.
    Bounds and plot limits use t*; differentiation and integration use tau.
    Metrics and figures retain their existing directory and filename layout.
    All fits use state_feature_library_case. Parameterization is explicit;
    VR and joint L/D–VR models require input_feature_library_case.
    """
    if evaluation_type not in ("derivative", "integral"):
        raise ValueError("evaluation_type must be 'derivative' or 'integral'.")
    if cylinder_type == "single":
        case_name = SINGLE_CYLINDER_CASE_NAME
        re = SINGLE_CYLINDER_RE if re is None else re
        mstar = SINGLE_CYLINDER_MSTAR if mstar is None else mstar
        vr_list = (3, 4, 5, 6, 7, 8) if vr_list is None else vr_list
        variable_names = ['h_0', r'\dot{h}_0', 'C_{L0}']
        lpd_list = (None,)
    elif cylinder_type == "two_tandem":
        case_name = TWO_TANDEM_CYLINDERS_CASE_NAME
        re = TWO_TANDEM_CYLINDERS_RE if re is None else re
        mstar = TWO_TANDEM_CYLINDERS_MSTAR if mstar is None else mstar
        vr_list = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14) if vr_list is None else vr_list
        lpd_list = (1.5, 2.0, 3.0, 3.5, 4.0) if lpd_list is None else lpd_list
        lpd_list = validate_selection(lpd_list, "lpd_list")
        variable_names = ['h_0', r'\dot{h}_0', 'C_{L0}', 'h_1', r'\dot{h}_1', 'C_{L1}']
    else:
        raise ValueError("cylinder_type must be 'single' or 'two_tandem'.")
    vr_list = validate_selection(vr_list, "vr_list")
    if parameterization not in ("individual", "vr", "vr_lpd"):
        raise ValueError("parameterization must be 'individual', 'vr', or 'vr_lpd'.")
    if parameterization == "vr_lpd" and cylinder_type != "two_tandem":
        raise ValueError("parameterization='vr_lpd' requires cylinder_type='two_tandem'.")
    # Require library configurations to agree with the explicitly selected model type.
    if parameterization == "individual" and input_feature_library_case is not None:
        raise ValueError("Individual evaluation cannot use input_feature_library_case; omit it or select a parametric model.")
    if parameterization != "individual" and input_feature_library_case is None:
        raise ValueError("Parametric evaluation requires input_feature_library_case.")
    if state_feature_library_case is None:
        raise ValueError("All evaluation requires state_feature_library_case.")
    if parameterization == "individual":
        fit_mode = "individual"
    elif parameterization == "vr" or parameterization == "vr_lpd":
        if parameterization == "vr":
            fit_mode = "parametric_vr"
        else:
            fit_mode = "parametric_vr_lpd"
    model_differentiator = model_differentiator_name(dynamics_case, differentiator_case, weak)
    if weak:
        fit_mode = "weak_" + fit_mode
    reference_case = evaluation_differentiator_case if evaluation_differentiator_case is not None else differentiator_case
    # Integral case 1 constructs X directly; all other paths need reference differentiation.
    if evaluation_type == "derivative":
        if reference_case is None:
            raise ValueError("Derivative evaluation requires evaluation_differentiator_case or differentiator_case.")
        differentiator_config = load_config(reference_case)
        reference_name = Path(reference_case).stem
    elif dynamics_case == 2:
        if reference_case is None:
            raise ValueError("Case 2 integral evaluation requires a reference differentiator.")
        differentiator_config = load_config(reference_case)
        reference_name = Path(reference_case).stem
    else:
        differentiator_config = None
        reference_name = "no_differentiator"
    state_library_config = load_config(state_feature_library_case)
    if parameterization != "individual":
        input_library_config = load_config(input_feature_library_case)

    evaluations = {}
    for lpd in lpd_list:
        for vr in vr_list:
            if cylinder_type == "single":
                time, data = load_data_single_cylinder(re, mstar, vr, dtype=dtype, variables=variable_names)
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=dtype, variables=variable_names)
            region = select_time_region(time, vr, tstar0, tstarf)
            differentiator = initiate_object(**differentiator_config) if differentiator_config is not None else None
            if evaluation_type == "derivative":
                # Differentiate the full signal before selecting the scored interval.
                if cylinder_type == "single":
                    t, X, X_dot, state_names = prepare_input_single_individual(
                        time, data, variable_names, differentiator, case=dynamics_case,
                    )
                else:
                    t, X, X_dot, state_names = prepare_input_two_tandem_individual(
                        time, data, variable_names, differentiator, case=dynamics_case,
                    )
                t, X, reference = t[region].copy(), X[region].copy(), X_dot[region].copy()
                del X_dot
                variable_names_to_report = add_dot_text(state_names)
            else:
                X, state_names = prepare_integral_states(
                    time, data, variable_names, differentiator, dynamics_case, cylinder_type,
                )
                t, X = np.asarray(time, dtype=np.float64)[region].copy(), X[region].copy()
                reference = X
                variable_names_to_report = state_names
            # Selected arrays own their storage, allowing full-trajectory buffers to be released.
            del time, data, differentiator

            if parameterization == "individual":
                parameter_names, parameters = [], ()
                library = initiate_object(**state_library_config)
            else:
                if parameterization == "vr":
                    parameter_names, parameters = ['V_R'], (vr,)
                else:
                    # Joint parameters remain ordered L/D then VR, matching preparation and training.
                    parameter_names, parameters = ['L/D', 'V_R'], (lpd, vr)
                library = ps.ParameterizedLibrary(
                    feature_library=initiate_object(**state_library_config),
                    parameter_library=prepare_parameter_library(initiate_object(**input_library_config)),
                    num_features=len(state_names), num_parameters=len(parameter_names),
                )
            model_dir = Path(results_root) / case_name / f"dynamics_case_{dynamics_case}" / model_differentiator / fit_mode
            if parameterization == "individual":
                model_dir /= Path(state_feature_library_case).stem
            else:
                model_dir = model_dir / f"state_{Path(state_feature_library_case).stem}" / f"input_{Path(input_feature_library_case).stem}"
            model_dir /= Path(optimizer_case).stem
            if cylinder_type == "single":
                model_dir = model_dir / f"Re = {re}" / f"Mstar = {mstar}"
                if parameterization == "individual":
                    model_dir /= f"VR = {vr}"
            elif parameterization == "individual":
                model_dir = model_dir / f"LpD = {lpd}" / f"VR = {vr}"
            elif parameterization == "vr":
                model_dir /= f"LpD = {lpd}"
            coefficients = load_coefficients(model_dir, library, X, state_names, parameters, parameter_names)
            if evaluation_type == "derivative":
                prediction = predict_derivative(library, coefficients, X, parameters, block_size=block_size)
                solver = None
            else:
                prediction, solver = simulate(
                    library, coefficients, t, reference, parameters,
                    method=method, rtol=rtol, atol=atol, solver_options=solver_options,
                )
            report_dir = model_dir / "evaluation"
            parameter_values = {'Re': re, 'm^*': mstar, 'V_R': vr}
            if cylinder_type == "single":
                key = vr
            else:
                key = (lpd, vr)
                parameter_values['L/D'] = lpd
                report_dir /= f"LpD = {lpd}"
            report_dir /= f"VR = {vr}"
            evaluations[key] = report_evaluation(
                evaluation_type, model_dir, report_dir, t, vr, reference, prediction,
                variable_names_to_report, parameter_values, reference_name,
                save_plot=save_plot, xlim=xlim, ylim_mean_time=ylim_mean_time,
                block_size=block_size, solver=solver,
            )
    return evaluations
