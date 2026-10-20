"""Derivative and integral evaluation selected by explicit case branches."""

from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import DTypeLike
from os import PathLike
from sindy_viv._typing import CylinderType, DynamicsCase, EvaluationRecord, Parameterization
from typing import Any, Literal

from pathlib import Path

import numpy as np
import pysindy as ps

from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME, SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR,
    TWO_TANDEM_CYLINDERS_CASE_NAME, TWO_TANDEM_CYLINDERS_RE, TWO_TANDEM_CYLINDERS_MSTAR,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.plot import select_plot_variables
from sindy_viv.case_bryan.prepare_sindy.trajectory import append_parameter_columns
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
from sindy_viv.feature_library.functions import uses_shape_only_fit
from sindy_viv.feature_library import prepare_library
from sindy_viv.case_bryan.model_storage import resolve_model_directory, model_base_directory
from sindy_viv.utils import add_dot_text, initiate_object, load_config
from .model_evaluation import (
    load_coefficients, model_differentiator_name, predict_derivative,
    report_evaluation, select_time_region, simulate, validate_selection,
)

def evaluate_sindy(
    dynamics_case: DynamicsCase,
    optimizer_case: str | PathLike[str],
    *,
    evaluation_type: Literal["derivative", "integral"] = "derivative",
    cylinder_type: CylinderType = "single",
    parameterization: Parameterization = "individual",
    differentiator_case: str | PathLike[str] | None = None,
    state_feature_library_case: str | PathLike[str] | None = None,
    input_feature_library_case: str | PathLike[str] | None = None,
    vr_list: Sequence[float] | None = None,
    lpd_list: Sequence[float] | None = None,
    tstar0: float | None = None,
    tstarf: float | None = None,
    evaluation_differentiator_case: str | PathLike[str] | None = None,
    weak: bool = False,
    save_plot: bool = True,
    return_data: bool = True,
    xlim: Sequence[float] | None = None,
    ylim_mean_time: Sequence[float | None] = (None, None),
    ylim_source: Literal["reference", "prediction", "both", "true", "pred"] = 'reference',
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
    method: str | type = "RK45",
    rtol: float | np.ndarray = 1e-6,
    atol: float | np.ndarray = 1e-9,
    solver_options: dict[str, Any] | None = None,
    block_size: int = 10_000,
    results_root: str | PathLike[str] = "results",
    re: float | None = None,
    mstar: float | None = None,
    dtype: DTypeLike = np.float64,
    model_dir: str | PathLike[str] | None = None,
    training_vr_list: Sequence[float] | None = None,
    training_lpd_list: Sequence[float] | None = None,
    fit_window_case: str | PathLike[str] | None = None,
) -> dict[float | tuple[float, float], EvaluationRecord]:
    """Evaluate saved models by derivative reconstruction or initial-value simulation.

    Select ``evaluation_type='derivative'`` or ``'integral'``; cylinder and
    parameterization options match compute_sindy. Returns {vr: evaluation}
    for single cylinders or {(lpd, vr): evaluation} for tandem cylinders.
    Bounds and plot limits use t*; differentiation and integration use tau.
    Metrics and figures retain their existing directory and filename layout.
    All fits use state_feature_library_case. Parameterization is explicit;
    VR, L/D, and joint L/D-VR models require input_feature_library_case.
    Training selectors identify a fitted model independently of evaluation VR/L/D.
    fit_window_case selects a named training-window JSON config, not evaluation bounds.
    Pass model_dir when multiple datasets or fitting windows match.
    Cases 3 and 4 prescribe measured controls during integration; only model
    states evolve.

    Parameters:
        dynamics_case (int): Physical dynamics case, one of 1, 2, 3, or 4.

        optimizer_case (str or PathLike[str]): Training optimizer configuration path identifying the saved model.

        evaluation_type (Literal['derivative', 'integral']): Evaluation method: derivative reconstruction or integral simulation. Default: 'derivative'

        cylinder_type (str): Cylinder configuration: single or two_tandem. Default: 'single'
        parameterization (str): Explicit fitted model family: individual, vr, lpd, or vr_lpd. Default: 'individual'

        differentiator_case (str, PathLike[str], or None): Training differentiator configuration path identifying the saved model. Default: None
        state_feature_library_case (str, PathLike[str], or None): Response-library configuration path for states and measured controls. Default: None
        input_feature_library_case (str, PathLike[str], or None): Parameter-library configuration path for explicitly parameterized models. Default: None

        vr_list (Sequence[float] or None): Reduced velocities to evaluate, including held-out trajectories. Default: None
        lpd_list (Sequence[float] or None): Cylinder spacings L/D to evaluate, including held-out trajectories. Default: None

        tstar0 (float or None): First evaluation time in recorded t*; None uses the available start. Default: None
        tstarf (float or None): Last evaluation time in recorded t*; None uses the available end. Default: None

        evaluation_differentiator_case (str, PathLike[str], or None): Optional reference differentiator config. Default: None

        weak (bool): Locate legacy weak coefficients; does not enable weak fitting. Default: False

        save_plot (bool): Whether to save the comparison figure alongside the metric report. Default: True
        return_data (bool): True returns full arrays; False retains small report records. Default: True

        xlim (Sequence[float] or None): Displayed time limits in recorded t*; None uses the available interval. Default: None
        ylim_mean_time (Sequence[float | None]): Recorded t* interval for y-limit statistics, with None for open endpoints. Default: (None, None)
        ylim_source (Literal['reference', 'prediction', 'both', 'true', 'pred']): Source of y-limit statistics: reference, prediction, or both; true and pred are aliases. Default: 'reference'
        plot_variables (Sequence[str] or None): Variable families or exact output-channel names to plot; derivative evaluation uses derivative names. Metrics retain all channels. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to plot; metrics retain all cylinders. Default: None

        method (str or type): Integration solver passed to scipy.integrate.solve_ivp. Default: 'RK45'
        rtol (float or numpy.ndarray): Relative integration tolerance passed to scipy.integrate.solve_ivp. Default: 1e-06
        atol (float or numpy.ndarray): Absolute integration tolerance passed to scipy.integrate.solve_ivp. Default: 1e-09
        solver_options (dict[str, Any] or None): Additional keyword settings passed to scipy.integrate.solve_ivp. Default: None

        block_size (int): Positive row limit for derivative features/metric scratch. Default: 10000

        results_root (str or PathLike[str]): Root directory containing saved model results. Default: 'results'

        re (float or None): Reynolds number used in dataset labels; None selects the cylinder-specific value. Default: None
        mstar (float or None): Mass ratio; workflow entry points use the cylinder-specific value when None. Default: None

        dtype (numpy.dtype, type, or str): Floating state/control/derivative dtype; time remains float64. Default: np.float64

        model_dir (str, PathLike[str], or None): Explicit model directory; None searches using the supplied model and training selectors. Default: None

        training_vr_list (Sequence[float] or None): Reduced velocities used to fit the saved model, independently of the evaluation selection. Default: None
        training_lpd_list (Sequence[float] or None): Cylinder spacings used to fit the saved model, independently of the evaluation selection. Default: None
        fit_window_case (str, PathLike[str], or None): Training-window JSON path used to select saved coefficients; None leaves window selection open. Default: None

    Returns:
        {VR: record} for single cylinders or {(LpD, VR): record} for tandem.
        Records always contain metrics, paths, solver diagnostics, counts and
        actual time regions. Full time/reference/prediction arrays are included
        only with return_data=True. CLI defaults to summary-only records.

    Raises:
        ValueError for invalid selections/configuration or ambiguous models,
        FileNotFoundError for missing coefficients, RuntimeError for incomplete
        integration. Matching metric/figure filenames are overwritten.
    """
    if evaluation_type not in ("derivative", "integral"):
        raise ValueError("evaluation_type must be 'derivative' or 'integral'.")

    if ylim_source not in ('reference', 'prediction', 'both', 'true', 'pred'):
        raise ValueError("ylim_source must be 'reference', 'prediction', or 'both'.")

    if cylinder_type == "single":
        case_name = SINGLE_CYLINDER_CASE_NAME
        re = SINGLE_CYLINDER_RE if re is None else re
        mstar = SINGLE_CYLINDER_MSTAR if mstar is None else mstar
        vr_list = (3, 4, 5, 6, 7, 8) if vr_list is None else vr_list
        lpd_list = (None,)
    elif cylinder_type == "two_tandem":
        case_name = TWO_TANDEM_CYLINDERS_CASE_NAME
        re = TWO_TANDEM_CYLINDERS_RE if re is None else re
        mstar = TWO_TANDEM_CYLINDERS_MSTAR if mstar is None else mstar
        vr_list = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14) if vr_list is None else vr_list
        lpd_list = (1.5, 2.0, 3.0, 3.5, 4.0) if lpd_list is None else lpd_list
        lpd_list = validate_selection(lpd_list, "lpd_list")
    else:
        raise ValueError("cylinder_type must be 'single' or 'two_tandem'.")

    layout = get_case_layout(dynamics_case, cylinder_type)
    variable_names = list(layout.measured_names)
    vr_list = validate_selection(vr_list, "vr_list")
    if parameterization not in ("individual", "vr", "lpd", "vr_lpd"):
        raise ValueError("parameterization must be 'individual', 'vr', 'lpd', or 'vr_lpd'.")

    if parameterization in ("lpd", "vr_lpd") and cylinder_type != "two_tandem":
        raise ValueError(f"parameterization={parameterization!r} requires cylinder_type='two_tandem'.")

    # Require library configurations to agree with the explicitly selected model type.
    if parameterization == "individual" and input_feature_library_case is not None:
        raise ValueError("Individual evaluation cannot use input_feature_library_case; omit it or select a parametric model.")

    if parameterization != "individual" and input_feature_library_case is None:
        raise ValueError("Parametric evaluation requires input_feature_library_case.")

    if state_feature_library_case is None:
        raise ValueError("All evaluation requires state_feature_library_case.")

    if parameterization == "individual":
        fit_mode = "individual"
    elif parameterization == "vr":
        fit_mode = "parametric_vr"
    elif parameterization == "lpd":
        fit_mode = "parametric_lpd"
    else:
        fit_mode = "parametric_vr_lpd"

    if training_vr_list is not None:
        training_vr_list = validate_selection(training_vr_list, 'training_vr_list')

    if training_lpd_list is not None:
        training_lpd_list = validate_selection(training_lpd_list, 'training_lpd_list')

    requested_model_dir = model_dir
    model_differentiator = model_differentiator_name(dynamics_case, differentiator_case, weak)
    if weak:
        fit_mode = "weak_" + fit_mode

    reference_case = evaluation_differentiator_case if evaluation_differentiator_case is not None else differentiator_case

    # Integral structural cases need no differentiation; lift-rate states do.
    if evaluation_type == "derivative":
        if reference_case is None:
            raise ValueError("Derivative evaluation requires evaluation_differentiator_case or differentiator_case.")

        differentiator_config = load_config(reference_case)
        reference_name = Path(reference_case).stem
    elif dynamics_case in (2, 4):
        if reference_case is None:
            raise ValueError(f"Case {dynamics_case} integral evaluation requires a reference differentiator.")

        differentiator_config = load_config(reference_case)
        reference_name = Path(reference_case).stem
    else:
        differentiator_config = None
        reference_name = "no_differentiator"

    state_library_config = load_config(state_feature_library_case)
    if parameterization != "individual":
        input_library_config = load_config(input_feature_library_case)

    evaluations = {}
    fitted_libraries = {}
    for lpd in lpd_list:
        for vr in vr_list:

            # Select the model directory using the requested model and training selectors.
            model_dir = model_base_directory(
                results_root, case_name, dynamics_case, model_differentiator, fit_mode,
                cylinder_type, parameterization, state_feature_library_case,
                input_feature_library_case, optimizer_case, re, mstar, vr, lpd,
            )

            # Training datasets and fitting windows are independent of the trajectories being evaluated.
            model_dir = resolve_model_directory(
                model_dir, cylinder_type, parameterization, vr, lpd, model_dir=requested_model_dir,
                training_vr_list=training_vr_list, training_lpd_list=training_lpd_list,
                fit_window_case=fit_window_case,
            )

            if cylinder_type == "single":
                time, data = load_data_single_cylinder(re, mstar, vr, dtype=dtype, variables=variable_names)
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=dtype, variables=variable_names)

            region = select_time_region(time, vr, tstar0, tstarf)
            differentiator = initiate_object(**differentiator_config) if differentiator_config is not None else None

            # The same case layout constructs training and integral reference states.
            trajectory = prepare_trajectory(
                time, data, variable_names, differentiator, dynamics_case, cylinder_type,
                dtype=dtype, with_derivatives=evaluation_type == "derivative",
            )
            t, X = trajectory.time[region].copy(), trajectory.states[region].copy()
            state_names, input_names = trajectory.state_names, trajectory.input_names
            U = None if trajectory.inputs is None else trajectory.inputs[region].copy()

            if evaluation_type == "derivative":
                reference = trajectory.derivatives[region].copy()
                variable_names_to_report = add_dot_text(state_names)
            else:
                reference, variable_names_to_report = X, state_names

            # Validate figure filters without removing states needed by prediction or integration.
            if save_plot:
                select_plot_variables(variable_names_to_report, plot_variables, plot_cylinders)

            control_count = len(input_names)

            del time, data, differentiator, trajectory

            if parameterization == "individual":
                parameter_names, parameters = [], ()
            else:
                if parameterization == "vr":
                    parameter_names, parameters = ['V_R'], (vr,)
                elif parameterization == "lpd":
                    parameter_names, parameters = ['L/D'], (lpd,)
                else:

                    # Joint parameters remain ordered L/D then VR, matching preparation and training.
                    parameter_names, parameters = ['L/D', 'V_R'], (lpd, vr)

            if parameter_names and control_count:
                U, input_names = append_parameter_columns(X, U, input_names, parameters, parameter_names)

            # Constant-only models retain broadcast parameters without allocating or interpolating U.
            if model_dir in fitted_libraries:
                library, coefficients = fitted_libraries[model_dir]
            else:

                # Build feature libraries only when this saved model has no reusable cached metadata.
                if parameterization == "individual":
                    library = prepare_library(state_library_config)
                else:
                    library = ps.ParameterizedLibrary(
                        feature_library=prepare_library(state_library_config),
                        parameter_library=prepare_library(input_library_config),
                        num_features=len(state_names) + control_count, num_parameters=len(parameter_names),
                    )
                coefficients = load_coefficients(
                    model_dir, library, X, state_names, parameters, parameter_names,
                    input_data=U, input_names=input_names if U is not None else None,
                )

                # Data-adaptive/validation-dependent libraries keep their original per-trajectory fit.
                if uses_shape_only_fit(library):
                    fitted_libraries[model_dir] = (library, coefficients)

            if evaluation_type == "derivative":
                prediction = predict_derivative(library, coefficients, X, parameters, input_data=U, block_size=block_size)
                solver = None
            else:
                prediction, solver = simulate(
                    library, coefficients, t, reference, parameters,
                    method=method, rtol=rtol, atol=atol, solver_options=solver_options,
                    input_data=U,
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
                save_plot=save_plot, xlim=xlim, ylim_mean_time=ylim_mean_time, ylim_source=ylim_source,
                plot_variables=plot_variables, plot_cylinders=plot_cylinders,
                block_size=block_size, solver=solver, return_data=return_data,
            )

            # Summary mode releases this trajectory before loading the next dataset.
            del t, X, reference, prediction, U

    return evaluations
