"""Cylinder SINDy computations selected by explicit case branches."""

from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import DTypeLike
from os import PathLike
from sindy_viv._typing import CylinderType, DynamicsCase, Parameterization
from typing import NoReturn

import json
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
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.validation import validate_selection
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
from sindy_viv.case_bryan.prepare_sindy.optimizer import prepare_optimizer
from sindy_viv.case_bryan.model_storage import (
    crop_training_trajectory, training_directory,
    load_fit_window_config, prepare_fit_window_bounds, model_base_directory,
)
from sindy_viv.feature_library import prepare_library
from sindy_viv.sindy.functions import perform_sindy
from sindy_viv.utils import add_dot_text, initiate_object, load_config

def compute_sindy(
    dynamics_case: DynamicsCase,
    differentiator_case: str | PathLike[str] | None,
    optimizer_case: str | PathLike[str],
    *,
    cylinder_type: CylinderType = "single",
    parameterization: Parameterization = "individual",
    state_feature_library_case: str | PathLike[str] | None = None,
    input_feature_library_case: str | PathLike[str] | None = None,
    vr_list: Sequence[float] | None = None,
    lpd_list: Sequence[float] | None = None,
    save_pkl: bool = False,
    results_root: str | PathLike[str] = "results",
    re: float | None = None,
    mstar: float | None = None,
    dtype: DTypeLike = np.float64,
    constraint_mstar: float | None = None,
    fit_window_case: str | PathLike[str] | None = None,
) -> ps.SINDy | dict[float | tuple[float, float], ps.SINDy]:
    """Fit individual, VR-only, L/D-only, or joint L/D-VR SINDy models.

    Parameters:
        dynamics_case (int): Dynamics case 1 through 4.

        differentiator_case (str, PathLike[str], or None): Path to the differentiator configuration file.
        optimizer_case (str or PathLike[str]): Path to the optimizer configuration file.

        cylinder_type (str): Type of cylinder configuration ('single' or 'two_tandem'). Default: 'single'
        parameterization (str): Explicit parameterization ('individual', 'vr', 'lpd', or 'vr_lpd'). Default: 'individual'

        state_feature_library_case (str, PathLike[str], or None): Path to the state feature library configuration file for every fit. Default: None
        input_feature_library_case (str, PathLike[str], or None): Parameter library, required for every parametric fit. Default: None

        vr_list (Sequence[float] or None): Reduced velocities selected for individual or parametric fitting. Default: None
        lpd_list (Sequence[float] or None): L/D ratios selected for individual or parametric fitting (only for two_tandem). Default: None

        save_pkl (bool): Whether to save the fitted model as a pickle file. Default: False
        results_root (str or PathLike[str]): Root directory for saving results. Default: 'results'

        re (float or None): Reynolds number (optional, defaults to case-specific value). Default: None
        mstar (float or None): Mass ratio (optional, defaults to case-specific value). Default: None

        dtype (numpy.dtype, type, or str): Data type for the loaded data. Default: np.float64

        constraint_mstar (float or None): Mass ratio for physical constraints in the optimizer (optional, defaults to case-specific value). Default: None

        fit_window_case (str, PathLike[str], or None): JSON path containing trajectory windows keyed by VR (single) or LpD then VR (tandem); None selects the cylinder-specific overall.json. Default: None

    Returns:
        A fitted SINDy for single-cylinder VR or joint LpD-VR fitting.
        Individual fits return {VR: model} for single cylinders or
        {(LpD, VR): model} for tandem cylinders. Tandem VR fits return
        {LpD: model}; LpD-only fits return {VR: model}.

    Notes:
        Trajectories are loaded, differentiated on full float64 tau, and
        cropped one at a time. dtype is also forwarded to preparation.
        model.csv and training.json (and optional model.pkl) are written
        beneath dataset/window-specific paths; matching paths overwrite files.
        Invalid selections/configurations raise ValueError; I/O, library and
        optimizer errors propagate. The physical case definitions are unchanged.
    """

    # Validate the dynamics case
    if dynamics_case not in (1, 2, 3, 4):
        raise ValueError("dynamics_case must be 1, 2, 3, or 4.")

    if cylinder_type == "single":
        case_name = SINGLE_CYLINDER_CASE_NAME
        re = SINGLE_CYLINDER_RE if re is None else re
        mstar = SINGLE_CYLINDER_MSTAR if mstar is None else mstar
        constraint_mstar = SINGLE_CYLINDER_MSTAR_TRUE if constraint_mstar is None else constraint_mstar
        vr_list = (3, 4, 5, 6, 7, 8) if vr_list is None else vr_list
        lpd_list = (None,)
    elif cylinder_type == "two_tandem":
        case_name = TWO_TANDEM_CYLINDERS_CASE_NAME
        re = TWO_TANDEM_CYLINDERS_RE if re is None else re
        mstar = TWO_TANDEM_CYLINDERS_MSTAR if mstar is None else mstar
        constraint_mstar = TWO_TANDEM_CYLINDERS_MSTAR_TRUE if constraint_mstar is None else constraint_mstar
        vr_list = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14) if vr_list is None else vr_list
        lpd_list = (1.5, 2.0, 3.0, 3.5, 4.0) if lpd_list is None else lpd_list
    else:
        raise ValueError("cylinder_type must be 'single' or 'two_tandem'.")

    layout = get_case_layout(dynamics_case, cylinder_type)
    variable_names = list(layout.measured_names)
    vr_list = validate_selection(vr_list, "vr_list")
    lpd_list = validate_selection(lpd_list, "lpd_list") if cylinder_type == "two_tandem" else (None,)

    if parameterization not in ("individual", "vr", "lpd", "vr_lpd"):
        raise ValueError("parameterization must be 'individual', 'vr', 'lpd', or 'vr_lpd'.")

    if parameterization in ("lpd", "vr_lpd") and cylinder_type != "two_tandem":
        raise ValueError(f"parameterization={parameterization!r} requires cylinder_type='two_tandem'.")

    # Validate the library arguments against the explicitly selected fit mode.
    if parameterization == "individual" and input_feature_library_case is not None:
        raise ValueError("Individual fits cannot use input_feature_library_case; omit it or select a parametric fit.")

    if parameterization != "individual" and input_feature_library_case is None:
        raise ValueError("Parametric fits require input_feature_library_case.")

    if state_feature_library_case is None:
        raise ValueError("All fits require state_feature_library_case.")

    # Even the full-data default is read from JSON so every fit has one window source.
    if fit_window_case is None:
        fit_window_case = Path(__file__).resolve().parents[4] / "configs" / "fit_window" / cylinder_type / "overall.json"
    window_config = load_fit_window_config(fit_window_case, cylinder_type)
    fitting_bounds = prepare_fit_window_bounds(
        window_config, [(lpd, vr) for lpd in lpd_list for vr in vr_list], cylinder_type,
    )
    state_library_config = load_config(state_feature_library_case)

    # Each group corresponds to one fitted model; joint fits use the complete selected grid.
    if parameterization == "individual":
        trajectory_groups = [(lpd, vr) for lpd in lpd_list for vr in vr_list]
        fit_mode = "individual"
    else:
        input_library_config = load_config(input_feature_library_case)

        if parameterization == "vr":
            trajectory_groups = [(lpd, None) for lpd in lpd_list]
            fit_mode = "parametric_vr"
        elif parameterization == "lpd":
            trajectory_groups = [(None, vr) for vr in vr_list]
            fit_mode = "parametric_lpd"
        else:
            trajectory_groups = [(None, None)]
            fit_mode = "parametric_vr_lpd"

    differentiator_config = load_config(differentiator_case)
    optimizer_config = load_config(optimizer_case)
    models = {}
    for lpd, vr in trajectory_groups:
        differentiator = initiate_object(**differentiator_config)
        if parameterization == "individual":
            fitted_pairs = [(lpd, vr)]
        elif parameterization == "vr":
            fitted_pairs = [(lpd, value) for value in vr_list]
        elif parameterization == "lpd":
            fitted_pairs = [(value, vr) for value in lpd_list]
        else:
            fitted_pairs = [(spacing, value) for spacing in lpd_list for value in vr_list]

        if parameterization == "individual":
            parameter_names = []
        elif parameterization == "vr":
            parameter_names = ['V_R']
        elif parameterization == "lpd":
            parameter_names = ['L/D']
        else:
            parameter_names = ['L/D', 'V_R']

        cropped, training_trajectories = [], []
        for trajectory_lpd, trajectory_vr in fitted_pairs:
            if cylinder_type == "single":
                time, data = load_data_single_cylinder(re, mstar, trajectory_vr, dtype=dtype, variables=variable_names)
            else:
                time, data = load_data_two_tandem_cylinders(
                    re, mstar, trajectory_vr, trajectory_lpd, dtype=dtype, variables=variable_names,
                )
            trajectory = prepare_trajectory(time, data, variable_names, differentiator, dynamics_case, cylinder_type, dtype=dtype)
            del time, data
            if parameterization == "vr":
                trajectory = trajectory.with_parameters((trajectory_vr,), parameter_names)
            elif parameterization == "lpd":
                trajectory = trajectory.with_parameters((trajectory_lpd,), parameter_names)
            elif parameterization == "vr_lpd":
                trajectory = trajectory.with_parameters((trajectory_lpd, trajectory_vr), parameter_names)

            # Crop each full-resolution derivative immediately, before loading another raw trajectory.
            time, states, derivatives, inputs, metadata = crop_training_trajectory(
                trajectory.time, trajectory.states, trajectory.derivatives, trajectory.inputs,
                trajectory_lpd, trajectory_vr, fitting_bounds[(trajectory_lpd, trajectory_vr)],
            )
            state_names, input_names = trajectory.state_names, trajectory.input_names
            cropped.append((time, states, derivatives, inputs))
            training_trajectories.append(metadata)
            del trajectory, time, states, derivatives, inputs
        del differentiator
        if parameterization == "individual":
            t, X, X_dot, U = cropped[0]
            library = prepare_library(state_library_config)
        else:
            t, X, X_dot, U = [list(values) for values in zip(*cropped)]

            # Measured motion/lift controls stay in the response basis; only VR/LpD are parameters.
            library = ps.ParameterizedLibrary(
                feature_library=prepare_library(state_library_config),
                parameter_library=prepare_library(input_library_config),
                num_features=len(state_names) + len(input_names) - len(parameter_names),
                num_parameters=len(parameter_names),
            )
        del cropped

        # Resolve physical constraints only after the states and parameter features are known.
        optimizer = prepare_optimizer(
            optimizer_config, library, X, state_names,
            case=dynamics_case, cylinder_type=cylinder_type, mstar=constraint_mstar,
            vr=vr, input_data=U, input_names=input_names, parameter_names=parameter_names,
        )
        model = perform_sindy(
            time=t, state_variable_data=X, feature_library=library, optimizer=optimizer,
            state_variable_differentiated_data=X_dot,
            feature_names=state_names + input_names, input_variable_data=U,
        )

        # Returned models predict from new X/U; prepared training arrays are no longer needed here.
        del t, X, X_dot, U

        results_dir = model_base_directory(
            results_root, case_name, dynamics_case, Path(differentiator_case).stem, fit_mode,
            cylinder_type, parameterization, state_feature_library_case,
            input_feature_library_case, optimizer_case, re, mstar, vr, lpd,
        )
        results_dir = training_directory(results_dir, parameterization, training_trajectories, fit_window_case)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Record ordered requested bounds and actual samples beside all artifacts for this fit.
        training_metadata = {
            'schema_version': 1,
            'cylinder_type': cylinder_type,
            'parameterization': parameterization,
            'dynamics_case': dynamics_case,
            'time_coordinate': 'tau=t*/VR',
            'trajectory_order': 'L/D outermost, VR innermost; preserves CLI selection order',
            'trajectories': training_trajectories,
            'data_dtype': np.dtype(dtype).name,
            'Re': float(re),
            'mstar': float(mstar),
            'constraint_mstar': float(constraint_mstar),
            'configurations': {
                'differentiator': differentiator_config,
                'optimizer': optimizer_config,
                'state_feature_library': state_library_config
            },
        }
        if parameterization != "individual":
            training_metadata['configurations']['parameter_feature_library'] = input_library_config

        # Snapshot the full config as well as each model's resolved trajectory bounds.
        training_metadata['fit_window'] = {
            'name': Path(fit_window_case).stem, 'source': str(fit_window_case), 'config': window_config,
        }
        (results_dir / 'training.json').write_text(
            json.dumps(training_metadata, indent=2, allow_nan=False), encoding='utf-8',
        )

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
        elif parameterization == "lpd":
            models[vr] = model
        else:
            return model

    return models

def compute_weak_sindy(
    dynamics_case: DynamicsCase,
    optimizer_case: str | PathLike[str],
    *,
    cylinder_type: CylinderType = "single",
    parameterization: Parameterization = "individual",
    state_feature_library_case: str | PathLike[str] | None = None,
    input_feature_library_case: str | PathLike[str] | None = None,
    vr_list: Sequence[float] | None = None,
    lpd_list: Sequence[float] | None = None,
    save_pkl: bool = False,
    differentiator_case: str | PathLike[str] | None = None,
    tweak0: float | None = None,
    tweakf: float | None = None,
    K: int = 100,
    H_xt: float | None = None,
    p: int = 4,
    random_state: int | None = 0,
    results_root: str | PathLike[str] = "results",
    re: float | None = None,
    mstar: float | None = None,
    dtype: DTypeLike = np.float64,
    constraint_mstar: float | None = None,
) -> NoReturn:
    """Reserve the WeakSINDy API and legacy integration-window arguments.

    No data, libraries or optimizers are loaded. Always raises NotImplementedError.
    Retained tweak0/tweakf denote future integration bounds in recorded t*.

    Parameters:
        dynamics_case (int): Physical dynamics case, one of 1, 2, 3, or 4.

        optimizer_case (str or PathLike[str]): Optimizer configuration path.

        cylinder_type (str): Cylinder configuration: single or two_tandem. Default: 'single'
        parameterization (str): Explicit fitting family: individual, vr, lpd, or vr_lpd. Default: 'individual'

        state_feature_library_case (str, PathLike[str], or None): Response-library configuration path for states and measured controls. Default: None
        input_feature_library_case (str, PathLike[str], or None): Parameter-library configuration path, required for explicit parametric fitting. Default: None

        vr_list (Sequence[float] or None): Reduced velocities selected for the requested workflow. Default: None
        lpd_list (Sequence[float] or None): Cylinder spacings L/D selected for tandem-cylinder workflows. Default: None

        save_pkl (bool): Whether to save the fitted model as model.pkl. Default: False

        differentiator_case (str, PathLike[str], or None): Differentiator configuration path identifying the fitted model. Default: None
        tweak0 (float or None): Reserved first weak-integration time in recorded t*. Default: None
        tweakf (float or None): Reserved last weak-integration time in recorded t*. Default: None
        K (int): Reserved number of weak integration subdomains. Default: 100
        H_xt (float or None): Reserved weak-subdomain half width in dynamic time tau. Default: None
        p (int): Reserved weak test-function polynomial degree. Default: 4
        random_state (int or None): Reserved random seed for weak-subdomain sampling. Default: 0
        results_root (str or PathLike[str]): Root directory for fitted models and evaluation artifacts. Default: 'results'

        re (float or None): Reynolds number used in dataset labels; None selects the cylinder-specific value. Default: None
        mstar (float or None): Mass ratio; workflow entry points use the cylinder-specific value when None. Default: None

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64

        constraint_mstar (float or None): Mass ratio for structural constraints; None selects the cylinder-specific physical value. Default: None
    """
    raise NotImplementedError("WeakSINDy is not implemented.")
