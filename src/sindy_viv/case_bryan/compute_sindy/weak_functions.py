"""Weak-form cylinder computations with pointwise models for evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import DTypeLike
from os import PathLike
from sindy_viv._typing import CylinderType, DynamicsCase, Parameterization

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
from sindy_viv.sindy.functions import perform_weak_sindy
from sindy_viv.feature_library.weak import resolve_weak_library_config, validate_weak_settings
from sindy_viv.utils import add_dot_text, initiate_object, load_config

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
    K: int | None = None,
    H_xt: float | None = None,
    p: int | None = None,
    random_state: int | None = 0,
    results_root: str | PathLike[str] = "results",
    re: float | None = None,
    mstar: float | None = None,
    dtype: DTypeLike = np.float64,
    constraint_mstar: float | None = None,
    fit_window_case: str | PathLike[str] | None = None,
    weak_feature_library_case: str | PathLike[str] | None = None,
) -> ps.SINDy | dict[float | tuple[float, float], ps.SINDy]:
    """Fit weak dynamics for Cases 1-5 and all cylinder/parameter selections.

    The selectors, return mapping, libraries, optimizers and artifact formats
    match compute_sindy. Only Cases 2/4/5 require differentiator_case, to build
    lift-rate states on full trajectories; weak targets are never differentiated.
    tweak0/tweakf select inclusive recorded t* bounds (None means open ends).
    Alternatively fit_window_case supplies per-trajectory bounds; do not combine
    it with tweak0/tweakf. Time remains float64 tau=t*/VR throughout fitting.

    weak_feature_library_case selects a WeakPDELibrary integration descriptor
    (default configs/weak_feature_library/default.json), combined with the
    response/parameter configs. Explicit K/H_xt/p override its settings.
    K windows per trajectory use phi=(1-z**2)**p, p>=1. H_xt is a half width
    in tau, defaulting to each cropped trajectory span/20. random_state seeds
    a local generator, leaving NumPy's global state untouched. Windows need
    at least three samples and never span separate trajectories. Controls and
    parameter products are integrated together in the pointwise library order.

    Writes model.csv, training.json and optional model.pkl under
    weak_*/<weak-config-stem>/<pointwise-library> paths.
    Metadata includes requested/actual bounds and resolved integration widths.
    Returned PySINDy models predict pointwise derivatives and can be simulated
    or evaluated using weak=True. Matching destinations overwrite prior results.
    Invalid settings raise ValueError; data/configuration/optimizer errors propagate.
    """
    for name, value in (("tweak0", tweak0), ("tweakf", tweakf)):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite or None.")
    if tweak0 is not None and tweakf is not None and tweak0 >= tweakf:
        raise ValueError("tweak0 must be smaller than tweakf.")
    if fit_window_case is not None and (tweak0 is not None or tweakf is not None):
        raise ValueError("Use either fit_window_case or tweak0/tweakf, not both.")

    # Validate the dynamics case
    if dynamics_case not in (1, 2, 3, 4, 5):
        raise ValueError("dynamics_case must be 1, 2, 3, 4, or 5.")

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
    if layout.rate_names and differentiator_case is None:
        raise ValueError(f"Case {dynamics_case} requires differentiator_case to build lift-rate states.")
    differentiator_name = Path(differentiator_case).stem if layout.rate_names else "no_differentiator"
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

    if weak_feature_library_case is None:
        weak_feature_library_case = Path(__file__).resolve().parents[4] / "configs" / "weak_feature_library" / "default.json"
    weak_library_config = resolve_weak_library_config(load_config(weak_feature_library_case), K, H_xt, p)
    K, H_xt, p = (weak_library_config['kwargs'][name] for name in ('K', 'H_xt', 'p'))
    validate_weak_settings(K, H_xt, p, random_state)

    pairs = [(lpd, vr) for lpd in lpd_list for vr in vr_list]
    window_config = None
    if fit_window_case is None:
        fitting_bounds = dict.fromkeys(pairs, (tweak0, tweakf))
    else:
        window_config = load_fit_window_config(fit_window_case, cylinder_type)
        fitting_bounds = prepare_fit_window_bounds(window_config, pairs, cylinder_type)
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

    differentiator_config = load_config(differentiator_case) if layout.rate_names else None
    optimizer_config = load_config(optimizer_case)
    models = {}
    for lpd, vr in trajectory_groups:
        differentiator = initiate_object(**differentiator_config) if differentiator_config is not None else None
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
            trajectory = prepare_trajectory(
                time, data, variable_names, differentiator, dynamics_case, cylinder_type,
                dtype=dtype, with_derivatives=False,
            )
            del time, data
            if parameterization == "vr":
                trajectory = trajectory.with_parameters((trajectory_vr,), parameter_names)
            elif parameterization == "lpd":
                trajectory = trajectory.with_parameters((trajectory_lpd,), parameter_names)
            elif parameterization == "vr_lpd":
                trajectory = trajectory.with_parameters((trajectory_lpd, trajectory_vr), parameter_names)

            # Build any lift-rate states before cropping; no pointwise derivative targets are needed.
            time, states, _, inputs, metadata = crop_training_trajectory(
                trajectory.time, trajectory.states, trajectory.derivatives, trajectory.inputs,
                trajectory_lpd, trajectory_vr, fitting_bounds[(trajectory_lpd, trajectory_vr)],
            )
            state_names, input_names = trajectory.state_names, trajectory.input_names
            cropped.append((time, states, inputs))
            training_trajectories.append(metadata)
            del trajectory, time, states, inputs
        del differentiator
        if parameterization == "individual":
            t, X, U = cropped[0]
            library = prepare_library(state_library_config)
        else:
            t, X, U = [list(values) for values in zip(*cropped)]

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
        model = perform_weak_sindy(
            time=t, state_variable_data=X, feature_library=library, optimizer=optimizer,
            K=K, H_xt=H_xt, p=p, random_state=random_state,
            weak_library_config=weak_library_config,
            feature_names=state_names + input_names, input_variable_data=U,
        )

        # Returned models predict from new X/U; prepared training arrays are no longer needed here.
        del t, X, U

        results_dir = model_base_directory(
            results_root, case_name, dynamics_case, differentiator_name, "weak_" + fit_mode,
            cylinder_type, parameterization, state_feature_library_case,
            input_feature_library_case, optimizer_case, re, mstar, vr, lpd,
            weak_feature_library_case=weak_feature_library_case,
        )
        results_dir = training_directory(results_dir, parameterization, training_trajectories, fit_window_case)
        results_dir.mkdir(parents=True, exist_ok=True)

        # Record ordered requested bounds and actual samples beside all artifacts for this fit.
        training_metadata = {
            'schema_version': 1,
            'weak': model.weak_settings_,
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
        training_metadata['configurations']['weak_feature_library'] = weak_library_config
        training_metadata['weak_feature_library_source'] = str(weak_feature_library_case)
        if parameterization != "individual":
            training_metadata['configurations']['parameter_feature_library'] = input_library_config

        if fit_window_case is not None:
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
