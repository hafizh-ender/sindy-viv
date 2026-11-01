"""Couple learned lift dynamics to theoretical or saved structural equations."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
from pysindy.feature_library.base import BaseFeatureLibrary
from sindy_viv._typing import CylinderType
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from .model_evaluation import pointwise_predictor, solve_state_ivp, simulate
from .complete_lift_model import complete_state_names, compile_complete_model

def coupled_reference(
    states: np.ndarray, state_names: Sequence[str], inputs: np.ndarray,
    input_names: Sequence[str], cylinder_type: CylinderType, case: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the complete measured reference by channel name (three states in Case 6)."""
    names = complete_state_names(case, cylinder_type)
    values = {name: states[:, i] for i, name in enumerate(state_names)}
    values.update({name: inputs[:, i] for i, name in enumerate(input_names)})
    return np.column_stack([values[name] for name in names]), names

def simulate_coupled_lift(
    library: BaseFeatureLibrary, coefficients: np.ndarray, time: np.ndarray,
    initial_state: np.ndarray, *, case: int, cylinder_type: CylinderType, vr: float,
    mstar: float, parameters: Sequence[float] = (), method: str | type = 'RK45',
    rtol: float | np.ndarray = 1e-6, atol: float | np.ndarray = 1e-9,
    solver_options: dict[str, Any] | None = None,
    allow_partial: bool = False,
    lpd: float | None = None, motion_models=None, base_model_dir=None,
    output_directory=None, reference: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Integrate the complete coupled state without prescribed motion histories.

    States are grouped per cylinder in ID order: [h_i, hdot_i, CL_i] for Case 6;
    [h_i, hdot_i, CL_i, CLdot_i] for Cases 4/7.
    h'=hdot and hdot'=-4*pi**2*h+2*VR**2*CL/(pi*mstar) are exact.
    Case 4 retains both fitted lift equations; its acceleration input is computed
    from the evolving h/CL. Cases 6/7 compile their independently substituted
    lift/motion models and exact kinematic identities before integration.
    Constant parameters follow the original
    pointwise response inputs in the library. All rates and solver times use
    tau=t*/VR. Only initial_state is taken from measured trajectories.
    """
    if case not in (4, 6, 7):
        raise ValueError('Coupled lift integration supports only Cases 4, 6, and 7.')
    if case in (6, 7):
        if reference is None:
            reference = np.broadcast_to(initial_state, (len(time), len(initial_state)))
        complete_library, complete_coefficients, manifest = compile_complete_model(
            library, coefficients, reference, case=case, cylinder_type=cylinder_type,
            vr=vr, lpd=lpd, mstar=mstar, parameters=parameters, motion_models=motion_models,
            base_model_dir=base_model_dir, output_directory=output_directory)
        prediction, diagnostics = simulate(complete_library, complete_coefficients, time,
            reference, (), method=method, rtol=rtol, atol=atol, solver_options=solver_options,
            allow_partial=allow_partial)
        diagnostics.update(coupled_motion=True, prescribed_motion=False,
            motion_source=manifest['motion']['source'], parameters_substituted=manifest['parameters_substituted'],
            kinematic_identities=manifest['kinematic_identities'])
        if manifest['motion']['source'] == 'theoretical':
            diagnostics['structural_mstar'] = float(mstar)
        return prediction, diagnostics
    if motion_models is not None:
        raise ValueError('Saved Case 3 motion donors are supported for Cases 6/7.')
    if not np.isfinite(vr) or vr <= 0 or not np.isfinite(mstar) or mstar <= 0:
        raise ValueError('vr and structural mstar must be finite and positive.')
    layout = get_case_layout(case, cylinder_type)
    count = len(layout.cylinder_ids)
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if initial_state.shape != (4*count,):
        raise ValueError('Coupled initial state must have four entries per cylinder.')

    predict = pointwise_predictor(library, coefficients)
    lift_coefficient = 2*vr**2/(np.pi*mstar)

    def rhs(tau: float, state: np.ndarray) -> np.ndarray:
        """Construct the lift-library inputs from the evolving coupled state."""
        h, velocity, lift, lift_rate = state.reshape(count, 4).T
        acceleration = -4*np.pi**2*h+lift_coefficient*lift
        motion = np.column_stack((h, velocity, acceleration))
        # The fitted library orders all CL states before all CLdot states;
        # measured motion controls are grouped by cylinder, then parameters.
        learned = predict(np.concatenate((lift, lift_rate)),
                          np.concatenate((motion.ravel(), parameters)))
        return np.column_stack((velocity, acceleration, learned[:count], learned[count:])).ravel()

    prediction, diagnostics = solve_state_ivp(
        rhs, time, initial_state, parameters=parameters, method=method, rtol=rtol,
        atol=atol, solver_options=solver_options, allow_partial=allow_partial,
    )
    diagnostics.update(coupled_motion=True, structural_mstar=float(mstar),
                       prescribed_motion=False, lift_equations='both fitted lift equations')
    return prediction, diagnostics
