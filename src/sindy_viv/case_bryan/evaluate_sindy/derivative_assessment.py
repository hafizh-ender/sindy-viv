"""Assess numerical derivatives before SINDy fitting: IVP and recorded channels."""

from __future__ import annotations
from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any
import json
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

from sindy_viv._typing import CylinderType, DynamicsCase
from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME, SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR,
    TWO_TANDEM_CYLINDERS_CASE_NAME, TWO_TANDEM_CYLINDERS_RE, TWO_TANDEM_CYLINDERS_MSTAR,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.analysis.paths import analysis_directory
from sindy_viv.case_bryan.model_storage.paths import training_directory
from sindy_viv.case_bryan.model_storage.fit_windows import load_fit_window_config, prepare_fit_window_bounds
from sindy_viv.case_bryan.prepare_sindy.case_layout import get_case_layout
from sindy_viv.case_bryan.prepare_sindy.trajectory import prepare_trajectory
from sindy_viv.case_bryan.validation import validate_selection
from sindy_viv.utils import initiate_object, load_config
from .model_evaluation import report_evaluation, select_time_region, solve_state_ivp


def reconstruct_from_derivatives(
    time: np.ndarray, derivatives: np.ndarray, initial_state: np.ndarray, *,
    method: str | type = 'trapezoid', rtol: float | np.ndarray = 1e-7,
    atol: float | np.ndarray = 1e-9, solver_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve X'=interpolated numerical Xdot(tau), using only measured X(tau0).

    Each derivative column is a prescribed time history; subsequent measured
    states do not drive the reconstruction. This checks differentiation and
    reintegration consistency, independently of any SINDy model or structural
    equation. Derivatives/time are aligned N-by-state and N-vectors in tau.
    The default integrates the piecewise-linear derivative exactly by cumulative
    trapezoids. SciPy IVP methods such as RK45/DOP853 are optional alternatives.
    """
    time = np.asarray(time, dtype=np.float64)
    derivatives = np.asarray(derivatives, dtype=np.float64)
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)
            or initial_state.ndim != 1 or derivatives.shape != (len(time), len(initial_state))
            or not np.all(np.isfinite(derivatives)) or not np.all(np.isfinite(initial_state))):
        raise ValueError('Derivative reconstruction requires aligned finite times, derivatives and an initial state.')
    if method == 'trapezoid':
        if solver_options:
            raise ValueError('solver_options apply only to adaptive IVP methods, not trapezoid reconstruction.')
        prediction = initial_state+cumulative_trapezoid(derivatives, time, axis=0, initial=0.)
        if not np.all(np.isfinite(prediction)):
            raise RuntimeError('Derivative IVP reconstruction produced nonfinite values.')
        return prediction, dict(method='trapezoid', status=0, nfev=0,
                                message='Exact IVP integration of piecewise-linear derivative histories.')
    derivative_at_time = interp1d(time, derivatives, axis=0, assume_sorted=True, bounds_error=True, copy=False)
    # Resolve every interpolation segment, so adaptive solvers cannot skip short-lived derivative structure.
    options = dict(solver_options or {})
    options.setdefault('max_step', float(np.min(np.diff(time))))
    return solve_state_ivp(lambda tau, state: derivative_at_time(tau), time, initial_state,
                           method=method, rtol=rtol, atol=atol, solver_options=options)


def assess_derivatives(
    dynamics_case: DynamicsCase, differentiator_case: str | PathLike[str], *,
    cylinder_type: CylinderType = 'single', vr_list: Sequence[float] | None = None,
    lpd_list: Sequence[float] | None = None, tstar0: float | None = None, tstarf: float | None = None,
    results_root: str | PathLike[str] = 'results', re: float | None = None, mstar: float | None = None,
    method: str | type = 'trapezoid', rtol: float | np.ndarray = 1e-7, atol: float | np.ndarray = 1e-9,
    solver_options: dict[str, Any] | None = None, save_plot: bool = True, return_data: bool = False,
    filtering_case: str | PathLike[str] | None = None,
    fit_window_case: str | PathLike[str] | None = None,
) -> dict[float | tuple[float, float], dict[str, Any]]:
    """Assess full-record numerical derivatives on a selected reporting interval.

    First construct X and Xdot on full tau records using prepare_trajectory,
    then crop both. Integrate every numerical derivative from the selected
    initial X and report NRMSE versus X. For Cases 1/2/3, also compare D(h)
    with recorded hdot and D(recorded hdot) with recorded hddot, per cylinder.
    Recorded comparisons use matching timestamps without shifting the data.
    Cases 4/5 contain no motion states, so only their lift-state IVP is checked.
    Optional filtering_case smooths full records before differentiation.
    fit_window_case supplies per-trajectory reporting bounds; do not combine it
    with explicit tstar0/tstarf. Full-record filtering/differentiation precede cropping.
    No SINDy fitting is performed. Outputs include separate
    JSON/plots for the IVP and recorded-channel checks, plus a config snapshot.
    Unfiltered reports use an explicit filtering=None folder beside named filters.
    """
    layout = get_case_layout(dynamics_case, cylinder_type)
    single = cylinder_type == 'single'
    case_name = SINGLE_CYLINDER_CASE_NAME if single else TWO_TANDEM_CYLINDERS_CASE_NAME
    re = (SINGLE_CYLINDER_RE if single else TWO_TANDEM_CYLINDERS_RE) if re is None else re
    mstar = (SINGLE_CYLINDER_MSTAR if single else TWO_TANDEM_CYLINDERS_MSTAR) if mstar is None else mstar
    if single and lpd_list is not None:
        raise ValueError('lpd_list requires cylinder_type=two_tandem.')
    vrs = validate_selection(((3, 4, 5, 6, 7, 8) if single else tuple(range(5, 15)))
                             if vr_list is None else vr_list, 'vr_list')
    lpds = (None,) if single else validate_selection((1.5, 2., 3., 3.5, 4.) if lpd_list is None else lpd_list, 'lpd_list')
    if fit_window_case is not None and (tstar0 is not None or tstarf is not None):
        raise ValueError('Use either fit_window_case or tstar0/tstarf, not both.')
    window_config = load_fit_window_config(fit_window_case, cylinder_type) if fit_window_case is not None else None
    bounds = (prepare_fit_window_bounds(window_config, [(lpd, vr) for lpd in lpds for vr in vrs], cylinder_type)
              if window_config is not None else dict.fromkeys([(lpd, vr) for lpd in lpds for vr in vrs], (tstar0, tstarf)))
    config = load_config(differentiator_case)
    filtering_config = load_config(filtering_case) if filtering_case is not None else None
    filtering = initiate_object(**filtering_config) if filtering_case is not None else None
    names = list(layout.measured_names)
    motion_states = dynamics_case in (1, 2, 3)
    if motion_states:
        names += [rf'\ddot{{h}}_{i}' for i in layout.cylinder_ids]
    result = {}
    for lpd in lpds:
        for vr in vrs:
            if single:
                time, data = load_data_single_cylinder(re, mstar, vr, variables=names)
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, variables=names)
            if filtering is not None:
                data = filtering(data, time)
            prepared = prepare_trajectory(time, data, names, initiate_object(**config), dynamics_case, cylinder_type)
            start, stop = bounds[(lpd, vr)]
            region = select_time_region(prepared.time, vr, start, stop)
            t = prepared.time[region].copy()
            states = prepared.states[region].copy()
            derivatives = prepared.derivatives[region].copy()
            recorded_reference, computed_derivatives, recorded_names = [], [], []
            if motion_states:
                for i in layout.cylinder_ids:
                    h, v, a = f'h_{i}', rf'\dot{{h}}_{i}', rf'\ddot{{h}}_{i}'
                    recorded_reference.extend((data[region, names.index(v)].copy(), data[region, names.index(a)].copy()))
                    computed_derivatives.extend((derivatives[:, prepared.state_names.index(h)],
                                                 derivatives[:, prepared.state_names.index(v)]))
                    recorded_names.extend((v, a))
            del data, time, prepared
            prediction, solver = reconstruct_from_derivatives(t, derivatives, states[0], method=method,
                                                              rtol=rtol, atol=atol, solver_options=solver_options)
            folder = (analysis_directory(results_root, case_name, cylinder_type, re, mstar, lpd=lpd, vr=vr)
                      / 'derivative_assessment' / f'dynamics_case_{dynamics_case}'
                      / ('filtering=' + (Path(filtering_case).stem if filtering_case is not None else 'None'))
                      / Path(differentiator_case).stem)
            folder = training_directory(folder, 'individual', [dict(
                VR=vr, LpD=lpd, requested_tstar0=start, requested_tstarf=stop)], fit_window_case)
            labels = {'Re': re, 'm^*': mstar, 'V_R': vr}
            if lpd is not None:
                labels['L/D'] = lpd
            # Canonical state ordering comes from the same preparation used in training.
            state_names = list(layout.state_names)
            kwargs = dict(model_dir=None, report_dir=folder, time=t, vr=vr, parameter_values=labels,
                          reference_differentiator_name=Path(differentiator_case).stem,
                          save_plot=save_plot, xlim=None, ylim_mean_time=(None, None), block_size=10000,
                          return_data=return_data)
            record = {'ivp': report_evaluation('derivative_ivp', reference=states, prediction=prediction,
                                               variable_names=state_names, solver=solver, **kwargs)}
            if motion_states:
                record['recorded_derivatives'] = report_evaluation(
                    'recorded_derivatives', reference=np.column_stack(recorded_reference),
                    prediction=np.column_stack(computed_derivatives), variable_names=recorded_names, **kwargs)
            snapshot = dict(dynamics_case=dynamics_case, cylinder_type=cylinder_type, differentiator=config,
                            time_coordinate='tau=t*/VR', recorded_comparison_alignment='same timestamp; no shifts',
                            differentiation_scope='full trajectory before cropping')
            if filtering_case is not None:
                snapshot['filtering'] = dict(name=Path(filtering_case).stem, source=str(filtering_case), config=filtering_config)
            if fit_window_case is not None:
                snapshot['fit_window'] = dict(name=Path(fit_window_case).stem, source=str(fit_window_case), config=window_config)
            # Each report has its own time bounds; the snapshot records processing conventions.
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'assessment_config.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False), encoding='utf-8')
            result[vr if single else (lpd, vr)] = record
            del t, states, derivatives, prediction, recorded_reference, computed_derivatives
    return result
