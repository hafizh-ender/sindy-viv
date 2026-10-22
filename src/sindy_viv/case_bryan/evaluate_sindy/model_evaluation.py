"""Shared validation, coefficient loading, and reporting for evaluation workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from numpy.typing import ArrayLike
from os import PathLike
from pysindy.feature_library.base import BaseFeatureLibrary
from sindy_viv._typing import DynamicsCase, EvaluationRecord
from typing import Any, Literal

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from sindy_viv.case_bryan.plot import plot_compare_prior_check, plot_selection_suffix
from sindy_viv.case_bryan.validation import validate_selection
from sindy_viv.feature_library.functions import fit_library_metadata
from sindy_viv.metrics.numerical import normalized_root_mean_squared_error, relative_l2_error
from sindy_viv.utils import add_dot_text

def model_differentiator_name(
    dynamics_case: DynamicsCase,
    differentiator_case: str | PathLike[str] | None,
    weak: bool,
) -> str:
    """Validate the dynamics case and return its training differentiator path label.

    Weak Cases 1/3 use no_differentiator. Other cases require a config
    path; only its stem is returned, without opening the file.

    Parameters:
        dynamics_case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.

        differentiator_case (str, PathLike[str], or None): Differentiator configuration path identifying the fitted model.
        weak (bool): Whether to locate weak-model results.
    """
    match dynamics_case:
        case 1 | 2 | 3 | 4 | 5:
            pass
        case _:
            raise ValueError("dynamics_case must be 1, 2, 3, 4, or 5.")

    # Cases without lift-rate states require no training differentiation for weak fits.
    if weak and dynamics_case in (1, 3):
        return "no_differentiator"

    if differentiator_case is None:
        raise ValueError("differentiator_case is required to identify this fitted model.")

    return Path(differentiator_case).stem

def select_time_region(time: ArrayLike, vr: float, tstar0: float | None, tstarf: float | None) -> slice:
    """Return an inclusive row slice selected by recorded t* bounds on tau input.

    Time must be finite, strictly increasing and contain at least two samples.
    Bounds default to the available endpoints; no interpolation/resampling is
    performed. Out-of-range/reversed bounds or fewer than two rows raise ValueError.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        tstar0 (float or None): First evaluation time in recorded t*; None uses the available start.
        tstarf (float or None): Last evaluation time in recorded t*; None uses the available end.
    """
    time = np.asarray(time, dtype=np.float64)

    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError("Evaluation needs at least two finite, strictly increasing time samples.")

    # Convert model time tau to recorded t* for selecting or displaying the data region.
    time_star = time * vr
    start = time_star[0] if tstar0 is None else float(tstar0)
    stop = time_star[-1] if tstarf is None else float(tstarf)

    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("tstar0 and tstarf must define a finite increasing interval.")

    if start < time_star[0] or stop > time_star[-1]:
        raise ValueError(f"Requested t* region [{start}, {stop}] is outside [{time_star[0]}, {time_star[-1]}].")

    # Include both time bounds using a slice rather than allocating a Boolean mask.
    region = slice(np.searchsorted(time_star, start, side='left'), np.searchsorted(time_star, stop, side='right'))
    if len(time[region]) < 2:
        raise ValueError("The selected t* region must contain at least two data samples.")

    return region

def load_coefficients(
    model_dir: str | PathLike[str],
    library: BaseFeatureLibrary,
    states: np.ndarray,
    state_names: Sequence[str],
    parameters: Sequence[float] = (),
    parameter_names: Sequence[str] = (),
    *,
    input_data: np.ndarray | None = None,
    input_names: Sequence[str] | None = None,
) -> np.ndarray:
    """Fit feature metadata and load an aligned float64 feature-by-target matrix.

    states are N-by-state; input_data, when supplied, contains measured controls
    followed by parameter columns described by input_names. Otherwise parameters
    are constant values described by parameter_names. CSV names must match the
    configured library and differentiated state names exactly. The passed library
    is fitted/mutated, but no optimizer is refitted. Invalid names, dimensions
    or nonfinite states/coefficients raise ValueError.

    Parameters:
        model_dir (str or PathLike[str]): Resolved directory containing the saved model.csv.

        library (BaseFeatureLibrary): Feature-library instance to prepare, fit, or evaluate.

        states (numpy.ndarray): State samples with one column per state name.
        state_names (Sequence[str]): Ordered names of the fitted state columns.
        parameters (Sequence[float]): Constant parameter values in parameter_names order. Default: ()
        parameter_names (Sequence[str]): Ordered names of the constant parameter columns. Default: ()
        input_data (numpy.ndarray or None): Aligned control and parameter samples, with one row per state sample. Default: None
        input_names (Sequence[str] or None): Ordered measured-control names, followed by parameter names when applicable. Default: None
    """
    if not np.all(np.isfinite(states)):
        raise ValueError("Selected reference states contain nonfinite values; select an interior region or check the differentiator.")

    if input_data is not None and (input_names is None or input_data.shape != (len(states), len(input_names))):
        raise ValueError("input_data must align with state samples and input_names.")

    # Only known shape-only libraries use a metadata sample; other fits retain full validation.
    fit_library_metadata(library, states, input_data, parameters)
    feature_names = library.get_feature_names(list(state_names) + list(parameter_names if input_names is None else input_names))
    target_names = add_dot_text(state_names)

    # Set the index after parsing so pandas retains numeric-looking feature names as text.
    table = pd.read_csv(Path(model_dir) / 'model.csv', converters={0: str})
    table = table.set_index(table.columns[0])

    if len(set(feature_names)) != len(feature_names) or table.index.has_duplicates or table.columns.has_duplicates:
        raise ValueError("Coefficient and library names must be unique.")

    if set(table.index) != set(feature_names) or set(table.columns) != set(target_names):
        raise ValueError(f"Model CSV feature/target names do not match the configured library and dynamics case: {model_dir}")
    coefficients = table.loc[feature_names, target_names].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Model coefficients must be finite.")

    return coefficients

def predict_derivative(
    library: BaseFeatureLibrary,
    coefficients: np.ndarray,
    states: np.ndarray,
    parameters: Sequence[float] = (),
    *,
    input_data: np.ndarray | None = None,
    block_size: int = 10000,
) -> np.ndarray:
    """Evaluate a fitted pointwise library with bounded feature-matrix scratch storage.

    states are N-by-state, coefficients are feature-by-target, and input_data
    contains aligned controls/parameters. Without input_data, parameters are
    broadcast constants. block_size limits temporary feature rows. Returns a
    float64 N-by-target array; invalid block size or nonfinite output raises ValueError.

    Parameters:
        library (BaseFeatureLibrary): Feature-library instance to prepare, fit, or evaluate.
        coefficients (numpy.ndarray): Feature-by-state coefficient matrix, with rows matching library features and columns matching state derivatives.
        states (numpy.ndarray): State samples with one column per state name.
        parameters (Sequence[float]): Constant parameter values in parameter_names order. Default: ()
        input_data (numpy.ndarray or None): Aligned control and parameter samples, with one row per state sample. Default: None
        block_size (int): Maximum rows per temporary feature or metric block; must be positive. Default: 10000
    """
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    prediction = np.empty((len(states), coefficients.shape[1]), dtype=np.float64)
    if input_data is not None and (input_data.ndim != 2 or len(input_data) != len(states) or not np.all(np.isfinite(input_data))):
        raise ValueError("input_data must contain finite control rows aligned with states.")

    # Limit each temporary feature matrix to the configured number of samples.
    for start in range(0, len(states), block_size):
        stop = min(start + block_size, len(states))
        samples = states[start:stop]
        if input_data is not None:
            samples = np.concatenate((samples, input_data[start:stop]), axis=1)
        elif len(parameters):

            # Broadcast the trajectory's constant parameters without allocating repeated rows.
            inputs = np.broadcast_to(parameters, (len(samples), len(parameters)))
            samples = np.concatenate((samples, inputs), axis=1)
        prediction[start:stop] = np.asarray(library.transform(samples)) @ coefficients

    if not np.all(np.isfinite(prediction)):
        raise ValueError("The fitted equations produced nonfinite derivatives.")

    return prediction

def simulate(
    library: BaseFeatureLibrary,
    coefficients: np.ndarray,
    time: ArrayLike,
    reference: np.ndarray,
    parameters: Sequence[float],
    *,
    method: str | type,
    rtol: float | np.ndarray,
    atol: float | np.ndarray,
    solver_options: dict[str, Any] | None,
    input_data: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Integrate evolving states in tau using the initial reference state only.

    library is fitted and coefficients are feature-by-state. input_data, when
    present, is linearly interpolated at solver stages; otherwise parameters
    are constant. method/rtol/atol and allowed solver_options go to solve_ivp.
    Returns (prediction, diagnostics) on exactly the supplied time samples.
    Failed, event-stopped, incomplete or nonfinite solves raise RuntimeError;
    reserved solver overrides raise ValueError. Reference trajectories are not
    used to drive the modeled states.

    Parameters:
        library (BaseFeatureLibrary): Feature-library instance to prepare, fit, or evaluate.
        coefficients (numpy.ndarray): Feature-by-state coefficient matrix, with rows matching library features and columns matching state derivatives.
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        reference (numpy.ndarray): Reference state or derivative samples aligned with time.
        parameters (Sequence[float]): Constant parameter values in parameter_names order.
        method (str or type): Integration solver passed to scipy.integrate.solve_ivp.
        rtol (float or numpy.ndarray): Relative numerical tolerance for the operation.
        atol (float or numpy.ndarray): Absolute tolerance passed to scipy.integrate.solve_ivp.
        solver_options (dict[str, Any] or None): Additional solve_ivp keyword settings, or None.
        input_data (numpy.ndarray or None): Aligned control and parameter samples, with one row per state sample. Default: None
    """
    options = dict(solver_options or {})

    # Preserve the model equation, initial state, and selected times when applying solver options.
    forbidden = {'fun', 't_span', 'y0', 't_eval', 'args', 'vectorized', 'method', 'rtol', 'atol'}
    if forbidden.intersection(options):
        raise ValueError(f"solver_options cannot override {sorted(forbidden.intersection(options))}.")

    if input_data is not None:
        if input_data.ndim != 2 or len(input_data) != len(time) or not np.all(np.isfinite(input_data)):
            raise ValueError("input_data must contain finite control rows aligned with simulation time.")

        # Interpolate prescribed controls at solver stages in tau, not at nearest reference rows.
        control_at_time = interp1d(time, input_data, axis=0, assume_sorted=True, bounds_error=True, copy=False)

    def rhs(tau: float, state: np.ndarray) -> np.ndarray:

        # Evaluate the evolving model state rather than following the measured reference.
        """Evaluate model derivatives at the evolving state and interpolated controls.

        Parameters:
            tau (float): Current simulation time in the dynamic coordinate tau.
            state (numpy.ndarray): Current simulated state vector.
        """
        if input_data is not None:
            return predict_derivative(library, coefficients, state[None, :], input_data=control_at_time(tau)[None, :])[0]
        return predict_derivative(library, coefficients, state[None, :], parameters)[0]

    return solve_state_ivp(rhs, time, reference[0], parameters=parameters,
                           method=method, rtol=rtol, atol=atol, solver_options=options)


def solve_state_ivp(
    rhs: Callable[[float, np.ndarray], np.ndarray], time: ArrayLike, initial_state: ArrayLike, *,
    parameters: Sequence[float] = (), method: str | type = 'RK45',
    rtol: float | np.ndarray = 1e-6, atol: float | np.ndarray = 1e-9,
    solver_options: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve an IVP on exactly the requested tau grid and validate full coverage.

    Only initial_state supplies reference information. Nonfinite or incomplete
    solves raise RuntimeError with the solver's original diagnostic. Reserved
    equation/grid/solver overrides in solver_options raise ValueError.
    """
    time = np.asarray(time, dtype=np.float64)
    initial_state = np.asarray(initial_state, dtype=np.float64)
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)
            or initial_state.ndim != 1 or not len(initial_state) or not np.all(np.isfinite(initial_state))):
        raise ValueError('IVP requires finite increasing times and a finite one-dimensional initial state.')
    options = dict(solver_options or {})
    forbidden = {'fun', 't_span', 'y0', 't_eval', 'args', 'vectorized', 'method', 'rtol', 'atol'}
    if forbidden.intersection(options):
        raise ValueError(f"solver_options cannot override {sorted(forbidden.intersection(options))}.")
    solution = solve_ivp(
        rhs,
        t_span=(float(time[0]), float(time[-1])),
        y0=initial_state,
        t_eval=time,
        method=method,
        rtol=rtol,
        atol=atol,
        **options,
    )

    # A stopped or failed solve must not be scored as if it covered the requested region.
    if not solution.success or len(solution.t) != len(time) or not np.array_equal(solution.t, time):

        # Report returned samples rather than rejected solver stages as the completed interval.
        last_time = f"{solution.t[-1]:.17g}" if len(solution.t) else "none"
        state_magnitude = f"{np.max(np.abs(solution.y[:, -1])):.6g}" if len(solution.t) else "unavailable"
        covered = f"[{solution.t[0]:.17g}, {last_time}]" if len(solution.t) else "none (no samples returned)"
        method_name = method if isinstance(method, str) else method.__name__

        raise RuntimeError(
            f"Model integration did not cover the selected region: {solution.message}. "
            f"method={method_name}; parameters={tuple(parameters)}; "
            f"requested tau=[{time[0]:.17g}, {time[-1]:.17g}]"
            f"; covered tau={covered}; "
            f"max |state| at last returned sample={state_magnitude}."
        )

    prediction = np.asarray(solution.y.T, dtype=np.float64)

    if prediction.shape != (len(time), len(initial_state)) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("Model integration returned an invalid state trajectory.")

    diagnostics = {
        'method': method if isinstance(method, str) else method.__name__,
        'rtol': np.asarray(rtol).tolist(),
        'atol': np.asarray(atol).tolist(),
        'status': int(solution.status),
        'message': solution.message,
        'nfev': int(solution.nfev),
        'njev': int(solution.njev),
        'nlu': int(solution.nlu),
    }
    return prediction, diagnostics

def report_evaluation(
    kind: str,
    model_dir: str | PathLike[str] | None,
    report_dir: str | PathLike[str],
    time: ArrayLike,
    vr: float,
    reference: np.ndarray,
    prediction: np.ndarray,
    variable_names: Sequence[str],
    parameter_values: Mapping[str, float],
    reference_differentiator_name: str,
    *,
    save_plot: bool,
    xlim: Sequence[float] | None,
    ylim_mean_time: Sequence[float | None],
    block_size: int,
    solver: dict[str, Any] | None = None,
    ylim_source: Literal['reference', 'prediction', 'both', 'true', 'pred'] = 'reference',
    return_data: bool = True,
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
) -> EvaluationRecord:
    """Save selected-window metrics and optionally a comparison figure closed after saving.

    Time is tau; filenames and figures use t*=VR*tau. Reference/prediction are
    aligned N-by-channel arrays. Metrics are written before figure rendering,
    and undefined normalized values become JSON null. Existing matching artifact
    names are overwritten. return_data=False returns a small record containing
    metrics, paths, diagnostics and time-region/sample metadata. The default True
    also returns time/time_star, an owning reference copy and the prediction.

    Parameters:
        kind (str): Evaluation label used in report filenames.
        model_dir (str, PathLike[str], or None): Saved-model directory; None for pre-fit derivative assessment.
        report_dir (str or PathLike[str]): Destination directory for evaluation metrics and figures.

        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        reference (numpy.ndarray): Reference state or derivative samples aligned with time.
        prediction (numpy.ndarray): Predicted state or derivative samples aligned with the reference.
        variable_names (Sequence[str]): Ordered names identifying measured or evaluated columns.

        parameter_values (Mapping[str, float]): Dataset parameter labels and values written to reports and figures.
        reference_differentiator_name (str): Reference differentiator label used in report paths.
        save_plot (bool): Whether to save the comparison figure with the metric report.

        xlim (Sequence[float] or None): Displayed time limits in recorded t*, or None for the available range.
        ylim_mean_time (Sequence[float | None]): Recorded t* interval used for y-limit statistics, with None for open endpoints.

        block_size (int): Maximum rows per temporary feature or metric block; must be positive.
        solver (dict[str, Any] or None): Integration status and diagnostic values included in the report. Default: None
        ylim_source (Literal['reference', 'prediction', 'both', 'true', 'pred']): Data used to calculate y limits: reference, prediction, or both; true and pred are aliases. Default: 'reference'

        return_data (bool): Whether returned reports retain time, reference, and prediction arrays. Default: True
        plot_variables (Sequence[str] or None): Variable families or exact output-channel names to display, without filtering metrics. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display, without filtering metrics. Default: None
    """
    if reference.shape != prediction.shape or not np.all(np.isfinite(reference)):
        raise ValueError("Reference and prediction must have matching finite arrays.")

    # Score each channel using the same selected samples supplied to the comparison plot.
    nrmse = normalized_root_mean_squared_error(prediction, reference, reduction='none', block_size=block_size)
    rl2e = relative_l2_error(prediction, reference, reduction='none', block_size=block_size)
    metrics = {
        # name: {'NRMSE': float(nrmse[column]), 'RL2E': float(rl2e[column])}
        name: {'NRMSE': float(nrmse[column])}
        for column, name in enumerate(variable_names)
    }
    time = np.asarray(time, dtype=np.float64)
    if return_data:
        time = time.copy()

    # Convert model time tau to recorded t* for selecting or displaying the data region.
    time_star = time * vr if save_plot or return_data else np.asarray([time[0] * vr, time[-1] * vr])
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    parameter_values = {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in parameter_values.items()
    }

    # Preserve float64 endpoint precision so nearby evaluation windows have distinct filenames.
    suffix = f"t={time_star[0]:.17g}-{time_star[-1]:.17g}_diff={reference_differentiator_name}"
    metrics_path = report_dir / f"{kind}_error_metrics_{suffix}.json"

    # Undefined normalization (constant or zero reference channels) is valid JSON null.
    json_metrics = {
        name: {key: value if np.isfinite(value) else None for key, value in channel.items()}
        for name, channel in metrics.items()
    }
    payload = {
        'evaluation': kind,
        'model_path': None if model_dir is None else str(Path(model_dir) / 'model.csv'),
        # 'time_coordinate': 'tau; t_star = VR * tau',
        'time_region': [float(time[0]), float(time[-1])],
        'tstar_region': [float(time_star[0]), float(time_star[-1])],
        'sample_count': len(time),
        'parameters': parameter_values,
        'reference_differentiator': reference_differentiator_name,
        'solver': solver,
        'metrics': json_metrics,
    }

    # Preserve computed metrics even if figure rendering or saving fails afterward.
    with metrics_path.open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)

    figure_path = None
    if save_plot:
        if len(ylim_mean_time) != 2:
            raise ValueError("ylim_mean_time must contain a start and an end.")

        start, stop = ylim_mean_time
        figure, _ = plot_compare_prior_check(
            time=time_star, data_true=reference, data_pred=prediction,
            true_state_variable_names=variable_names, pred_state_variable_names=variable_names,
            parameter_values_dict=parameter_values, error_metrics_dict=metrics,
            xlim=xlim, ylim_mean_time_start=time_star[0] if start is None else start,
            ylim_mean_time_end=stop, ylim_source=ylim_source,
            plot_variables=plot_variables, plot_cylinders=plot_cylinders,
        )
        selection_suffix = plot_selection_suffix(variable_names, plot_variables, plot_cylinders)
        figure_path = report_dir / f"{kind}_comparison_{suffix}{selection_suffix}.png"

        try:
            figure.savefig(figure_path, dpi=300)
        finally:

            # Release the figure even if saving fails during a multi-trajectory evaluation.
            plt.close(figure)

    result = {
        'variable_names': list(variable_names),
        'metrics': metrics,
        'model_path': None if model_dir is None else Path(model_dir) / 'model.csv',
        'metrics_path': metrics_path, 'figure_path': figure_path, 'solver': solver,
        'sample_count': len(time),
        'time_region': payload['time_region'], 'tstar_region': payload['tstar_region'],
    }

    if return_data:
        result.update(time=time, time_star=time_star, reference=reference.copy(), prediction=prediction)

    return result
