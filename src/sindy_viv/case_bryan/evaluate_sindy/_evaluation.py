"""Shared validation, coefficient loading, and reporting for evaluation workflows."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from sindy_viv.case_bryan.plot import plot_compare_prior_check
from sindy_viv.differentiation import perform_derivative
from sindy_viv.metrics.numerical import normalized_root_mean_squared_error, relative_l2_error
from sindy_viv.utils import add_dot_text

def model_differentiator_name(dynamics_case, differentiator_case, weak):
    match dynamics_case:
        case 1 | 2:
            pass
        case _:
            raise ValueError("dynamics_case must be 1 or 2.")
        
    # WeakSINDy on Case 1 is saved without a training differentiator in its model path.
    if weak and dynamics_case == 1:
        return "no_differentiator"
    
    if differentiator_case is None:
        raise ValueError("differentiator_case is required to identify this fitted model.")
    
    return Path(differentiator_case).stem


def validate_selection(values, name):
    values = tuple(values)
    
    if not values or any(not np.isscalar(value) or not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain finite positive values.")
    
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")
    
    # Convert NumPy scalars to native values for dataset paths and JSON reports.
    return tuple(value.item() if isinstance(value, np.generic) else value for value in values)


def select_time_region(time, vr, tstar0, tstarf):
    """Select existing samples inclusively in t*=VR*tau, without interpolation."""
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

def prepare_integral_states(time, data, variable_names, differentiator, dynamics_case, cylinder_type):
    """Construct X without computing unused regression targets for integration."""
    if cylinder_type == 'single':
        cylinder_ids = (0,)
    elif cylinder_type == 'two_tandem':
        cylinder_ids = (0, 1)
    else:
        raise ValueError("Unknown cylinder_type.")
    
    state_names = list(variable_names)
    match dynamics_case:
        case 1:
            return data, state_names
        case 2:
            if differentiator is None:
                raise ValueError("Case 2 requires a differentiator to construct lift-rate reference states.")
            state_names += [rf'\dot{{C}}_{{L{cylinder_id}}}' for cylinder_id in cylinder_ids]
        case _:
            raise ValueError("dynamics_case must be 1 or 2.")
        
    states = np.empty((len(time), len(state_names)), dtype=data.dtype)
    
    for name in variable_names:
        states[:, state_names.index(name)] = data[:, variable_names.index(name)]
    lift_names = [f'C_{{L{cylinder_id}}}' for cylinder_id in cylinder_ids]
    
    # Differentiate the complete lift signal before selecting a simulation window.
    lift_rates = perform_derivative(
        time, data[:, [variable_names.index(name) for name in lift_names]], differentiator,
    )
    
    for column, cylinder_id in enumerate(cylinder_ids):
        states[:, state_names.index(rf'\dot{{C}}_{{L{cylinder_id}}}')] = lift_rates[:, column]
        
    return states, state_names

def load_coefficients(model_dir, library, states, state_names, parameters=(), parameter_names=()):
    """Align CSV coefficients by feature and derivative names, never row positions."""
    if not np.all(np.isfinite(states)):
        raise ValueError("Selected reference states contain nonfinite values; select an interior region or check the differentiator.")
    
    samples = states
    if parameter_names:
        # Broadcast the trajectory's constant parameters without allocating repeated rows.
        inputs = np.broadcast_to(parameters, (len(states), len(parameter_names)))
        samples = np.concatenate((states, inputs), axis=1)
        
    # Fit only feature metadata; no optimizer or model coefficients are refitted.
    library.fit(samples)
    feature_names = library.get_feature_names(list(state_names) + list(parameter_names))
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


def predict_derivative(library, coefficients, states, parameters=(), *, block_size=10_000):
    """Bound feature-matrix memory by transforming one state block at a time."""
    if not isinstance(block_size, (int, np.integer)) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    
    prediction = np.empty((len(states), coefficients.shape[1]), dtype=np.float64)
    
    # Limit each temporary feature matrix to the configured number of samples.
    for start in range(0, len(states), block_size):
        stop = min(start + block_size, len(states))
        samples = states[start:stop]
        if len(parameters):
            # Broadcast the trajectory's constant parameters without allocating repeated rows.
            inputs = np.broadcast_to(parameters, (len(samples), len(parameters)))
            samples = np.concatenate((samples, inputs), axis=1)
        prediction[start:stop] = np.asarray(library.transform(samples)) @ coefficients
        
    if not np.all(np.isfinite(prediction)):
        raise ValueError("The fitted equations produced nonfinite derivatives.")
    
    return prediction

def simulate(library, coefficients, time, reference, parameters, *, method, rtol, atol, solver_options):
    """Evolve the model state on tau; reference samples supply only the initial state."""
    options = dict(solver_options or {})
    
    # Preserve the model equation, initial state, and selected times when applying solver options.
    forbidden = {'fun', 't_span', 'y0', 't_eval', 'args', 'vectorized', 'method', 'rtol', 'atol'}
    if forbidden.intersection(options):
        raise ValueError(f"solver_options cannot override {sorted(forbidden.intersection(options))}.")

    def rhs(tau, state):
        # Evaluate the evolving model state rather than following the measured reference.
        return predict_derivative(library, coefficients, state[None, :], parameters)[0]

    solution = solve_ivp(
        rhs, t_span=(float(time[0]), float(time[-1])),
        y0=np.asarray(reference[0], dtype=np.float64), t_eval=time,
        method=method, rtol=rtol, atol=atol, **options,
    )
    
    # A stopped or failed solve must not be scored as if it covered the requested region.
    if not solution.success or len(solution.t) != len(time) or not np.array_equal(solution.t, time):
        # Report returned samples rather than rejected solver stages as the completed interval.
        last_time = f"{solution.t[-1]:.17g}" if len(solution.t) else "none"
        
        state_magnitude = f"{np.max(np.abs(solution.y[:, -1])):.6g}" if len(solution.t) else "unavailable"
        
        method_name = method if isinstance(method, str) else method.__name__
        
        raise RuntimeError(
            f"Model integration did not cover the selected region: {solution.message}. "
            f"method={method_name}; parameters={tuple(parameters)}; "
            f"requested tau=[{time[0]:.17g}, {time[-1]:.17g}]"
            f"covered tau=[{solution.t[0]:.17g}, {last_time}];"
            f"max |state| at last returned sample={state_magnitude}."
        )
        
    prediction = np.asarray(solution.y.T, dtype=np.float64)
    
    if prediction.shape != reference.shape or not np.all(np.isfinite(prediction)):
        raise RuntimeError("Model integration returned an invalid state trajectory.")
    
    diagnostics = {
        'method': method if isinstance(method, str) else method.__name__,
        'rtol': np.asarray(rtol).tolist(), 'atol': np.asarray(atol).tolist(),
        'status': int(solution.status), 'message': solution.message,
        'nfev': int(solution.nfev), 'njev': int(solution.njev), 'nlu': int(solution.nlu),
    }
    return prediction, diagnostics

def report_evaluation(
    kind, model_dir, report_dir, time, vr, reference, prediction, variable_names,
    parameter_values, reference_differentiator_name, *, save_plot, xlim,
    ylim_mean_time, block_size, solver=None,
):
    """Save metrics for the scored samples and optionally a closed comparison figure."""
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
    
    time = np.asarray(time, dtype=np.float64).copy()
    
    # Convert model time tau to recorded t* for selecting or displaying the data region.
    time_star = time * vr
    
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
        'model_path': str(Path(model_dir) / 'model.csv'),
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
            ylim_mean_time_end=stop,
        )
        
        figure_path = report_dir / f"{kind}_comparison_{suffix}.png"
        
        try:
            figure.savefig(figure_path, dpi=300)
        finally:
            # Release the figure even if saving fails during a multi-trajectory evaluation.
            plt.close(figure)
            
    return {
        'time': time,
        'time_star': time_star,
        'reference': reference.copy(),
        'prediction': prediction,
        'variable_names': list(variable_names),
        'metrics': metrics,
        'model_path': Path(model_dir) / 'model.csv',
        'metrics_path': metrics_path, 'figure_path': figure_path, 'solver': solver,
    }
