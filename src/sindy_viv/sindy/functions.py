from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import ArrayLike
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.optimizers import BaseOptimizer
from sindy_viv._typing import Differentiator
from typing import Any
import numpy as np
import pysindy as ps
from pysindy.utils import AxesArray, comprehend_axes

from sindy_viv.feature_library.weak import (
    build_weak_system, resolve_weak_library_config, validate_weak_settings, weak_half_width,
)

# Generic fitting accepts already prepared libraries, optimizers and optional derivatives.

def perform_sindy(
    time: ArrayLike,
    state_variable_data: np.ndarray | list[np.ndarray],
    feature_library: BaseFeatureLibrary,
    optimizer: BaseOptimizer,
    state_variable_differentiated_data: np.ndarray | list[np.ndarray] | None = None,
    differentiation_method: Differentiator | None = None,
    feature_names: Sequence[str] | None = None,
    input_variable_data: np.ndarray | list[np.ndarray] | None = None,
) -> ps.SINDy:
    """Perform SINDy on the given time-series data.

    Parameters:
        time (array-like): Time points or a positive scalar sample step in the same coordinate as x_dot. Shape: (N,) or a list of trajectory axes.
        state_variable_data (numpy.ndarray or list[numpy.ndarray]): State variable data. Shape: (N, num_state_variables) or list of arrays for multiple trajectories.

        feature_library (BaseFeatureLibrary): Feature library object.
        optimizer (BaseOptimizer): Optimizer object.

        state_variable_differentiated_data (numpy.ndarray, list[numpy.ndarray], or None): Pre-computed derivative of the state variables. Shape: (N, num_state_variables) or list of arrays for multiple trajectories. Default: None

        differentiation_method (Differentiator or None): Differentiation method object. Default: None

        feature_names (Sequence[str] or None): Raw state names followed by control/parameter names; excludes expanded library terms. Default: None
        input_variable_data (numpy.ndarray, list[numpy.ndarray], or None): Input variable data. Shape: (N, num_input_variables) or list of arrays for multiple trajectories. Default: None

    Returns:
        pysindy.SINDy: Fitted SINDy model.

    Precomputed derivatives take precedence over internal differentiation.
    The optimizer's training design matrix Theta_ is released after fitting.
    model.predict(X, u=U) evaluates derivatives at new state/control samples
    using the fitted library and coefficients; it does not reuse training X/U.
    """
    if differentiation_method is None and state_variable_differentiated_data is None:
        raise ValueError("Either 'differentiation_method' or 'state_variable_differentiated_data' must be provided.")

    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
        differentiation_method=differentiation_method,
    )

    # Explicit x_dot skips internal differentiation without coupling time and state dtypes.
    model.fit(
        x=state_variable_data,
        t=time,
        x_dot=state_variable_differentiated_data,
        feature_names=feature_names,
        u=input_variable_data
    )

    # Prediction uses new X/U and fitted coefficients; discard the retained training design matrix.
    if hasattr(model.optimizer, 'Theta_'):
        del model.optimizer.Theta_

    return model

def perform_weak_sindy(
    time: ArrayLike,
    state_variable_data: np.ndarray | list[np.ndarray],
    feature_library: BaseFeatureLibrary,
    optimizer: BaseOptimizer,
    feature_names: Sequence[str] | None = None,
    input_variable_data: np.ndarray | list[np.ndarray] | None = None,
    parameter_library: BaseFeatureLibrary | None = None,
    *,
    K: int | None = None,
    H_xt: float | None = None,
    p: int | None = None,
    random_state: int | None = 0,
    weak_library_config: dict[str, Any] | None = None,
) -> ps.SINDy:
    """Fit temporal weak equations and return a pointwise PySINDy model.

    time is a sample step, one shared time axis, or a list of trajectory axes.
    State and optional input arrays have matching rows and consistent columns
    across trajectories; grids and row counts may differ. feature_library is
    an ordinary pointwise library over states followed by controls/parameters.
    If parameter_library is supplied separately, all input columns are treated
    as parameters and tensored with the state library before integration.

    weak_library_config describes PySINDy's WeakPDELibrary integration settings;
    the pointwise feature_library supplies its data basis. Explicit K/H_xt/p
    override this descriptor. Defaults are K=100, H_xt=None and p=4.
    K windows are sampled per trajectory with a local random_state generator.
    H_xt is their half width in the supplied time coordinate (default span/20
    per trajectory); p>=1 sets phi=(1-z**2)**p. Integration uses piecewise
    linear samples and PySINDy's integration by parts, without state differentiation.
    Windows never cross trajectory boundaries. Both libraries and optimizer
    are fitted/mutated. The returned model retains the pointwise library for
    prediction, simulation, feature names and coefficient export, and stores
    weak_settings_ for reproducibility. Training design arrays are released.
    """
    weak_library_config = resolve_weak_library_config(weak_library_config, K, H_xt, p)
    settings = weak_library_config['kwargs']
    K, H_xt, p = settings['K'], settings['H_xt'], settings['p']
    validate_weak_settings(K, H_xt, p, random_state)
    states = state_variable_data if isinstance(state_variable_data, (list, tuple)) else [state_variable_data]
    if not len(states):
        raise ValueError("state_variable_data must contain at least one trajectory.")
    states = [np.asarray(x) for x in states]
    for x in states:
        if x.ndim != 2 or x.shape[1] == 0 or x.dtype.kind not in 'biuf' or not np.all(np.isfinite(x)):
            raise ValueError("States must be finite real 2-D arrays with at least one column.")
    n_states = states[0].shape[1]
    if any(x.shape[1] != n_states for x in states):
        raise ValueError("All trajectories must have the same number of state columns.")
    # A list of arrays supplies separate grids; a list of numbers is one shared axis.
    if isinstance(time, (list, tuple)) and len(time) and np.ndim(time[0]) > 0:
        times = time
    else:
        times = [time] * len(states)
    if len(times) != len(states):
        raise ValueError("Time and state trajectory counts must match.")
    axes, widths = [], []
    for t, x in zip(times, states):
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 0:
            if not np.isfinite(t) or t <= 0:
                raise ValueError("The sample step must be finite and positive.")
            t = np.arange(len(x), dtype=np.float64) * t
        if len(t) != len(x):
            raise ValueError("Time and state sample counts must match.")
        widths.append(weak_half_width(t, H_xt))
        axes.append(t)
    if input_variable_data is None:
        inputs = [None] * len(states)
        n_inputs = 0
    else:
        inputs = input_variable_data if isinstance(input_variable_data, (list, tuple)) else [input_variable_data]
        if len(inputs) != len(states):
            raise ValueError("Input and state trajectory counts must match.")
        inputs = [np.asarray(u) for u in inputs]
        n_inputs = inputs[0].shape[1] if inputs[0].ndim == 2 else 0
        for x, u in zip(states, inputs):
            if (u.ndim != 2 or not n_inputs or u.shape != (len(x), n_inputs)
                    or u.dtype.kind not in 'biuf' or not np.all(np.isfinite(u))):
                raise ValueError("Inputs must be finite real 2-D arrays aligned with states and consistent columns.")
    if feature_names is not None:
        feature_names = list(feature_names)
        if (len(feature_names) != n_states + n_inputs or len(set(feature_names)) != len(feature_names)
                or any(not isinstance(name, str) or not name for name in feature_names)):
            raise ValueError("feature_names must contain one unique nonempty name per state/input column.")
    if parameter_library is not None:
        if not n_inputs:
            raise ValueError("parameter_library requires input_variable_data.")
        feature_library = ps.ParameterizedLibrary(
            feature_library=feature_library, parameter_library=parameter_library,
            num_features=n_states, num_parameters=n_inputs,
        )
    samples = []
    for x, u in zip(states, inputs):
        values = x if u is None else np.concatenate((x, u), axis=1)
        samples.append(AxesArray(values, comprehend_axes(values)))
    feature_library.fit(samples)
    rng = np.random.default_rng(random_state)
    systems = []
    for t, x, values in zip(axes, states, samples):
        features = np.asarray(feature_library.transform(values), dtype=np.float64)
        if (features.ndim != 2 or len(features) != len(x) or not features.shape[1]
                or not np.all(np.isfinite(features))):
            raise ValueError("Weak fitting requires finite pointwise library features aligned with states.")
        systems.append(build_weak_system(t, x, features, K=K, H_xt=H_xt, p=p, rng=rng,
                                         weak_library_config=weak_library_config))
    design, targets = (np.concatenate(parts, axis=0) for parts in zip(*systems))
    optimizer.fit(design, targets)
    model = ps.SINDy(feature_library=feature_library, optimizer=optimizer)
    model.n_control_features_ = n_inputs
    model.feature_names = feature_names
    # The same shape finalization used by SINDy.fit, after fitting weak arrays directly.
    model._fit_shape()
    model.weak_settings_ = dict(K=int(K), H_xt=None if H_xt is None else float(H_xt), p=int(p),
                                random_state=None if random_state is None else int(random_state),
                                trajectory_half_widths=widths)
    if hasattr(optimizer, 'Theta_'):
        del optimizer.Theta_
    return model
