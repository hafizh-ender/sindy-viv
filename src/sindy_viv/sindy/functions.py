from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import ArrayLike
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.optimizers import BaseOptimizer
from sindy_viv._typing import Differentiator
from typing import NoReturn
import numpy as np
import pysindy as ps

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
) -> NoReturn:
    """Reserve the generic weak-fitting interface without performing any work.

    Always raises NotImplementedError; all inputs are intentionally retained
    for the future implementation and are neither validated nor mutated.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        state_variable_data (numpy.ndarray or list[numpy.ndarray]): State sample matrix, or a list of matrices for multiple trajectories.

        feature_library (BaseFeatureLibrary): Feature-library instance used to construct the regression basis.
        optimizer (BaseOptimizer): Optimizer instance used to fit the feature coefficients.

        feature_names (Sequence[str] or None): State names followed by control and parameter names, before feature expansion. Default: None
        input_variable_data (numpy.ndarray, list[numpy.ndarray], or None): Control and parameter samples aligned with each state trajectory, or None. Default: None
        parameter_library (BaseFeatureLibrary or None): Reserved parameter basis for future weak fitting. Default: None
    """
    raise NotImplementedError("WeakSINDy is not implemented.")
