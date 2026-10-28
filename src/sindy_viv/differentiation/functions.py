from __future__ import annotations

from copy import deepcopy
from numpy.typing import ArrayLike
from sindy_viv._typing import Differentiator
import numpy as np
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference, SpectralDerivative

from .modules import SavitzkyGolayDerivative

def perform_derivative(time: ArrayLike, data: ArrayLike, differentiator: Differentiator | None) -> np.ndarray:
    """Call differentiator(data, time), preserving its returned shape and precision.

    Time is an array in the desired coordinate or a scalar sample step. Data is
    N-by-channel. Bryan preparation supplies float64 tau and requires a first
    derivative callable; this generic function applies no time/state rescaling.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        data (array-like): Measured samples with one column per supplied variable name.

        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.
    """

    # Apply the differentiator to the data
    return differentiator(data, time)


def perform_second_derivative(
    time: ArrayLike,
    data: ArrayLike,
    differentiator: Differentiator,
    *,
    first_derivative: ArrayLike | None = None,
) -> np.ndarray:
    """Differentiate measurements directly twice when the method supports it.

    Native PySINDy finite differences (including smoothed finite differences)
    and spectral derivatives use a fresh instance with d=2. The project's
    SavitzkyGolayDerivative uses deriv=2 with its original window, polynomial,
    edge and precision settings. Reconstructing the method updates finite-
    difference stencil sizes without modifying the caller's method or caches.

    Unknown callables/subclasses retain repeated first differentiation. The
    same fallback applies to finite-difference accuracy order 1, whose native
    second-derivative stencil is unsupported by PySINDy, and SG polyorder < 2.
    Errors from supported native methods propagate rather than silently
    switching algorithms. Known methods must initially be first derivatives.

    Parameters:
        time: Sample times or scalar step in the desired time coordinate.
        data: Original measurements, already filtered if filtering is wanted.
        differentiator: Configured first-derivative callable; never mutated here.
        first_derivative: Optional aligned first derivative reused only by the
            fallback, avoiding a redundant call when preparation has it already.

    Returns:
        Second derivatives on the supplied time axis, without time rescaling
        or any additional filtering outside the selected method.
    """
    method_type = type(differentiator)
    second = None
    if method_type in (FiniteDifference, SmoothedFiniteDifference, SpectralDerivative):
        if differentiator.d != 1:
            raise ValueError("Second-derivative preparation requires a first-derivative method (d=1).")
        if method_type is SpectralDerivative:
            second = method_type(d=2, axis=differentiator.axis)
        elif differentiator.order >= 2:
            parameters = {
                name: getattr(differentiator, name)
                for name in ('order', 'axis', 'is_uniform', 'drop_endpoints', 'periodic')
            }
            if method_type is SmoothedFiniteDifference:
                parameters.update(
                    smoother=differentiator.smoother,
                    smoother_kws=deepcopy(differentiator.smoother_kws),
                    save_smooth=differentiator.save_smooth,
                )
            second = method_type(d=2, **parameters)
    elif method_type is SavitzkyGolayDerivative:
        if differentiator.deriv != 1:
            raise ValueError("Second-derivative preparation requires a first-derivative method (deriv=1).")
        if differentiator.polyorder >= 2:
            parameters = differentiator.get_params(deep=False)
            second = method_type(**dict(parameters, deriv=2))

    if second is not None:
        return perform_derivative(time, data, second)
    if first_derivative is None:
        first_derivative = perform_derivative(time, data, differentiator)
    if np.shape(first_derivative) != np.shape(data):
        raise ValueError("The first derivative must have the same shape as its measurements.")
    return perform_derivative(time, first_derivative, differentiator)
