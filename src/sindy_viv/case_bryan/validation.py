"""Shared validation of the physical trajectories selected for fitting/evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real

import numpy as np

def validate_selection(values: Iterable[Real], name: str) -> tuple[float | int, ...]:
    """Return distinct positive real values, rejecting booleans, None and nonfinite values.

    NumPy scalars become native Python scalars for paths and JSON. Selection
    order is retained; an empty selection or invalid member raises ValueError.

    Parameters:
        values (Iterable[Real]): Ordered positive, finite, distinct VR or L/D values.
        name (str): Argument or parameter label used in validation messages.
    """
    name = f"{name} (VR)" if name == "vr_list" else f"{name} (L/D)" if name == "lpd_list" else name
    try:
        values = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a nonempty sequence of finite positive real values.") from None
    if not values:
        label = "VR" if "(VR)" in name else "L/D" if "(L/D)" in name else name
        raise ValueError(f"At least one {label} value is required in {name}.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        or not np.isfinite(value) or value <= 0 for value in values
    ):
        raise ValueError(f"{name} must contain finite positive real values.")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(value.item() if isinstance(value, np.generic) else value for value in values)
