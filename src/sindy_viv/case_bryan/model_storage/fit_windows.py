"""Trajectory-owned fitting-window validation and cropping."""

from __future__ import annotations

from collections.abc import Sequence
from numpy.typing import ArrayLike
from os import PathLike
from sindy_viv._typing import CylinderType, TimeArray
from typing import Any

import json
from numbers import Real
from pathlib import Path

import numpy as np

def _parameter_key(value: Any, name: str) -> float:
    """Interpret JSON object keys numerically, including equivalent forms such as 2 and 2.0.

    Parameters:
        value (object): JSON object key representing a positive finite VR or L/D value.
        name (str): Argument or parameter label used in validation messages.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Fit-window {name} keys must be finite positive numbers.") from None

    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"Fit-window {name} keys must be finite positive numbers.")

    return number

def _index_fit_windows(
    config: dict[str, Any],
    cylinder_type: CylinderType,
) -> dict[tuple[float | None, float], dict[str, Any]]:
    """Validate nested trajectory dictionaries and index them by numeric (LpD, VR) pairs.

    Parameters:
        config (dict[str, Any]): Nested trajectory-window configuration dictionary.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
    """

    if not isinstance(config, dict) or not config:
        raise ValueError("A fit-window config must be a nonempty nested JSON dictionary.")

    # The outer dictionary is keyed by L/D for tandem cylinders, or a single entry for one cylinder.
    # Thus, for a single cylinder, substitute a single-item list of (None, config) to unify the iteration below.
    if cylinder_type == 'single':
        overall_time_windows = [(None, config)]
    elif cylinder_type == 'two_tandem':
        overall_time_windows = config.items()
    else:
        raise ValueError("Fit-window cylinder_type must be 'single' or 'two_tandem'.")

    records, seen_overall_time_windows = {}, set()
    for lpd_key, vr_time_windows in overall_time_windows:
        lpd = None if cylinder_type == 'single' else _parameter_key(lpd_key, 'LpD')

        if lpd in seen_overall_time_windows:
            raise ValueError(f"Duplicate numeric LpD key in fit-window config: {lpd}.")
        seen_overall_time_windows.add(lpd)

        if not isinstance(vr_time_windows, dict) or not vr_time_windows:
            raise ValueError(f"Fit-window LpD={lpd} must contain a nonempty VR dictionary.")

        for vr_key, bounds in vr_time_windows.items():
            vr = _parameter_key(vr_key, 'VR')

            if (lpd, vr) in records:
                raise ValueError(f"Duplicate numeric VR key for LpD={lpd}, VR={vr}.")

            if not isinstance(bounds, dict) or set(bounds) != {'tstar0', 'tstarf'}:
                raise ValueError("Every trajectory window must contain exactly tstar0 and tstarf.")

            for name, value in bounds.items():
                if value is None or value == 'TO_BE_FILLED':
                    continue

                if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                    raise ValueError(f"Fit-window {name} must be a finite number, null, or 'TO_BE_FILLED'.")

            records[(lpd, vr)] = bounds

    return records

def load_fit_window_config(fit_window_config_path: str | PathLike[str], cylinder_type: CylinderType) -> dict[str, Any]:
    """Read and validate trajectory-owned JSON fitting windows without modifying them.

    Single configs map VR to tstar0/tstarf; tandem configs map LpD then VR.
    Numeric keys are normalized for matching. null leaves an endpoint open and
    TO_BE_FILLED is allowed until that trajectory is selected. Returns the raw
    configuration dictionary; malformed structures raise ValueError.

    Parameters:
        fit_window_config_path (str or PathLike[str]): Path to the nested trajectory-window JSON file.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
    """
    config = json.loads(Path(fit_window_config_path).read_text(encoding='utf-8'))
    _index_fit_windows(config, cylinder_type)
    return config

def prepare_fit_window_bounds(
    config: dict[str, Any],
    trajectory_pairs: Sequence[tuple[float | None, float]],
    cylinder_type: CylinderType,
) -> dict[tuple[float | None, float], tuple[float | None, float | None]]:
    """Resolve requested (LpD, VR) pairs to t* endpoint tuples.

    Selection order and dynamics case do not affect trajectory-owned windows.
    None leaves an endpoint open. Missing trajectories, selected placeholders
    and nonincreasing finite endpoint pairs raise ValueError.

    Parameters:
        config (dict[str, Any]): Nested trajectory-window configuration dictionary.
        trajectory_pairs (Sequence[tuple[float | None, float]]): Requested (LpD, VR) pairs; single-cylinder pairs use LpD=None.
        cylinder_type (str): Cylinder configuration: single or two_tandem.
    """
    records = _index_fit_windows(config, cylinder_type)
    bounds = {}

    for lpd, vr in trajectory_pairs:

        # Check that the requested trajectory pair exists in the fit-window config.
        if (lpd, vr) not in records:
            raise ValueError(f"Fit-window config has no window for LpD={lpd}, VR={vr}.")
        record = records[(lpd, vr)]
        values = (record['tstar0'], record['tstarf'])
        if 'TO_BE_FILLED' in values:
            raise ValueError(f"Fill the fit-window bounds for LpD={lpd}, VR={vr} with finite t* numbers or null endpoints.")

        start, stop = (None if value is None else float(value) for value in values)
        if start is not None and stop is not None and start >= stop:
            raise ValueError(f"Fit-window tstar0 must be smaller than tstarf for LpD={lpd}, VR={vr}.")

        bounds[(lpd, vr)] = (start, stop)

    return bounds

def crop_training_trajectory(
    time: ArrayLike,
    states: np.ndarray,
    derivatives: np.ndarray | None,
    inputs: np.ndarray | None,
    lpd: float,
    vr: float,
    bounds: tuple[float | None, float | None],
) -> tuple[TimeArray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Crop aligned prepared arrays inclusively after full-signal differentiation.

    Input time is float64 tau; bounds are requested t* endpoints with None for
    open ends. Returns (time, states, derivatives, inputs, metadata). Explicit
    windows return owning copies so full buffers can be released; fully open
    windows reuse arrays. metadata records requested/actual windows and counts.
    Invalid time axes or windows with fewer than two samples raise ValueError.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        states (numpy.ndarray): State samples with one column per state name.
        derivatives (numpy.ndarray or None): Prepared targets, omitted for weak/integral workflows.
        inputs (numpy.ndarray or None): Aligned measured controls and constant parameter columns, or None.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        bounds (tuple[float | None, float | None]): Inclusive fitting endpoints (tstar0, tstarf) in recorded t*, with None for open endpoints.
    """
    time = np.asarray(time, dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
        raise ValueError("Fitting needs at least two finite, strictly increasing time samples.")

    time_star = time * vr
    start, stop = bounds
    first = time_star[0] if start is None else start
    last = time_star[-1] if stop is None else stop

    if first < time_star[0] or last > time_star[-1] or first >= last:
        raise ValueError(f"Fitting t* region [{first}, {last}] is outside or invalid for L/D={lpd}, VR={vr}: [{time_star[0]}, {time_star[-1]}].")

    region = slice(np.searchsorted(time_star, first, side='left'), np.searchsorted(time_star, last, side='right'))
    if len(time[region]) < 2:
        raise ValueError("Each fitting t* region must contain at least two data samples.")

    metadata = {
        'VR': float(vr), 'LpD': None if lpd is None else float(lpd),
        'requested_tstar0': start, 'requested_tstarf': stop,
        'tstar_region': [float(time_star[region][0]), float(time_star[region][-1])],
        'tau_region': [float(time[region][0]), float(time[region][-1])],
        'samples': len(time[region]), 'total_samples': len(time),
    }

    if start is None and stop is None:
        return time, states, derivatives, inputs, metadata

    # Copies let the full prepared buffers be released before constructing regression features.
    return (
        time[region].copy(),
        states[region].copy(),
        None if derivatives is None else derivatives[region].copy(),
        None if inputs is None else inputs[region].copy(),
        metadata
    )
