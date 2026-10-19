from __future__ import annotations

from collections.abc import Sequence

from collections.abc import Mapping
from numpy.typing import DTypeLike
from sindy_viv._typing import TimeArray
import pandas as pd
import numpy as np

from .constants import (
    SINGLE_CYLINDER_DATA_DIR,
    SINGLE_CYLINDER_STATE_VARIABLES,
    SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES,

    TWO_TANDEM_CYLINDERS_DATA_DIR,
    TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
    TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES,
)

# Compatibility imports; historical DataFrame loaders live in compatibility.dataframe_loading.
from .compatibility.dataframe_loading import load_raw_data_single_cylinder, load_raw_data_two_tandem_cylinders

from .preprocessing import solve_time_duplicates

def _load_processed_data(
    file_paths: Mapping[str, str],
    vr: float,
    dtype: DTypeLike,
    strategy: str,
) -> tuple[TimeArray, np.ndarray]:
    """Load ordered CSV channels into one state matrix and a float64 tau axis.

    CSV columns time/error contain t* and measurements. Channels are stably
    sorted and must share exactly the same recorded timestamps. Bryan's uniform
    reconstruction runs once, then tau=t*/VR. Signal storage follows dtype;
    empty selections, unknown strategies and misaligned time axes raise ValueError.

    Parameters:
        file_paths (Mapping[str, str]): Mapping from variable names to processed CSV paths.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64.
        strategy (str): Duplicate-time handling strategy, as described by this function.
    """
    if not file_paths:
        raise ValueError("At least one state variable is required.")
    if strategy != 'uniform':
        raise ValueError(f"Unknown strategy '{strategy}'. Supported strategies: 'uniform'.")
    reference_time = None

    # Fill the data array column-wise to avoid repeated memory allocation.
    for column, (variable, file_path) in enumerate(file_paths.items()):

        # Parse timestamps independently so a float32 state dtype cannot erase time precision.
        frame = pd.read_csv(
            file_path,
            dtype={'time': np.float64, 'error': dtype},
            usecols=['time', 'error']
        )

        # Sort timestamps and values together, retaining the stored order among duplicate timestamps.
        if np.any(frame['time'].to_numpy()[1:] < frame['time'].to_numpy()[:-1]):
            frame = frame.sort_values('time', kind='stable')

        # The first variable establishes the row count and shared time reference.
        if reference_time is None:
            reference_time = frame['time'].to_numpy(copy=True)

            # Initialize the data array with the appropriate shape and dtype.
            data_array = np.empty((len(frame), len(file_paths)), dtype=frame['error'].dtype)
        elif not np.array_equal(reference_time, frame['time'].to_numpy()):
            raise ValueError(f"Time array for variable {variable} does not match the reference time array.")

        data_array[:, column] = frame['error'].to_numpy()

        # Release the DataFrame before parsing the next CSV.
        del frame

    # Handle duplicate timestamps according to the specified strategy.
    time_array, data_array = solve_time_duplicates(reference_time, data_array, strategy=strategy)

    # Recall that time_array is in t* units. For derivation purposes, we need to convert it to tau units.
    # Recall that t* = t * U / D, and tau = t * f_N = t * U / (D * V_R), so tau = t* / V_R.
    time_array /= vr

    return time_array, data_array

def load_data_single_cylinder(
    re: float,
    mstar: float,
    vr: float,
    dtype: DTypeLike = np.float64,
    variables: Sequence[str] = SINGLE_CYLINDER_STATE_VARIABLES,
    strategy: str = 'uniform',
) -> tuple[TimeArray, np.ndarray]:
    """Return float64 tau and an N-by-variable signal array for one single-cylinder dataset.

    variables determines column order. dtype controls signals only (default
    float64); timestamps are parsed independently as float64. CSV t* is uniformly
    reconstructed for Bryan's LBM data, then divided by VR exactly once.
    Missing files/channels and timestamp mismatches raise the corresponding errors.

    Parameters:
        re (float): Reynolds number used in dataset labels; None selects the cylinder-specific value.
        mstar (float): Mass ratio; workflow entry points use the cylinder-specific value when None.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
        variables (Sequence[str]): Measured variable names to load in the requested order. Default: SINGLE_CYLINDER_STATE_VARIABLES
        strategy (str): Duplicate-time handling strategy, as described by this function. Default: 'uniform'
    """
    directory = f"{SINGLE_CYLINDER_DATA_DIR}/Re = {re}/Mstar = {mstar}/VR = {vr}"
    file_paths = {
        var: f"{directory}/{SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES[var]}"
        for var in variables
    }
    return _load_processed_data(file_paths, vr, dtype, strategy)

def load_data_two_tandem_cylinders(
    re: float,
    mstar: float,
    vr: float,
    lpd: float,
    dtype: DTypeLike = np.float64,
    variables: Sequence[str] = TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
    strategy: str = 'uniform',
) -> tuple[TimeArray, np.ndarray]:
    """Return float64 tau and selected tandem signals in variables order.

    The fixed Bryan tandem dataset is selected by LpD and VR; re and mstar remain
    accepted compatibility labels. dtype controls signal storage, not timestamp
    precision. The same timestamp-alignment/reconstruction rules as the single
    loader apply; no physical state or force scaling is performed.

    Parameters:
        re (float): Reynolds number used in dataset labels; None selects the cylinder-specific value.
        mstar (float): Mass ratio; workflow entry points use the cylinder-specific value when None.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
        variables (Sequence[str]): Measured variable names to load in the requested order. Default: TWO_TANDEM_CYLINDERS_STATE_VARIABLES
        strategy (str): Duplicate-time handling strategy, as described by this function. Default: 'uniform'
    """
    directory = f"{TWO_TANDEM_CYLINDERS_DATA_DIR}/LpD = {lpd}/VR = {vr}"
    file_paths = {
        var: f"{directory}/{TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES[var]}"
        for var in variables
    }
    return _load_processed_data(file_paths, vr, dtype, strategy)
