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

from .preprocessing import remove_repeated_tandem_records, solve_time_duplicates

def _load_processed_data(
    file_paths: Mapping[str, str],
    vr: float,
    dtype: DTypeLike,
    strategy: str,
    *,
    tandem: bool = False,
) -> tuple[TimeArray, np.ndarray]:
    """Load ordered CSV channels into one state matrix and a float64 tau axis.

    CSV columns time/error contain t* and measurements. Channels are stably
    sorted and must share exactly the same recorded timestamps. Bryan's uniform
    reconstruction runs once. With tandem=True, all ten channels are required;
    exact consecutive reporting repeats are removed in float64 first.
    The optional, locally toggled alignment block
    advances recorded hddot by one sample and trims all channels/time to N-1
    rows before tau=t*/VR. Signal storage follows dtype;
    empty selections, unknown strategies and misaligned time axes raise ValueError.

    Parameters:
        file_paths (Mapping[str, str]): Mapping from variable names to processed CSV paths.
        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64.
        strategy (str): Duplicate-time handling strategy, as described by this function.
        tandem (bool): Detect reporting repeats using the complete tandem state matrix.
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
            dtype={'time': np.float64, 'error': np.float64 if tandem else dtype},
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

    # Detect reporting repeats before downcasting, rebuilding time, or shifting acceleration.
    # A shared mask from every tandem channel also makes subset loads consistent.
    if tandem:
        reference_time, data_array = remove_repeated_tandem_records(reference_time, data_array)
        data_array = data_array.astype(dtype, copy=False)

    # Reconstruct rounded timestamps on the retained sample grid.
    time_array, data_array = solve_time_duplicates(reference_time, data_array, strategy=strategy)

    # hddot[k+1] belongs with h[k], hdot[k], CL[k], etc. at time[k].
    # Set the flag to False, or comment out this entire alignment section,
    # to restore unshifted loading (after tandem reporting-repeat removal).
    align_recorded_acceleration = True
    if align_recorded_acceleration:
        if len(time_array) < 3:
            raise ValueError("Acceleration alignment requires at least three recorded samples.")
        aligned_data = data_array[:-1].copy()
        
        for column, variable in enumerate(file_paths):
            if variable in {r'\ddot{h}_0', r'\ddot{h}_1'}:
                aligned_data[:, column] = data_array[1:, column]
                
        data_array = aligned_data
        time_array = time_array[:-1].copy()

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
    With the shared acceleration-alignment block enabled, recorded hddot is
    shifted left one sample and every selected channel/time has N-1 rows,
    including selections that do not request hddot.
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
    precision. All ten motion/force channels are read in float64 to identify
    exact consecutive reporting repeats using their shared timestamps and values,
    even for subset selections. Repeats are removed before uniform reconstruction
    using the nominal step from the first two retained timestamps. Rounded time
    collisions with differing measurements are retained. Raw files are unchanged.
    With shared acceleration alignment enabled, both recorded accelerations
    shift left one retained sample; the final aligned output has N_clean-1 rows.
    No physical state or force scaling is performed.

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
    # Validate the selection before reading the complete repeat-detection matrix.
    selected = dict.fromkeys(variables)
    if not selected:
        raise ValueError("At least one state variable is required.")
    names = list(TWO_TANDEM_CYLINDERS_STATE_VARIABLES)
    columns = [names.index(var) for var in selected]
    file_paths = {
        var: f"{directory}/{TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES[var]}"
        for var in names
    }
    time_array, data_array = _load_processed_data(file_paths, vr, dtype, strategy, tandem=True)
    return time_array, data_array[:, columns]
