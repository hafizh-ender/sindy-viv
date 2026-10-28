
from __future__ import annotations
import numpy as np

from .constants import TWO_TANDEM_CYLINDERS_STATE_VARIABLES

# Compatibility import for older callers; active pipelines load processed arrays directly.
from .compatibility.dataframe_conversion import convert_to_array


def remove_repeated_tandem_records(
    time_array: np.ndarray,
    data_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove consecutive reporting repeats using all ten tandem channels.

    A row is removed only if its recorded timestamp AND every measured channel
    exactly equal the preceding row. Equal timestamps alone are expected when
    six-significant-digit output becomes coarser than the physical time step;
    equal values in one channel can also be genuine rounded plateaus.

    Supply all TWO_TANDEM_CYLINDERS_STATE_VARIABLES, parsed in float64 before
    any storage downcast. Column order does not affect this comparison. Use
    the returned shared row selection before uniform time reconstruction and
    acceleration alignment. This is specific to Bryan's reporting repeats,
    not a generic method for inferring missing samples or variable time steps.
    Inputs must be finite, time-sorted and row-aligned; they are not modified.
    """
    time_array = np.asarray(time_array)
    data_array = np.asarray(data_array)
    if time_array.ndim != 1 or len(time_array) == 0:
        raise ValueError("Time must be a nonempty 1-D array.")
    if (data_array.ndim != 2
            or data_array.shape != (len(time_array), len(TWO_TANDEM_CYLINDERS_STATE_VARIABLES))):
        raise ValueError("Repeat detection requires all ten tandem channels aligned with time.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(data_array)):
        raise ValueError("Tandem repeat detection requires finite time and channel values.")
    if np.any(time_array[1:] < time_array[:-1]):
        raise ValueError("Tandem repeat detection requires nondecreasing recorded times.")

    repeated = ((time_array[1:] == time_array[:-1])
                & np.all(data_array[1:] == data_array[:-1], axis=1))
    if not np.any(repeated):
        return time_array, data_array
    keep = np.r_[True, ~repeated]
    return time_array[keep], data_array[keep]


def solve_time_duplicates(
    time_array: np.ndarray,
    data_array: np.ndarray,
    strategy: str = 'uniform',
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct Bryan's uniform recorded time axis without dropping data rows.

    This dataset-specific strategy trusts the first two timestamps to define the
    step. It reconstructs every timestamp, even if later inputs contain gaps;
    it is not a general duplicate-merging or interpolation routine. time_array
    must be a floating 1-D array with at least two entries and a positive initial
    step; data_array has the same row count. Returns a new time axis and the
    original data object. The only supported strategy is 'uniform'.

    Parameters:
        time_array (numpy.ndarray): Recorded sample times aligned with data_array.
        data_array (numpy.ndarray): Sample values aligned with time_array.
        strategy (str): Duplicate-time handling strategy, as described by this function. Default: 'uniform'
    """
    if strategy == 'uniform':

        # This is the physically correct strategy
        # For Bryan's data, the sampling frequency is supposed to be constant
        # Infer the expected step from the first two recorded timestamps.
        time_array = np.asarray(time_array)

        if time_array.ndim != 1 or len(time_array) < 2:
            raise ValueError("Time must be a 1-D array with at least two entries.")

        if len(time_array) != len(data_array):
            raise ValueError("Time and data must have the same number of rows.")

        # Promote before subtraction to retain the input sampling precision.
        expected_time_step = float(time_array[1]) - float(time_array[0])
        if not np.isfinite(expected_time_step) or expected_time_step <= 0:
            raise ValueError("The initial time step must be finite and positive.")

        # Update it in place to avoid temporary arrays.
        processed_time_array = np.arange(len(time_array), dtype=time_array.dtype)
        processed_time_array *= expected_time_step
        processed_time_array += time_array[0]

        return processed_time_array, data_array
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for solving time duplicates. Supported strategies: 'uniform'.")
