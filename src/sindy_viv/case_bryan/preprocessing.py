
from __future__ import annotations
import numpy as np

# Compatibility import for older callers; active pipelines load processed arrays directly.
from .compatibility.dataframe_conversion import convert_to_array

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
