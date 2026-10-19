"""Convert historical channel-to-DataFrame dictionaries into aligned arrays."""

from __future__ import annotations

from collections.abc import Mapping
from numpy.typing import DTypeLike
import numpy as np
import pandas as pd


def convert_to_array(
    data_dict: Mapping[str, pd.DataFrame],
    dtype: DTypeLike = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a dictionary of data to a NumPy array.

    Parameters:
        data_dict (Mapping[str, pd.DataFrame]): Dictionary containing the data with variable names as keys.
        dtype (numpy.dtype, type, or str): Output data dtype. Default: np.float64

    Returns:
        tuple: A tuple containing the time array and the data array. Shape: (N,) and (N, num_variables) respectively.
    """

    # First, store all variable names in a list to maintain the order
    variable_names = list(data_dict.keys())
    if not variable_names:
        raise ValueError("At least one state variable is required.")

    reference_time = data_dict[variable_names[0]]['time'].to_numpy()

    # Check whether all variables has the same time array. If not, raise an error.
    for var in variable_names[1:]:
        if not np.array_equal(reference_time, data_dict[var]['time'].to_numpy()):
            raise ValueError(f"Time arrays for variable {var} do not match the time array for {variable_names[0]}.")

    # Now, we can create a NumPy array to hold the data. The shape of the array will be (num_time_points, num_variables).
    num_time_points = len(data_dict[variable_names[0]])
    num_variables = len(variable_names)

    # Infer a common value dtype if dtype is None, otherwise use the provided dtype.
    if dtype is None:
        dtype = np.result_type(*(data_dict[var]['error'].dtype for var in variable_names))

    # Initialize the data array with the appropriate shape and dtype.
    data_array = np.empty((num_time_points, num_variables), dtype=dtype)

    # Fill the array with the data
    for i, var in enumerate(variable_names):
        data_array[:, i] = data_dict[var]['error']

    # Keep timestamp precision independent of the chosen state dtype.
    time_array = np.array(reference_time, dtype=reference_time.dtype, copy=True)

    return time_array, data_array
