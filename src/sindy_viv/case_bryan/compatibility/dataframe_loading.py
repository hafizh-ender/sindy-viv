"""Historical DataFrame-based CSV loading, retained for older analysis code.

New fitting/evaluation workflows use case_bryan.input.load_data_* directly.
Raw loaders return t* timestamps and honor dtype for both time and signals;
choose float64 to retain timestamp precision. Tandem Re/Mstar arguments remain
accepted for compatibility with the fixed Bryan tandem dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import DTypeLike

import numpy as np
import pandas as pd

from ..constants import (
    SINGLE_CYLINDER_DATA_DIR, SINGLE_CYLINDER_STATE_VARIABLES, SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES,
    TWO_TANDEM_CYLINDERS_DATA_DIR, TWO_TANDEM_CYLINDERS_STATE_VARIABLES, TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES,
)

def load_raw_data_single_cylinder(
    re: float,
    mstar: float,
    vr: float,
    dtype: DTypeLike = np.float64,
    variables: Sequence[str] = SINGLE_CYLINDER_STATE_VARIABLES,
) -> dict[str, pd.DataFrame]:
    """Load single cylinder VIV data for a given Reynolds number, mass ratio, and reduced velocity.

    Parameters:
        re (float): Reynolds number.
        mstar (float): Mass ratio.
        vr (float): Reduced velocity.

        dtype (numpy.dtype, type, or str): Data type for the loaded data. Default: np.float64
        variables (Sequence[str]): Measured variable names to load in order. Default: SINGLE_CYLINDER_STATE_VARIABLES

    Returns:
        dict: A dictionary containing the loaded data for each variable
    """
    single_cylinder_data = {}

    for var in variables:

        # Get the corresponding filename for the variable
        filename = SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES[var]

        # Construct the file path based on the provided parameters
        file_path = f"{SINGLE_CYLINDER_DATA_DIR}/Re = {re}/Mstar = {mstar}/VR = {vr}/{filename}"

        # Read a DataFrame; raw time remains in t* and uses the requested dtype.
        # Expected shape: (N, 2) where the first column is time and the second column is the variable data
        data = pd.read_csv(file_path, dtype=dtype)

        # Store the loaded data in the dictionary with the variable name as the key
        single_cylinder_data[var] = data

    return single_cylinder_data


def load_raw_data_two_tandem_cylinders(
    re: float,
    mstar: float,
    vr: float,
    lpd: float,
    dtype: DTypeLike = np.float64,
    variables: Sequence[str] = TWO_TANDEM_CYLINDERS_STATE_VARIABLES,
) -> dict[str, pd.DataFrame]:
    """Load two tandem cylinders VIV data for a given Reynolds number, mass ratio, and reduced velocity.

    Parameters:
        re (float): Reynolds number.
        mstar (float): Mass ratio.
        vr (float): Reduced velocity.
        lpd (float): L/D ratio.

        dtype (numpy.dtype, type, or str): Data type for the loaded data. Default: np.float64
        variables (Sequence[str]): Measured variable names to load in order. Default: TWO_TANDEM_CYLINDERS_STATE_VARIABLES

    Returns:
        dict: A dictionary containing the loaded data for each variable
    """
    two_tandem_cylinders_data = {}

    for var in variables:

        # Get the corresponding filename for the variable
        filename = TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES[var]

        # Construct the file path based on the provided parameters
        file_path = f"{TWO_TANDEM_CYLINDERS_DATA_DIR}/LpD = {lpd}/VR = {vr}/{filename}"

        # Read a DataFrame; raw time remains in t* and uses the requested dtype.
        # Expected shape: (N, 2) where the first column is time and the second column is the variable data
        data = pd.read_csv(file_path, dtype=dtype)

        # Store the loaded data in the dictionary with the variable name as the key
        two_tandem_cylinders_data[var] = data

    return two_tandem_cylinders_data
