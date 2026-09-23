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

"""
Raw data
"""
def load_raw_data_single_cylinder(re, mstar, vr, dtype=np.float32, variables=SINGLE_CYLINDER_STATE_VARIABLES):
    """
    Load single cylinder VIV data for a given Reynolds number, mass ratio, and reduced velocity.
    
    Parameters:
        re (float): Reynolds number
        mstar (float): Mass ratio
        vr (float): Reduced velocity
        dtype (numpy.dtype): Data type for the loaded data
        variables (tuple): Tuple of variable names to load. Defaults to SINGLE_CYLINDER_STATE_VARIABLES.
    
    Returns:
        dict: A dictionary containing the loaded data for each variable
    """
    single_cylinder_data = {}
    
    for var in variables:
        # Get the corresponding filename for the variable
        filename = SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES[var]
        
        # Construct the file path based on the provided parameters
        file_path = f"{SINGLE_CYLINDER_DATA_DIR}/Re = {re}/Mstar = {mstar}/VR = {vr}/{filename}"
        
        # Load the data from the CSV file and convert it to a NumPy array with the specified dtype
        # Expected shape: (N, 2) where the first column is time and the second column is the variable data
        data = pd.read_csv(file_path, dtype=dtype)
        
        # Store the loaded data in the dictionary with the variable name as the key
        single_cylinder_data[var] = data
    
    return single_cylinder_data

def load_raw_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=np.float32, variables=TWO_TANDEM_CYLINDERS_STATE_VARIABLES):
    """
    Load two tandem cylinders VIV data for a given Reynolds number, mass ratio, and reduced velocity.
    
    Parameters:
        re (float): Reynolds number
        mstar (float): Mass ratio
        vr (float): Reduced velocity
        lpd (float): L/D ratio
        dtype (numpy.dtype): Data type for the loaded data
        variables (tuple): Tuple of variable names to load. Defaults to TWO_TANDEM_CYLINDERS_STATE_VARIABLES.
    
    Returns:
        dict: A dictionary containing the loaded data for each variable
    """
    two_tandem_cylinders_data = {}
    
    for var in variables:
        # Get the corresponding filename for the variable
        filename = TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES[var]
        
        # Construct the file path based on the provided parameters
        file_path = f"{TWO_TANDEM_CYLINDERS_DATA_DIR}/LpD = {lpd}/VR = {vr}/{filename}"
        
        # Load the data from the CSV file and convert it to a NumPy array with the specified dtype
        # Expected shape: (N, 2) where the first column is time and the second column is the variable data
        data = pd.read_csv(file_path, dtype=dtype)
        
        # Store the loaded data in the dictionary with the variable name as the key
        two_tandem_cylinders_data[var] = data
    
    return two_tandem_cylinders_data

"""
Preprocessed data
"""
from .preprocessing import convert_to_array, solve_time_duplicates

def load_data_single_cylinder(re, mstar, vr, dtype=np.float32, variables=SINGLE_CYLINDER_STATE_VARIABLES, strategy='uniform'):
    # Load the raw data for a single cylinder and convert it to a NumPy array, resolving any time duplicates.
    data_dict = load_raw_data_single_cylinder(re, mstar, vr, dtype=dtype, variables=variables)
    
    # Convert the dictionary of data to a NumPy array
    time_array, data_array = convert_to_array(data_dict)
    
    # Solve duplicates in the time array by averaging the corresponding data points
    processed_time_array, processed_data_array = solve_time_duplicates(time_array, data_array, strategy=strategy)
    
    # Recall that the time array recorded in Bryan's LBM program is not t*
    # It is actually Uinfinity * t / D, so we need to divide it by VR to get t*
    processed_time_array /= vr
    
    return processed_time_array, processed_data_array

def load_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=np.float32, variables=TWO_TANDEM_CYLINDERS_STATE_VARIABLES, strategy='uniform'):
    # Load the raw data for two tandem cylinders and convert it to a NumPy array, resolving any time duplicates.
    data_dict = load_raw_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=dtype, variables=variables)
    
    # Convert the dictionary of data to a NumPy array
    time_array, data_array = convert_to_array(data_dict)
    
    # Solve duplicates in the time array by averaging the corresponding data points
    processed_time_array, processed_data_array = solve_time_duplicates(time_array, data_array, strategy=strategy)
    
    # Recall that the time array recorded in Bryan's LBM program is not t*
    # It is actually Uinfinity * t / D, so we need to divide it by VR to get t*
    processed_time_array /= vr
    
    return processed_time_array, processed_data_array
    