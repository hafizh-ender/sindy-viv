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
def load_raw_data_single_cylinder(re, mstar, vr, dtype=np.float64, variables=SINGLE_CYLINDER_STATE_VARIABLES):
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

def load_raw_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=np.float64, variables=TWO_TANDEM_CYLINDERS_STATE_VARIABLES):
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
from .preprocessing import solve_time_duplicates

def _load_processed_data(file_paths, vr, dtype, strategy):
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

def load_data_single_cylinder(re, mstar, vr, dtype=np.float64, variables=SINGLE_CYLINDER_STATE_VARIABLES, strategy='uniform'):
    """Load selected states directly into an array"""
    directory = f"{SINGLE_CYLINDER_DATA_DIR}/Re = {re}/Mstar = {mstar}/VR = {vr}"
    file_paths = {
        var: f"{directory}/{SINGLE_CYLINDER_STATE_VARIABLE_FILENAMES[var]}"
        for var in variables
    }
    return _load_processed_data(file_paths, vr, dtype, strategy)

def load_data_two_tandem_cylinders(re, mstar, vr, lpd, dtype=np.float64, variables=TWO_TANDEM_CYLINDERS_STATE_VARIABLES, strategy='uniform'):
    """Load selected states directly into an array"""
    directory = f"{TWO_TANDEM_CYLINDERS_DATA_DIR}/LpD = {lpd}/VR = {vr}"
    file_paths = {
        var: f"{directory}/{TWO_TANDEM_CYLINDERS_STATE_VARIABLE_FILENAMES[var]}"
        for var in variables
    }
    return _load_processed_data(file_paths, vr, dtype, strategy)