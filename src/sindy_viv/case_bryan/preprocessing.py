import numpy as np
import pandas as pd

def convert_to_array(data_dict):
    """
    Convert a dictionary of data to a NumPy array.
    
    Parameters:
        data_dict (dict): Dictionary containing the data with variable names as keys
    
    Returns:
        tuple: A tuple containing the time array and the data array. Shape: (N,) and (N, num_variables) respectively.
    """
    # First, store all variable names in a list to maintain the order
    variable_names = list(data_dict.keys())
    
    # Check whether all variables has the same time array. If not, raise an error.
    for var in variable_names:
        if not np.array_equal(data_dict[variable_names[0]]['time'], data_dict[var]['time']):
            raise ValueError(f"Time arrays for variable {var} do not match the time array for {variable_names[0]}.")
    
    # Now, we can create a NumPy array to hold the data. The shape of the array will be (num_time_points, num_variables).
    num_time_points = len(data_dict[variable_names[0]])
    num_variables = len(variable_names)
    
    # Create the NumPy array
    data_array = np.zeros((num_time_points, num_variables), dtype=np.float32)
    
    # Fill the array with the data
    for i, var in enumerate(variable_names):
        data_array[:, i] = data_dict[var]['error']
    
    # Store time array as well
    time_array = np.array(data_dict[variable_names[0]]['time'].values, dtype=np.float32)
    
    return time_array, data_array

def solve_time_duplicates(time_array, data_array, strategy='uniform'):
    """
    Solve duplicates in the time array by averaging the corresponding data points.
    
    Parameters:
        time_array (np.ndarray): Array of time points
        data_array (np.ndarray): Array of data points corresponding to the time points
        
    Returns:
        tuple: A tuple containing the new time array and the new data array with duplicates resolved
    """
    if strategy == 'uniform':
        # This is the physically correct strategy
        # For Bryan's data, the sampling frequency is supposed to be constant
        # The first element of time_array is also the sampling frequency, so we can use it to determine the expected time step
        expected_time_step = time_array[1] - time_array[0]
        
        # Create a new time array and data array to hold the results
        processed_time_array = np.arange(len(time_array), dtype=np.float32) * expected_time_step + time_array[0]
        
        return processed_time_array, data_array
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for solving time duplicates. Supported strategies: 'uniform'.")
    