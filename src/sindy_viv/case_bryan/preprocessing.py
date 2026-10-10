import numpy as np

def convert_to_array(data_dict, dtype=None):
    """
    Convert a dictionary of data to a NumPy array.
    
    Parameters:
        data_dict (dict): Dictionary containing the data with variable names as keys
        dtype (numpy.dtype or None): Output data dtype. None preserves the
            common dtype of the input value columns. Time is float64.
    
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
    
    # Infer a common value dtype instead of silently downcasting float64 inputs.
    if dtype is None:
        dtype = np.result_type(*(data_dict[var]['error'].dtype for var in variable_names))
        
    # Every column is filled below, so zero-initializing the output is unnecessary.
    data_array = np.empty((num_time_points, num_variables), dtype=dtype)
    
    # Fill the array with the data
    for i, var in enumerate(variable_names):
        data_array[:, i] = data_dict[var]['error']
    
    # Keep timestamp precision independent of the chosen state dtype.
    time_array = np.array(reference_time, dtype=np.float64, copy=True)
    
    return time_array, data_array

def solve_time_duplicates(time_array, data_array, strategy='uniform'):
    """
    Reconstruct a uniform float64 time axis without dropping or averaging rows.
    
    Parameters:
        time_array (np.ndarray): Array of time points
        data_array (np.ndarray): Array of data points corresponding to the time points
        
    Returns:
        tuple: A tuple containing the new time array and the new data array with duplicates resolved
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
        
        # Build the grid in float64 and update it in place to avoid temporary arrays.
        processed_time_array = np.arange(len(time_array), dtype=np.float64)
        processed_time_array *= expected_time_step
        processed_time_array += time_array[0]
        
        return processed_time_array, data_array
    else:
        raise ValueError(f"Unknown strategy '{strategy}' for solving time duplicates. Supported strategies: 'uniform'.")