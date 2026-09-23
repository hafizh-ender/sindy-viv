import importlib

import pysindy as ps

from sindy_viv.utils import kwargs_to_object, initiate_object

# Data-Information-Agnostic SINDy
# Basically does three things: prepare feature library, prepare optimizer, and perform SINDy
# Differentiation is perfor

# Input:
# - Data: t, X (state variables), U (control input), state variable names, control input names, X_dot (derivative of state variables)
# - Settings:
#   > feature library: object of type pysindy.feature_library.FeatureLibrary and keyword arguments for the feature library
#   > optimizer: object of type pysindy.optimizers.Optimizer and keyword arguments for the optimizer,
#                except for ps.ConstrainedSR3 which requires additional keyword arguments for the constraints based feature library
# Output: SINDy model

# Process:
# 1. Load all information (t, X, U, state variable names, control input names, X_dot)
# 2. Get feature library and optimizer based on the provided settings
# 3. Perform SINDy using the provided data and the prepared feature library and optimizer

def perform_sindy(
    time, 
    state_variable_data,
    feature_library, 
    optimizer,
    state_variable_differentiated_data=None,
    differentiation_method=None,
    feature_names=None,
    input_variable_data=None,
):
    """
    Perform SINDy on the given time-series data.

    Parameters:
        time (numpy.ndarray or list[numpy.ndarray]): Time points corresponding to the state variable data. Shape: (N,) or list of arrays for multiple trajectories.
        state_variable_data (numpy.ndarray or list[numpy.ndarray]): State variable data. Shape: (N, num_state_variables) or list of arrays for multiple trajectories.
        feature_library (pysindy.feature_library.FeatureLibrary): Feature library object.
        optimizer (pysindy.optimizers.Optimizer): Optimizer object.
        state_variable_differentiated_data (numpy.ndarray or list[numpy.ndarray], optional): Pre-computed derivative of the state variables. Shape: (N, num_state_variables) or list of arrays for multiple trajectories.
        differentiation_method (pysindy.differentiation.DifferentiationMethod, optional): Differentiation method object.
        feature_names (list of str, optional): Names of the features.
        input_variable_data (numpy.ndarray or list[numpy.ndarray], optional): Input variable data. Shape: (N, num_input_variables) or list of arrays for multiple trajectories.

    Returns:
        pysindy.SINDy: Fitted SINDy model.
    """
    if differentiation_method is None and state_variable_differentiated_data is None:
        raise ValueError("Either 'differentiation_method' or 'state_variable_differentiated_data' must be provided.")
    
    model = ps.SINDy(
        feature_library=feature_library,
        optimizer=optimizer,
        differentiation_method=differentiation_method,
    )
    
    model.fit(
        x=state_variable_data, 
        t=time, 
        x_dot=state_variable_differentiated_data, 
        feature_names=feature_names, 
        u=input_variable_data
    )
    
    return model