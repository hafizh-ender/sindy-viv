import numpy as np

from sindy_viv.differentiation import perform_derivative

"""
Single cylinder case
"""
# Case 1:
# State variables: h_0, \dot{h}_0, C_{L0}
# Control input: None

# Case 2:
# State variables: h_0, \dot{h}_0, \ddot{h}_0, C_{L0}, \dot{C}_{L0}
# Control input: None

def prepare_input_single_case_1(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', 'C_{L0}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in chosen_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    # Extract time
    t = time
    
    # Extract chosen state variable data
    X = full_state_variable_data[:, [full_state_variable_names.index(var) for var in chosen_state_variables]]

    # Perform derivative on chosen state variable data
    X_dot = perform_derivative(time, X, differentiator=differentiator)

    return t, X, X_dot, chosen_state_variables

def prepare_input_single_case_2(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in chosen_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")

    # Extract time
    t = time

    # Extract chosen state variable data
    X = full_state_variable_data[:, [full_state_variable_names.index(var) for var in chosen_state_variables]]
    
    # Perform derivative on chosen state variable data
    X_dot = perform_derivative(time, X, differentiator=differentiator)
    
    # Now, we have
    # \dot{h}_0, \ddot{h}_0, \dddot{h}_0, \dot{C}_{L0} in the X_dot
    # We want \dot{C}_{L0} to be the input as well, so let's extract them, perform derivative individually, and attach them to the X_dot
    derivative_C_L0_data = X_dot[:, [chosen_state_variables.index('C_{L0}')]]  # Extract \dot{C}_{L0}
    
    # Perform derivative on \dot{C}_{L0} to get \ddot{C}_{L0}
    double_derivative_C_L0_data = perform_derivative(time, derivative_C_L0_data, differentiator=differentiator)
    
    # Combine the chosen state variable data and the derivative data
    X = np.hstack((X, derivative_C_L0_data))
    X_dot = np.hstack((X_dot, double_derivative_C_L0_data))
    chosen_state_variables.append('\\dot{C}_{L0}')

    return t, X, X_dot, chosen_state_variables

def prepare_input_single_parametric_vr(
    times: dict[float, np.ndarray],                         # Dictionary of time arrays for each VR
    full_state_variables: dict[float, np.ndarray],          # Dictionary of full state variable data arrays for each VR
    full_state_variable_names: list[str],                   # List of full state variable names
    input_variables: list[float],                           # List of input variable names (VR values)
    differentiators: object | list[object],                 # Differentiator object or list of differentiator objects for each VR
    case: int = 1,                               # Individual case number (1 or 2)
):
    # Choose state variables
    required_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in required_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    if not all(vr in full_state_variables for vr in input_variables):
        raise ValueError("Some input variables are not present in the full state variables.")
    
    if not all(vr in times for vr in input_variables):
        raise ValueError("Some input variables are not present in the times dictionary.")

    if case not in [1, 2]:
        raise ValueError(f"Invalid case value. Must be 1 or 2. Got {case}.")
    
    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)  # Use the same differentiator for all VRs
    
    # Prepare data for each input variable (VR)
    # To make it PySINDy-ready, return as a list of arrays for each VR, where each array contains the time, chosen state variable data, and derivative data for that VR
    t_s = []
    X_s = []
    X_dot_s = []
    U_s = []
    
    for vr in input_variables:
        # Get time and full state variable data for the current VR
        time = times[vr]
        full_state_variable_data = full_state_variables[vr]
        differentiator = differentiators[input_variables.index(vr)]        
        
        # Extract chosen state variable data
        if case == 1:
            time, X, X_dot, chosen_state_variables = prepare_input_single_case_1(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        elif case == 2:
            time, X, X_dot, chosen_state_variables = prepare_input_single_case_2(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        
        # Prepare input variable data as a column vector
        U = np.full((X.shape[0], 1), vr, dtype=X.dtype)
        
        # Append to lists
        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
        
    # Define chosen input variables (in this case, just the VR value)
    chosen_input_variables = ['V_R']
        
    return t_s, X_s, X_dot_s, chosen_state_variables, U_s, chosen_input_variables

def prepare_input_single_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
    case: int = 1,
):
    if case == 1:
        return prepare_input_single_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator
        )
    elif case == 2:
        return prepare_input_single_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator
        )
    else:
        raise ValueError    (f"Invalid case value. Must be 1 or 2. Got {case}")
    
    
"""
Two cylinder in tandem case
"""
# Case 1:
# State variables: h_0, \dot{h}_0, C_{L0}, h_1, \dot{h}_1, C_{L1}
# Control input: None

# Case 2:
# State variables: h_0, \dot{h}_0, \ddot{h}_0, C_{L0}, \dot{C}_{L0}, h_1, \dot{h}_1, \ddot{h}_1, C_{L1}, \dot{C}_{L1}
# Control input: None

def prepare_input_two_tandem_case_1(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', 'C_{L1}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in chosen_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    # Extract time
    t = time
    
    # Extract chosen state variable data
    X = full_state_variable_data[:, [full_state_variable_names.index(var) for var in chosen_state_variables]]

    # Perform derivative on chosen state variable data
    X_dot = perform_derivative(time, X, differentiator=differentiator)

    return t, X, X_dot, chosen_state_variables

def prepare_input_two_tandem_case_2(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', '\\ddot{h}_1', 'C_{L1}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in chosen_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    # Extract time
    t = time
    
    # Extract chosen state variable data
    X = full_state_variable_data[:, [full_state_variable_names.index(var) for var in chosen_state_variables]]
    
    # Perform derivative on chosen state variable data
    X_dot = perform_derivative(time, X, differentiator=differentiator)
    
    # Now, we have \dot{h}_0, \ddot{h}_0, \dddot{h}_0, \dot{C}_{L0}, \dot{h}_1, \ddot{h}_1, \dddot{h}_1, \dot{C}_{L1} in the X_dot
    # We want \dot{C}_{L0} and \dot{C}_{L1} to be the inputs as well, so let's extract them, perform derivative individually, and attach them to the X_dot
    derivative_C_L0_data = X_dot[:, [chosen_state_variables.index('C_{L0}')]]  # Extract \dot{C}_{L0}
    derivative_C_L1_data = X_dot[:, [chosen_state_variables.index('C_{L1}')]]  # Extract \dot{C}_{L1}
    
    # Perform derivative on \dot{C}_{L0} and \dot{C}_{L1} to get \ddot{C}_{L0} and \ddot{C}_{L1}
    double_derivative_C_L0_data = perform_derivative(time, derivative_C_L0_data, differentiator=differentiator)
    double_derivative_C_L1_data = perform_derivative(time, derivative_C_L1_data, differentiator=differentiator)
    
    # Combine the chosen state variable data and the derivative data
    X = np.hstack((X, derivative_C_L0_data, derivative_C_L1_data))
    X_dot = np.hstack((X_dot, double_derivative_C_L0_data, double_derivative_C_L1_data))
    chosen_state_variables.append('\\dot{C}_{L0}')
    chosen_state_variables.append('\\dot{C}_{L1}')
    
    return t, X, X_dot, chosen_state_variables

def prepare_input_two_tandem_parametric_vr(
    times: dict[float, np.ndarray],                         # Dictionary of time arrays for each VR on one LpD
    full_state_variables: dict[float, np.ndarray],          # Dictionary of full state variable data arrays for each VR on one LpD
    full_state_variable_names: list[str],                   # List of full state variable names
    input_variables: list[float],                           # List of input variable names (VR values)
    differentiators: object | list[object],                 # Differentiator object or list of differentiator objects for each VR
    case: int = 1,                               # Individual case number (1 or 2)
):
    # Choose state variables
    required_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', '\\ddot{h}_1', 'C_{L1}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in required_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    if not all(vr in full_state_variables for vr in input_variables):
        raise ValueError("Some input variables are not present in the full state variables.")
    
    if not all(vr in times for vr in input_variables):
        raise ValueError("Some input variables are not present in the times dictionary.")

    if case not in [1, 2]:
        raise ValueError("Invalid case value. Must be 1 or 2.")
    
    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)  # Use the same differentiator for all VRs
    
    # Prepare data for each input variable (VR)
    t_s = []
    X_s = []
    X_dot_s = []
    U_s = []
    
    for vr in input_variables:
        # Get time and full state variable data for the current VR
        time = times[vr]
        full_state_variable_data = full_state_variables[vr]
        differentiator = differentiators[input_variables.index(vr)]        
        
        # Extract chosen state variable data
        if case == 1:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_1(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        elif case == 2:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_2(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        
        # Prepare input variable data as a column vector
        U = np.full((X.shape[0], 1), vr, dtype=X.dtype)
        
        # Append to lists
        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
        
    # Define chosen input variables (in this case, just the VR value)
    chosen_input_variables = ['V_R']
        
    return t_s, X_s, X_dot_s, chosen_state_variables, U_s, chosen_input_variables

def prepare_input_two_tandem_parametric_vr_lpd(
    times: dict[float, dict[float, np.ndarray]],                         # Dictionary of time arrays for each VR and each LpD
    full_state_variables: dict[float, dict[float, np.ndarray]],          # Dictionary of full state variable data arrays for each VR and each LpD
    full_state_variable_names: list[str],                                # List of full state variable names
    input_variables: list[tuple[float, float]],                          # List of input variable names (VR values) and LpD values
    differentiators: object | list[object],                              # Differentiator object or list of differentiator objects for each VR and each LpD
    case: int = 1,                                            # Individual case number (1 or 2)
):
    # Choose state variables
    required_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', '\\ddot{h}_1', 'C_{L1}']
    
    # Check input validity
    if not all(var in full_state_variable_names for var in required_state_variables):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    if not all(lpd in full_state_variables and vr in full_state_variables[lpd] for lpd, vr in input_variables):
        raise ValueError("Some input variables are not present in the full state variables.")
    
    if not all(lpd in times and vr in times[lpd] for lpd, vr in input_variables):
        raise ValueError("Some input variables are not present in the times dictionary.")

    if case not in [1, 2]:
        raise ValueError("Invalid case value. Must be 1 or 2.")
    
    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)  # Use the same differentiator for all VRs and LpDs
    
    # Prepare data for each input variable (VR and LpD)
    t_s = []
    X_s = []
    X_dot_s = []
    U_s = []
    
    for (lpd, vr) in input_variables:
        # Get time and full state variable data for the current VR and LpD
        time = times[lpd][vr]
        full_state_variable_data = full_state_variables[lpd][vr]
        differentiator = differentiators[input_variables.index((lpd, vr))]        
        
        # Extract chosen state variable data
        if case == 1:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_1(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        elif case == 2:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_2(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        
        # Prepare input variable data as a column vector
        U = np.concat([np.full((X.shape[0], 1), lpd, dtype=X.dtype), np.full((X.shape[0], 1), vr, dtype=X.dtype)], axis=-1)
        
        # Append to lists
        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
        
    # Define chosen input variables (in this case, VR and LpD values)
    chosen_input_variables = ['L/D', 'V_R']
    
    return t_s, X_s, X_dot_s, chosen_state_variables, U_s, chosen_input_variables

def prepare_input_two_tandem_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
    case: int = 1,
):
    if case == 1:
        return prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator
        )
    elif case == 2:
        return prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator
        )
    else:
        raise ValueError(f"Invalid case value. Must be 1 or 2. Got {case}.")