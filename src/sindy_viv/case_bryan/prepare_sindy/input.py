import numpy as np

from sindy_viv.differentiation import perform_derivative

def _prepare_input(time, data, names, differentiator, chosen, lift_variables=()):
    """Allocate final state arrays once; retain float64 time precision.

    The returned time can share storage with a float64 input and should be
    treated as read-only. State and derivative arrays own their storage.
    """
    if not all(var in names for var in chosen):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    # Preserve small time increments even when the state arrays use float32.
    t = np.asarray(time, dtype=np.float64)
    if t.ndim != 1 or data.ndim != 2 or len(t) != len(data):
        raise ValueError("Time must be 1-D and data 2-D with the same number of rows.")
    base_columns = len(chosen)
    
    # Allocate base and derived state columns together to avoid stacking copies.
    X = np.empty((len(t), base_columns + len(lift_variables)), dtype=data.dtype)
    for column, variable in enumerate(chosen):
        X[:, column] = data[:, names.index(variable)]
        
    # Let each differentiator validate and interpret the complete time array.
    first_derivative = perform_derivative(t, X[:, :base_columns], differentiator)
    
    # Case 1 needs no extra lift-derivative state columns.
    if not lift_variables:
        return t, X, first_derivative.astype(X.dtype, copy=False), list(chosen)
    
    # Assign second derivatives into the state dtype to prevent float64 promotion.
    X_dot = np.empty_like(X)
    X_dot[:, :base_columns] = first_derivative
    del first_derivative
    for column, variable in enumerate(lift_variables, start=base_columns):
        X[:, column] = X_dot[:, chosen.index(variable)]
        
    # Filter lift channels together and assign into the final output dtype.
    X_dot[:, base_columns:] = perform_derivative(
        t, X[:, base_columns:], differentiator
    )
    
    chosen = list(chosen) + [f"\\dot{{{var.split('_')[0]}}}_{var.split('_')[1]}" for var in lift_variables]
    
    return t, X, X_dot, chosen

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
    
    return _prepare_input(time, full_state_variable_data, full_state_variable_names, differentiator, chosen_state_variables)

def prepare_input_single_case_2(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}']
    
    return _prepare_input(time, full_state_variable_data, full_state_variable_names, differentiator, chosen_state_variables, ['C_{L0}'])

def prepare_input_single_parametric_vr(
    times: dict[float, np.ndarray],                         # Dictionary of time arrays for each VR
    full_state_variables: dict[float, np.ndarray],          # Dictionary of full state variable data arrays for each VR
    full_state_variable_names: list[str],                   # List of full state variable names
    input_variables: list[float],                           # List of input variable names (VR values)
    differentiators: object | list[object],                 # Differentiator object or list of differentiator objects for each VR
    case: int = 1,                               # Individual case number (1 or 2)
):
    # Choose state variables
    required_state_variables = ['h_0', '\\dot{h}_0', 'C_{L0}']
    # Only case 2 requires measured acceleration, allowing case 1 to load less data.
    if case == 2:
        required_state_variables.append('\\ddot{h}_0')
    
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
    
    if not input_variables:
        raise ValueError("At least one input variable is required.")
    # Match differentiators by position, including repeated parameter values.
    for vr, differentiator in zip(input_variables, differentiators):
        # Get time and full state variable data for the current VR
        time = times[vr]
        full_state_variable_data = full_state_variables[vr]
        
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
    
    return _prepare_input(time, full_state_variable_data, full_state_variable_names, differentiator, chosen_state_variables)

def prepare_input_two_tandem_case_2(
    time: np.ndarray, 
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    # Choose state variables
    chosen_state_variables = ['h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', '\\ddot{h}_1', 'C_{L1}']
    
    return _prepare_input(time, full_state_variable_data, full_state_variable_names, differentiator, chosen_state_variables, ['C_{L0}', 'C_{L1}'])

def prepare_input_two_tandem_parametric_vr(
    times: dict[float, np.ndarray],                         # Dictionary of time arrays for each VR on one LpD
    full_state_variables: dict[float, np.ndarray],          # Dictionary of full state variable data arrays for each VR on one LpD
    full_state_variable_names: list[str],                   # List of full state variable names
    input_variables: list[float],                           # List of input variable names (VR values)
    differentiators: object | list[object],                 # Differentiator object or list of differentiator objects for each VR
    case: int = 1,                               # Individual case number (1 or 2)
):
    # Choose state variables
    required_state_variables = ['h_0', '\\dot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', 'C_{L1}']
    
    # Only case 2 requires measured acceleration, allowing case 1 to load less data.
    if case == 2:
        required_state_variables.extend(['\\ddot{h}_0', '\\ddot{h}_1'])
    
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
    
    if not input_variables:
        raise ValueError("At least one input variable is required.")
    
    # Match differentiators by position, including repeated parameter values.
    for vr, differentiator in zip(input_variables, differentiators):
        # Get time and full state variable data for the current VR
        time = times[vr]
        full_state_variable_data = full_state_variables[vr]
        
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
    required_state_variables = ['h_0', '\\dot{h}_0', 'C_{L0}', 'h_1', '\\dot{h}_1', 'C_{L1}']
    # Only case 2 requires measured acceleration, allowing case 1 to load less data.
    if case == 2:
        required_state_variables.extend(['\\ddot{h}_0', '\\ddot{h}_1'])
    
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
    
    if not input_variables:
        raise ValueError("At least one input variable is required.")
    
    # Match differentiators by position, including repeated parameter pairs.
    for (lpd, vr), differentiator in zip(input_variables, differentiators):
        # Get time and full state variable data for the current VR and LpD
        time = times[lpd][vr]
        full_state_variable_data = full_state_variables[lpd][vr]
        
        # Extract chosen state variable data
        if case == 1:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_1(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        elif case == 2:
            time, X, X_dot, chosen_state_variables = prepare_input_two_tandem_case_2(
                time, full_state_variable_data, full_state_variable_names, differentiator
            )
        
        # Fill both parameter columns directly instead of concatenating temporary arrays.
        U = np.empty((X.shape[0], 2), dtype=X.dtype)
        U[:, 0] = lpd
        U[:, 1] = vr
        
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

def iter_prepared_trajectories(
    trajectories, full_state_variable_names, differentiator, case=1, *, tandem=False
):
    """Yield prepared (t, X, X_dot, names) tuples without collecting raw data.

    ``trajectories`` yields (time, data) pairs, for example from a generator
    loading one case at a time. Consume results incrementally to bound memory;
    collecting them into a list retains all prepared arrays. This does not
    make PySINDy's batch model fitting a streaming operation.
    """
    prepare = prepare_input_two_tandem_individual if tandem else prepare_input_single_individual
    
    for time, data in trajectories:
        yield prepare(time, data, full_state_variable_names, differentiator, case=case)
