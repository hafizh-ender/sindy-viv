"""Prepare SINDy states and optional constant parameter columns.

Individual functions return (t, X, X_dot, state_names). Case-specific
parametric functions return (t, X, X_dot, state_names, U, parameter_names)
for one trajectory; dictionary wrappers return lists for t, X, X_dot and U.

Case 1:
    [\dot{h}, \ddot{h}, \dot{CL}] = f(h, \dot{h}, CL, [L/D, V_R])
    
Case 2:
    [\dot{h}, \ddot{h}, \dot{CL}, \ddot{CL}] = f(h, \dot{h}, CL, \dot{CL}, [L/D, V_R])
    
To-Be-Added:
Case 3:
    [\dot{h}, \ddot{h}] = f(h, \dot{h}, [CL, L/D, V_R])

Case 4:
    [\dot{CL}, \ddot{CL}] = f(CL, \dot{CL}, [h, \dot{h}, \ddot{h}, L/D, V_R])
"""

import numpy as np

from sindy_viv.differentiation import perform_derivative

def _prepare_input(time, data, names, differentiator, case, cylinder_type):
    """
    Construct single-cylinder or tandem states and derivatives for one case.
    """
    # Select the cylinder layout before the shared case preparation.
    if cylinder_type == "single":
        base_state_names = [
            'h_0', r'\dot{h}_0', 'C_{L0}'
        ]
        cylinder_ids = (0,)
    elif cylinder_type == "two_tandem":
        base_state_names = [
            'h_0', r'\dot{h}_0', 'C_{L0}', 'h_1', r'\dot{h}_1', 'C_{L1}',
        ]
        cylinder_ids = (0, 1)
    else:
        raise ValueError(f"Unknown cylinder type: {cylinder_type}")

    match case:
        case 1:
            state_names = base_state_names
        case 2:
            state_names = base_state_names + [
                rf'\dot{{C}}_{{L{cylinder_id}}}' for cylinder_id in cylinder_ids
            ]
        case _:
            raise ValueError(f"Invalid case value. Must be 1 or 2. Got {case}.")

    if not all(name in names for name in base_state_names):
        raise ValueError("Some chosen state variables are not present in the full state variable names.")
    
    t = np.asarray(time, dtype=time.dtype)
    if t.ndim != 1 or data.ndim != 2 or len(t) != len(data):
        raise ValueError(f"Time must be a 1-D array and data must be a 2-D array with the same number of rows as time. Got {t.shape}, {data.shape}")

    # Prepare the state array X with the selected base state variables in order.
    X = np.empty((len(t), len(state_names)), dtype=data.dtype)
    for column, name in enumerate(base_state_names):
        X[:, column] = data[:, names.index(name)]
        
    base_columns = len(base_state_names)
    
    # Compute the first derivative of the base state variables using the provided differentiator.
    first_derivative = perform_derivative(t, X[:, :base_columns], differentiator)

    match case:
        case 1:
            return t, X, first_derivative.astype(X.dtype, copy=False), state_names
        case 2:
            X_dot = np.empty_like(X)
            X_dot[:, :base_columns] = first_derivative
            
            del first_derivative
            
            # Resolve each cylinder's lift and lift-rate columns by state name.
            for cylinder_id in cylinder_ids:
                lift_name = f'C_{{L{cylinder_id}}}'
                lift_rate_name = rf'\dot{{C}}_{{L{cylinder_id}}}'
                X[:, state_names.index(lift_rate_name)] = X_dot[:, state_names.index(lift_name)]
                
            X_dot[:, base_columns:] = perform_derivative(t, X[:, base_columns:], differentiator)
            
            return t, X, X_dot, state_names

def _select_case(case, case_1, case_2):
    """
    Helper function to select the appropriate case function based on the provided case number.
    
    This can be used to return class, function, or any other callable based on the case number.
    """
    match case:
        case 1:
            return case_1
        case 2:
            return case_2
        case _:
            raise ValueError(f"Invalid case value. Must be 1 or 2. Got {case}.")

def _attach_constant_parameters(prepared_output, parameter_values, parameter_names):
    """Attach constant parameter columns to one prepared trajectory."""
    time, X, X_dot, state_names = prepared_output
    
    # Broadcast parameters into final storage using the state dtype.
    U = np.empty((len(time), len(parameter_values)), dtype=X.dtype)
    U[:] = parameter_values
    
    return time, X, X_dot, state_names, U, parameter_names

def _prepare_parametric_vr_inputs(
    times, full_state_variables, full_state_variable_names,
    input_variables, differentiators, prepare_input_case_function,
):
    """Validate selected VR keys and collect their prepared trajectories."""
    if not input_variables:
        raise ValueError("At least one input variable is required.")

    # Validate the entire VR selection before differentiating any trajectory.
    if not all(vr in full_state_variables for vr in input_variables):
        raise ValueError("Some input variables are not present in the full state variables.")
    
    if not all(vr in times for vr in input_variables):
        raise ValueError("Some input variables are not present in the times dictionary.")

    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)

    t_s, X_s, X_dot_s, U_s = [], [], [], []
    
    # Match differentiators by selection position, including repeated VR values.
    for vr, differentiator in zip(input_variables, differentiators):
        time, X, X_dot, state_names, U, parameter_names = prepare_input_case_function(
            times[vr], full_state_variables[vr], full_state_variable_names,
            differentiator, vr=vr,
        )
        
        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
        
    return t_s, X_s, X_dot_s, state_names, U_s, parameter_names

def _prepare_parametric_vr_lpd_inputs(
    times, full_state_variables, full_state_variable_names,
    input_variables, differentiators, prepare_input_case_function,
):
    """Validate selected (lpd, vr) pairs in dictionaries[lpd][vr] and collect."""
    if not input_variables:
        raise ValueError("At least one input variable is required.")

    # Validate both spacing and VR keys before differentiating any trajectory.
    if not all(
        lpd in full_state_variables and vr in full_state_variables[lpd]
        for lpd, vr in input_variables
    ):
        raise ValueError("Some input variables are not present in the full state variables.")
    if not all(
        lpd in times and vr in times[lpd] for lpd, vr in input_variables
    ):
        raise ValueError("Some input variables are not present in the times dictionary.")

    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)

    t_s, X_s, X_dot_s, U_s = [], [], [], []
    # Preserve pair ordering and assign differentiators by trajectory position.
    for (lpd, vr), differentiator in zip(input_variables, differentiators):
        time, X, X_dot, state_names, U, parameter_names = prepare_input_case_function(
            times[lpd][vr], full_state_variables[lpd][vr], full_state_variable_names,
            differentiator, vr=vr, lpd=lpd,
        )
        
        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
        
    return t_s, X_s, X_dot_s, state_names, U_s, parameter_names

"""
Single cylinder case
"""
def prepare_input_single_case_1(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=1, cylinder_type="single",
    )

def prepare_input_single_case_2(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=2, cylinder_type="single",
    )

def prepare_input_single_case_1_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
):
    return _attach_constant_parameters(
        prepare_input_single_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (vr,), ['V_R'],
    )

def prepare_input_single_case_2_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
):
    return _attach_constant_parameters(
        prepare_input_single_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (vr,), ['V_R'],
    )

def prepare_input_single_parametric_vr(
    times: dict[float, np.ndarray],
    full_state_variables: dict[float, np.ndarray],
    full_state_variable_names: list[str],
    input_variables: list[float],
    differentiators: object | list[object],
    case: int = 1,
):
    """Return lists of trajectories for the ordered VR subset in input_variables.

    Each selected VR must exist in both dictionaries. A differentiator list
    follows selection order; a single differentiator is reused for every VR.
    Returns (times, states, derivatives, state_names, parameters, parameter_names).
    """
    prepare_input_case_function = _select_case(
        case,
        prepare_input_single_case_1_parametric_vr,
        prepare_input_single_case_2_parametric_vr,
    )
    return _prepare_parametric_vr_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
    )

def prepare_input_single_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
    case: int = 1,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=case, cylinder_type="single",
    )

"""
Two cylinder in tandem case
"""
def prepare_input_two_tandem_case_1(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=1, cylinder_type="two_tandem",
    )

def prepare_input_two_tandem_case_2(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=2, cylinder_type="two_tandem",
    )

def prepare_input_two_tandem_case_1_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
):
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (vr,), ['V_R'],
    )

def prepare_input_two_tandem_case_2_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
):
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (vr,), ['V_R'],
    )

def prepare_input_two_tandem_case_1_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
    lpd: float,
):
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )


def prepare_input_two_tandem_case_2_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator,
    vr: float,
    lpd: float,
):
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )

def prepare_input_two_tandem_parametric_vr(
    times: dict[float, np.ndarray],
    full_state_variables: dict[float, np.ndarray],
    full_state_variable_names: list[str],
    input_variables: list[float],
    differentiators: object | list[object],
    case: int = 1,
):
    prepare_input_case_function = _select_case(
        case,
        prepare_input_two_tandem_case_1_parametric_vr,
        prepare_input_two_tandem_case_2_parametric_vr,
    )
    return _prepare_parametric_vr_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
    )

def prepare_input_two_tandem_parametric_vr_lpd(
    times: dict[float, dict[float, np.ndarray]],
    full_state_variables: dict[float, dict[float, np.ndarray]],
    full_state_variable_names: list[str],
    input_variables: list[tuple[float, float]],
    differentiators: object | list[object],
    case: int = 1,
):
    prepare_input_case_function = _select_case(
        case,
        prepare_input_two_tandem_case_1_parametric_vr_lpd,
        prepare_input_two_tandem_case_2_parametric_vr_lpd,
    )
    return _prepare_parametric_vr_lpd_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
    )

def prepare_input_two_tandem_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator,
    case: int = 1,
):
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=case, cylinder_type="two_tandem",
    )

def iter_prepared_trajectories(
    trajectories, full_state_variable_names, differentiator, case=1, *, tandem=False
):
    """Yield prepared (t, X, X_dot, names) tuples without collecting raw data.

    ``trajectories`` yields (time, data) pairs, for example from a generator
    loading one case at a time. Consume results incrementally to bound memory;
    collecting them into a list retains all prepared arrays. This does not
    make PySINDy's batch model fitting a streaming operation.
    """
    prepare_input_case_function = prepare_input_two_tandem_individual if tandem else prepare_input_single_individual
    
    for time, data in trajectories:
        yield prepare_input_case_function(time, data, full_state_variable_names, differentiator, case=case)
