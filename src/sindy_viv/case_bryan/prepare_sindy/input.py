r"""Prepare SINDy states, time-varying inputs, and constant parameters.

Individual cases 1 and 2 return (t, X, X_dot, state_names). Cases 3, 4 and 5
also return (U, input_names) because they have time-varying inputs.
All case-specific parametric functions return
(t, X, X_dot, state_names, U, input_names) for one trajectory; dictionary
wrappers return lists for t, X, X_dot and U. Time-varying input columns
precede constant parameters, ordered V_R or L/D then V_R.
Time is already tau; differentiation does not rescale it by V_R again.

Case 1:
    [\dot{h}, \ddot{h}, \dot{CL}] = f(h, \dot{h}, CL, [L/D, V_R])

Case 2:
    [\dot{h}, \ddot{h}, \dot{CL}, \ddot{CL}] = f(h, \dot{h}, CL, \dot{CL}, [L/D, V_R])

Case 3:
    [\dot{h}, \ddot{h}] = f(h, \dot{h}, [CL, L/D, V_R])

Case 4:
    [\dot{CL}, \ddot{CL}] = f(CL, \dot{CL}, [h, \dot{h}, \ddot{h}, L/D, V_R])

Case 5:
    [\dot{CL}, \ddot{CL}] = f(CL, \dot{CL}, [h, \dot{h}, L/D, V_R])

Proposal only (not implemented): change the derivative scheme. Instead of using \dot{} = d()/d\tau, \tau = f_N t = U t / (VR D) = t* / VR,
we would use \tilde{} = d()/d\theta, \theta = 2 \pi f_N t = 2 \pi U t / (VR D) = 2 \pi t* / VR. This is equivalent to scaling the time axis by 2 \pi.
In this case, the derivative data, namely raw \dot{h} and \ddot{h} would have to be rescaled as well.

\dot{} = d()/d\tau = d()/d\theta * d\theta/d\tau = \tilde{} * 2 \pi
\ddot{} = \tilde{\tilde{}} * (2 \pi)^2

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from numpy.typing import ArrayLike
from sindy_viv._typing import ControlledInputs, CylinderType, Differentiator, DynamicsCase, IndividualInputs, ParametricInputs, PreparedInputs
from typing import Any

import numpy as np

from numpy.typing import DTypeLike

from sindy_viv.case_bryan.prepare_sindy.trajectory import append_parameter_columns
from .trajectory import prepare_trajectory

def _prepare_input(
    time: ArrayLike,
    data: ArrayLike,
    names: Sequence[str],
    differentiator: Differentiator | None,
    case: DynamicsCase,
    cylinder_type: CylinderType,
    *,
    dtype: DTypeLike = np.float64,
) -> PreparedInputs:
    """Return the original training tuple from the shared named trajectory.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        data (array-like): Measured samples with one column per supplied variable name.
        names (Sequence[str]): Ordered variable names identifying array columns.

        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        cylinder_type (str): Cylinder configuration: single or two_tandem.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return prepare_trajectory(time, data, names, differentiator, case, cylinder_type, dtype=dtype).as_tuple()


def _select_case(
    case: DynamicsCase,
    case_1: Callable[..., Any],
    case_2: Callable[..., Any],
    case_3: Callable[..., Any],
    case_4: Callable[..., Any],
    case_5: Callable[..., Any],
) -> Callable[..., Any]:
    """Helper function to select the appropriate case function based on the provided case number.

    All supplied alternatives must be callable; invalid case numbers raise ValueError.

    Parameters:
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        case_1 (Callable[..., Any]): Preparation callable for dynamics Case 1.
        case_2 (Callable[..., Any]): Preparation callable for dynamics Case 2.
        case_3 (Callable[..., Any]): Preparation callable for dynamics Case 3.
        case_4 (Callable[..., Any]): Preparation callable for dynamics Case 4.
        case_5 (Callable[..., Any]): Preparation callable for dynamics Case 5.
    """
    if isinstance(case, (bool, np.bool_)):
        raise ValueError("Invalid case value. Must be 1, 2, 3, 4, or 5.")
    match case:
        case 1:
            return case_1
        case 2:
            return case_2
        case 3:
            return case_3
        case 4:
            return case_4
        case 5:
            return case_5
        case _:
            raise ValueError(f"Invalid case value. Must be 1, 2, 3, 4, or 5. Got {case}.")

def _attach_constant_parameters(
    prepared_output: PreparedInputs,
    parameter_values: Sequence[float],
    parameter_names: Sequence[str],
) -> ControlledInputs:
    """Append constant parameters after any time-varying input columns.

    Parameters:
        prepared_output (PreparedInputs): Prepared four- or six-item trajectory tuple to augment with parameter controls.
        parameter_values (Sequence[float]): Constant values appended to each trajectory as model controls.
        parameter_names (Sequence[str]): Ordered names of the constant parameter columns.
    """
    if len(prepared_output) == 4:
        time, X, X_dot, state_names = prepared_output
        input_data, input_names = None, []
    else:
        time, X, X_dot, state_names, input_data, input_names = prepared_output

    U, input_names = append_parameter_columns(X, input_data, input_names, parameter_values, parameter_names)
    return time, X, X_dot, state_names, U, input_names


def _prepare_parametric_vr_inputs(
    times: Mapping[float, Any],
    full_state_variables: Mapping[float, Any],
    full_state_variable_names: Sequence[str],
    input_variables: Sequence[Any],
    differentiators: Differentiator | list[Differentiator],
    prepare_input_case_function: Callable[..., PreparedInputs],
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Validate selected VR keys and collect their prepared trajectories.

    Parameters:
        times (Mapping[float, Any]): Trajectory time arrays keyed by VR, or by LpD then VR for nested datasets.
        full_state_variables (Mapping[float, Any]): Measured trajectory arrays keyed by VR, or by LpD then VR.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        input_variables (Sequence[Any]): Selected VR keys or (LpD, VR) pairs, preserving the requested trajectory order.
        differentiators (Differentiator or list[Differentiator]): One derivative callable for all selections, or a list aligned with input_variables.
        prepare_input_case_function (Callable[..., PreparedInputs]): Individual trajectory preparation callable selected for the physical case.
        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
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
        time, X, X_dot, state_names, U, input_names = prepare_input_case_function(
            times[vr], full_state_variables[vr], full_state_variable_names,
            differentiator, vr=vr,
            dtype=dtype,
        )

        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)

    return t_s, X_s, X_dot_s, state_names, U_s, input_names

def _prepare_parametric_vr_lpd_inputs(
    times: Mapping[float, Any],
    full_state_variables: Mapping[float, Any],
    full_state_variable_names: Sequence[str],
    input_variables: Sequence[Any],
    differentiators: Differentiator | list[Differentiator],
    prepare_input_case_function: Callable[..., PreparedInputs],
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Validate selected (lpd, vr) pairs in dictionaries[lpd][vr] and collect.

    Parameters:
        times (Mapping[float, Any]): Trajectory time arrays keyed by VR, or by LpD then VR for nested datasets.
        full_state_variables (Mapping[float, Any]): Measured trajectory arrays keyed by VR, or by LpD then VR.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        input_variables (Sequence[Any]): Selected VR keys or (LpD, VR) pairs, preserving the requested trajectory order.
        differentiators (Differentiator or list[Differentiator]): One derivative callable for all selections, or a list aligned with input_variables.
        prepare_input_case_function (Callable[..., PreparedInputs]): Individual trajectory preparation callable selected for the physical case.
        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
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
        time, X, X_dot, state_names, U, input_names = prepare_input_case_function(
            times[lpd][vr], full_state_variables[lpd][vr], full_state_variable_names,
            differentiator, vr=vr, lpd=lpd,
            dtype=dtype,
        )

        t_s.append(time)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)

    return t_s, X_s, X_dot_s, state_names, U_s, input_names

# Single cylinder case
def prepare_input_single_case_1(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> IndividualInputs:
    """Prepare Case 1 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=1, cylinder_type="single",
        dtype=dtype,
    )

def prepare_input_single_case_2(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> IndividualInputs:
    """Prepare Case 2 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=2, cylinder_type="single",
        dtype=dtype,
    )

def prepare_input_single_case_3(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 3 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=3, cylinder_type="single",
        dtype=dtype,
    )


def prepare_input_single_case_4(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 4 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=4, cylinder_type="single",
        dtype=dtype,
    )


def prepare_input_single_case_5(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 5 lift states/derivatives with measured h and hdot controls.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=5, cylinder_type="single",
        dtype=dtype,
    )


def prepare_input_single_case_1_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 1 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_single_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )

def prepare_input_single_case_2_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 2 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_single_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )

def prepare_input_single_case_3_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 3 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_single_case_3(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )


def prepare_input_single_case_4_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 4 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_single_case_4(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )


def prepare_input_single_case_5_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 5 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_single_case_5(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
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
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Collect prepared trajectories for ordered VR keys.

    input_variables guards dictionary membership and selects/order trajectories.
    A differentiator list follows selection position; one callable is otherwise
    reused. Time is already tau and dtype controls output storage (default
    float64). Returns lists for time/states/derivatives/inputs and shared name
    lists. Measured controls precede constant parameter columns.

    Parameters:
        times (dict[float, numpy.ndarray]): Trajectory time arrays keyed by VR, or by LpD then VR for nested datasets.
        full_state_variables (dict[float, numpy.ndarray]): Measured trajectory arrays keyed by VR, or by LpD then VR.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        input_variables (list[float]): Selected VR keys or (LpD, VR) pairs, preserving the requested trajectory order.
        differentiators (object or list[object]): One derivative callable for all selections, or a list aligned with input_variables.
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    prepare_input_case_function = _select_case(
        case,
        prepare_input_single_case_1_parametric_vr,
        prepare_input_single_case_2_parametric_vr,
        prepare_input_single_case_3_parametric_vr,
        prepare_input_single_case_4_parametric_vr,
        prepare_input_single_case_5_parametric_vr,
    )
    return _prepare_parametric_vr_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
        dtype=dtype,
    )

def prepare_input_single_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    case: int = 1,
    *,
    dtype: DTypeLike = np.float64,
) -> PreparedInputs:
    """Prepare one selected dynamics case on an existing float64 tau axis.

    dtype controls state/derivative/control storage (default float64). Cases 1/2
    return (time, states, derivatives, names); Cases 3/4/5 also return inputs and
    input_names. Required channels follow get_case_layout and no time rescaling
    is applied. Invalid dimensions, dtype, case or missing channels raise ValueError.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=case, cylinder_type="single",
        dtype=dtype,
    )

# Two cylinder in tandem case
def prepare_input_two_tandem_case_1(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> IndividualInputs:
    """Prepare Case 1 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=1, cylinder_type="two_tandem",
        dtype=dtype,
    )

def prepare_input_two_tandem_case_2(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> IndividualInputs:
    """Prepare Case 2 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=2, cylinder_type="two_tandem",
        dtype=dtype,
    )

def prepare_input_two_tandem_case_3(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 3 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=3, cylinder_type="two_tandem",
        dtype=dtype,
    )


def prepare_input_two_tandem_case_4(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 4 states/first derivatives on an existing tau axis.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=4, cylinder_type="two_tandem",
        dtype=dtype,
    )


def prepare_input_two_tandem_case_5(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare Case 5 lift states/derivatives with measured h and hdot controls.

    dtype controls floating state/control storage (default float64). Channel
    order follows get_case_layout. Returns (time, states, derivatives, state_names, inputs, input_names).
    The differentiator receives the full tau array, without another VR scaling.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=5, cylinder_type="two_tandem",
        dtype=dtype,
    )


def prepare_input_two_tandem_case_1_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 1 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )

def prepare_input_two_tandem_case_2_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 2 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )

def prepare_input_two_tandem_case_3_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 3 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_3(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )


def prepare_input_two_tandem_case_4_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 4 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_4(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )


def prepare_input_two_tandem_case_5_parametric_vr(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 5 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_5(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (vr,), ['V_R'],
    )


def prepare_input_two_tandem_case_1_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 1 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )


def prepare_input_two_tandem_case_2_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 2 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )

def prepare_input_two_tandem_case_3_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 3 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_3(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )


def prepare_input_two_tandem_case_4_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 4 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_4(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd, vr), ['L/D', 'V_R'],
    )


def prepare_input_two_tandem_case_5_parametric_vr_lpd(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list[str],
    differentiator: Differentiator | None,
    vr: float,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 5 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        vr (float): Reduced velocity for this trajectory or fixed-VR model.
        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_5(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
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
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Collect prepared trajectories for ordered VR keys.

    input_variables guards dictionary membership and selects/order trajectories.
    A differentiator list follows selection position; one callable is otherwise
    reused. Time is already tau and dtype controls output storage (default
    float64). Returns lists for time/states/derivatives/inputs and shared name
    lists. Measured controls precede constant parameter columns.

    Parameters:
        times (dict[float, numpy.ndarray]): Trajectory time arrays keyed by VR, or by LpD then VR for nested datasets.
        full_state_variables (dict[float, numpy.ndarray]): Measured trajectory arrays keyed by VR, or by LpD then VR.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        input_variables (list[float]): Selected VR keys or (LpD, VR) pairs, preserving the requested trajectory order.
        differentiators (object or list[object]): One derivative callable for all selections, or a list aligned with input_variables.
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    prepare_input_case_function = _select_case(
        case,
        prepare_input_two_tandem_case_1_parametric_vr,
        prepare_input_two_tandem_case_2_parametric_vr,
        prepare_input_two_tandem_case_3_parametric_vr,
        prepare_input_two_tandem_case_4_parametric_vr,
        prepare_input_two_tandem_case_5_parametric_vr,
    )
    return _prepare_parametric_vr_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
        dtype=dtype,
    )

def prepare_input_two_tandem_parametric_vr_lpd(
    times: dict[float, dict[float, np.ndarray]],
    full_state_variables: dict[float, dict[float, np.ndarray]],
    full_state_variable_names: list[str],
    input_variables: list[tuple[float, float]],
    differentiators: object | list[object],
    case: int = 1,
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Collect prepared trajectories for ordered (LpD, VR) pairs in dictionaries[LpD][VR].

    input_variables guards dictionary membership and selects/order trajectories.
    A differentiator list follows selection position; one callable is otherwise
    reused. Time is already tau and dtype controls output storage (default
    float64). Returns lists for time/states/derivatives/inputs and shared name
    lists. Measured controls precede constant parameter columns.

    Parameters:
        times (dict[float, dict[float, numpy.ndarray]]): Trajectory time arrays keyed by VR, or by LpD then VR for nested datasets.
        full_state_variables (dict[float, dict[float, numpy.ndarray]]): Measured trajectory arrays keyed by VR, or by LpD then VR.
        full_state_variable_names (list[str]): Names identifying the measured trajectory columns.
        input_variables (list[tuple[float, float]]): Selected VR keys or (LpD, VR) pairs, preserving the requested trajectory order.
        differentiators (object or list[object]): One derivative callable for all selections, or a list aligned with input_variables.
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    prepare_input_case_function = _select_case(
        case,
        prepare_input_two_tandem_case_1_parametric_vr_lpd,
        prepare_input_two_tandem_case_2_parametric_vr_lpd,
        prepare_input_two_tandem_case_3_parametric_vr_lpd,
        prepare_input_two_tandem_case_4_parametric_vr_lpd,
        prepare_input_two_tandem_case_5_parametric_vr_lpd,
    )
    return _prepare_parametric_vr_lpd_inputs(
        times, full_state_variables, full_state_variable_names,
        input_variables, differentiators, prepare_input_case_function,
        dtype=dtype,
    )

def prepare_input_two_tandem_individual(
    time: np.ndarray,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: list,
    differentiator: Differentiator | None,
    case: int = 1,
    *,
    dtype: DTypeLike = np.float64,
) -> PreparedInputs:
    """Prepare one selected dynamics case on an existing float64 tau axis.

    dtype controls state/derivative/control storage (default float64). Cases 1/2
    return (time, states, derivatives, names); Cases 3/4/5 also return inputs and
    input_names. Required channels follow get_case_layout and no time rescaling
    is applied. Invalid dimensions, dtype, case or missing channels raise ValueError.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (list): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _prepare_input(
        time, full_state_variable_data, full_state_variable_names, differentiator,
        case=case, cylinder_type="two_tandem",
        dtype=dtype,
    )

def iter_prepared_trajectories(
    trajectories: Iterable[Any],
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    case: DynamicsCase = 1,
    *,
    tandem: bool = False,
    dtype: DTypeLike = np.float64,
) -> Iterator[PreparedInputs]:
    """Yield prepared trajectories, including U and input names for cases 3, 4 and 5.

    ``trajectories`` yields (time, data) pairs, for example from a generator
    loading one case at a time. Consume results incrementally to bound memory;
    collecting them into a list retains all prepared arrays. This does not
    make PySINDy's batch model fitting a streaming operation.

    Parameters:
        trajectories (Iterable[Any]): Trajectory records or iterable entries in the format described by this function.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1
        tandem (bool): Whether the iterable contains tandem-cylinder trajectories. Default: False
        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    prepare_input_case_function = prepare_input_two_tandem_individual if tandem else prepare_input_single_individual

    for time, data in trajectories:
        yield prepare_input_case_function(
            time, data, full_state_variable_names, differentiator, case=case,
            dtype=dtype,
        )


def prepare_input_two_tandem_parametric_lpd(
    times: Mapping[float, Any], full_state_variables: Mapping[float, Any], full_state_variable_names: Sequence[str],
    input_variables: Sequence[Any], differentiators: Differentiator | list[Differentiator], case: DynamicsCase = 1,
    *,
    dtype: DTypeLike = np.float64,
) -> ParametricInputs:
    """Collect prepared trajectories for ordered L/D keys at fixed VR.

    input_variables guards dictionary membership and selects/orders trajectories.
    A differentiator list follows selection position; one callable is otherwise
    reused. Time is already tau and dtype controls output storage (default
    float64). Returns lists for time/states/derivatives/inputs and shared name
    lists. Measured controls precede constant parameter columns.

    Parameters:
        times (Mapping[float, Any]): Trajectory tau arrays keyed directly by L/D at one fixed VR.
        full_state_variables (Mapping[float, Any]): Measured trajectory arrays keyed directly by L/D at one fixed VR.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        input_variables (Sequence[Any]): Selected L/D keys, preserving the requested trajectory order.
        differentiators (Differentiator or list[Differentiator]): One derivative callable for all selections, or a list aligned with input_variables.
        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5. Default: 1

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    if not input_variables:
        raise ValueError("At least one L/D input variable is required.")
    if not all(lpd in times and lpd in full_state_variables for lpd in input_variables):
        raise ValueError("Some L/D input variables are missing from times or full_state_variables.")
    if isinstance(differentiators, list):
        if len(differentiators) != len(input_variables):
            raise ValueError("Length of differentiators list must match the number of L/D input variables.")
    else:
        differentiators = [differentiators] * len(input_variables)
    prepare_case = _select_case(
        case,
        prepare_input_two_tandem_case_1_parametric_lpd,
        prepare_input_two_tandem_case_2_parametric_lpd,
        prepare_input_two_tandem_case_3_parametric_lpd,
        prepare_input_two_tandem_case_4_parametric_lpd,
        prepare_input_two_tandem_case_5_parametric_lpd,
    )
    t_s, X_s, X_dot_s, U_s = [], [], [], []
    for lpd, differentiator in zip(input_variables, differentiators):
        t, X, X_dot, state_names, U, input_names = prepare_case(
            times[lpd], full_state_variables[lpd], full_state_variable_names,
            differentiator, lpd,
            dtype=dtype,
        )
        t_s.append(t)
        X_s.append(X)
        X_dot_s.append(X_dot)
        U_s.append(U)
    return t_s, X_s, X_dot_s, state_names, U_s, input_names


def prepare_input_two_tandem_case_1_parametric_lpd(
    time: ArrayLike,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 1 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_1(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd,), ['L/D'],
    )


def prepare_input_two_tandem_case_2_parametric_lpd(
    time: ArrayLike,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 2 trajectory with constant parameter inputs only.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_2(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd,), ['L/D'],
    )


def prepare_input_two_tandem_case_3_parametric_lpd(
    time: ArrayLike,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 3 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_3(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd,), ['L/D'],
    )


def prepare_input_two_tandem_case_4_parametric_lpd(
    time: ArrayLike,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 4 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_4(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd,), ['L/D'],
    )


def prepare_input_two_tandem_case_5_parametric_lpd(
    time: ArrayLike,
    full_state_variable_data: np.ndarray,
    full_state_variable_names: Sequence[str],
    differentiator: Differentiator | None,
    lpd: float,
    *,
    dtype: DTypeLike = np.float64,
) -> ControlledInputs:
    """Prepare one Case 5 trajectory with measured controls followed by constant parameters.

    Time is already tau; dtype sets states, derivatives and inputs (default
    float64). Returns (time, states, derivatives, state_names, inputs, input_names).
    Parameter columns are V_R, L/D, or L/D then V_R as named by the function.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        full_state_variable_data (numpy.ndarray): Measured trajectory columns in full_state_variable_names order.
        full_state_variable_names (Sequence[str]): Names identifying the measured trajectory columns.
        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        lpd (float): Cylinder spacing L/D; None denotes the single-cylinder configuration.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
    """
    return _attach_constant_parameters(
        prepare_input_two_tandem_case_5(
            time, full_state_variable_data, full_state_variable_names, differentiator,
            dtype=dtype,
        ),
        (lpd,), ['L/D'],
    )
