"""Prepare named trajectories and append constant parameters without rescaling time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
import numpy as np
from numpy.typing import DTypeLike

from sindy_viv._typing import CylinderType, DynamicsCase, Differentiator, FloatArray, TimeArray, PreparedInputs
from sindy_viv.differentiation import perform_derivative
from .case_layout import get_case_layout


def append_parameter_columns(
    states: FloatArray,
    inputs: FloatArray | None,
    input_names: Sequence[str],
    parameter_values: Sequence[float],
    parameter_names: Sequence[str],
) -> tuple[FloatArray | None, list[str]]:
    """Append constant parameter columns after measured controls in one allocation.

    Parameters:
        states (numpy.ndarray): State matrix defining the sample count and output dtype.
        inputs (numpy.ndarray or None): Measured controls aligned with the state rows.
        input_names (Sequence[str]): Ordered names of the measured control columns.

        parameter_values (Sequence[float]): Finite constants appended to every sample row.
        parameter_names (Sequence[str]): Ordered names of the appended constants.
    """
    names = list(input_names) + list(parameter_names)
    if len(parameter_values) != len(parameter_names):
        raise ValueError("Parameter values and names must have equal lengths.")

    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Control and parameter names must be nonempty unique strings.")

    if inputs is None:
        if len(input_names):
            raise ValueError("Control names require aligned input data.")

    elif inputs.shape != (len(states), len(input_names)):
        raise ValueError("Input data must align with state rows and control names.")

    if not len(parameter_values):
        return inputs, names

    values = np.asarray(parameter_values, dtype=states.dtype)
    if values.shape != (len(parameter_names),) or not np.all(np.isfinite(values)):
        raise ValueError("Parameter values must be a finite one-dimensional sequence.")

    # Use state storage precision without converting or copying the time coordinate.
    combined = np.empty((len(states), len(names)), dtype=states.dtype)
    if inputs is not None:
        combined[:, :len(input_names)] = inputs
    combined[:, len(input_names):] = values

    return combined, names


@dataclass
class PreparedTrajectory:
    """One trajectory with float64 tau and independently chosen floating data dtype.

    Arrays have N rows; derivatives may be omitted during integral evaluation.
    State/control names describe columns exactly. Compatibility wrappers expose
    the original tuples; internal workflows access these descriptive fields.
    """

    time: TimeArray
    states: FloatArray
    derivatives: FloatArray | None
    state_names: list[str]
    inputs: FloatArray | None
    input_names: list[str]

    def as_tuple(self) -> PreparedInputs:
        """Return the original four-item or six-item training preparation tuple."""
        if self.derivatives is None:
            raise ValueError("Training tuples require derivative targets.")
        output = (self.time, self.states, self.derivatives, self.state_names)
        return output if self.inputs is None else output + (self.inputs, self.input_names)

    def with_parameters(self, values: Sequence[float], names: Sequence[str]) -> "PreparedTrajectory":
        """Copy controls into one array followed by constant VR/L/D columns.

        Parameters:
            values (Sequence[float]): Finite VR/L/D constants appended to each sample row.

            names (Sequence[str]): Names of the constants in values order.
        """
        inputs, input_names = append_parameter_columns(self.states, self.inputs, self.input_names, values, names)
        if not len(values):
            return self
        return replace(self, inputs=inputs, input_names=input_names)


def prepare_trajectory(
    time: np.ndarray, data: np.ndarray, names: list[str],
    differentiator: Differentiator | None, case: DynamicsCase, cylinder_type: CylinderType,
    *, dtype: DTypeLike = np.float64, with_derivatives: bool = True,
) -> PreparedTrajectory:
    """Prepare canonical states and controls on float64 tau without rescaling time.

    dtype (float32 or float64, default float64) controls state, derivative and
    control storage, including integer measurements. The differentiator must
    return first derivatives in tau with the same shape as its input.
    with_derivatives=False computes only lift rates needed for Cases 2/4/5 and
    omits regression targets. Differentiate full trajectories before cropping.
    Invalid names, dimensions or dtypes raise ValueError.

    Parameters:
        time (numpy.ndarray): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        data (numpy.ndarray): Measured samples with one column per supplied variable name.
        names (list[str]): Ordered variable names identifying array columns.

        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.

        case (int): Physical dynamics case, one of 1, 2, 3, 4, or 5.
        cylinder_type (str): Cylinder configuration: single or two_tandem.

        dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
        with_derivatives (bool): Whether to compute full derivative targets; False computes only derivatives required to construct states. Default: True
    """
    layout = get_case_layout(case, cylinder_type)
    value_dtype = np.dtype(dtype)
    if value_dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("dtype must be float32 or float64.")
    t, data = np.asarray(time, dtype=np.float64), np.asarray(data)
    if t.ndim != 1 or data.ndim != 2 or len(t) != len(data) or data.shape[1] != len(names):
        raise ValueError("Time must be 1-D and data must have one aligned column per variable name.")
    if data.dtype.kind not in "biuf":
        raise ValueError("Measurements must be real numeric values.")
    if len(set(names)) != len(names):
        raise ValueError("Variable names must be unique.")
    indices = {name: column for column, name in enumerate(names)}
    if any(name not in indices for name in layout.measured_names):
        raise ValueError("Some chosen state or input variables are not present in the full state variable names.")
    state_names, input_names = list(layout.state_names), list(layout.input_names)
    states = np.empty((len(t), len(state_names)), dtype=value_dtype)
    state_indices = {name: column for column, name in enumerate(state_names)}
    for name in layout.base_names:
        states[:, state_indices[name]] = data[:, indices[name]]
    derivatives = None
    if with_derivatives or layout.rate_names:
        if differentiator is None:
            raise ValueError(f"Case {case} requires a differentiator for the requested states or derivatives.")

        # Integral evaluation needs lift rates only; training differentiates every measured base state.
        source_names = layout.base_names if with_derivatives else layout.lift_names
        sources = states[:, :len(layout.base_names)] if with_derivatives else states[:, [state_indices[name] for name in source_names]]
        first = np.asarray(perform_derivative(t, sources, differentiator), dtype=value_dtype)
        if first.shape != sources.shape:
            raise ValueError("The differentiator must return the same shape as its input.")
        if with_derivatives:
            derivatives = np.empty_like(states)
            derivatives[:, :len(layout.base_names)] = first
        for lift, rate in zip(layout.lift_names, layout.rate_names):
            states[:, state_indices[rate]] = first[:, source_names.index(lift)]
        del first, sources
        if with_derivatives and layout.rate_names:
            rate_columns = [state_indices[name] for name in layout.rate_names]
            derivatives[:, rate_columns] = perform_derivative(t, states[:, rate_columns], differentiator)
    inputs = None
    if input_names:
        inputs = np.empty((len(t), len(input_names)), dtype=value_dtype)
        for column, name in enumerate(input_names):
            inputs[:, column] = data[:, indices[name]]
    return PreparedTrajectory(t, states, derivatives, state_names, inputs, input_names)
