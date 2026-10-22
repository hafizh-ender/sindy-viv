"""Shared scientific array and callable contracts; annotations do not coerce inputs."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32] | NDArray[np.float64]
TimeArray = NDArray[np.float64]
DynamicsCase = Literal[1, 2, 3, 4, 5]
CylinderType = Literal["single", "two_tandem"]
Parameterization = Literal["individual", "vr", "lpd", "vr_lpd"]

class Differentiator(Protocol):
    """A callable returning derivatives with respect to its supplied time coordinate."""

    def __call__(self, data: FloatArray, time: TimeArray | float) -> FloatArray:
        """Return an array with the same shape as data.

        Parameters:
            data (numpy.ndarray): Measured samples with one column per supplied variable name.
            time (numpy.ndarray or float): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        """
        ...

IndividualInputs = tuple[TimeArray, FloatArray, FloatArray, list[str]]
ControlledInputs = tuple[TimeArray, FloatArray, FloatArray, list[str], FloatArray, list[str]]
PreparedInputs = IndividualInputs | ControlledInputs
ParametricInputs = tuple[list[TimeArray], list[FloatArray], list[FloatArray], list[str], list[FloatArray], list[str]]

class EvaluationSummary(TypedDict):
    """Small evaluation record returned independently of trajectory retention."""

    variable_names: list[str]
    metrics: dict[str, dict[str, float]]
    model_path: Path | None
    metrics_path: Path
    figure_path: Path | None
    solver: dict[str, Any] | None
    sample_count: int
    time_region: list[float]
    tstar_region: list[float]

class EvaluationRecord(EvaluationSummary, total=False):
    """Optional complete trajectories available when return_data=True."""

    time: TimeArray
    time_star: TimeArray
    reference: FloatArray
    prediction: FloatArray
