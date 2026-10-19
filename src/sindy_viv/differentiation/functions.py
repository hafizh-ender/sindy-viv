from __future__ import annotations

from numpy.typing import ArrayLike
from sindy_viv._typing import Differentiator
import numpy as np

def perform_derivative(time: ArrayLike, data: ArrayLike, differentiator: Differentiator | None) -> np.ndarray:
    """Call differentiator(data, time), preserving its returned shape and precision.

    Time is an array in the desired coordinate or a scalar sample step. Data is
    N-by-channel. Bryan preparation supplies float64 tau and requires a first
    derivative callable; this generic function applies no time/state rescaling.

    Parameters:
        time (array-like): Sample times in the coordinate used by the data; Bryan preparation and evaluation use float64 tau.
        data (array-like): Measured samples with one column per supplied variable name.

        differentiator (Differentiator or None): First-derivative callable evaluated on the supplied time axis; None is valid only when no derivative is needed.
    """

    # Apply the differentiator to the data
    return differentiator(data, time)
