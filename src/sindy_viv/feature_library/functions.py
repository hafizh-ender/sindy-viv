"""General feature-library preparation for SINDy workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pysindy.feature_library.base import BaseFeatureLibrary
from typing import Any

import numpy as np
import pysindy as ps

from .selected_polynomial_library import SelectedPolynomialLibrary

from sindy_viv.utils import initiate_object

def prepare_library(library_config: dict[str, Any]) -> BaseFeatureLibrary:
    """Instantiate an unfitted feature library from a module/class_name/kwargs dictionary.

    A bare TensoredLibrary is wrapped in GeneralizedLibrary to preserve outer
    factor input groups in SINDy/control/parameter workflows. Other libraries
    are returned directly. Nested descriptors use initiate_object validation.

    Parameters:
        library_config (dict[str, Any]): Feature-library object descriptor with module, constructor, and keyword settings.
    """
    library = initiate_object(**library_config)

    if not isinstance(library, BaseFeatureLibrary):
        raise ValueError("The configuration must construct a PySINDy feature library.")

    if isinstance(library, ps.TensoredLibrary):

        # Preserve one outer input group per factor when PySINDy multiplies libraries.
        return ps.GeneralizedLibrary([library])

    return library


def uses_shape_only_fit(library: BaseFeatureLibrary) -> bool:
    """Recognize exact shape-only implementations; subclasses/unknown libraries use full fitting.

    Reciprocal libraries deliberately retain their full-data fit validation,
    including checks for zero/nonfinite inputs. Composite libraries qualify
    only when every factor is shape-only.

    Parameters:
        library (BaseFeatureLibrary): Feature-library instance to prepare, fit, or evaluate.
    """
    if type(library) in (ps.PolynomialLibrary, ps.FourierLibrary, ps.CustomLibrary, SelectedPolynomialLibrary):
        return True

    if type(library) in (ps.GeneralizedLibrary, ps.TensoredLibrary, ps.ParameterizedLibrary):
        return all(uses_shape_only_fit(child) for child in library.libraries)

    return False

def fit_library_metadata(
    library: BaseFeatureLibrary,
    states: np.ndarray,
    input_data: np.ndarray | Sequence[np.ndarray] | None = None,
    parameters: Sequence[float] = (),
) -> BaseFeatureLibrary:
    """Fit supported metadata on one row; preserve full fitting for other libraries.

    Only explicitly recognized shape-only libraries are sampled. Unknown,
    data-adaptive, and reciprocal libraries receive all rows as before.
    Concatenation happens after row selection, limiting temporary allocations
    for ordinary polynomial/Fourier/wake features. The library is mutated and
    returned; no optimizer is fitted.

    Parameters:
        library (BaseFeatureLibrary): Feature-library instance to prepare, fit, or evaluate.

        states (numpy.ndarray): State samples with one column per state name.
        input_data (numpy.ndarray, Sequence[numpy.ndarray], or None): Measured control samples aligned with the state rows, or None. Default: None
        parameters (Sequence[float]): Constants appended after the state columns only when input_data is omitted, in the library parameter order. Default: ()
    """
    rows = slice(0, 1) if uses_shape_only_fit(library) else slice(None)

    # Start with the state samples...
    samples = np.asarray(states)[rows]

    #...then append any measured controls or constant parameters.
    if input_data is not None:
        samples = np.concatenate((samples, np.asarray(input_data)[rows]), axis=1)
    elif len(parameters):
        samples = np.concatenate((samples, np.broadcast_to(parameters, (len(samples), len(parameters)))), axis=1)

    library.fit(samples)

    return library
