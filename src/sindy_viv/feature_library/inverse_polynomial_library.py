"""Reciprocal-power features for PySINDy state or parameter libraries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from numbers import Integral

import numpy as np
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library.base import x_sequence_or_item
from pysindy.utils import AxesArray
from sklearn.utils.validation import check_is_fitted


class InversePolynomialLibrary(PolynomialLibrary):
    """Generate PolynomialLibrary features evaluated on reciprocal inputs.

    Parameters:
        degree (int): Highest total degree of the reciprocal products. Must be nonnegative; degree zero requires ``include_bias=True``. Default: 2
        include_interaction (bool): Include mixed reciprocal products, such as ``1/(x0*x1)``. If False, include only individual reciprocal powers. Default: True
        interaction_only (bool): Use distinct inputs in each product, omitting repeated powers such as ``1/x0^2``. Requires ``include_interaction=True``. Default: False
        include_bias (bool): Prepend a constant feature equal to one. Default: True
        order (str): Memory order of the dense output array. Default: 'C'

    Attributes
    ----------
    powers_ : ndarray of shape (n_output_features, n_features_in)
        Exponents of the original inputs; reciprocal powers are negative.

    Notes
    -----
    Feature ordering and constructor defaults match PolynomialLibrary. For two
    inputs and degree two, the default features are ``1``, ``1/x0``, ``1/x1``,
    ``1/x0^2``, ``1/(x0*x1)``, and ``1/x1^2``. Dense inputs must be finite and
    nonzero whenever reciprocal terms are requested. Output uses at least
    float64 precision.

    This library can be cloned and pickled with the fitted SINDy model.
    """

    def __init__(
        self, degree: int = 2, include_interaction: bool = True, interaction_only: bool = False,
        include_bias: bool = True, order: str = "C",
    ) -> None:
        """Store constructor parameters; see the class docstring for their scientific meaning.

        Parameters:
            degree (int): Highest total reciprocal polynomial degree; zero requires include_bias=True. Default: 2
            include_interaction (bool): Whether to include products of different reciprocal input variables. Default: True
            interaction_only (bool): Whether products must use distinct inputs without repeated powers. Default: False
            include_bias (bool): Whether to include a constant feature equal to one. Default: True
            order (str): Dense feature-array memory order: C or F. Default: 'C'
        """
        super().__init__(
            degree=degree, include_interaction=include_interaction,
            interaction_only=interaction_only, include_bias=include_bias, order=order,
        )

    def _validate_trajectory(self, x: np.ndarray) -> np.ndarray:
        """Validate one trajectory without allocating reciprocal features.

        Parameters:
            x (numpy.ndarray): Input samples with features on the final axis.
        """
        values = np.asarray(x)
        if values.ndim < 2 or values.shape[-1] == 0:
            raise ValueError("Expected an array with a nonempty final feature axis.")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError("Inverse polynomial inputs must be finite numeric values.")
        if self.degree > 0 and np.any(values == 0):
            raise ValueError("Inverse polynomial inputs must be nonzero.")
        return values

    @x_sequence_or_item
    def fit(self, x_full: np.ndarray | Sequence[np.ndarray], y: Any = None) -> InversePolynomialLibrary:
        """Validate trajectories and determine the number of output features.

        Parameters:
            x_full (numpy.ndarray or Sequence[numpy.ndarray]): One sample array or a sequence of trajectory arrays with features on the final axis.
            y (object): Unused target accepted for estimator compatibility. Default: None
        """
        if isinstance(self.degree, bool) or not isinstance(self.degree, Integral) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer.")
        for name in ("include_interaction", "interaction_only", "include_bias"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean.")
        if not self.include_interaction and self.interaction_only:
            raise ValueError("interaction_only=True requires include_interaction=True.")
        if self.order not in ("C", "F"):
            raise ValueError("order must be 'C' or 'F'.")
        if self.degree == 0 and not self.include_bias:
            raise ValueError("degree=0 requires include_bias=True to produce a feature.")
        n_features = self._validate_trajectory(x_full[0]).shape[-1]
        for x in x_full[1:]:
            if self._validate_trajectory(x).shape[-1] != n_features:
                raise ValueError("All trajectories must have the same number of input features.")
        self.n_features_in_ = n_features

        # Reuse PySINDy's combinations so feature counts and ordering stay identical.
        self.n_output_features_ = sum(1 for _ in self._combinations(
            n_features, self.degree, self.include_interaction,
            self.interaction_only, self.include_bias,
        ))
        return self

    @property
    def powers_(self) -> np.ndarray:
        """Return exponents with respect to the original, uninverted inputs."""
        return -super().powers_

    @x_sequence_or_item
    def transform(self, x_full: np.ndarray | Sequence[np.ndarray]) -> AxesArray | list[AxesArray]:
        """Evaluate reciprocal powers while preserving trajectory axes.

        Parameters:
            x_full (numpy.ndarray or Sequence[numpy.ndarray]): One sample array or a sequence of trajectory arrays with features on the final axis.
        """
        check_is_fitted(self, ("n_features_in_", "n_output_features_"))
        transformed = []
        for x in x_full:
            values = self._validate_trajectory(x)
            if values.shape[-1] != self.n_features_in_:
                raise ValueError("Input feature count does not match the fitted library.")

            # Promote integer inputs before inversion so reciprocal features cannot be truncated.
            dtype = np.result_type(values.dtype, np.float64)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                values = values.astype(dtype, copy=False)
                inverse = np.reciprocal(values) if self.degree else values

                # PolynomialLibrary supplies interaction products, axis handling, and output order.
                transformed.append(super().transform(AxesArray(inverse, x.axes)))
        return transformed

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return reciprocal names in PolynomialLibrary's feature order.

        Parameters:
            input_features (Sequence[str] or None): Names for the fitted input columns; None generates positional names. Default: None
        """
        check_is_fitted(self, ("n_features_in_", "n_output_features_"))
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        if len(input_features) != self.n_features_in_:
            raise ValueError("input_features must match the fitted number of input features.")
        names = []
        for powers in -self.powers_:
            factors = [
                str(name) if power == 1 else f"{name}^{power}"
                for name, power in zip(input_features, powers) if power
            ]
            if not factors:
                names.append("1")
            elif len(factors) == 1:
                names.append(f"1/{factors[0]}")
            else:
                names.append(f"1/({'*'.join(factors)})")
        return names
