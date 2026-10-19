"""Explicitly selected monomials without full-arity callable workarounds."""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence

import numpy as np
from pysindy.feature_library.base import BaseFeatureLibrary, x_sequence_or_item
from pysindy.utils import AxesArray
from sklearn.utils.validation import check_is_fitted


class SelectedPolynomialLibrary(BaseFeatureLibrary):
    """Evaluate ordered monomials on the final input axis.

    input_names defines the exact state/control order. term_powers contains one
    nonnegative integer exponent vector per monomial; include_bias prepends a
    constant feature. Duplicate exponent vectors are removed in insertion order.
    Floating precision and trajectory axes are preserved; integer inputs are
    promoted. No fitted callables, hard-coded arities or dynamic code are used.
    """

    def __init__(
        self,
        input_names: Sequence[str],
        term_powers: Sequence[Sequence[int]],
        include_bias: bool = False,
    ) -> None:
        """Store constructor parameters unchanged so sklearn cloning remains valid.

        Parameters:
            input_names (Sequence[str]): Ordered state and measured-control names expected by this library.
            term_powers (Sequence[Sequence[int]]): Ordered exponent vectors selecting polynomial monomials in input_names order.
            include_bias (bool): Whether to include a constant feature equal to one. Default: False
        """
        self.input_names = input_names
        self.term_powers = term_powers
        self.include_bias = include_bias

    @x_sequence_or_item
    def fit(self, x_full: np.ndarray | Sequence[np.ndarray], y: Any = None) -> "SelectedPolynomialLibrary":
        """Validate exponent/layout metadata and record input/output dimensions.

        Parameters:
            x_full (numpy.ndarray or Sequence[numpy.ndarray]): One sample array or a sequence of trajectory arrays with features on the final axis.
            y (object): Unused target accepted for estimator compatibility. Default: None
        """
        names = tuple(self.input_names)
        if not names or len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
            raise ValueError("input_names must be nonempty unique strings.")

        if not isinstance(self.include_bias, (bool, np.bool_)):
            raise ValueError("include_bias must be a boolean.")

        powers = np.asarray(self.term_powers)
        if powers.size == 0:
            powers = np.empty((0, len(names)), dtype=np.intp)

        if powers.ndim != 2 or powers.shape[1] != len(names) or powers.dtype.kind not in "iu" or np.any(powers < 0):
            raise ValueError("term_powers must contain nonnegative integer rows matching input_names.")

        ordered = ([tuple([0] * len(names))] if self.include_bias else []) + [tuple(row) for row in powers]
        self.powers_ = np.asarray(list(dict.fromkeys(ordered)), dtype=np.intp).reshape(-1, len(names))
        if not len(self.powers_):
            raise ValueError("Select at least one monomial.")

        for x in x_full:
            self._validate_input(x)

        self.n_features_in_ = len(names)
        self.n_output_features_ = len(self.powers_)

        return self

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        """Require real numeric arrays with a final feature axis matching the layout.

        Parameters:
            x (numpy.ndarray): Input samples with features on the final axis.
        """
        values = np.asarray(x)
        if values.ndim < 2 or values.shape[-1] != len(self.input_names) or values.dtype.kind not in "biuf":
            raise ValueError("Input must have a final numeric feature axis matching input_names.")

        return values

    @x_sequence_or_item
    def transform(self, x_full: np.ndarray | Sequence[np.ndarray]) -> list[AxesArray]:
        """Evaluate selected products using the output column as working storage.

        Parameters:
            x_full (numpy.ndarray or Sequence[numpy.ndarray]): One sample array or a sequence of trajectory arrays with features on the final axis.
        """
        check_is_fitted(self, ("powers_", "n_features_in_", "n_output_features_"))
        outputs = []
        for x in x_full:
            values = self._validate_input(x)
            output = np.ones((*values.shape[:-1], self.n_output_features_), dtype=np.result_type(values.dtype, np.float32))
            for column, powers in enumerate(self.powers_):

                # Avoid an N-by-terms-by-inputs temporary when evaluating all monomials.
                for index in np.flatnonzero(powers):
                    factor = values[..., index] if powers[index] == 1 else values[..., index] ** powers[index]
                    np.multiply(output[..., column], factor, out=output[..., column])

            outputs.append(AxesArray(output, x.axes))

        return outputs

    def get_feature_names(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Name features in their original order, rejecting swapped state/control layouts.

        Parameters:
            input_features (Sequence[str] or None): Names for the fitted input columns; None generates positional names. Default: None
        """
        check_is_fitted(self, ("powers_", "n_features_in_"))

        names = tuple(f"x{i}" for i in range(self.n_features_in_)) if input_features is None else tuple(input_features)
        defaults = tuple(f"x{i}" for i in range(self.n_features_in_))
        if names not in (tuple(self.input_names), defaults):
            raise ValueError(f"Library expects state/control order {tuple(self.input_names)}, received {names}.")

        return [
            " ".join(name if power == 1 else f"{name}^{power}" for name, power in zip(names, row) if power) or "1"
            for row in self.powers_
        ]
