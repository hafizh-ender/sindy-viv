"""Problem-independent correlation and conditioning of dense SINDy feature matrices.

Statistics and reduced QR factors are merged in batches. The condition number
comes from an SVD of the QR factor, never from the squared-condition Gram matrix.
Unit-L2-column conditioning uses that factor's column norms and a second SVD;
it does not center columns or require another feature-data pass.
No data loader, physical case, optimizer, plotting, or filesystem logic lives here.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from pysindy.feature_library.base import BaseFeatureLibrary

from .functions import uses_shape_only_fit


@dataclass(frozen=True)
class FeatureMatrixAssessment:
    """Pearson correlations plus raw and unit-L2-column matrix conditioning."""

    feature_names: tuple[str, ...]
    sample_count: int
    correlation: np.ndarray
    constant_columns: tuple[int, ...]
    rank: int
    rank_tolerance: float
    singular_values: np.ndarray
    condition_number: float
    column_l2_norms: np.ndarray
    normalized_rank: int
    normalized_rank_tolerance: float
    normalized_singular_values: np.ndarray
    normalized_condition_number: float

    @property
    def feature_count(self) -> int:
        """Return the number of ordered feature columns retained by this assessment."""
        return len(self.feature_names)

    @property
    def rank_deficient(self) -> bool:
        """Report whether numerical rank is smaller than the feature-column count."""
        return self.rank < self.feature_count

    @property
    def normalized_rank_deficient(self) -> bool:
        """Report deficient column rank using the normalized matrix's own tolerance."""
        return self.normalized_rank < self.feature_count


@dataclass(frozen=True)
class LibraryAssessment:
    """Per-dataset and vertically pooled assessments using one fitted feature basis."""

    individual: tuple[FeatureMatrixAssessment, ...]
    pooled: FeatureMatrixAssessment


def _matrix(values, columns=None):
    """Return finite real float64 samples with optional column-count validation.

    Require a nonempty 2-D matrix; reject unsupported dtypes and nonfinite
    values instead of silently dropping feature columns or sample rows.
    """
    values = np.asarray(values)
    if (values.ndim != 2 or not all(values.shape) or values.dtype.kind not in 'biuf'
            or (columns is not None and values.shape[1] != columns)):
        raise ValueError('Samples must be a nonempty real 2-D matrix with consistent columns.')
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('Feature-library samples must be finite; rows/columns are never silently dropped.')
    return values


def _batch_size(value):
    """Require a positive integer batch row count, rejecting booleans and fractions."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError('batch_size must be a positive integer.')


class _Statistics:
    """Mergeable centered moments and uncentered QR factors; never store full Theta."""

    def __init__(self):
        """Start an empty accumulator; the first update establishes its column count."""
        self.count = 0

    def update(self, values):
        """Merge one finite feature block into centered moments and an uncentered QR factor.

        No sample rows are retained. Covariance overflow raises ValueError so
        invalid conditioning/correlation reports cannot be written silently.
        """
        values = _matrix(values, None if not self.count else len(self.mean))
        batch = _Statistics()
        batch.count = len(values)
        batch.mean = values.mean(axis=0)
        centered = values-batch.mean
        batch.scatter = centered.T @ centered
        batch.minimum, batch.maximum = values.min(axis=0), values.max(axis=0)
        batch.factor = np.linalg.qr(values, mode='r')
        if not np.all(np.isfinite(batch.scatter)):
            raise ValueError('Feature covariance overflowed; rescale inputs before building the library.')
        self.merge(batch)

    def merge(self, other):
        """Combine another accumulator without concatenating its original sample rows.

        Update counts, means, centered cross-products, extrema and the reduced QR
        factor in place; both accumulators must use identical feature ordering.
        """
        if not other.count:
            return
        if not self.count:
            self.count = other.count
            for name in ('mean', 'scatter', 'minimum', 'maximum', 'factor'):
                setattr(self, name, getattr(other, name).copy())
            return
        if len(self.mean) != len(other.mean):
            raise ValueError('Pooled feature matrices must have the same ordered columns.')
        count = self.count+other.count
        delta = other.mean-self.mean
        self.scatter += other.scatter+np.outer(delta, delta)*(self.count/count)*other.count
        self.mean += delta*(other.count/count)
        self.minimum = np.minimum(self.minimum, other.minimum)
        self.maximum = np.maximum(self.maximum, other.maximum)
        self.factor = np.linalg.qr(np.vstack((self.factor, other.factor)), mode='r')
        self.count = count

    def result(self, names, rank_rtol):
        """Return correlations and both raw and unit-L2-column rank/conditioning.

        Require at least two samples and one name per feature. Constant-column
        correlations are NaN; deficient column rank yields an infinite condition
        number under the supplied relative tolerance or the standard SVD threshold.
        """
        if self.count < 2:
            raise ValueError('At least two feature-matrix rows are required for correlation assessment.')
        names = tuple(names)
        if len(names) != len(self.mean) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError('feature_names must contain one nonempty string per matrix column.')
        if rank_rtol is not None and (isinstance(rank_rtol, (bool, np.bool_))
                or not np.isscalar(rank_rtol) or not np.isfinite(rank_rtol) or not 0 <= rank_rtol < 1):
            raise ValueError('rank_rtol must be finite in [0, 1), or None for the standard SVD tolerance.')
        if not np.all(np.isfinite(self.scatter)):
            raise ValueError('Pooled covariance overflowed; rescale inputs before building the library.')
        deviation = np.sqrt(np.maximum(np.diag(self.scatter), 0.))
        constant = (self.minimum == self.maximum) | (deviation == 0)
        denominator = np.outer(deviation, deviation)
        correlation = np.full(self.scatter.shape, np.nan)
        valid = ~constant[:, None] & ~constant[None, :]
        np.divide(self.scatter, denominator, out=correlation, where=valid)
        np.clip(correlation, -1., 1., out=correlation)
        singular_values = np.linalg.svd(self.factor, compute_uv=False)
        relative = (np.finfo(np.float64).eps*max(self.count, len(names))
                    if rank_rtol is None else rank_rtol)
        tolerance = float(relative*singular_values[0])
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition = (float('inf') if rank < len(names) else
                     float(singular_values[0]/singular_values[-1]))
        # Theta=Q R with orthonormal Q: column norms and the singular values
        # after column scaling can be obtained directly from the small R.
        # Scale in two steps to avoid squaring extremely large/small entries.
        column_maximum = np.max(np.abs(self.factor), axis=0)
        scaled_factor = self.factor/np.where(column_maximum > 0, column_maximum, 1.)
        scaled_norms = np.linalg.norm(scaled_factor, axis=0)
        column_norms = column_maximum*scaled_norms
        normalized_factor = scaled_factor/np.where(scaled_norms > 0, scaled_norms, 1.)
        normalized_values = np.linalg.svd(normalized_factor, compute_uv=False)
        normalized_tolerance = float(relative*normalized_values[0])
        normalized_rank = int(np.count_nonzero(normalized_values > normalized_tolerance))
        normalized_condition = (float('inf') if normalized_rank < len(names) else
                                float(normalized_values[0]/normalized_values[-1]))
        return FeatureMatrixAssessment(names, self.count, correlation,
            tuple(map(int, np.flatnonzero(constant))), rank, tolerance, singular_values, condition,
            column_norms, normalized_rank, normalized_tolerance, normalized_values, normalized_condition)


def assess_feature_matrix(
    matrix: np.ndarray, feature_names: Sequence[str] | None = None, *,
    batch_size: int = 8192, rank_rtol: float | None = None,
) -> FeatureMatrixAssessment:
    """Assess a precomputed dense Theta from any SINDy problem, without fitting.

    Pearson correlation centers each column. Exactly constant columns (including
    a bias) have undefined correlations represented by NaN. Conditioning uses
    the original, uncentered, unnormalized Theta, with all columns retained.
    Numerically deficient column rank, including N < P, yields infinity.
    Default numerical-rank tolerance is eps*max(N,P)*sigma_max; rank_rtol can
    supply a relative threshold instead. No subsampling or regularization is used.
    Also report conditioning after dividing every nonzero feature column by
    its full-record L2 norm, without centering or dropping bias/zero columns.
    Normalized numerical rank uses the same relative tolerance independently.
    """
    _batch_size(batch_size)
    matrix = _matrix(matrix)
    names = feature_names if feature_names is not None else [f'f_{i}' for i in range(matrix.shape[1])]
    statistics = _Statistics()
    for first in range(0, len(matrix), batch_size):
        statistics.update(matrix[first:first+batch_size])
    return statistics.result(names, rank_rtol)


def assess_feature_library(
    library: BaseFeatureLibrary, samples: np.ndarray | Sequence[np.ndarray],
    input_names: Sequence[str] | None = None, *, batch_size: int = 8192,
    rank_rtol: float | None = None,
) -> LibraryAssessment:
    """Fit a copied pointwise library and assess each dataset plus their row pool.

    samples contains already prepared states, controls and optional parameters in
    library input order. For parameterized models, pass a ParameterizedLibrary
    and every trajectory used by that fit. The caller's library is not mutated.
    Known shape-only libraries transform in batches; unknown/data-adaptive
    libraries fit on full data and transform whole trajectories, preserving their
    semantics. Row-changing libraries require a precomputed feature matrix and
    assess_feature_matrix instead. Only small moments/QR factors are pooled.
    """
    _batch_size(batch_size)
    if not isinstance(library, BaseFeatureLibrary):
        raise ValueError('library must be a PySINDy BaseFeatureLibrary.')
    datasets = [samples] if isinstance(samples, np.ndarray) else list(samples)
    if not datasets:
        raise ValueError('At least one sample matrix is required.')
    datasets = [_matrix(values) for values in datasets]
    columns = datasets[0].shape[1]
    if any(values.shape[1] != columns for values in datasets):
        raise ValueError('All datasets must have the same ordered input columns.')
    names = list(input_names) if input_names is not None else [f'x{i}' for i in range(columns)]
    if len(names) != columns or len(set(names)) != len(names):
        raise ValueError('input_names must uniquely identify every input column.')
    # PySINDy 2.1 ParameterizedLibrary does not expose every constructor argument
    # as an attribute, so sklearn.clone cannot copy it. Preserve all nested settings.
    fitted = deepcopy(library)
    pointwise = uses_shape_only_fit(fitted)
    fitted.fit(datasets[0][:1] if pointwise else (datasets[0] if len(datasets) == 1 else datasets))
    feature_names = fitted.get_feature_names(names)
    pooled, results = _Statistics(), []
    for values in datasets:
        statistics = _Statistics()
        step = batch_size if pointwise else len(values)
        for first in range(0, len(values), step):
            block = values[first:first+step]
            features = np.asarray(fitted.transform(block))
            if features.ndim != 2 or len(features) != len(block) or features.shape[1] != len(feature_names):
                raise ValueError('Library transform must preserve sample rows; assess its precomputed matrix instead.')
            statistics.update(features)
        results.append(statistics.result(feature_names, rank_rtol))
        pooled.merge(statistics)
    return LibraryAssessment(tuple(results), pooled.result(feature_names, rank_rtol))
