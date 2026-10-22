"""Temporal weak regression for arbitrary pointwise feature libraries."""

from __future__ import annotations

from numbers import Integral, Real
import numpy as np
from typing import Any

import pysindy as ps
from pysindy.utils import AxesArray, comprehend_axes


def validate_weak_settings(K: int, H_xt: float | None, p: int, random_state: int | None) -> None:
    """Validate window count, half width, test degree and local RNG seed."""
    for name, value in (("K", K), ("p", p)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if H_xt is not None and (
        isinstance(H_xt, (bool, np.bool_)) or not isinstance(H_xt, Real)
        or not np.isfinite(H_xt) or H_xt <= 0
    ):
        raise ValueError("H_xt must be a finite positive half width or None.")
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_)) or not isinstance(random_state, Integral)
        or random_state < 0
    ):
        raise ValueError("random_state must be a nonnegative integer or None.")


def weak_half_width(time: np.ndarray, H_xt: float | None) -> float:
    """Resolve a half width on a finite, strictly increasing temporal grid."""
    if time.ndim != 1 or len(time) < 3 or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("Weak fitting needs at least three finite, strictly increasing time samples.")
    span = float(time[-1] - time[0])
    width = span / 20 if H_xt is None else float(H_xt)
    if not 0 < width < span / 2:
        raise ValueError("H_xt must be smaller than half the selected trajectory span.")
    return width


def _sample_centers(time: np.ndarray, width: float, K: int, rng: np.random.Generator) -> np.ndarray:
    """Sample uniformly from centers whose windows contain at least three rows.

    Merge feasible center intervals before sampling; sparse/irregular grids
    therefore fail promptly rather than entering an unbounded rejection loop.
    """
    intervals: list[list[float]] = []
    for left, right in zip(time[2:] - width, time[:-2] + width):
        left, right = max(float(left), time[0] + width), min(float(right), time[-1] - width)
        if left >= right:
            continue
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    if not intervals:
        raise ValueError("H_xt is too small: no integration window contains three samples.")
    intervals = np.asarray(intervals)
    lengths = intervals[:, 1] - intervals[:, 0]
    ends = np.cumsum(lengths)
    offsets = rng.uniform(0, ends[-1], size=K)
    indices = np.searchsorted(ends, offsets, side='right')
    starts = np.concatenate(([0.], ends[:-1]))
    return intervals[indices, 0] + offsets - starts[indices]


class _TemporalWeakPDELibrary(ps.WeakPDELibrary):
    """PySINDy weak integration with local, bounded sampling on temporal grids.

    Only weight setup changes: feasible centers are sampled without global RNG
    mutation or rejection loops. PySINDy supplies the analytic piecewise-linear
    weights, feature integration, and integration-by-parts targets.
    """

    def __init__(self, time: np.ndarray, rng: np.random.Generator, **kwargs: Any):
        self.rng = rng
        self.time = time
        super().__init__(function_library=ps.IdentityLibrary(), spatiotemporal_grid=time, **kwargs)

    def _set_up_weights(self) -> None:
        centers = _sample_centers(self.time, float(self.H_xt[0]), self.K, self.rng)
        self.inds_k, self.fullweights0, self.fulltweights, self.fullweights1 = [], [], [], []
        for center in centers:
            indices = np.flatnonzero((self.time > center-self.H_xt[0]) & (self.time < center+self.H_xt[0]))
            t = self.time[indices]
            half = (t[-1] - t[0]) / 2
            z = (t - (t[0] + half)) / half
            self.inds_k.append([indices])
            for derivative, destination, scale in ((0, self.fullweights0, half), (1, self.fulltweights, 1.)):
                weights = self._linear_weights(z, derivative, self.p)
                weights[0] = self._left_weights(z[:1], z[1:2], derivative, self.p)[0]
                weights[-1] = self._right_weights(z[-2:-1], z[-1:], derivative, self.p)[0]
                destination.append(np.asarray(weights) * scale)
            self.fullweights1.append([])

    def fit(self, x: AxesArray, y: Any = None) -> _TemporalWeakPDELibrary:
        """Fit the identity integrand on a 2-D trajectory, retaining its axes.

        PySINDy 2.1's weak fit drops the sample axis when fitting one row.
        This temporal adapter avoids that warning for its identity integrand.
        """
        self.function_library.fit(x)
        self.n_features_in_ = x.shape[-1]
        self.n_output_features_ = self.function_library.n_output_features_
        return self


def resolve_weak_library_config(
    config: dict[str, Any] | None, K: int | None, H_xt: float | None, p: int | None,
) -> dict[str, Any]:
    """Combine a WeakPDELibrary descriptor with optional explicit overrides.

    The response/parameter basis and temporal grid are supplied by fitting.
    Extra bias and PDE derivatives are disallowed because they would change
    the configured pointwise feature names and coefficient ordering.
    """
    if config is None:
        config = {'module': 'pysindy', 'class_name': 'WeakPDELibrary', 'kwargs': {}}
    if (not isinstance(config, dict) or config.get('class_name') != 'WeakPDELibrary'
            or config.get('module') not in ('pysindy', 'pysindy.feature_library',
                                           'pysindy.feature_library.weak_pde_library')
            or set(config) - {'module', 'class_name', 'kwargs'}):
        raise ValueError("Weak-library config must describe PySINDy's WeakPDELibrary.")
    kwargs = config.get('kwargs', {})
    allowed = {'K', 'H_xt', 'p', 'derivative_order', 'include_bias', 'include_interaction', 'implicit_terms'}
    if not isinstance(kwargs, dict) or set(kwargs) - allowed:
        raise ValueError("Weak-library kwargs may contain only temporal integration settings; the feature basis and grid are supplied separately.")
    if kwargs.get('derivative_order', 0) != 0 or kwargs.get('implicit_terms', False) or kwargs.get('include_bias', False):
        raise ValueError("Temporal weak fitting requires derivative_order=0, implicit_terms=False and include_bias=False; configure bias in the pointwise library.")
    resolved = dict(K=100, H_xt=None, p=4, derivative_order=0,
                    include_bias=False, include_interaction=False, implicit_terms=False)
    resolved.update(kwargs)
    resolved.update({name: value for name, value in (('K', K), ('H_xt', H_xt), ('p', p)) if value is not None})
    validate_weak_settings(resolved['K'], resolved['H_xt'], resolved['p'], None)
    resolved['K'], resolved['p'] = int(resolved['K']), int(resolved['p'])
    if resolved['H_xt'] is not None:
        resolved['H_xt'] = float(resolved['H_xt'])
    return {'module': config['module'], 'class_name': 'WeakPDELibrary', 'kwargs': resolved}


def build_weak_system(
    time: np.ndarray, states: np.ndarray, features: np.ndarray, *,
    K: int, H_xt: float | None, p: int, rng: np.random.Generator,
    weak_library_config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate pointwise features and states with PySINDy's WeakPDELibrary.

    An identity function library integrates the already expanded response and
    parameter products without refitting the user's pointwise basis. Both
    sides share subdomains and are normalized by integral(phi), giving each
    window equal weight without changing coefficient units. Returned arrays
    have K rows; temporal integration never differentiates the state samples.
    """
    width = weak_half_width(time, H_xt)
    descriptor = resolve_weak_library_config(weak_library_config, K, width, p)
    weak = _TemporalWeakPDELibrary(time, rng, **descriptor['kwargs'])
    features = AxesArray(np.asarray(features, dtype=np.float64), comprehend_axes(features))
    design = np.asarray(weak.fit_transform(features))
    targets = weak.convert_u_dot_integral(np.asarray(states, dtype=np.float64))
    mass = np.array([weights.sum() for weights in weak.fullweights0])[:, None]
    return design / mass, targets / mass
