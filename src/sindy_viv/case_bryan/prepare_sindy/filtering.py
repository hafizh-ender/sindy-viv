"""SciPy smoothing callables applied to measured channels before differentiation.

All filters operate along sample axis 0, preserve shape and floating precision,
and leave the supplied measurements and float64 tau timestamps untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt

from sindy_viv._typing import FloatArray, TimeArray

Filter = Callable[[FloatArray, TimeArray], FloatArray]


def _positive_integer(value: int, name: str) -> None:
    """Reject noninteger and nonpositive sample counts or filter orders."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _samples(data: ArrayLike, time: ArrayLike) -> tuple[np.ndarray, float]:
    """Validate finite aligned samples on uniform, increasing float64 tau."""
    values, t = np.asarray(data), np.asarray(time, dtype=np.float64)
    if values.ndim not in (1, 2) or t.ndim != 1 or len(values) != len(t) or len(t) < 2:
        raise ValueError("Filtering needs at least two aligned samples and a 1-D time array.")
    if values.dtype.kind not in 'biuf' or not np.all(np.isfinite(values)):
        raise ValueError("Filtering measurements must be finite real numeric values.")
    steps = np.diff(t)
    dt = float(steps[0])
    if not np.all(np.isfinite(t)) or dt <= 0 or not np.allclose(steps, dt, rtol=1e-6, atol=0):
        raise ValueError("Filtering requires finite, strictly increasing, uniformly spaced time.")
    dtype = values.dtype if values.dtype in (np.dtype('float32'), np.dtype('float64')) else np.dtype('float64')
    return np.asarray(values, dtype=dtype), dt


def _window(window_length: int, sample_count: int) -> None:
    """Validate a centered odd window without silently changing its bandwidth."""
    _positive_integer(window_length, 'window_length')
    if window_length % 2 != 1 or window_length > sample_count:
        raise ValueError("window_length must be odd and no larger than the number of samples.")


class SavgolFilter:
    """Smooth with scipy.signal.savgol_filter (deriv=0) along sample axis 0.

    window_length is an odd sample count; polyorder is smaller than the window.
    mode and cval use SciPy's edge-extension conventions.
    """

    def __init__(self, window_length: int, polyorder: int, mode: str = 'interp', cval: float = 0.) -> None:
        """Validate and store Savitzky-Golay smoothing settings.

        window_length is a positive sample count and polyorder must be smaller.
        mode/cval are forwarded to SciPy; record-length validation occurs on call.
        """
        _positive_integer(window_length, 'window_length')
        if (isinstance(polyorder, (bool, np.bool_)) or not isinstance(polyorder, Integral)
                or not 0 <= polyorder < window_length):
            raise ValueError("polyorder must be a nonnegative integer smaller than window_length.")
        self.window_length, self.polyorder = window_length, polyorder
        self.mode, self.cval = mode, cval

    def __call__(self, data: ArrayLike, time: ArrayLike) -> FloatArray:
        """Return smoothed measurements without computing a derivative."""
        values, _ = _samples(data, time)
        _window(self.window_length, len(values))
        return savgol_filter(values, self.window_length, self.polyorder, deriv=0,
                             axis=0, mode=self.mode, cval=self.cval)


class ButterworthFilter:
    """Zero-phase Butterworth filtering using SciPy second-order sections.

    cutoff is a fraction of Nyquist (0 < cutoff < 1) by default. Set
    normalized=False for cycles per unit tau; sampling frequency is then
    inferred from the original float64 time array, without VR rescaling.
    Bandpass/bandstop use a pair of increasing cutoff values. order, btype,
    padtype and padlen are passed to scipy.signal.butter/sosfiltfilt.
    """

    def __init__(
        self, cutoff: float | tuple[float, float], order: int = 4,
        btype: str = 'lowpass', normalized: bool = True,
        padtype: str | None = 'odd', padlen: int | None = None,
    ) -> None:
        """Validate and store zero-phase Butterworth filter settings.

        cutoff is a Nyquist fraction when normalized=True, otherwise cycles per
        unit of the supplied time coordinate. SciPy receives order, btype and
        padding settings when the filter processes a full record.
        """
        _positive_integer(order, 'order')
        if not isinstance(normalized, bool):
            raise ValueError("normalized must be a boolean.")
        self.cutoff, self.order, self.btype = cutoff, order, btype
        self.normalized, self.padtype, self.padlen = normalized, padtype, padlen

    def __call__(self, data: ArrayLike, time: ArrayLike) -> FloatArray:
        """Filter the full record forward and backward, preserving precision."""
        values, dt = _samples(data, time)
        sos = butter(self.order, self.cutoff, btype=self.btype, output='sos',
                     fs=None if self.normalized else 1. / dt)
        return sosfiltfilt(sos, values, axis=0, padtype=self.padtype, padlen=self.padlen).astype(values.dtype)


class MovingAverageFilter:
    """Centered moving average using scipy.ndimage.uniform_filter1d.

    window_length is an odd sample count. mode (default 'reflect') and cval
    specify SciPy edge padding; origin=0 keeps the window centered.
    """

    def __init__(self, window_length: int, mode: str = 'reflect', cval: float = 0.) -> None:
        """Validate and store a centered moving-average sample window.

        mode and cval select SciPy boundary handling; the window must fit the
        loaded record when the object is called.
        """
        _positive_integer(window_length, 'window_length')
        self.window_length, self.mode, self.cval = window_length, mode, cval

    def __call__(self, data: ArrayLike, time: ArrayLike) -> FloatArray:
        """Return a same-shape average of each measured channel independently."""
        values, _ = _samples(data, time)
        _window(self.window_length, len(values))
        return uniform_filter1d(values, self.window_length, axis=0, mode=self.mode, cval=self.cval, origin=0)
