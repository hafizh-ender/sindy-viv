from __future__ import annotations

from numpy.typing import ArrayLike, DTypeLike
import numpy as np

from scipy.signal import savgol_filter

from pysindy.differentiation import BaseDifferentiation

class SavitzkyGolayDerivative(BaseDifferentiation):
    """Savitzky-Golay differentiation for PySINDy.

    Fits a local polynomial of degree `polyorder` over a sliding window of
    `window_length` samples and evaluates its `deriv`-th derivative. Requires
    a uniformly spaced time grid.

    Parameters:
        window_length (int): Number of samples in the filter window.
        polyorder (int): Degree of the local polynomial; must be < window_length.
        deriv (int): Order of the derivative to return. Default: 1
        mode (str): Edge handling, passed to scipy. Default: 'interp'
        cval (float): Fill value when mode='constant'. Default: 0.0
        rtol (float): Relative tolerance for the uniform-spacing check. Default: 1e-06

        dtype (numpy.dtype, type, or str): Computation dtype. Default: np.float64

    Notes:
        SciPy returns zero for derivative orders above polyorder. Bryan state
        preparation requires a first-derivative callable, so use deriv=1 there.
        Time spacing is checked in float64 independently of working precision;
        no fluid/dynamic time conversion is performed by this differentiator.
    """
    def __init__(
        self,
        window_length: int,
        polyorder: int,
        deriv: int = 1,
        mode: str = "interp",
        cval: float = 0.0,
        rtol: float = 1e-6,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """Store constructor parameters; see the class docstring for their scientific meaning.

        Parameters:
            window_length (int): Number of samples in each Savitzky-Golay filter window.
            polyorder (int): Local polynomial degree; must be smaller than window_length.
            deriv (int): Derivative order returned by the Savitzky-Golay filter. Default: 1
            mode (str): Savitzky-Golay edge handling mode passed to SciPy. Default: 'interp'
            cval (float): Padding value used when mode is constant. Default: 0.0
            rtol (float): Relative numerical tolerance for the operation. Default: 1e-06

            dtype (numpy.dtype, type, or str): Floating computation or storage dtype; Bryan time arrays remain float64. Default: np.float64
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.mode = mode
        self.cval = cval
        self.rtol = rtol
        self.dtype = dtype

    def _uniform_dt(self, t: ArrayLike | float) -> float:

        # Scalar steps avoid allocating and scanning a full timestamp array.
        """Return a positive scalar step or validate uniform float64 timestamp spacing.

        Parameters:
            t (array-like or float): Timestamp array or positive scalar sample step in the data time coordinate.
        """
        if np.isscalar(t):
            dt = float(t)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Time step must be positive, got {dt}.")
            return dt

        # Check spacing in float64 independently of the state computation dtype.
        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"Time array must be 1-D, got shape {t.shape}.")
        if t.size < 2:
            raise ValueError(
                "Time array must have at least two entries to define a spacing."
            )

        diffs = np.diff(t)
        dt = diffs[0]
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Time array must be strictly increasing.")
        if not np.allclose(diffs, dt, rtol=self.rtol, atol=0.0):
            raise ValueError(
                "Time array must be uniformly spaced for Savitzky-Golay "
                f"differentiation. Spacing ranges from {diffs.min():.6g} "
                f"to {diffs.max():.6g}."
            )
        return float(dt)

    def _differentiate(self, x: np.ndarray, t: ArrayLike | float) -> np.ndarray:

        # Ensure input is a NumPy array with the specified dtype for computation.
        """Return Savitzky-Golay derivatives along axis zero using the configured working dtype.

        Parameters:
            x (numpy.ndarray): Input samples with features on the final axis.
            t (array-like or float): Timestamp array or positive scalar sample step in the data time coordinate.
        """
        x = np.asarray(x, dtype=self.dtype)

        # Convert unsupported input dtypes to SciPy's float64 working format.
        if x.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            x = np.asarray(x, dtype=np.float64)

        # Check if t is a scalar or an array and compute the uniform time step
        dt = self._uniform_dt(t)

        # Check if the window length is appropriate for the number of samples
        n_samples = x.shape[0]
        if self.mode == "interp" and self.window_length > n_samples:
            raise ValueError(
                f"window_length ({self.window_length}) exceeds the number of "
                f"samples ({n_samples}); required when mode='interp'."
            )

        return savgol_filter(
            x,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=dt,
            axis=0,
            mode=self.mode,
            cval=self.cval,
        )
