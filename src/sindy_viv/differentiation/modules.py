import numpy as np

from scipy.signal import savgol_filter

from pysindy.differentiation import BaseDifferentiation

class SavitzkyGolayDerivative(BaseDifferentiation):
    """
    Savitzky-Golay differentiation for PySINDy.

    Fits a local polynomial of degree `polyorder` over a sliding window of
    `window_length` samples and evaluates its `deriv`-th derivative. Requires
    a uniformly spaced time grid.

    Parameters:
        window_length (int): Number of samples in the filter window.
        polyorder (int): Degree of the local polynomial; must be < window_length.
        deriv (int): Order of the derivative to return. Default 1.
        mode (str): Edge handling, passed to scipy. Default 'interp'.
        cval (float): Fill value when mode='constant'.
        rtol (float): Relative tolerance for the uniform-spacing check.
    """

    def __init__(
        self,
        window_length,
        polyorder,
        deriv=1,
        mode="interp",
        cval=0.0,
        rtol=1e-6,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.mode = mode
        self.cval = cval
        self.rtol = rtol

    def _uniform_dt(self, t):
        if np.isscalar(t):
            dt = float(t)
            if dt <= 0:
                raise ValueError(f"Time step must be positive, got {dt}.")
            return dt

        t = np.asarray(t, dtype=float)
        if t.ndim != 1:
            raise ValueError(f"Time array must be 1-D, got shape {t.shape}.")
        if t.size < 2:
            raise ValueError(
                "Time array must have at least two entries to define a spacing."
            )

        diffs = np.diff(t)
        dt = diffs[0]
        if dt <= 0:
            raise ValueError("Time array must be strictly increasing.")
        if not np.allclose(diffs, dt, rtol=self.rtol, atol=0.0):
            raise ValueError(
                "Time array must be uniformly spaced for Savitzky-Golay "
                f"differentiation. Spacing ranges from {diffs.min():.6g} "
                f"to {diffs.max():.6g}."
            )
        return float(dt)

    def _differentiate(self, x, t):
        # Check input shapes and convert to float
        x = np.asarray(x, dtype=float)
        
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