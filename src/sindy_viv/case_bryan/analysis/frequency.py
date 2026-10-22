"""Pure FFT computation for uniformly sampled signals in fluid time t*.

No data loading, plotting, model evaluation or file writes occur here.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.signal.windows import get_window

FFT_WINDOWS = ('rectangular', 'hann', 'hamming', 'blackman', 'blackmanharris', 'bartlett', 'flattop')


@dataclass
class FrequencyAnalysis:
    """Single-sided amplitudes and cosine phases, with one column per channel.

    Frequencies use f*=cycles per unit t*. Phase is in radians relative to the
    first selected sample. DC is removed before tapering. Unresolved phases
    are NaN; peak_indices=-1 denotes a channel with no resolved oscillation.
    """
    frequencies: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    peak_indices: np.ndarray
    variable_names: list[str]
    means: np.ndarray
    sample_count: int
    tstar_region: tuple[float, float]
    time_step: float
    fft_window: str

    def summary(self) -> dict[str, Any]:
        """Return JSON-compatible dominant-bin summaries; undefined values are None."""
        channels = {}
        for column, name in enumerate(self.variable_names):
            peak = int(self.peak_indices[column])
            channels[name] = {
                'frequency': None if peak < 0 else float(self.frequencies[peak]),
                'amplitude': 0. if peak < 0 else float(self.amplitudes[peak, column]),
                'phase_rad': None if peak < 0 else float(self.phases[peak, column]),
                'mean': float(self.means[column]),
                'bin': None if peak < 0 else peak,
            }
        return dict(frequency_coordinate='f*=cycles/t*', phase_coordinate='cosine phase in radians',
                    phase_origin_tstar=self.tstar_region[0], sample_count=self.sample_count,
                    tstar_region=list(self.tstar_region), time_step=self.time_step,
                    frequency_resolution=float(self.frequencies[1]), fft_window=self.fft_window,
                    mean_removed=True, channels=channels)


def compute_frequency_analysis(
    tstar: np.ndarray, data: np.ndarray, variable_names: Sequence[str] | None = None, *,
    fft_window: str = 'rectangular', amplitude_floor: float = 0.,
) -> FrequencyAnalysis:
    """Compute full spectra and dominant positive-frequency bins without I/O.

    tstar must be finite, increasing and uniformly spaced. data is a real
    N-vector or N-by-channel array. Names default to x0, x1, ... . Subtract
    channel means, apply a periodic FFT window and divide by its coherent
    gain. Double positive-frequency amplitudes except the even-N Nyquist bin.
    No padding, peak interpolation, resampling or frequency-unit conversion is
    applied. Phase follows numpy.rfft's cosine convention at tstar[0].

    amplitude_floor is an absolute nonnegative resolution threshold. A small
    additional numerical floor relative to each channel's scale suppresses
    phases from roundoff. Constant channels have no dominant frequency/phase.
    Input arrays are not modified. Results retain the full spectra.
    """
    tstar = np.asarray(tstar, dtype=np.float64)
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[:, None]
    if tstar.ndim != 1 or len(tstar) < 2 or not np.all(np.isfinite(tstar)):
        raise ValueError('FFT requires at least two finite time samples.')
    if (data.ndim != 2 or data.shape[0] != len(tstar) or data.shape[1] == 0
            or data.dtype.kind not in 'biuf' or not np.all(np.isfinite(data))):
        raise ValueError('FFT requires aligned, finite real data with one or more channels.')
    names = [f'x{i}' for i in range(data.shape[1])] if variable_names is None else list(variable_names)
    if (len(names) != data.shape[1] or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Supply one unique nonempty variable name per data column.')
    dt = float(tstar[1]-tstar[0])
    if dt <= 0 or not np.allclose(np.diff(tstar), dt, rtol=1e-5, atol=dt*1e-4):
        raise ValueError('FFT requires uniformly spaced, increasing time samples.')
    if fft_window not in FFT_WINDOWS:
        raise ValueError(f'fft_window must be one of {FFT_WINDOWS!r}.')
    if not np.isscalar(amplitude_floor) or not np.isfinite(amplitude_floor) or amplitude_floor < 0:
        raise ValueError('amplitude_floor must be finite and nonnegative.')
    values = np.asarray(data, dtype=np.float64)
    means = values.mean(axis=0)
    weights = get_window('boxcar' if fft_window == 'rectangular' else fft_window, len(tstar), fftbins=True)
    normalization = weights.sum()
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError('The FFT window must have a finite positive sum.')
    spectrum = np.fft.rfft((values-means)*weights[:, None], axis=0)
    amplitudes = np.abs(spectrum)/normalization
    amplitudes[1:-1 if len(tstar) % 2 == 0 else None] *= 2
    floor = np.maximum(float(amplitude_floor), 100*np.finfo(float).eps*np.max(np.abs(values), axis=0))
    phases = np.angle(spectrum)
    phases[amplitudes <= floor] = np.nan
    phases[0] = np.nan
    peaks = np.argmax(amplitudes[1:], axis=0)+1
    peaks[amplitudes[peaks, np.arange(data.shape[1])] <= floor] = -1
    return FrequencyAnalysis(np.fft.rfftfreq(len(tstar), d=dt), amplitudes, phases, peaks,
                             names, means, len(tstar), (float(tstar[0]), float(tstar[-1])), dt, fft_window)


def complete_frequency_analysis(*args: Any, **kwargs: Any) -> FrequencyAnalysis:
    """Return the full numerical frequency analysis of supplied arrays."""
    return compute_frequency_analysis(*args, **kwargs)


def dominant_frequency_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return only the dominant-bin summary and analysis metadata of supplied arrays."""
    return compute_frequency_analysis(*args, **kwargs).summary()
