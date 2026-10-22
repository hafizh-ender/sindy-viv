"""Plot computed FFT amplitudes and phases; frequency numerics live in analysis.frequency."""

from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from sindy_viv.case_bryan.plot import create_subplots, LINE_WIDTH, select_plot_variables
from sindy_viv.case_bryan.analysis.frequency import FrequencyAnalysis, compute_frequency_analysis

def plot_frequency_analysis(
    tstar: np.ndarray,
    data: np.ndarray,
    variable_names: Sequence[str],
    *,
    xlim: Sequence[float] | None = None,
    title: str = "",
    plot_variables: Sequence[str] | None = None,
    plot_cylinders: Sequence[int] | None = None,
    fft_window: str = 'rectangular',
) -> tuple[Figure, np.ndarray]:
    """Plot centered single-sided amplitudes and FFT phases in square panels.

    Each channel's dominant positive-frequency bin determines both annotations.
    Phase is relative to the first selected sample and is displayed in units of pi.
    Periodic FFT windows are applied after mean removal. Amplitudes are divided
    by the window sum to compensate for its coherent gain.
    The caller owns the returned figure and must close it after use.

    Parameters:
        tstar (numpy.ndarray): Uniform fluid-time samples t* = t U_inf / D.
        data (numpy.ndarray): Finite real samples, with one column per variable name.
        variable_names (Sequence[str]): Ordered variable names used as row labels.

        xlim (Sequence[float] or None): Displayed f* limits; None uses zero through min(1, Nyquist). Default: None
        title (str): Dataset and time-window information added to the figure title. Default: ''
        plot_variables (Sequence[str] or None): Variable families or exact channel names to display. Default: None
        plot_cylinders (Sequence[int] or None): Cylinder IDs to display. Default: None
        fft_window (str): Window from FFT_WINDOWS: rectangular, hann, hamming, blackman, blackmanharris, bartlett, or flattop. Default: 'rectangular'
    """
    tstar = np.asarray(tstar, dtype=np.float64)
    data = np.asarray(data)
    if not variable_names or data.shape != (len(tstar), len(variable_names)):
        raise ValueError("FFT data must have one aligned column per variable name.")
    selected_names = select_plot_variables(variable_names, plot_variables, plot_cylinders)
    if selected_names != list(variable_names):
        data = data[:, [variable_names.index(name) for name in selected_names]]
    variable_names = selected_names
    analysis = compute_frequency_analysis(tstar, data, variable_names, fft_window=fft_window)
    frequencies, magnitude, phase = analysis.frequencies, analysis.amplitudes, analysis.phases
    peak_indices = analysis.peak_indices
    frequency_limits = (0, min(1., frequencies[-1])) if xlim is None else tuple(xlim)
    if (len(frequency_limits) != 2 or not np.all(np.isfinite(frequency_limits))
            or not 0 <= frequency_limits[0] < frequency_limits[1]):
        raise ValueError("Frequency limits must be finite, increasing, and nonnegative.")
    if frequency_limits[0] >= frequencies[-1]:
        raise ValueError(f"Frequency limits start at or above the Nyquist frequency {frequencies[-1]:g}.")

    # Crop plotted bins so frequency zooming also rescales the amplitude axis.
    visible = slice(
        max(0, np.searchsorted(frequencies, frequency_limits[0]) - 1),
        min(len(frequencies), np.searchsorted(frequencies, frequency_limits[1], side='right') + 1),
    )

    figure, axes = create_subplots(
        len(variable_names), 2, figsize=(6, 3 * len(variable_names) + 1),
        sharex=True, squeeze=False,
    )

    try:
        figure.suptitle("Frequency Analysis" + ("\n" + title if title else ""), fontsize=14)
        for column, name in enumerate(variable_names):
            magnitude_ax, phase_ax = axes[column]
            peak = peak_indices[column]
            magnitude_ax.plot(frequencies[visible], magnitude[visible, column], linewidth=LINE_WIDTH)
            magnitude_ax.margins(y=.2)
            phase_ax.plot(frequencies[visible], phase[visible, column], linewidth=LINE_WIDTH)
            magnitude_ax.set_ylabel(rf"$|{name}|$")
            phase_ax.set_ylabel(rf"$\phi_{{{name}}}$")
            for ax in (magnitude_ax, phase_ax):
                ax.set_xlim(frequency_limits)

                # A square panel preserves independent frequency and amplitude/phase scales.
                ax.set_box_aspect(1)
            phase_ax.set_ylim(-np.pi, np.pi)
            phase_ax.set_yticks(
                [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                labels=[r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"],
            )
            if peak < 0:
                magnitude_ax.text(.95, .95, 'No resolved oscillation', transform=magnitude_ax.transAxes,
                                  ha='right', va='top', fontsize=10)
                phase_ax.text(.95, .95, 'Phase undefined', transform=phase_ax.transAxes,
                              ha='right', va='top', fontsize=10)
                continue
            magnitude_ax.text(
                .95, .95, rf"$f^*_\mathrm{{max}}={frequencies[peak]:.4f}$",
                transform=magnitude_ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=.8, edgecolor='none'),
            )
            phase_ax.text(
                .95, .95, rf"$\phi(f^*_\mathrm{{max}})={phase[peak, column] / np.pi:.4f}\pi$",
                transform=phase_ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=.8, edgecolor='none'),
            )

            # An out-of-view peak must not expand the zoomed amplitude limits.
            if frequency_limits[0] <= frequencies[peak] <= frequency_limits[1]:
                magnitude_ax.plot(frequencies[peak], magnitude[peak, column], 'o', color='red', markersize=3)
                phase_ax.plot(frequencies[peak], phase[peak, column], 'o', color='red', markersize=3)

        axes[0, 0].set_title("Amplitude")
        axes[0, 1].set_title("Phase")

        for ax in axes[-1]:
            ax.set_xlabel(r"$f^* = fD/U_\infty$")

    except Exception:
        plt.close(figure)
        raise
    return figure, axes


def plot_frequency_comparison(
    prediction: FrequencyAnalysis, reference: FrequencyAnalysis | None = None, *, title: str = '', xlim: Sequence[float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot precomputed model/reference spectra without recomputing frequencies."""
    frequencies = prediction.frequencies
    limits = (0., min(1., frequencies[-1])) if xlim is None else tuple(xlim)
    if len(limits) != 2 or not np.all(np.isfinite(limits)) or not 0 <= limits[0] < limits[1]:
        raise ValueError('Frequency plot limits must be finite, increasing and nonnegative.')
    visible = (frequencies >= limits[0]) & (frequencies <= limits[1])
    figure, axes = create_subplots(len(prediction.variable_names), 2,
                                  figsize=(8, 3*len(prediction.variable_names)), squeeze=False)
    try:
        for j, name in enumerate(prediction.variable_names):
            for result, label in ((reference, 'Reference'), (prediction, 'SINDy')):
                if result is None:
                    continue
                axes[j, 0].plot(frequencies[visible], result.amplitudes[visible, j], label=label)
                axes[j, 1].plot(frequencies[visible], result.phases[visible, j], label=label)
            axes[j, 0].set_ylabel(f'|{name}|')
            axes[j, 1].set_ylabel(f'phase({name}) [rad]')
            axes[j, 1].set_ylim(-np.pi, np.pi)
            for ax in axes[j]:
                ax.set_xlim(limits)
                ax.set_xlabel('f* [cycles/t*]')
                ax.legend(frameon=False)
        figure.suptitle('Frequency assessment' + ('\n'+title if title else ''))
        return figure, axes
    except Exception:
        plt.close(figure)
        raise
