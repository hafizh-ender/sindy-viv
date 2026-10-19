"""Experimental state, frequency, energy-transfer, and relative-phase plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_CASE_NAME,
    SINGLE_CYLINDER_MSTAR,
    SINGLE_CYLINDER_RE,
    SINGLE_CYLINDER_STATE_VARIABLES,
)
from sindy_viv.case_bryan.input import load_data_single_cylinder
from sindy_viv.case_bryan.plot import create_subplots, LINE_WIDTH
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region

def parse_args() -> argparse.Namespace:
    """Parse and validate this command line; --help describes available settings."""
    argument_parser = argparse.ArgumentParser(description="Plot states, frequency spectra, structural energy transfer, and phase differences for Bryan's single-cylinder data.")
    argument_parser.add_argument(
        "--vr_list",
        type=int,
        nargs='+',
        default=[3, 4, 5, 6, 7, 8],
        help="List of VR cases to analyze."
    )
    argument_parser.add_argument(
        "--tstar0",
        type=float,
        default=None,
        help="Start time for the analysis in t*."
    )
    argument_parser.add_argument(
        "--tstarf",
        type=float,
        default=None,
        help="End time for the analysis in t*. If None, use the last time point."
    )
    argument_parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        help="Frequency-axis limits in f* = f D / U_inf; defaults to 0 through 1 (up to Nyquist)."
    )
    args = argument_parser.parse_args()
    if args.xlim is not None:
        if not np.all(np.isfinite(args.xlim)) or not 0 <= args.xlim[0] < args.xlim[1]:
            argument_parser.error("--xlim requires finite increasing nonnegative frequency limits.")
    return args

def main() -> None:

    # Load the arguments
    """Run this experimental analysis/plot command."""
    args = parse_args()
    re = SINGLE_CYLINDER_RE
    mstar = SINGLE_CYLINDER_MSTAR

    # Analyze each VR separately so FFT arrays and figures are released between cases.
    for vr in args.vr_list:

        # Load data for current VR
        time, data = load_data_single_cylinder(re, mstar, vr)

        # Construct the region
        region = select_time_region(time, vr, args.tstar0, args.tstarf)

        # Detach the selected window so FFT buffers do not retain complete source trajectories.
        time = time[region].copy()
        data = np.array(data[region, :], dtype=np.float64, copy=True)

        # The loader returns tau; using t* here gives frequencies f* = f D / U_inf.
        tstar = time * vr
        sample_count = len(tstar)

        # Ensure we have enough samples to even compute a step
        if sample_count < 2:
            raise ValueError(f"VR={vr}: Not enough data points in the selected time region.")

        time_step = (tstar[1] - tstar[0])

        # Allow roundoff in a reconstructed uniform time axis relative to its sample spacing.
        if not np.allclose(np.diff(tstar), time_step, rtol=1e-5, atol=time_step * 1e-4):
            raise ValueError(f"VR={vr}: FFT requires uniformly spaced time samples.")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"VR={vr}: FFT requires finite data values.")

        # Center only the FFT input; rfft retains the complex coefficients needed for phase.
        fft_data = data - np.mean(data, axis=0)
        spectrum = np.fft.rfft(fft_data, axis=0)
        frequencies = np.fft.rfftfreq(sample_count, d=time_step)

        # Normalize by the sample count before converting to single-sided amplitudes.
        magnitude = np.abs(spectrum) / sample_count
        phase = np.angle(spectrum)

        # Double paired positive-frequency amplitudes, excluding DC and an even-length Nyquist bin.
        if sample_count % 2 == 0:
            magnitude[1:-1] *= 2
        else:
            magnitude[1:] *= 2

        # Exclude DC and the even-length Nyquist bin unless it is the only positive bin.
        peak_stop = len(frequencies) - 1 if sample_count % 2 == 0 and len(frequencies) > 2 else len(frequencies)
        max_freq_indices = np.argmax(magnitude[1:peak_stop], axis=0) + 1
        max_frequencies = frequencies[max_freq_indices]
        fNstar = 1 / vr
        frequency_limits = (0, min(1., frequencies[-1])) if args.xlim is None else args.xlim
        if frequency_limits[0] > frequencies[-1]:
            raise ValueError(f"VR={vr}: --xlim starts above the Nyquist frequency {frequencies[-1]:g}.")

        # Crop plotted bins so frequency zooming also rescales the amplitude axis.
        visible = slice(
            max(0, np.searchsorted(frequencies, frequency_limits[0]) - 1),
            min(len(frequencies), np.searchsorted(frequencies, frequency_limits[1], side='right') + 1),
        )
        results_path = (
            Path("results") / SINGLE_CYLINDER_CASE_NAME / "figures"
            / f"Re = {re}" / f"Mstar = {mstar}" / f"VR = {vr}"
        )
        results_path.mkdir(parents=True, exist_ok=True)
        if args.tstar0 is None and args.tstarf is None:
            window = "full_range"
        else:
            window = f"{args.tstar0}_{args.tstarf}"
        frequency_filename = f"frequency_analysis_{window}.png"
        if args.xlim is not None:
            frequency_filename = f"frequency_analysis_{window}_frequency_{args.xlim[0]}_{args.xlim[1]}.png"

        title = (
            rf"$Re={re},\quad m^*={mstar},\quad V_R={vr}$"
            f"\n$t^* \\in [{tstar[0]:.3f}, {tstar[-1]:.3f}]$"
        )
        figures = []
        try:
            time_fig, time_axes = create_subplots(
                len(SINGLE_CYLINDER_STATE_VARIABLES), 1,
                figsize=(6, len(SINGLE_CYLINDER_STATE_VARIABLES) + 1),
                sharex=True, squeeze=False,
            )
            figures.append((time_fig, f"time_states_{window}.png"))
            time_fig.suptitle("State histories\n" + title, fontsize=14)
            for column, name in enumerate(SINGLE_CYLINDER_STATE_VARIABLES):
                ax = time_axes[column, 0]
                ax.plot(tstar, data[:, column], linewidth=LINE_WIDTH)
                ax.set_ylabel(rf"${name}$")
                ax.set_xlim(tstar[0], tstar[-1])
            time_axes[-1, 0].set_xlabel(r"$t^*$")

            fig, axs = create_subplots(
                len(SINGLE_CYLINDER_STATE_VARIABLES), 2,
                figsize=(12, len(SINGLE_CYLINDER_STATE_VARIABLES) + 1),
                sharex=True, squeeze=False,
            )
            figures.append((fig, frequency_filename))
            fig.suptitle("Frequency analysis\n" + title, fontsize=14)
            for column, name in enumerate(SINGLE_CYLINDER_STATE_VARIABLES):
                magnitude_ax, phase_ax = axs[column]
                magnitude_ax.plot(frequencies[visible], magnitude[visible, column], linewidth=LINE_WIDTH)
                phase_ax.plot(frequencies[visible], phase[visible, column], linewidth=LINE_WIDTH)
                magnitude_ax.set_ylabel(rf"${name}$")
                phase_ax.set_ylabel(rf"${name}$")
                phase_ax.set_ylim(-np.pi, np.pi)
                for ax in (magnitude_ax, phase_ax):
                    ax.set_xlim(frequency_limits)

                # Annotate the frequency at maximum amplitude for each state variable.
                magnitude_ax.text(
                    0.95, 0.95,
                    rf"$f^*_\mathrm{{max}}={max_frequencies[column]:.4f}$",
                    transform=magnitude_ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                )

                # Compare frequencies in the same units: f_N* = 1/VR, so f*/f_N* = VR*f*.
                magnitude_ax.text(
                    0.95, 0.05,
                    rf"$f^*_\mathrm{{max}}/f_N^*={max_frequencies[column] / fNstar:.4f}$",
                    transform=magnitude_ax.transAxes, fontsize=10,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                )

            axs[0, 0].set_title("Single-sided amplitude spectrum")
            axs[0, 1].set_title("Phase [rad] relative to the first selected sample")
            for ax in axs[-1]:
                ax.set_xlabel(r"$f^* = fD/U_\infty$")

            # CL*hdot is the instantaneous energy-transfer proxy, using uncentered states.
            lift_column = SINGLE_CYLINDER_STATE_VARIABLES.index('C_{L0}')
            velocity_column = SINGLE_CYLINDER_STATE_VARIABLES.index(r'\dot{h}_0')
            structural_energy = data[:, lift_column] * data[:, velocity_column]
            energy_fig, energy_ax = create_subplots(1, figsize=(6, 2))
            figures.append((energy_fig, f"structural_energy_{window}.png"))
            energy_fig.suptitle("Structural energy transfer\n" + title, fontsize=14)
            energy_ax.plot(tstar, structural_energy, linewidth=LINE_WIDTH)
            energy_ax.axhline(0, color='black', linewidth=.5, alpha=.5)
            energy_ax.set_xlabel(r"$t^*$")
            energy_ax.set_ylabel(r"$C_{L0}\dot{h}_0$")
            energy_ax.set_xlim(tstar[0], tstar[-1])

            comparison_names = ('h_0', r'\dot{h}_0', r'\ddot{h}_0')
            comparison_fig, comparison_axes = create_subplots(
                len(comparison_names), 1, figsize=(6, len(comparison_names) + 1),
                sharex=True, squeeze=False,
            )
            phase_filename = frequency_filename.replace("frequency_analysis_", "phase_comparison_", 1)
            figures.append((comparison_fig, phase_filename))
            comparison_fig.suptitle(
                "Phase relative to lift (positive = state leads lift)\n" + title, fontsize=14
            )
            for row, name in enumerate(comparison_names):
                state_column = SINGLE_CYLINDER_STATE_VARIABLES.index(name)

                # The cross-spectrum gives phi(state)-phi(CL), wrapped to [-pi, pi].
                relative_phase = np.angle(
                    spectrum[visible, state_column] * np.conj(spectrum[visible, lift_column])
                )

                # Centering removes DC; phase is also undefined when either FFT amplitude is zero.
                relative_phase[
                    (frequencies[visible] == 0)
                    | (magnitude[visible, state_column] == 0)
                    | (magnitude[visible, lift_column] == 0)
                ] = np.nan
                ax = comparison_axes[row, 0]
                ax.plot(frequencies[visible], relative_phase, linewidth=LINE_WIDTH)
                ax.axhline(0, color='black', linewidth=.5, alpha=.5)

                # The title defines the lift reference, keeping per-row labels short at paper size.
                ax.set_ylabel(rf"$\Delta\phi_{{{name}}}$" + "\n[rad]")
                ax.set_xlim(frequency_limits)
                ax.set_ylim(-np.pi, np.pi)

                # Label the wrapped phase range in multiples of pi while retaining radians.
                ax.set_yticks(
                    [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                    labels=[r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"],
                )
            comparison_axes[-1, 0].set_xlabel(r"$f^* = fD/U_\infty$")

            for figure, filename in figures:
                output_path = results_path / filename
                figure.savefig(output_path, dpi=300)
                print(f"Saved VR={vr} figure to {output_path}")

            # Display all four experimental figures together for the current VR.
            plt.show()
        finally:
            for figure, _ in figures:
                plt.close(figure)
        del time, data, tstar, fft_data, spectrum, frequencies, magnitude, phase, structural_energy

if __name__ == "__main__":
    main()
