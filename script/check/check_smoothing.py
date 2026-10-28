"""Check how smoothing measured lift affects a prescribed structural simulation.

Recorded h, hdot, hddot, and CL are loaded for single or tandem cylinders. The model is
dh/dtau = hdot and dhdot/dtau = -4*pi**2*h + 2*VR**2/(pi*mstar)*CL. Raw and
smoothed lift drive separate simulations with identical measured initial states.
Both start at the first selected sample. Time bounds,
plots, and FFT frequencies use fluid time t*, while all integrators use tau=t*/VR.
Choose SciPy's adaptive DOP853/RK45 or
leapfrog_euler_velocity/leapfrog_trapezoid_velocity. Both LBM variants use the
leapfrog position update with their named velocity update.
They evaluate acceleration from the previous
state and lift sample, then use the same position update. They use the recorded
uniform tau step, which may differ from the LBM internal step.
The trapezoidal velocity update retains acceleration from the sample immediately
before the selected interval, using zero when no preceding record is available.

Edit the filter settings below to switch between Savitzky-Golay and zero-phase
Butterworth smoothing. Displacement, velocity, and acceleration from both simulations
are compared with the recorded channels using the shared NRMSE metric,
RMSE/std(reference channel), as a dimensionless ratio (1 = 100%). Figures are displayed only.
A separate displacement figure compares integration of recorded hddot[n] and
hddot[n-1], using the selected method and identical measured initial h/hdot.
No figures, metrics, or simulation arrays are saved.

Recorded hddot[n] is computed from h[n-1] and CL[n-1] in the source data.
Define a[n-1] = recorded hddot[n], equivalently a[n] = recorded hddot[n+1].
The a view aligns with time[:-1]; its final timestamp has no available successor.
Before simulation, retain only the NRMSE check of that relation on target rows
with a predecessor, including the pre-window record for cropped intervals.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.signal.windows import get_window

from sindy_viv.case_bryan.analysis.frequency import FFT_WINDOWS
from sindy_viv.case_bryan.analysis.integration import integrate_acceleration as integrate_recorded_acceleration
from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_MSTAR, SINGLE_CYLINDER_MSTAR_TRUE, SINGLE_CYLINDER_RE,
    TWO_TANDEM_CYLINDERS_MSTAR, TWO_TANDEM_CYLINDERS_MSTAR_TRUE, TWO_TANDEM_CYLINDERS_RE,
)
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.plot import create_subplots, format_parameter_title, LINE_WIDTH, PLOT_COLORS, select_plot_variables
from sindy_viv.metrics.numerical import normalized_root_mean_squared_error


# Modify filter settings here; they intentionally have no CLI arguments.
FILTER_TYPE = 'savitzky_golay'  # Use 'savitzky_golay' or 'butterworth'.
SAVGOL_WINDOW_LENGTH = 867  # Modify this sample-window length to adjust the smoothing.
SAVGOL_POLYORDER = 3
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF_FSTAR = .5  # Cutoff frequency in f* = f D / U_inf.



def parse_args() -> argparse.Namespace:
    """Parse dataset selections and t* bounds; filter settings remain in the code."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    # Physical case and dataset selection
    parser.add_argument(
        '--cylinder_type',
        choices=('single', 'two_tandem'),
        default='single',
        help='Dataset configuration. Default: single.',
    )
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=None,
        help="VR datasets; defaults to the selected cylinder configuration's available cases.",
    )
    parser.add_argument(
        '--lpd_list',
        type=float,
        nargs='+',
        default=None,
        help='L/D datasets for two_tandem; defaults to all available spacings.',
    )

    # Analysis interval
    parser.add_argument(
        '--tstar0',
        type=float,
        default=None,
        help='Start of the simulation/comparison interval in t*. Default: first sample.',
    )
    parser.add_argument(
        '--tstarf',
        type=float,
        default=None,
        help='End of the simulation/comparison interval in t*. Default: last sample.',
    )

    # Frequency analysis settings
    parser.add_argument(
        '--fft_window',
        choices=FFT_WINDOWS,
        default='rectangular',
        help=(
            'Window for the lift spectra; independent of the smoothing filter. Default: '
            'rectangular.'
        ),
    )

    # Numerical integration settings
    parser.add_argument(
        '--integration_method',
        choices=('DOP853', 'RK45', 'leapfrog_euler_velocity', 'leapfrog_trapezoid_velocity'),
        default='DOP853',
        help=(
            "SciPy's adaptive DOP853/RK45, or the LBM leapfrog position update with Euler "
            'velocity (leapfrog_euler_velocity) or trapezoidal velocity '
            '(leapfrog_trapezoid_velocity). Default: DOP853.'
        ),
    )

    # Plot settings
    parser.add_argument(
        '--plot_variables',
        nargs='+',
        default=None,
        help=(
            'Variables h, hdot, hddot, and/or CL, or exact channel names; CL includes its time '
            'and FFT comparisons. Default: all.'
        ),
    )
    parser.add_argument(
        '--plot_cylinders',
        type=int,
        nargs='+',
        choices=(0, 1),
        default=None,
        help='Cylinder IDs: 0 for single/upstream, 1 for downstream. Default: all available.',
    )
    args = parser.parse_args()
    if args.cylinder_type == 'single' and args.lpd_list is not None:
        parser.error('--lpd_list requires --cylinder_type two_tandem.')
    for name in ('vr_list', 'lpd_list'):
        values = getattr(args, name)
        if values is not None and (not np.all(np.isfinite(values)) or any(value <= 0 for value in values)):
            parser.error(f'--{name} requires finite positive values.')
    return args


def smooth_lift(tstar: np.ndarray, lift: np.ndarray) -> np.ndarray:
    """Smooth a full uniformly sampled lift record without differentiating it.

    Parameters:
        tstar (numpy.ndarray): Finite, uniformly increasing fluid-time samples.
        lift (numpy.ndarray): Finite lift coefficient samples aligned with tstar.
    """
    tstar = np.asarray(tstar, dtype=np.float64)
    lift = np.asarray(lift, dtype=np.float64)
    if tstar.ndim != 1 or len(tstar) < 2 or lift.shape != tstar.shape:
        raise ValueError('Smoothing requires aligned one-dimensional time and lift arrays with at least two samples.')
    if not np.all(np.isfinite(tstar)) or not np.all(np.isfinite(lift)):
        raise ValueError('Time and lift samples must be finite.')
    time_step = tstar[1] - tstar[0]
    if time_step <= 0 or not np.allclose(np.diff(tstar), time_step, rtol=1e-5, atol=time_step * 1e-4):
        raise ValueError('Smoothing and FFT require uniformly spaced, increasing t* samples.')

    match FILTER_TYPE:
        case 'savitzky_golay':
            if (SAVGOL_WINDOW_LENGTH <= SAVGOL_POLYORDER or SAVGOL_POLYORDER < 0
                    or SAVGOL_WINDOW_LENGTH % 2 != 1 or SAVGOL_WINDOW_LENGTH > len(lift)):
                raise ValueError('Savitzky-Golay needs an odd window no longer than the record and 0 <= polyorder < window_length.')
            return savgol_filter(lift, window_length=SAVGOL_WINDOW_LENGTH, polyorder=SAVGOL_POLYORDER, mode='interp')
        case 'butterworth':
            if not np.isfinite(BUTTERWORTH_CUTOFF_FSTAR) or not 0 < BUTTERWORTH_CUTOFF_FSTAR < .5 / time_step:
                raise ValueError('The Butterworth cutoff must be positive and below the Nyquist frequency in f*.')

            # Forward/backward filtering avoids adding an artificial lift-to-motion phase lag.
            sections = butter(BUTTERWORTH_ORDER, BUTTERWORTH_CUTOFF_FSTAR, btype='lowpass', fs=1 / time_step, output='sos')
            return sosfiltfilt(sections, lift)
        case _:
            raise ValueError("FILTER_TYPE must be 'savitzky_golay' or 'butterworth'.")


def simulate_response(
    time: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    lift: np.ndarray,
    vr: float,
    mstar: float,
    integration_method: str = 'DOP853',
    *,
    previous_acceleration: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the prescribed structural equation using measured initial states.

    Both sampled methods follow the supplied LBM convention in Eqs. (5.4)-(5.7):
        a[n] = -4*pi**2*h[n-1] + 2*VR**2/(pi*mstar)*CL[n-1]
        h[n] = h[n-1] + dt*v[n-1] + .5*dt**2*a[n]
    leapfrog_euler_velocity uses v[n] = v[n-1] + dt*a[n].
    leapfrog_trapezoid_velocity instead uses
    v[n] = v[n-1] + .5*dt*(a[n] + a[n-1]). The entire acceleration RHS,
    including prescribed lift, is evaluated at the previous sample. The first
    velocity update uses previous_acceleration from the record immediately before
    time[0], rather than repeating the acceleration computed from index 0. Use
    zero as a startup convention when no preceding record is available.
    These LBM methods require a uniform recorded tau grid. SciPy's DOP853 and
    RK45 use adaptive internal steps with interpolated lift.
    No further time conversion is applied.

    Return (h, hdot, hddot) as aligned float64 arrays. Velocity comes directly
    from the integrator. Acceleration is the structural RHS at each returned h
    and matching CL sample; in the LBM variants this drives the next step.
    It is not the pre-window previous_acceleration or a numerical derivative.

    Parameters:
        time (numpy.ndarray): Finite strictly increasing dynamic time tau=t*/VR.
        displacement (numpy.ndarray): Measured h aligned with time; its first value initializes the model.
        velocity (numpy.ndarray): Measured dh/dtau aligned with time; its first value initializes the model.
        lift (numpy.ndarray): Prescribed raw or filtered CL aligned with time.
        vr (float): Positive reduced velocity for this trajectory.
        mstar (float): Positive mass ratio used in the 2*VR**2/(pi*mstar) lift coefficient.
        integration_method (str): DOP853, RK45, leapfrog_euler_velocity, or leapfrog_trapezoid_velocity; both LBM options share the leapfrog position update. Default: 'DOP853'
        previous_acceleration (float): Acceleration evaluated from h and the chosen lift at the sample immediately before time[0]; used only by leapfrog_trapezoid_velocity. Default: 0.0

    References:
        Supplied LBM documentation, equations (5.4)-(5.7).
    """
    time = np.asarray(time, dtype=np.float64)
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
            or not np.all(np.diff(time) > 0)):
        raise ValueError('Simulation requires at least two finite, increasing tau samples.')
    for values in (displacement, velocity, lift):
        if np.shape(values) != time.shape or not np.all(np.isfinite(values)):
            raise ValueError('Displacement, velocity, and lift must be finite and aligned with time.')
    if not np.isfinite(vr) or vr <= 0 or not np.isfinite(mstar) or mstar <= 0:
        raise ValueError('VR and mstar must be finite and positive.')
    if integration_method not in ('DOP853', 'RK45', 'leapfrog_euler_velocity', 'leapfrog_trapezoid_velocity'):
        raise ValueError("integration_method must be 'DOP853', 'RK45', 'leapfrog_euler_velocity', or 'leapfrog_trapezoid_velocity'.")
    if np.ndim(previous_acceleration) != 0 or not np.isfinite(previous_acceleration):
        raise ValueError('previous_acceleration must be a finite scalar.')
    coupling = 2 * vr**2 / (np.pi * mstar)
    stiffness = 4 * np.pi**2

    if integration_method in ('leapfrog_euler_velocity', 'leapfrog_trapezoid_velocity'):
        time_step = (time[-1] - time[0]) / (len(time) - 1)
        if not np.allclose(np.diff(time), time_step, rtol=1e-5, atol=time_step * 1e-4):
            raise ValueError('Both LBM leapfrog variants require a uniformly sampled tau grid.')
        simulated_h = np.empty(len(time), dtype=np.float64)
        simulated_h[0] = displacement[0] # (n-1)
        simulated_velocity = np.empty(len(time), dtype=np.float64)
        simulated_velocity[0] = velocity[0] # (n-1)
        simulated_acceleration = np.empty(len(time), dtype=np.float64)
        current_velocity = float(velocity[0]) # (n-1)
        for index in range(len(time) - 1):

            # Eq. (5.4) labels this a[n], although it uses the previous state and lift sample.
            # acceleration = -stiffness * simulated_h[index] + coupling * lift[index]
            acceleration = -stiffness * simulated_h[index] + coupling * lift[index] # (n-1)
            simulated_acceleration[index] = acceleration # (n-1)
            simulated_h[index + 1] = (simulated_h[index] + time_step * current_velocity
                                      + .5 * time_step**2 * acceleration) # (n)
            match integration_method:
                case 'leapfrog_euler_velocity':
                    current_velocity += time_step * acceleration # (n)
                case 'leapfrog_trapezoid_velocity':

                    # Eq. (5.7) averages the stored accelerations only for the velocity update.
                    current_velocity += .5 * time_step * (acceleration + previous_acceleration)
            simulated_velocity[index + 1] = current_velocity
            previous_acceleration = acceleration

        # Report the final RHS at the final sample, without shifting the acceleration time axis.
        simulated_acceleration[-1] = -stiffness * simulated_h[-1] + coupling * lift[-1]
        if not all(np.all(np.isfinite(values)) for values in (simulated_h, simulated_velocity, simulated_acceleration)):
            raise RuntimeError(f'{integration_method} produced nonfinite response values.')
        return simulated_h, simulated_velocity, simulated_acceleration

    # Avoid repeated copies of strided CSV columns during adaptive force interpolation.
    lift = np.ascontiguousarray(lift, dtype=np.float64)

    def structural_rhs(tau: float, state: np.ndarray) -> list[float]:
        """Evaluate the structural response with lift interpolated at the solver's tau."""
        h, hdot = state
        lift_value = np.interp(tau, time, lift)
        return [hdot, -stiffness * h + coupling * lift_value]

    # Loaded velocities already use tau; neither time nor velocity is divided by VR again.
    solution = solve_ivp(
        structural_rhs, (time[0], time[-1]), [displacement[0], velocity[0]],
        t_eval=time, method=integration_method, rtol=1e-7, atol=1e-9,

        # Resolve the unit natural period with at least twenty integration steps.
        max_step=.05,
    )
    if not solution.success or len(solution.t) != len(time) or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f'Structural integration did not cover the selected interval: {solution.message}')
    simulated_h, simulated_velocity = solution.y
    simulated_acceleration = -stiffness * simulated_h + coupling * lift
    if not np.all(np.isfinite(simulated_acceleration)):
        raise RuntimeError(f'{integration_method} produced nonfinite acceleration values.')
    return simulated_h, simulated_velocity, simulated_acceleration


def main() -> None:
    """Compare simulated h, hdot, hddot, and lift with recordings without saving artifacts."""
    args = parse_args()
    if args.cylinder_type == 'single':
        re, mstar = SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR_TRUE
        mstar_data = SINGLE_CYLINDER_MSTAR
        vr_list = [3, 4, 5, 6, 7, 8] if args.vr_list is None else args.vr_list
        lpd_list, available_cylinders = [None], (0,)
    else:
        re, mstar = TWO_TANDEM_CYLINDERS_RE, TWO_TANDEM_CYLINDERS_MSTAR_TRUE
        mstar_data = TWO_TANDEM_CYLINDERS_MSTAR
        vr_list = list(range(5, 15)) if args.vr_list is None else args.vr_list
        lpd_list = [1.5, 2., 3., 3.5, 4.] if args.lpd_list is None else args.lpd_list
        available_cylinders = (0, 1)
    available_names = [name for cylinder in available_cylinders
                       for name in (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', rf'\ddot{{h}}_{cylinder}', f'C_{{L{cylinder}}}')]
    plot_names = select_plot_variables(available_names, args.plot_variables, args.plot_cylinders)
    cylinders = [cylinder for cylinder in available_cylinders
                 if any(name in plot_names for name in (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}',
                                                       rf'\ddot{{h}}_{cylinder}', f'C_{{L{cylinder}}}'))]
    variable_names = [name for cylinder in cylinders
                      for name in (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', rf'\ddot{{h}}_{cylinder}', f'C_{{L{cylinder}}}')]

    for lpd in lpd_list:
        for vr in vr_list:
            if args.cylinder_type == 'single':
                time, data = load_data_single_cylinder(re, mstar_data, vr, variables=variable_names)
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar_data, vr, lpd, variables=variable_names)
            region = select_time_region(time, vr, args.tstar0, args.tstarf)
            tstar = time * vr
            selected_time, selected_tstar = time[region], tstar[region]
            for cylinder in cylinders:
                h_name, velocity_name, lift_name = f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', f'C_{{L{cylinder}}}'
                acceleration_name = rf'\ddot{{h}}_{cylinder}'

                # Keep the original hddot channel and expose its left-shifted a view without circular padding.
                a = data[1:, variable_names.index(acceleration_name)]
                h = data[region, variable_names.index(h_name)]
                velocity = data[region, variable_names.index(velocity_name)]
                acceleration = data[region, variable_names.index(acceleration_name)]
                lift = data[:, variable_names.index(lift_name)]

                description = f'{args.cylinder_type}, VR={vr}, cylinder={cylinder}'
                if lpd is not None:
                    description += f', L/D={lpd}'
                figure_parameters = {'V_R': vr}
                if lpd is not None:
                    figure_parameters['L/D'] = lpd
                figure_description = f'{args.cylinder_type}, cylinder={cylinder}, '+format_parameter_title(figure_parameters)

                # Validate the established n-1 relation without inventing or wrapping a missing predecessor.
                comparison_start = max(1, region.start)
                if region.stop - comparison_start >= 2:
                    previous_rows = slice(comparison_start - 1, region.stop - 1)
                    equation_acceleration = (-4 * np.pi**2 * data[previous_rows, variable_names.index(h_name)]
                                             + 2 * vr**2 / (np.pi * mstar) * lift[previous_rows])
                    target_acceleration = a[previous_rows]
                    if not np.all(np.isfinite(equation_acceleration)) or not np.all(np.isfinite(target_acceleration)):
                        raise ValueError('Recorded acceleration and preceding h/CL samples must be finite.')
                    relation_error = normalized_root_mean_squared_error(equation_acceleration, target_acceleration)
                    print(f'{description}, recorded acceleration relation:')
                    print(f'  Recorded hddot[n] vs F(h[n-1], CL[n-1]): NRMSE={relation_error:.8g} '
                          f'(N={region.stop - comparison_start})', flush=True)
                    del equation_acceleration, target_acceleration
                else:
                    print(f'{description}, recorded hddot[n] relation NRMSE unavailable: '
                          'fewer than two samples have a preceding record.', flush=True)

                # Filter before cropping so the selected boundaries retain their original neighbors.
                smoothed_lift = smooth_lift(tstar, lift)

                # Retain the preceding sample for each forcing history; never wrap to the record's last row.
                previous_acceleration_raw = previous_acceleration_smoothed = 0.0
                previous_recorded_acceleration = (data[region.start - 1, variable_names.index(acceleration_name)]
                                                  if region.start > 0 else 0.0)
                if args.integration_method == 'leapfrog_trapezoid_velocity' and region.start > 0:
                    previous_index = region.start - 1
                    previous_h = data[previous_index, variable_names.index(h_name)]
                    coupling = 2 * vr**2 / (np.pi * mstar)
                    previous_acceleration_raw = -4 * np.pi**2 * previous_h + coupling * lift[previous_index]
                    previous_acceleration_smoothed = -4 * np.pi**2 * previous_h + coupling * smoothed_lift[previous_index]
                smoothed_lift = smoothed_lift[region].copy()
                lift = lift[region]

                # Keep initial conditions and solver settings identical when comparing lift inputs.
                simulated_raw = simulate_response(selected_time, h, velocity, lift, vr, mstar,
                                                  integration_method=args.integration_method,
                                                  previous_acceleration=previous_acceleration_raw)
                simulated_smoothed = simulate_response(selected_time, h, velocity, smoothed_lift, vr, mstar,
                                                       integration_method=args.integration_method,
                                                       previous_acceleration=previous_acceleration_smoothed)

                # Reuse the shared reference-standard-deviation normalization and bounded metric buffers.
                response_names = (h_name, velocity_name, acceleration_name)
                reference_response = (h, velocity, acceleration)
                nrmse_raw = [normalized_root_mean_squared_error(predicted, reference)
                             for predicted, reference in zip(simulated_raw, reference_response)]
                nrmse_smoothed = [normalized_root_mean_squared_error(predicted, reference)
                                  for predicted, reference in zip(simulated_smoothed, reference_response)]
                print(f'{description}, filter={FILTER_TYPE}, integrator={args.integration_method}, '
                      f't*=[{selected_tstar[0]:.6g}, {selected_tstar[-1]:.6g}]:')
                for name, raw_error, smoothed_error in zip(response_names, nrmse_raw, nrmse_smoothed):
                    print(f'  {name}: raw CL NRMSE={raw_error:.8g}; smoothed CL NRMSE={smoothed_error:.8g}')

                show_lift = lift_name in plot_names
                row_count = sum(name in plot_names for name in response_names) + 2 * int(show_lift)
                figure, axes = create_subplots(row_count, sharex=False, squeeze=False, figsize=(6, 3 * row_count + 1))
                ax = None
                try:
                    figure.suptitle(f'Lift Smoothing Check ({FILTER_TYPE}, {args.integration_method})\n{figure_description}', fontsize=12)
                    row = 0
                    for column, (name, title) in enumerate(zip(response_names, ('Displacement', 'Velocity', 'Acceleration'))):
                        if name not in plot_names:
                            continue
                        ax = axes[row, 0]
                        ax.plot(selected_tstar, reference_response[column], color=PLOT_COLORS[0], linewidth=LINE_WIDTH, label='Reference (raw data)')
                        ax.plot(selected_tstar, simulated_raw[column], color=PLOT_COLORS[2], linewidth=LINE_WIDTH,
                                linestyle='-', label='Simulated (raw CL)')
                        ax.plot(selected_tstar, simulated_smoothed[column], color=PLOT_COLORS[1], linewidth=LINE_WIDTH,
                                label='Simulated (smoothed CL)')
                        ax.set_ylabel(rf'${name}$')
                        ax.set_xlabel(r'$t^*$')
                        ax.set_xlim(selected_tstar[0], selected_tstar[-1])
                        ax.set_title(title)
                        error_text = (f'Raw CL: NRMSE={nrmse_raw[column]:.4g}\n'
                                      f'Smoothed CL: NRMSE={nrmse_smoothed[column]:.4g}')
                        ax.text(.98, .95, error_text, transform=ax.transAxes,
                                ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                        ax.legend(frameon=False, fontsize=9, loc='lower right')
                        row += 1
                    if show_lift:
                        ax = axes[row, 0]
                        ax.plot(selected_tstar, lift, color=PLOT_COLORS[0], linewidth=LINE_WIDTH, label='Reference')
                        ax.plot(selected_tstar, smoothed_lift, color=PLOT_COLORS[1], linewidth=LINE_WIDTH, label='Smoothed')
                        ax.set_ylabel(rf'${lift_name}$')
                        ax.set_xlabel(r'$t^*$')
                        ax.set_xlim(selected_tstar[0], selected_tstar[-1])
                        ax.set_title('Lift coefficient')
                        ax.legend(frameon=False, fontsize=9)

                        # Compare both records on identical bins with the same coherent-gain correction.
                        fft_data = np.column_stack((lift, smoothed_lift))
                        fft_data -= fft_data.mean(axis=0)
                        weights = get_window('boxcar' if args.fft_window == 'rectangular' else args.fft_window,
                                             len(selected_tstar), fftbins=True)
                        fft_data *= weights[:, None]
                        magnitude = np.abs(np.fft.rfft(fft_data, axis=0)) / weights.sum()
                        if len(selected_tstar) % 2 == 0:
                            magnitude[1:-1] *= 2
                        else:
                            magnitude[1:] *= 2
                        frequencies = np.fft.rfftfreq(len(selected_tstar), d=selected_tstar[1] - selected_tstar[0])
                        frequency_limit = min(1., frequencies[-1])
                        visible = slice(0, min(len(frequencies), np.searchsorted(frequencies, frequency_limit, side='right') + 1))
                        ax = axes[row + 1, 0]
                        for column, label in enumerate(('Reference', 'Smoothed')):
                            ax.plot(frequencies[visible], magnitude[visible, column], color=PLOT_COLORS[column],
                                    linewidth=LINE_WIDTH, label=label)
                        ax.set_ylabel(rf'$|{lift_name}|$')
                        ax.set_xlabel(r'$f^* = fD/U_\infty$')
                        ax.set_xlim(0, frequency_limit)
                        ax.set_box_aspect(1)
                        ax.set_title('Amplitude')
                        ax.legend(frameon=False, fontsize=9)
                        del fft_data, weights, magnitude, frequencies
                    plt.show()
                finally:
                    plt.close(figure)

                    # Release the lift-driven simulation buffers before the separate timing reconstruction.
                    del figure, axes, ax
                    del reference_response, simulated_raw, simulated_smoothed

                if h_name in plot_names:

                    # Keep both acceleration timing hypotheses independent of the CL-based structural RHS.
                    reconstructed_current = integrate_recorded_acceleration(
                        selected_time, acceleration, h[0], velocity[0], integration_method=args.integration_method,
                        previous_acceleration=previous_recorded_acceleration, acceleration_timing='current',
                    )
                    reconstructed_previous = integrate_recorded_acceleration(
                        selected_time, acceleration, h[0], velocity[0], integration_method=args.integration_method,
                        previous_acceleration=previous_recorded_acceleration, acceleration_timing='previous',
                    )
                    nrmse_current_h = normalized_root_mean_squared_error(reconstructed_current, h)
                    nrmse_previous_h = normalized_root_mean_squared_error(reconstructed_previous, h)
                    print(f'{description}, recorded acceleration integration ({args.integration_method}):')
                    print(f'  {h_name} using hddot[n]: NRMSE={nrmse_current_h:.8g}')
                    print(f'  {h_name} using hddot[n-1]: NRMSE={nrmse_previous_h:.8g}', flush=True)
                    figure, ax = create_subplots(1, figsize=(6, 4))
                    try:
                        figure.suptitle(f'Recorded Acceleration Integration ({args.integration_method})\n{figure_description}', fontsize=12)
                        ax.plot(selected_tstar, h, color='black', linewidth=LINE_WIDTH, label='Reference (raw data)')
                        ax.plot(selected_tstar, reconstructed_current, color='red', linewidth=LINE_WIDTH,
                                label='Integrated (hddot[n])')
                        ax.plot(selected_tstar, reconstructed_previous, color='blue', linewidth=LINE_WIDTH,
                                label='Integrated (hddot[n-1])')
                        ax.set_title('Displacement')
                        ax.set_ylabel(rf'${h_name}$')
                        ax.set_xlabel(r'$t^*$')
                        ax.set_xlim(selected_tstar[0], selected_tstar[-1])
                        ax.text(.98, .95,
                                f'hddot[n]: NRMSE={nrmse_current_h:.4g}\n'
                                f'hddot[n-1]: NRMSE={nrmse_previous_h:.4g}', transform=ax.transAxes,
                                ha='right', va='top', fontsize=9, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
                        ax.legend(frameon=False, fontsize=9, loc='lower right')
                        plt.show()
                    finally:
                        plt.close(figure)
                        del figure, ax, reconstructed_current, reconstructed_previous
                del h, velocity, acceleration, lift, smoothed_lift, a
            del time, data, tstar, selected_time, selected_tstar


if __name__ == '__main__':
    main()
