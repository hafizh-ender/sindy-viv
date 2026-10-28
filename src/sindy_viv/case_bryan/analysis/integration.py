# """Integrate acceleration histories on the dynamic time coordinate tau."""

# import numpy as np
# from scipy.integrate import solve_ivp


# def integrate_acceleration(
#     time: np.ndarray,
#     acceleration: np.ndarray,
#     initial_h: float,
#     initial_velocity: float,
#     integration_method: str = 'DOP853',
#     *,
#     previous_acceleration: float = 0.0,
#     acceleration_timing: str = 'current',
# ) -> np.ndarray:
#     """Reconstruct h from sampled acceleration without evaluating the structural equation.

#     All time and derivative values use tau. For an LBM update from n-1 to n,
#     'current' uses recorded acceleration[n], while 'previous' uses acceleration[n-1].
#     Trapezoidal velocity averages consecutive values from that chosen sequence.
#     Adaptive solvers linearly interpolate either acceleration[n] or the record
#     shifted by one row to acceleration[n-1]. previous_acceleration supplies the
#     missing pre-window record for the latter, with zero as the startup convention.
#     No recorded h or hdot after the initial sample drives this reconstruction.

#     Parameters:
#         time (numpy.ndarray): Finite strictly increasing dynamic time tau=t*/VR.
#         acceleration (numpy.ndarray): Sampled d²h/dtau² aligned with time.
#         initial_h (float): Measured displacement at the selected interval's first sample.
#         initial_velocity (float): Measured dh/dtau at the selected interval's first sample.
#         integration_method (str): DOP853, RK45, leapfrog_euler_velocity, or leapfrog_trapezoid_velocity. Default: 'DOP853'
#         previous_acceleration (float): Recorded hddot immediately before time[0], used by the 'previous' timing option; zero when no preceding record is available. Default: 0.0
#         acceleration_timing (str): 'current' uses hddot[n]; 'previous' uses hddot[n-1]. Default: 'current'
#     """
#     time = np.asarray(time, dtype=np.float64)
#     acceleration = np.ascontiguousarray(acceleration, dtype=np.float64)
#     if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
#             or not np.all(np.diff(time) > 0)):
#         raise ValueError('Reconstruction requires at least two finite, increasing tau samples.')
#     if acceleration.shape != time.shape or not np.all(np.isfinite(acceleration)):
#         raise ValueError('Recorded acceleration must be finite and aligned with time.')
#     for value in (initial_h, initial_velocity, previous_acceleration):
#         if np.ndim(value) != 0 or not np.isfinite(value):
#             raise ValueError('Initial displacement, velocity, and previous acceleration must be finite scalars.')
#     if integration_method not in ('DOP853', 'RK45', 'leapfrog_euler_velocity', 'leapfrog_trapezoid_velocity'):
#         raise ValueError("integration_method must be 'DOP853', 'RK45', 'leapfrog_euler_velocity', or 'leapfrog_trapezoid_velocity'.")
#     if acceleration_timing not in ('current', 'previous'):
#         raise ValueError("acceleration_timing must be 'current' or 'previous'.")

#     if integration_method in ('leapfrog_euler_velocity', 'leapfrog_trapezoid_velocity'):
#         time_step = (time[-1] - time[0]) / (len(time) - 1)
#         if not np.allclose(np.diff(time), time_step, rtol=1e-5, atol=time_step * 1e-4):
#             raise ValueError('Both LBM leapfrog variants require a uniformly sampled tau grid.')
#         reconstructed_h = np.empty(len(time), dtype=np.float64)
#         reconstructed_h[0] = initial_h
#         current_velocity = float(initial_velocity)
#         if acceleration_timing == 'current':
#             previous_acceleration = acceleration[0]
#         for index in range(len(time) - 1):

#             # Here n is the destination sample index + 1, so the alternatives differ by one row.
#             current_acceleration = acceleration[index + 1] if acceleration_timing == 'current' else acceleration[index]
#             reconstructed_h[index + 1] = (reconstructed_h[index] + time_step * current_velocity
#                                            + .5 * time_step**2 * current_acceleration)
#             match integration_method:
#                 case 'leapfrog_euler_velocity':
#                     current_velocity += time_step * current_acceleration
#                 case 'leapfrog_trapezoid_velocity':
#                     current_velocity += .5 * time_step * (current_acceleration + previous_acceleration)
#             previous_acceleration = current_acceleration
#         if not np.all(np.isfinite(reconstructed_h)):
#             raise RuntimeError(f'{integration_method} produced nonfinite reconstructed displacement values.')
#         return reconstructed_h

#     if acceleration_timing == 'previous':

#         # Shift by recorded rows without wrapping the last value into the interval's first sample.
#         shifted_acceleration = np.empty_like(acceleration)
#         shifted_acceleration[0] = previous_acceleration
#         shifted_acceleration[1:] = acceleration[:-1]
#         acceleration = shifted_acceleration

#     def recorded_rhs(tau: float, state: np.ndarray) -> list[float]:
#         """Advance velocity using the recorded acceleration at the solver's tau."""
#         return [state[1], np.interp(tau, time, acceleration)]

#     # Use the same adaptive solver settings as the two lift-driven simulations.
#     solution = solve_ivp(
#         recorded_rhs, (time[0], time[-1]), [initial_h, initial_velocity],
#         t_eval=time, method=integration_method, rtol=1e-7, atol=1e-9, max_step=.05,
#     )
#     if not solution.success or len(solution.t) != len(time) or not np.all(np.isfinite(solution.y)):
#         raise RuntimeError(f'Recorded-acceleration integration did not cover the selected interval: {solution.message}')
#     return solution.y[0].copy()
