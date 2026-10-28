"""Compare derivatives of recorded h and hdot with recorded hdot and hddot.

Load h, hdot, hddot, and CL, then differentiate h and recorded hdot using a
first-derivative JSON configuration. The loader already returns float64 tau=t*/VR;
no further time or response rescaling is applied. Differentiate full records
before selecting optional t* reporting bounds, retaining the filter's neighbors.
Recorded hddot[n] is computed from h[n-1] and CL[n-1] in the source data.
Define a[n-1] = recorded hddot[n], equivalently a[n] = recorded hddot[n+1].
The a view aligns with time[:-1]; its final timestamp has no available successor.
Retain a single NRMSE check of that relation, using the pre-window record when
available and excluding the first dataset row when it has no predecessor.

Compare h reconstructed from recorded hddot with h reconstructed from the
numerical derivative of recorded hdot, using identical measured initial h/hdot.
The integration method is leapfrog_euler_velocity, RK45, or DOP853. Leapfrog uses
the acceleration at destination sample n for the update n-1 to n; adaptive
solvers interpolate each acceleration series at solver stages, without a shift.

Print NRMSE only (RMSE/std(reference), with 1 = 100%). Display the h comparison
unless --no_plot is supplied; no figures, metrics files, or arrays are saved.
Derivative diagnostics compare matching timestamps without shifting references.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sindy_viv.case_bryan.analysis.integration import integrate_acceleration
from sindy_viv.case_bryan.constants import (
    SINGLE_CYLINDER_MSTAR, SINGLE_CYLINDER_MSTAR_TRUE, SINGLE_CYLINDER_RE,
    TWO_TANDEM_CYLINDERS_MSTAR, TWO_TANDEM_CYLINDERS_MSTAR_TRUE, TWO_TANDEM_CYLINDERS_RE,
)
from sindy_viv.case_bryan.evaluate_sindy.model_evaluation import select_time_region
from sindy_viv.case_bryan.input import load_data_single_cylinder, load_data_two_tandem_cylinders
from sindy_viv.case_bryan.plot import create_subplots, format_parameter_title, LINE_WIDTH, select_plot_variables
from sindy_viv.differentiation import perform_derivative
from sindy_viv.metrics.numerical import normalized_root_mean_squared_error
from sindy_viv.utils import initiate_object, load_config


def parse_args() -> argparse.Namespace:
    """Parse dataset, first-derivative config, and optional t* reporting bounds."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    # Physical case and dataset selection
    parser.add_argument(
        '--cylinder_type',
        choices=('single', 'two_tandem'),
        default='single',
        help='Cylinder configuration. Default: single.',
    )
    parser.add_argument(
        '--vr_list',
        type=int,
        nargs='+',
        default=None,
        help='VR datasets; defaults to all available VR values for the selected configuration.',
    )
    parser.add_argument(
        '--lpd_list',
        type=float,
        nargs='+',
        default=None,
        help='L/D datasets for two_tandem; defaults to all available spacings.',
    )

    # Differentiation settings
    parser.add_argument(
        '--differentiator_case',
        type=Path,
        default=None,
        help=(
            'First-derivative JSON config. Default: '
            'configs/differentiator/savitzky_golay_<cylinder_type>.json.'
        ),
    )

    # Analysis interval
    parser.add_argument(
        '--tstar0',
        type=float,
        default=None,
        help='Start of the reporting interval in t*. Default: first sample.',
    )
    parser.add_argument(
        '--tstarf',
        type=float,
        default=None,
        help='End of the reporting interval in t*. Default: last sample.',
    )

    # Numerical integration settings
    parser.add_argument(
        '--integration_method',
        choices=('leapfrog_euler_velocity', 'RK45', 'DOP853'),
        default='DOP853',
        help=(
            'Method for both recorded and differentiated acceleration reconstructions. Default: '
            'DOP853.'
        ),
    )

    # Plot settings
    parser.add_argument(
        '--plot_cylinders',
        type=int,
        nargs='+',
        choices=(0, 1),
        default=None,
        help='Cylinder IDs to display; metrics still cover every cylinder. Default: all available.',
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Print NRMSE without displaying figures.',
    )
    args = parser.parse_args()
    if args.cylinder_type == 'single' and args.lpd_list is not None:
        parser.error('--lpd_list requires --cylinder_type two_tandem.')
    for name in ('vr_list', 'lpd_list'):
        values = getattr(args, name)
        if values is not None and (not np.all(np.isfinite(values)) or any(value <= 0 for value in values)):
            parser.error(f'--{name} requires finite positive values.')
    return args


def main() -> None:
    """Compare derivatives and h reconstructions for each selected trajectory and cylinder."""
    args = parse_args()
    if args.cylinder_type == 'single':
        re, mstar = SINGLE_CYLINDER_RE, SINGLE_CYLINDER_MSTAR
        dynamics_mstar = SINGLE_CYLINDER_MSTAR_TRUE
        vr_list = [3, 4, 5, 6, 7, 8] if args.vr_list is None else args.vr_list
        lpd_list, cylinders = [None], (0,)
    else:
        re, mstar = TWO_TANDEM_CYLINDERS_RE, TWO_TANDEM_CYLINDERS_MSTAR
        dynamics_mstar = TWO_TANDEM_CYLINDERS_MSTAR_TRUE
        vr_list = list(range(5, 15)) if args.vr_list is None else args.vr_list
        lpd_list = [1.5, 2., 3., 3.5, 4.] if args.lpd_list is None else args.lpd_list
        cylinders = (0, 1)
    plot_names = select_plot_variables([f'h_{cylinder}' for cylinder in cylinders], plot_cylinders=args.plot_cylinders)
    config_path = (Path('configs') / 'differentiator' / f'savitzky_golay_{args.cylinder_type}.json'
                   if args.differentiator_case is None else Path(args.differentiator_case))
    differentiator_config = load_config(config_path)
    variable_names = [name for cylinder in cylinders
                      for name in (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}', rf'\ddot{{h}}_{cylinder}', f'C_{{L{cylinder}}}')]

    for lpd in lpd_list:
        for vr in vr_list:
            if args.cylinder_type == 'single':
                time, data = load_data_single_cylinder(re, mstar, vr, variables=variable_names)
            else:
                time, data = load_data_two_tandem_cylinders(re, mstar, vr, lpd, variables=variable_names)
            region = select_time_region(time, vr, args.tstar0, args.tstarf)
            selected_time = time[region]
            for cylinder in cylinders:
                h_name, velocity_name, acceleration_name = (f'h_{cylinder}', rf'\dot{{h}}_{cylinder}',
                                                            rf'\ddot{{h}}_{cylinder}')
                cl_name = f'C_{{L{cylinder}}}'
                
                recorded_h = data[region, variable_names.index(h_name)]
                recorded_hdot = data[region, variable_names.index(velocity_name)]
                recorded_hddot = data[region, variable_names.index(acceleration_name)]
                recorded_cl = data[region, variable_names.index(cl_name)]
                
                shifted_time = selected_time[:-1]
                shifted_h = recorded_h[:-1]
                shifted_hdot = recorded_hdot[:-1]
                shifted_cl = recorded_cl[:-1]
                shifted_hddot = recorded_hddot[1:]
                
                stiffness = 4 * np.pi**2
                coupling = 2 * vr**2 / (np.pi * dynamics_mstar)
                # coupling = 0.25 * vr**2
                
                recorded_hddot_dynamics = -stiffness * recorded_h + coupling * recorded_cl
                shifted_hddot_dynamics = -stiffness * shifted_h + coupling * shifted_cl
                
                # Compare recorded hddot with dynamics-based hddot
                nrmse_recorded_dynamics = normalized_root_mean_squared_error(recorded_hddot, recorded_hddot_dynamics)
                nrmse_shifted_dynamics = normalized_root_mean_squared_error(shifted_hddot, shifted_hddot_dynamics)
                
                print(f'VR={vr}, Cylinder={cylinder}, L/D={lpd if lpd is not None else "N/A"}')
                print(f'stiffness={stiffness:.4f}, coupling={coupling:.4f}')
                
                print(f'  NRMSE(recorded hddot vs hddot from dynamics): {nrmse_recorded_dynamics*100:.4f}%')
                print(f'  NRMSE(shifted hddot vs hddot from dynamics): {nrmse_shifted_dynamics*100:.4f}%')
                
                # """
                # Next: Compare constructing h
                # 1. Integrate recorded hddot and hdot to get h. Known: h[0], hdot, hddot, and dt
                #     Use the following relation:
                #         h[n] = h[n-1] + hdot[n-1] * dt + 0.5 * hddot[n] * dt^2
                # 2. Integrate recorded hdot to get h. Known: h[0], hdot[0], hddot, and dt
                #     Use the following relation:
                #         h[n] = h[n-1] + hdot[n-1] * dt + 0.5 * hddot[n] * dt^2
                #         hdot[n] = hdot[n-1] + hddot[n] * dt
                # 3. Integrate shifted recorded hddot and hdot to get h. Known: shifted_h[0], shifted_hdot, shifted_hddot, and dt
                #     Use the following relation:
                #         shifted_h[n] = shifted_h[n-1] + shifted_hdot[n-1] * dt + 0.5 * shifted_hddot[n] * dt^2
                # 4. Integrate shifted recorded hdot to get h. Known: shifted_h[0], shifted_hdot[0], shifted_hddot, and dt
                #     Use the following relation:
                #         shifted_h[n] = shifted_h[n-1] + shifted_hdot[n-1] * dt + 0.5 * shifted_hddot[n-1] * dt^2
                #         shifted_hdot[n] = shifted_hdot[n-1] + shifted_hddot[n-1] * dt
                # 5. Integrate shifted recorded h and CL to get next h. Known: shifted_h[0], shifted_hdot[0], shifted_CL, and dt
                #     Use the following relation:
                #         shifted_hddot[n-1] = -stiffness * shifted_h[n-1] + coupling * shifted_CL[n-1]
                #         shifted_hdot[n] = shifted_hdot[n-1] + shifted_hddot[n-1] * dt
                #         shifted_h[n] = shifted_h[n-1] + shifted_hdot[n-1] * dt + 0.5 * shifted_hddot[n-1] * dt^2
                        
                        
                # 6. Integrate recorded h and CL to get next h. Known: h[0], hdot[0], CL, and dt
                #     Use the following relation:
                #         hddot[n] = -stiffness * h[n-1] + coupling * CL[n-1]
                #         hdot[n] = hdot[n-1] + hddot[n] * dt
                #         h[n] = h[n-1] + hdot[n-1] * dt + 0.5 * hddot[n] * dt^2
                # """
                
                # 1st and 2nd integration: recorded hddot and hdot
                dt = time[1] - time[0]
                
                print(f'  dtau={dt:.6f}, dt*={dt*vr:.6f}')
                
                # Integrate recorded hddot and hdot to get h
                simulated_h_case_1 = np.zeros_like(recorded_h)
                simulated_h_case_1[0] = recorded_h[0]
                
                simulated_hdot_case_2 = np.zeros_like(recorded_hdot)
                simulated_h_case_2 = np.zeros_like(recorded_h)
                simulated_h_case_2[0] = recorded_h[0]
                simulated_hdot_case_2[0] = recorded_hdot[0]
                
                simulated_hddot_case_6 = np.zeros_like(recorded_hddot)
                simulated_hdot_case_6 = np.zeros_like(recorded_hdot)
                simulated_h_case_6 = np.zeros_like(recorded_h)
                simulated_h_case_6[0] = recorded_h[0]
                simulated_hdot_case_6[0] = recorded_hdot[0]
                
                # If selected_time starts at the first sample, then simulated_hddot_case_6[0] = 0
                # Otherwise, we use time 1 step before the selected_time to compute the first acceleration
                if region.start > 0:
                    simulated_hddot_case_6[0] = -stiffness * data[region.start - 1, variable_names.index(h_name)] + coupling * data[region.start - 1, variable_names.index(cl_name)]
                else:
                    simulated_hddot_case_6[0] = 0.0
                
                for i in range(1, len(recorded_h)):
                    simulated_h_case_1[i] = (
                        simulated_h_case_1[i-1]
                        + recorded_hdot[i-1] * dt
                        + 0.5 * recorded_hddot[i] * dt**2
                    )
                    
                    simulated_hdot_case_2[i] = (
                        simulated_hdot_case_2[i-1]
                        + recorded_hddot[i] * dt
                    )
                    simulated_h_case_2[i] = (
                        simulated_h_case_2[i-1]
                        + simulated_hdot_case_2[i-1] * dt
                        + 0.5 * recorded_hddot[i] * dt**2
                    )
                    
                    simulated_hddot_case_6[i] = -stiffness * simulated_h_case_6[i-1] + coupling * recorded_cl[i-1]
                    simulated_hdot_case_6[i] = (
                        simulated_hdot_case_6[i-1]
                        + simulated_hddot_case_6[i] * dt
                    )
                    simulated_h_case_6[i] = (
                        simulated_h_case_6[i-1]
                        + simulated_hdot_case_6[i-1] * dt
                        + 0.5 * simulated_hddot_case_6[i] * dt**2
                    )
                
                # 3rd and 4th and 5th integration: shifted recorded hddot and hdot
                # simulated_shifted_h_case_3 = np.zeros_like(shifted_h)
                # simulated_shifted_h_case_3[0] = shifted_h[0]
                
                # simulated_shifted_hdot_case_4 = np.zeros_like(shifted_hdot)
                # simulated_shifted_h_case_4 = np.zeros_like(shifted_h)
                # simulated_shifted_h_case_4[0] = shifted_h[0]
                # simulated_shifted_hdot_case_4[0] = shifted_hdot[0]
                
                # simulated_shifted_hddot_case_5 = np.zeros_like(shifted_hddot)
                # simulated_shifted_hdot_case_5 = np.zeros_like(shifted_hdot)
                # simulated_shifted_h_case_5 = np.zeros_like(shifted_h)
                
                # simulated_shifted_h_case_5[0] = shifted_h[0]
                # simulated_shifted_hdot_case_5[0] = shifted_hdot[0]
                # simulated_shifted_hddot_case_5[0] = -stiffness * shifted_h[0] + coupling * shifted_cl[0]
                
                # for i in range(1, len(shifted_h)):
                #     simulated_shifted_h_case_3[i] = (
                #         simulated_shifted_h_case_3[i-1]
                #         + shifted_hdot[i-1] * dt
                #         + 0.5 * shifted_hddot[i] * dt**2
                #     )
                    
                #     simulated_shifted_hdot_case_4[i] = (
                #         simulated_shifted_hdot_case_4[i-1]
                #         + shifted_hddot[i-1] * dt
                #     )
                #     simulated_shifted_h_case_4[i] = (
                #         simulated_shifted_h_case_4[i-1]
                #         + simulated_shifted_hdot_case_4[i-1] * dt
                #         + 0.5 * shifted_hddot[i-1] * dt**2
                #     )
                    
                    # simulated_shifted_hdot_case_5[i] = (
                    #     simulated_shifted_hdot_case_5[i-1]
                    #     + simulated_shifted_hddot_case_5[i-1] * dt
                    # )
                    # simulated_shifted_h_case_5[i] = (
                    #     simulated_shifted_h_case_5[i-1]
                    #     + simulated_shifted_hdot_case_5[i-1] * dt
                    #     + 0.5 * simulated_shifted_hddot_case_5[i-1] * dt**2
                    # )
                    # simulated_shifted_hddot_case_5[i] = -stiffness * simulated_shifted_h_case_5[i] + coupling * shifted_cl[i]
                    
                # Compute NRMSE for all cases
                nrmse_case_1 = normalized_root_mean_squared_error(simulated_h_case_1, recorded_h)
                nrmse_case_2 = normalized_root_mean_squared_error(simulated_h_case_2, recorded_h)
                nrmse_case_2_hdot = normalized_root_mean_squared_error(simulated_hdot_case_2, recorded_hdot)
                
                # nrmse_case_3 = normalized_root_mean_squared_error(simulated_shifted_h_case_3, shifted_h)
                # nrmse_case_4 = normalized_root_mean_squared_error(simulated_shifted_h_case_4, shifted_h)
                # nrmse_case_4_hdot = normalized_root_mean_squared_error(simulated_shifted_hdot_case_4, shifted_hdot)

                # nrmse_case_5 = normalized_root_mean_squared_error(simulated_shifted_h_case_5, shifted_h)
                # nrmse_case_5_hdot = normalized_root_mean_squared_error(simulated_shifted_hdot_case_5, shifted_hdot)
                # nrmse_case_5_hddot = normalized_root_mean_squared_error(simulated_shifted_hddot_case_5, shifted_hddot)
                
                nrmse_case_6 = normalized_root_mean_squared_error(simulated_h_case_6, recorded_h)
                nrmse_case_6_hdot = normalized_root_mean_squared_error(simulated_hdot_case_6, recorded_hdot)
                nrmse_case_6_hddot = normalized_root_mean_squared_error(simulated_hddot_case_6, recorded_hddot)

                # Print NRMSE results
                print(f'VR={vr}, Cylinder={cylinder}, L/D={lpd if lpd is not None else "N/A"}')
                print(f'Case 1: Integrate recorded hddot and hdot to get h')
                print(f'  NRMSE(Recorded h vs Simulated h from recorded hdot and hddot): {nrmse_case_1*100:.4f}%')
                print(f'Case 2: Integrate recorded hdot to get h')
                print(f'  NRMSE(Recorded h vs Simulated h from simulated hdot and recorded hddot): {nrmse_case_2*100:.4f}%')
                print(f'  NRMSE(Recorded hdot vs Simulated hdot from recorded hddot): {nrmse_case_2_hdot*100:.4f}%')
                # print(f'Case 3: Integrate shifted recorded hddot and hdot to get h')
                # print(f'  NRMSE(Shifted h vs Simulated h from Shifted hddot): {nrmse_case_3*100:.4f}%')
                # print(f'Case 4: Integrate shifted recorded h`dot to get h')
                # print(f'  NRMSE(Shifted h vs Simulated h from Shifted hdot): {nrmse_case_4*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Simulated hdot from Shifted hdot): {nrmse_case_4_hdot*100:.4f}%')
                # print(f'Case 5: Integrate shifted recorded h and CL to get next h')
                # print(f'  NRMSE(Shifted h vs Simulated h from Shifted h and CL): {nrmse_case_5*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Simulated hdot from Shifted h and CL): {nrmse_case_5_hdot*100:.4f}%')
                # print(f'  NRMSE(Shifted hddot vs Simulated hddot from Shifted h and CL): {nrmse_case_5_hddot*100:.4f}%')
                print(f'Case 6: Integrate recorded h and CL to get next h')
                print(f'  NRMSE(Recorded h vs Simulated h from simulated hdot and hddot from recorded CL): {nrmse_case_6*100:.4f}%')
                print(f'  NRMSE(Recorded hdot vs Simulated hdot from simulated hddot from recorded CL): {nrmse_case_6_hdot*100:.4f}%')
                print(f'  NRMSE(Recorded hddot vs Simulated hddot from recorded CL): {nrmse_case_6_hddot*100:.4f}%')
                
                fig, ax = plt.subplots(figsize=(8, 4))
                # ax.plot(shifted_time * vr, shifted_h, label='Shifted h', color='blue', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, simulated_shifted_h_case_5, label='Simulated h from Shifted h and CL', color='orange', linewidth=LINE_WIDTH)
                ax.plot(selected_time * vr, recorded_h, label='Recorded h', color='blue', linewidth=LINE_WIDTH)
                # ax.plot(selected_time * vr, simulated_h_case_6, label='Simulated h from recorded h and CL', color='orange', linewidth=LINE_WIDTH)
                ax.plot(selected_time * vr, simulated_h_case_1, label='Simulated h from recorded hdot and hddot', color='green', linewidth=LINE_WIDTH)
                # ax.plot(selected_time * vr, simulated_h_case_2, label='Simulated h from simulated hdot and recorded hddot', color='red', linewidth=LINE_WIDTH)
                ax.set_xlabel('t*')
                ax.set_ylabel('Displacement')
                figure_parameters = {'V_R': vr}
                if lpd is not None:
                    figure_parameters['L/D'] = lpd
                ax.set_title(f'Cylinder {cylinder}, '+format_parameter_title(figure_parameters))
                ax.legend()
                ax.grid()
                ax.set_xlim(shifted_time[0] * vr, shifted_time[-1] * vr)
                plt.show()

                # # Now we use numerical differentiation
                # # differentiator_config['kwargs']['order'] = 4
                # differentiator = initiate_object(**differentiator_config)
                
                # differentiated_hdot = perform_derivative(selected_time, recorded_h, differentiator)
                # differentiated_hddot = perform_derivative(selected_time, recorded_hdot, differentiator)
                
                # # differentiator_second_order_config = differentiator_config.copy()
                # # differentiator_second_order_config['kwargs']['d'] = 2
                # # differentiator_second_order = initiate_object(**differentiator_second_order_config)
                # differentiated_hddot_from_h = perform_derivative(selected_time, differentiated_hdot, differentiator)
                
                # differentiated_shifted_hdot = differentiated_hdot[:-1]
                # differentiated_shifted_hddot = differentiated_hddot[:-1]
                # differentiated_shifted_hddot_from_h = differentiated_hddot_from_h[:-1]
                
                # # Check length:
                # nrmse_differentiated_hdot = normalized_root_mean_squared_error(differentiated_hdot, recorded_hdot)
                # nrmse_differentiated_hddot = normalized_root_mean_squared_error(differentiated_hddot, recorded_hddot)
                # nrmse_differentiated_shifted_hdot = normalized_root_mean_squared_error(differentiated_shifted_hdot, shifted_hdot)
                # nrmse_differentiated_shifted_hddot = normalized_root_mean_squared_error(differentiated_shifted_hddot, shifted_hddot)
                # nrmse_differentiated_shifted_hddot_from_h = normalized_root_mean_squared_error(differentiated_shifted_hddot_from_h, shifted_hddot)
                
                # print(f'Differentiation method: {differentiator_config["class_name"]}')
                # print(f'  NRMSE(Recorded hdot vs Differentiated h): {nrmse_differentiated_hdot*100:.4f}%')
                # print(f'  NRMSE(Recorded hddot vs Differentiated hdot): {nrmse_differentiated_hddot*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Differentiated shifted h): {nrmse_differentiated_shifted_hdot*100:.4f}%')
                # print(f'  NRMSE(Shifted hddot vs Differentiated shifted hdot): {nrmse_differentiated_shifted_hddot*100:.4f}%')
                # print(f'  NRMSE(Shifted hddot vs Differentiated shifted hdot, differentiated from h): {nrmse_differentiated_shifted_hddot_from_h*100:.4f}%')
                
                # fig, ax = plt.subplots(figsize=(8, 4))
                # ax.plot(shifted_time * vr, shifted_hddot, label='Shifted hddot', color='blue', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, differentiated_shifted_hddot, label='Differentiated shifted hdot', color='orange', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, differentiated_shifted_hddot_from_h, label='Differentiated shifted hdot from h', color='green', linewidth=LINE_WIDTH)
                # ax.set_xlabel('t*')
                # ax.set_ylabel('Acceleration')
                # ax.set_title(f'VR={vr}, Cylinder={cylinder}, L/D={lpd if lpd is not None else "N/A"}')
                # ax.legend()
                # ax.grid()
                # ax.set_xlim(shifted_time[0] * vr, shifted_time[-1] * vr)
                # plt.show()
                
                
                # # Try integrate with differentiated hdot and hddot, using RK45
                # def rhs_hddot_shifted_dynamics(t, y):
                #     h, hdot = y
                #     return [hdot, -stiffness * h + coupling * np.interp(t, shifted_time, shifted_cl)]
                
                # def rhs_hddot_shifted_raw(t, y):
                #     h, hdot = y
                #     return [hdot, np.interp(t, shifted_time, shifted_hddot)]
                
                # def rhs_hddot_shifted_differentiated(t, y):
                #     h, hdot = y
                #     return [hdot, np.interp(t, shifted_time, differentiated_shifted_hddot)]
                
                # def rhs_hdot_shifted_raw(t, y):
                #     h = y
                #     return [np.interp(t, shifted_time, shifted_hdot)]
                
                # def rhs_hdot_shifted_differentiated(t, y):
                #     h = y
                #     return [np.interp(t, shifted_time, differentiated_shifted_hdot)]
                
                # from scipy.integrate import solve_ivp
                
                # solution_shifted_dynamics = solve_ivp(
                #     rhs_hddot_shifted_dynamics,
                #     (shifted_time[0], shifted_time[-1]),
                #     [shifted_h[0], shifted_hdot[0]],
                #     t_eval=shifted_time,
                #     method=args.integration_method
                # )
                
                # solution_shifted_raw = solve_ivp(
                #     rhs_hddot_shifted_raw,
                #     (shifted_time[0], shifted_time[-1]),
                #     [shifted_h[0], shifted_hdot[0]],
                #     t_eval=shifted_time,
                #     method=args.integration_method
                # )
                
                # solution_shifted_differentiated = solve_ivp(
                #     rhs_hddot_shifted_differentiated,
                #     (shifted_time[0], shifted_time[-1]),
                #     [shifted_h[0], shifted_hdot[0]],
                #     t_eval=shifted_time,
                #     method=args.integration_method
                # )
                
                # solution_shifted_hdot_raw = solve_ivp(
                #     rhs_hdot_shifted_raw,
                #     (shifted_time[0], shifted_time[-1]),
                #     [shifted_h[0]],
                #     t_eval=shifted_time,
                #     method=args.integration_method
                # )
                
                # solution_shifted_hdot_differentiated = solve_ivp(
                #     rhs_hdot_shifted_differentiated,
                #     (shifted_time[0], shifted_time[-1]),
                #     [shifted_h[0]],
                #     t_eval=shifted_time,
                #     method=args.integration_method
                # )
                
                # nrmse_integrated_shifted_h = normalized_root_mean_squared_error(solution_shifted_raw.y[0], shifted_h)
                # nrmse_integrated_shifted_hdot = normalized_root_mean_squared_error(solution_shifted_raw.y[1], shifted_hdot)
                
                # nrmse_integrated_shifted_h_dynamics = normalized_root_mean_squared_error(solution_shifted_dynamics.y[0], shifted_h)
                # nrmse_integrated_shifted_hdot_dynamics = normalized_root_mean_squared_error(solution_shifted_dynamics.y[1], shifted_hdot)
                
                # nrmse_integrated_shifted_h_differentiated = normalized_root_mean_squared_error(solution_shifted_differentiated.y[0], shifted_h)
                # nrmse_integrated_shifted_hdot_differentiated = normalized_root_mean_squared_error(solution_shifted_differentiated.y[1], shifted_hdot)
                
                # nrmse_integrated_shifted_h_from_hdot_raw = normalized_root_mean_squared_error(solution_shifted_hdot_raw.y[0], shifted_h)
                
                # nrmse_integrated_shifted_h_from_hdot_differentiated = normalized_root_mean_squared_error(solution_shifted_hdot_differentiated.y[0], shifted_h)

                # print(f'Integration method: {args.integration_method}')
                # print(f'  NRMSE(Shifted h vs Integrated from Shifted hddot raw): {nrmse_integrated_shifted_h*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Integrated from Shifted hddot raw): {nrmse_integrated_shifted_hdot*100:.4f}%')
                # print(f'  NRMSE(Shifted h vs Integrated from Shifted hddot dynamics): {nrmse_integrated_shifted_h_dynamics*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Integrated from Shifted hddot dynamics): {nrmse_integrated_shifted_hdot_dynamics*100:.4f}%')
                # print(f'  NRMSE(Shifted h vs Integrated from Shifted hddot differentiated): {nrmse_integrated_shifted_h_differentiated*100:.4f}%')
                # print(f'  NRMSE(Shifted hdot vs Integrated from Shifted hddot differentiated): {nrmse_integrated_shifted_hdot_differentiated*100:.4f}%')
                # print(f'  NRMSE(Shifted h vs Integrated from Shifted hdot raw): {nrmse_integrated_shifted_h_from_hdot_raw*100:.4f}%')
                # print(f'  NRMSE(Shifted h vs Integrated from Shifted hdot differentiated): {nrmse_integrated_shifted_h_from_hdot_differentiated*100:.4f}%')
                
                # # Create plot comparing shifted h and all integrated results
                # fig, ax = plt.subplots(figsize=(10, 6))
                # ax.plot(shifted_time * vr, shifted_h, label='Shifted Recorded h', color='black', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, solution_shifted_raw.y[0], label='Integrated from Shifted hddot raw', linestyle='--', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, solution_shifted_dynamics.y[0], label='Integrated from Shifted hddot dynamics', linestyle='--', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, solution_shifted_differentiated.y[0], label='Integrated from Shifted hddot differentiated', linestyle='--', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, solution_shifted_hdot_raw.y[0], label='Integrated from Shifted hdot raw', linestyle=':', linewidth=LINE_WIDTH)
                # ax.plot(shifted_time * vr, solution_shifted_hdot_differentiated.y[0], label='Integrated from Shifted hdot differentiated', linestyle=':', linewidth=LINE_WIDTH)
                # ax.set_xlabel('t*')
                # ax.set_ylabel('h')
                # ax.set_title(f'VR={vr}, Cylinder={cylinder}, L/D={lpd if lpd is not None else "N/A"}')
                # ax.legend()
                # ax.grid()
                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                # ax.set_xlim(shifted_time[0] * vr, shifted_time[-1] * vr)
                # plt.tight_layout()
                # plt.show()
            
            del time, data, selected_time


if __name__ == '__main__':
    main()
