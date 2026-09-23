import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pysindy as ps
import pysindy.optimizers as psopt

import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\providecommand{\dddot}[1]{\dot{\ddot{#1}}}",
})

DATA_DIR = "./Data/oscillating-cylinder-tandem-viv"

DRAG_FILENAME_OBJECT_0 = "Drag coefficient at object 0.csv"
DRAG_FILENAME_OBJECT_1 = "Drag coefficient at object 1.csv"

LIFT_FILENAME_OBJECT_0 = "Lift coefficient at object 0.csv"
LIFT_FILENAME_OBJECT_1 = "Lift coefficient at object 1.csv"

NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_0 = "Normalized displacement at object 0.csv"
NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_1 = "Normalized displacement at object 1.csv"

NORMALIZED_VELOCITY_FILENAME_OBJECT_0 = "Normalized velocity at object 0.csv"
NORMALIZED_VELOCITY_FILENAME_OBJECT_1 = "Normalized velocity at object 1.csv"

NORMALIZED_ACCELERATION_FILENAME_OBJECT_0 = "Normalized acceleration at object 0.csv"
NORMALIZED_ACCELERATION_FILENAME_OBJECT_1 = "Normalized acceleration at object 1.csv"

MSTAR = 2.54648
MSTAR_TRUE = 2 / (np.pi*0.25)
RE = 200

"""
Data Loading and Preprocessing Functions
"""
def load_data(vr, lpd, dtype=np.float64):
    drag_object_0 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{DRAG_FILENAME_OBJECT_0}", dtype=dtype)
    drag_object_1 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{DRAG_FILENAME_OBJECT_1}", dtype=dtype)
    
    lift_object_0 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{LIFT_FILENAME_OBJECT_0}", dtype=dtype)
    lift_object_1 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{LIFT_FILENAME_OBJECT_1}", dtype=dtype)
    
    normalized_displacement_object_0 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_0}", dtype=dtype)
    normalized_displacement_object_1 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_1}", dtype=dtype)
    
    normalized_velocity_object_0 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_VELOCITY_FILENAME_OBJECT_0}", dtype=dtype)
    normalized_velocity_object_1 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_VELOCITY_FILENAME_OBJECT_1}", dtype=dtype)

    normalized_acceleration_object_0 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_ACCELERATION_FILENAME_OBJECT_0}", dtype=dtype)
    normalized_acceleration_object_1 = pd.read_csv(f"{DATA_DIR}/LpD = {lpd}/VR = {vr}/{NORMALIZED_ACCELERATION_FILENAME_OBJECT_1}", dtype=dtype)

    return (drag_object_0, drag_object_1, lift_object_0, lift_object_1, normalized_displacement_object_0, normalized_displacement_object_1, normalized_velocity_object_0, normalized_velocity_object_1, normalized_acceleration_object_0, normalized_acceleration_object_1)

def solve_time_duplicates(t, h, h_dot, h_ddot, cl, cd, strategy='uniform', dt=None):
    """
        strategy: 
        - 'uniform'
        - 'mean'
        - 'first'
    """
    # Create a DataFrame to hold the data
    df = pd.DataFrame({
        'time': t,
        'h': h,
        'h_dot': h_dot,
        'h_ddot': h_ddot,
        'cl': cl,
        'cd': cd
    })
    
    df_solved = df.copy()
    
    if strategy == 'mean':
        # Group by time and take the mean of h, h_dot, h_ddot, cl, and cd for duplicate time entries
        df_solved = df.groupby('time').agg({
            'h': 'mean',
            'h_dot': 'mean',
            'h_ddot': 'mean',
            'cl': 'mean',
            'cd': 'mean'
        }).reset_index()
    elif strategy == 'first':
        # Keep the first occurrence of each time entry and drop duplicates
        df_solved = df_solved.drop_duplicates(subset='time', keep='first').reset_index(drop=True)
    elif strategy == 'uniform':
        if dt is None:
            raise ValueError("For uniform strategy, dt must be provided to create uniform time steps.")
        
        df_solved['time'] = df_solved['time'].iloc[0] + np.arange(len(df_solved)) * dt
        
    return df_solved['time'].values, df_solved['h'].values, df_solved['h_dot'].values, df_solved['h_ddot'].values, df_solved['cl'].values, df_solved['cd'].values

"""
Formatting Functions
"""
def add_dot_text(feature_names):
    feature_names_with_dot = []
    for name in feature_names:        
        if name.startswith('\\dot{'):
            # Replace \dot{ with \ddot{
            new_name = name.replace('\\dot{', '\\ddot{')
        elif name.startswith('\\ddot{'):
            # Retain \ddot{ with \dddot{
            new_name = name.replace('\\ddot{', '\\dddot{')
        else:
            # Get variable without subscript
            var_name = name.split('_')[0]
            subscript = name.split('_')[1] if '_' in name else ''
            
            new_name = f"\\dot{{{var_name}}}_{subscript}" if subscript else f"\\dot{{{var_name}}}"
            
        feature_names_with_dot.append(new_name)
    return feature_names_with_dot

def kwargs_to_string(kwargs):
    return '-'.join([f"{key}{value}" if not isinstance(value, dict) else f"{key}[{kwargs_to_string(value)}]" for key, value in kwargs.items()])

"""
Metrics Functions
"""
def normalized_root_mean_squared_error(input, target):
    rmse = np.sqrt(np.mean((input - target)**2))
    nrmse = rmse / (np.std(target)) * 100
    return nrmse

def relative_l2_error(input, target):
    l2_error = np.linalg.norm(input - target)
    relative_l2_error = l2_error / np.linalg.norm(target) * 100
    return relative_l2_error

"""
Plotting Functions
"""
def plot_full(t, X_true_df, feature_names, vr, lpd, X_pred_df=None, xlim='auto', t_stat_start=0, t_stat_end=None, nrmse_dict=None):
    if t_stat_end is None:
        t_stat_end = t[-1]
    
    
    fig, axs = plt.subplots(len(feature_names), 1, figsize=(8, 3 * len(feature_names)), sharex=True)
    
    fig.suptitle(f"$V_R = {vr}$, $L/D = {lpd}$", fontsize=14)
    
    for i in range(len(feature_names)):
        feature_name = feature_names[i]
        ax = axs[i] if len(feature_names) > 1 else axs
        
        if feature_name in X_true_df.columns:
            ax.plot(t, X_true_df[feature_name].values, label=f'${feature_name}$', color='tab:blue')
        
        if X_pred_df is not None and feature_name in X_pred_df.columns:
                ax.plot(t, X_pred_df[feature_name].values, label=f'${feature_name}$ (Predicted)', color='tab:orange', linestyle='--')
                
        if feature_name not in X_true_df.columns and (X_pred_df is None or feature_name not in X_pred_df.columns):
            print(f"Warning: Feature '{feature_name}' not found in both true and predicted DataFrames.")
            continue
        
        if i == len(feature_names) - 1:
            ax.set_xlabel('$t$ (s)', fontsize=12)
        
        ax.set_ylabel(f'${feature_name}$', fontsize=12)
        
        if feature_name in X_true_df.columns:
            mean_value = np.mean(X_true_df[feature_name].values[(t >= t_stat_start) & (t <= t_stat_end)])
            relative_max_value = np.max(np.abs(X_true_df[feature_name].values[(t >= t_stat_start) & (t <= t_stat_end)] - mean_value))
        elif X_pred_df is not None and feature_name in X_pred_df.columns:
            mean_value = np.mean(X_pred_df[feature_name].values[(t >= t_stat_start) & (t <= t_stat_end)])
            relative_max_value = np.max(np.abs(X_pred_df[feature_name].values[(t >= t_stat_start) & (t <= t_stat_end)] - mean_value))
        
        ax.set_ylim(mean_value - 1.25 * relative_max_value, mean_value + 1.25 * relative_max_value)
        
        if xlim == 'auto':
            ax.set_xlim(t[0], t[-1])
        elif xlim is not None and isinstance(xlim, (list, tuple)) and len(xlim) == 2:
            ax.set_xlim(xlim)
        
        # Put NRMSE value on the top left corner of the plot if nrmse_dict is provided
        if nrmse_dict is not None and feature_name in nrmse_dict:
            nrmse_value = nrmse_dict[feature_name][lpd][vr] 
            ax.text(0.02, 0.95, f"NRMSE: {nrmse_value:.2f}\\%", transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
        ax.grid()
        
        plt.tight_layout()
        
    return fig, axs

def __main__():
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current Directory: {current_dir}")
    
    # Check pysindy version
    print(f"pysindy version: {ps.__version__}")
    
    # Argparse for command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        '--vr_list',
        nargs='+',
        type=int,
        default=list(range(5, 15)),
        help='List of VR values to process.',
    )
    parser.add_argument(
        '--lpd_list',
        nargs='+',
        type=float,
        default=[1.5, 2.0, 3.0, 3.5, 4.0],
        help='List of LpD values to process.',
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='STLSQ',
        choices=['STLSQ', 'SR3', 'ConstrainedSR3', 'SSR', 'FROLS'],
        help='Optimizer type for SINDy.',
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        default='polynomial',
        choices=['polynomial', 'fourier'],
        help='Feature library type for SINDy.',
    )
    parser.add_argument(
        '--input_feature_type',
        type=str,
        default='polynomial',
        choices=['polynomial', 'fourier'],
        help='Feature library type for input (parameter) for SINDy.',
    )
    parser.add_argument(
        '--derivative_method',
        type=str,
        default='finite_difference',
        choices=['FiniteDifference', 'finite_difference', 'SmoothedFiniteDifference', 'savitzky_golay', 'spline', 'trend_filtered', 'spectral', 'scipy_savitzky_golay', 'kalman'],
        help='Differentiation method for SINDy.',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./results',
        help='Directory to save the results.'
    )
    parser.add_argument(
        '--num_closeups',
        type=int,
        default=0,
        help='Number of close-up plots to generate for each VR. If 0, no close-ups are generated.'
    )
    parser.add_argument(
        '--closeup_xlims',
        nargs='+',
        type=float,
        default=None,
        help='X-axis limits for the close-up plots. Provide pairs of values: min1 max1 min2 max2 ...'
    )
    parser.add_argument(
        '--generate_state_plots',
        action='store_true',
        default=False,
        help='Generate plots for the state variables.'
    )
    parser.add_argument(
        '--generate_derivative_plots',
        action='store_true',
        default=False,
        help='Generate plots for the true and estimated derivatives.'
    )

    args = parser.parse_args()
    
    cylinder_case = 'two_tandem'
    sindy_case = 'parametric_vr'
    
    base_save_dir = os.path.join(args.save_dir, cylinder_case)
    os.makedirs(base_save_dir, exist_ok=True)
    
    vr_list = args.vr_list
    lpd_list = args.lpd_list

    print(f"VR List: {vr_list}")
    print(f"LpD List: {lpd_list}")
    
    drag_data = {}
    lift_data = {}
    normalized_displacement_data = {}
    normalized_velocity_data = {}
    normalized_acceleration_data = {}

    for vr in vr_list:
        for lpd in lpd_list:
            drag_data[(vr, lpd, 0)], drag_data[(vr, lpd, 1)], lift_data[(vr, lpd, 0)], lift_data[(vr, lpd, 1)], normalized_displacement_data[(vr, lpd, 0)], normalized_displacement_data[(vr, lpd, 1)], normalized_velocity_data[(vr, lpd, 0)], normalized_velocity_data[(vr, lpd, 1)], normalized_acceleration_data[(vr, lpd, 0)], normalized_acceleration_data[(vr, lpd, 1)] = load_data(vr, lpd)
            print(f"Data loaded for VR = {vr}, LpD = {lpd}.")

    print("Data loaded successfully.")
    
    # Prepare the list for individual SINDy for each VR
    # If you change these, don't forget to adjust X in the loop below accordingly
    state_variable_names = [
        'h_0', '\\dot{h}_0', 'C_{L0}',
        'h_1', '\\dot{h}_1', 'C_{L1}',
    ]
    derivative_state_variable_names = add_dot_text(state_variable_names)
    
    parameter_variable_names = [
        'V_R'
    ]
    
    scaled_t0s = {}
    Xs = {}
    Us = {}
    
    X_dot_true_dfs = {}
    X_dot_derivs = {}
    # X_dot_deriv_dfs = {}
    
    nrmse_dict_defined = False
    
    for vr in vr_list:
        for lpd in lpd_list:
            print(f"Processing VR = {vr}, LpD = {lpd}...")
            
            # Extract the time and state variables for the current VR
            t0 = normalized_displacement_data[(vr, lpd, 0)]['time']
            h0 = normalized_displacement_data[(vr, lpd, 0)]['error']
            h_dot0 = normalized_velocity_data[(vr, lpd, 0)]['error']
            h_ddot0 = normalized_acceleration_data[(vr, lpd, 0)]['error']
            cl0 = lift_data[(vr, lpd, 0)]['error']
            cd0 = drag_data[(vr, lpd, 0)]['error']

            t1 = normalized_displacement_data[(vr, lpd, 1)]['time']
            h1 = normalized_displacement_data[(vr, lpd, 1)]['error']
            h_dot1 = normalized_velocity_data[(vr, lpd, 1)]['error']
            h_ddot1 = normalized_acceleration_data[(vr, lpd, 1)]['error']
            cl1 = lift_data[(vr, lpd, 1)]['error']
            cd1 = drag_data[(vr, lpd, 1)]['error']
            
            # Solve time duplicates using the uniform strategy
            t0, h0, h_dot0, h_ddot0, cl0, cd0 = solve_time_duplicates(t0, h0, h_dot0, h_ddot0, cl0, cd0, strategy='uniform', dt=t0[1] - t0[0])
            t1, h1, h_dot1, h_ddot1, cl1, cd1 = solve_time_duplicates(t1, h1, h_dot1, h_ddot1, cl1, cd1, strategy='uniform', dt=t1[1] - t1[0])
            
            # Prepare the state matrix X for the current VR
            X = np.vstack((
                h0, h_dot0, cl0,
                h1, h_dot1, cl1,
            )).T
            X_df = pd.DataFrame(X, columns=state_variable_names)
            
            vrs = np.full_like(t0, vr, dtype=np.float32)
            
            U = vrs.reshape(-1, 1)

            # Plot the true and derived derivatives for the current VR
            if args.generate_state_plots:
                fig, axs = plot_full(t0, X_df, feature_names=state_variable_names, vr=vr, lpd=lpd, xlim='auto', t_stat_start=0, t_stat_end=None)
                plot_filename = f"state_variables_vr{vr}_lpd{lpd}.png"
                plot_filepath = os.path.join(base_save_dir, plot_filename)
                os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
                print(f"Saving plot to {plot_filepath}")
                fig.savefig(plot_filepath, dpi=300)
                plt.close(fig)

                for i in range(args.num_closeups):
                    if args.closeup_xlims is not None and len(args.closeup_xlims) >= 2 * (i + 1):
                        closeup_xlim = args.closeup_xlims[2 * i: 2 * i + 2]
                        
                        fig, axs = plot_full(t0, X_df, feature_names=state_variable_names, vr=vr, lpd=lpd, xlim=closeup_xlim, t_stat_start=closeup_xlim[0], t_stat_end=closeup_xlim[1])
                        closeup_plot_filename = f"state_variables_vr{vr}_lpd{lpd}_closeup{i+1}.png"
                        closeup_plot_filepath = os.path.join(base_save_dir, closeup_plot_filename)
                        os.makedirs(os.path.dirname(closeup_plot_filepath), exist_ok=True)
                        print(f"Saving close-up plot to {closeup_plot_filepath}")
                        fig.savefig(closeup_plot_filepath, dpi=300)
                        plt.close(fig)
                    else:
                        print(f"Warning: Not enough close-up xlim values provided for close-up {i+1}. Skipping this close-up plot.")
            
            scaled_t0 = t0 / vr  # Normalize time by VR for parametric SINDy
            
            scaled_t0s[(vr, lpd)] = scaled_t0
            Xs[(vr, lpd)] = X
            Us[(vr, lpd)] = U

            # Compute the true derivatives for the current VR and store them in a DataFrame
            X_dot_true = np.vstack((
                h_dot0, h_ddot0,
                h_dot1, h_ddot1,
            )).T
            X_dot_true_df = pd.DataFrame(X_dot_true, columns=['\\dot{h}_0', '\\ddot{h}_0', '\\dot{h}_1', '\\ddot{h}_1'])
            X_dot_true_dfs[(vr, lpd)] = X_dot_true_df
            
            # Compute derivation using prescribed differentiation method
            if args.derivative_method == 'FiniteDifference':
                differentiation_kwargs = {
                    'order': 2,     # Order of accuracy (big-O) for the finite difference scheme
                    'd': 1,         # Order of derivative
                }
                
                differentiation_method = ps.FiniteDifference(**differentiation_kwargs)
            elif args.derivative_method == 'SmoothedFiniteDifference':
                smoother_kws = {
                    'window_length': 347,  # Length of the filter window (must be odd)
                    'polyorder': 3,         # Order of the polynomial used to fit the samples
                }
                
                differentiation_kwargs = {
                    'order': 2,     # Order of accuracy (big-O) for the finite difference scheme
                    'd': 1,         # Order of derivative
                    'smoother_kws': smoother_kws,
                }
                
                differentiation_method = ps.SmoothedFiniteDifference(**differentiation_kwargs)
            elif args.derivative_method == 'scipy_savitzky_golay':
                from scipy.signal import savgol_filter
                
                differentiation_kwargs = {
                    'window_length': 347,  # Length of the filter window (must be odd)
                    'polyorder': 3,         # Order of the polynomial used to fit the samples
                }
                
                differentiation_method = lambda x, t: savgol_filter(x, delta=t[0], axis=0, deriv=1, **differentiation_kwargs)
            else:
                if args.derivative_method == 'finite_difference':
                    differentiation_kwargs = {
                        'kind': 'finite_difference',
                        'k': 2,                 # Interpolate the data with a polynomial through 2k + 1 points
                        'periodic': False,      # If True, the data is assumed to be periodic and the window will wrap around the edges
                    }
                elif args.derivative_method == 'savitzky_golay':
                    differentiation_kwargs = {
                        'kind': 'savitzky_golay',
                        'left': 2,                  # Left of if the window is t - left
                        'right': 2,                 # Right of if the window is t + right
                        'order': 3,                 # Order of the polynomial. Expects 0 <= order < points in window
                        'iwindow': False,           # If True, left and right act as indicies for t instead of as lengths in units of t 
                        'periodic': False,
                        'period': None,             # If periodic is True, the period of the data. If None, the period is assumed to be t[-1] - t[0]
                    }
                elif args.derivative_method == 'spline':
                    differentiation_kwargs = {
                        'kind': 'spline',
                        's': 1.0,           # Amount of smoothing
                        'order': 3,         # Order of the spline. Expects 1 <= order <= 5
                        'periodic': False,
                    }
                elif args.derivative_method == 'trend_filtered':
                    differentiation_kwargs = {
                        'kind': 'trend_filtered',
                        'order': 2, # Order of the LASSO derivative
                        'alpha': 1e-2, # Regularization hyperparameter for the LASSO problem
                    }
                elif args.derivative_method == 'spectral':
                    differentiation_kwargs = {
                        'kind': 'spectral',
                        'order': 1,       # Order of the derivative
                        'basis': 'fourier', # Basis for the spectral method. Options: 'fourier', 'chebyshev'
                        'filter': None # Filter for the spectral method. Example: 'lambda k: k < 10' to filter out frequencies higher than the first 10. If None, no filter is applied.
                    }
                elif args.derivative_method == 'kalman':
                    differentiation_kwargs = {
                        'kind': 'kalman',
                        'alpha': 0.1,     # Ratio of measurement error variance to assumed process variance.
                    }
                    
                differentiation_method = ps.SINDyDerivative(**differentiation_kwargs)

            differentiation_kwargs_str = kwargs_to_string(differentiation_kwargs)
            differentiation_foldername = f"{args.derivative_method}-{differentiation_kwargs_str}"
            differentiation_save_dir = os.path.join(base_save_dir, differentiation_foldername)
            os.makedirs(differentiation_save_dir, exist_ok=True)
                
            # Forward the differentiation method to compute the derivatives
            X_dot_deriv = differentiation_method(x=X, t=scaled_t0)
            X_dot_derivs[(vr, lpd)] = X_dot_deriv
            
            X_dot_deriv_df = pd.DataFrame(X_dot_deriv, columns=derivative_state_variable_names)
            
            # Plot the true and derived derivatives for the current VR
            if args.generate_derivative_plots:
                fig, axs = plot_full(t0, X_dot_true_df, feature_names=derivative_state_variable_names, vr=vr, lpd=lpd, X_pred_df=X_dot_deriv_df, xlim='auto', t_stat_start=0)
                plot_filename = f"derivative_comparison_vr{vr}_lpd{lpd}.png"
                plot_filepath = os.path.join(differentiation_save_dir, plot_filename)
                print(f"Saving plot to {plot_filepath}")
                fig.savefig(plot_filepath, dpi=300)
                plt.close(fig)
                
                for i in range(args.num_closeups):
                    if args.closeup_xlims is not None and len(args.closeup_xlims) >= 2 * (i + 1):
                        closeup_xlim = args.closeup_xlims[2 * i: 2 * i + 2]
                        fig, axs = plot_full(t0, X_dot_true_df, feature_names=derivative_state_variable_names, vr=vr, lpd=lpd, X_pred_df=X_dot_deriv_df, xlim=closeup_xlim, t_stat_start=closeup_xlim[0], t_stat_end=closeup_xlim[1])
                        closeup_plot_filename = f"derivative_comparison_vr{vr}_lpd{lpd}_closeup{i+1}.png"
                        closeup_plot_filepath = os.path.join(differentiation_save_dir, closeup_plot_filename)
                        print(f"Saving close-up plot to {closeup_plot_filepath}")
                        fig.savefig(closeup_plot_filepath, dpi=300)
                        plt.close(fig)
                    else:
                        print(f"Warning: Not enough close-up xlim values provided for close-up {i+1}. Skipping this close-up plot.")
        
    overall_variable_names = state_variable_names + parameter_variable_names
        
    for lpd in lpd_list:
        print(f"Performing SINDy for VR values: {vr_list}, LpD value: {lpd}...")
        print(f"State Variable: {state_variable_names}")
        print(f"Parameter Variable: {parameter_variable_names}")

        # Define the feature library based on the command line argument
        if args.feature_type == 'polynomial':
            state_feature_library_kwargs = {
                'degree': 3,
            }
            state_feature_library = ps.PolynomialLibrary(**state_feature_library_kwargs)
        elif args.feature_type == 'fourier':
            state_feature_library_kwargs = {
                'n_frequencies': 1,
                'include_sin': True,
                'include_cos': True,
            }
            state_feature_library = ps.FourierLibrary(**state_feature_library_kwargs)
        
        if args.input_feature_type == 'polynomial':
            parameter_feature_library_kwargs = {
                'degree': 3,
            }
            parameter_feature_library = ps.PolynomialLibrary(**parameter_feature_library_kwargs)
        elif args.input_feature_type == 'fourier':
            parameter_feature_library_kwargs = {
                'n_frequencies': 1,
                'include_sin': True,
                'include_cos': True,
            }
            parameter_feature_library = ps.FourierLibrary(**parameter_feature_library_kwargs)
            
        state_feature_library_kwargs_str = kwargs_to_string(state_feature_library_kwargs)
        parameter_feature_library_kwargs_str = kwargs_to_string(parameter_feature_library_kwargs)
        
        state_feature_library_foldername = f"{args.feature_type}_state_{state_feature_library_kwargs_str}"
        parameter_feature_library_foldername = f"{args.input_feature_type}_parameter_{parameter_feature_library_kwargs_str}"
            
        overall_feature_library = ps.ParameterizedLibrary(
            feature_library=state_feature_library,
            parameter_library=parameter_feature_library,
            num_features=len(state_variable_names),
            num_parameters=len(parameter_variable_names)
        )
        overall_feature_library.fit(x=np.zeros_like(np.concat([Xs[(vr_list[0], lpd_list[0])], Us[(vr_list[0], lpd_list[0])]], axis=-1)))  # Fit the library with dummy data to initialize it

        # Check terms in the overall feature library
        overall_features = overall_feature_library.get_feature_names(overall_variable_names)

        # Define the optimizer based on the command line argument
        if args.optimizer == 'STLSQ':
            optimizer_kwargs = {
                # 'threshold': 1e-1,
                'threshold': 1e-1,
                'alpha': 0.05,
                'normalize_columns': False,
            }
            optimizer = ps.STLSQ(verbose=True, **optimizer_kwargs)
        elif args.optimizer == 'SR3':
            optimizer_kwargs = {
                # 'reg_weight_lam': 5e-3,
                # 'regularizer': 'L0',
                # 'max_iter': 30
                'reg_weight_lam': 5e-3,
                'regularizer': 'L0',
                'relax_coeff_nu': 1.0,
                'max_iter': 100,
                'normalize_columns': False,
            }
            optimizer = ps.SR3(verbose=True, **optimizer_kwargs)
        elif args.optimizer == 'ConstrainedSR3':
            n_features = len(overall_features)
            n_targets = len(derivative_state_variable_names)
            
            lhs_rows, rhs_values = [], []
            
            def pin(target_idx, feature_idx, value):
                lhs = np.zeros(n_features * n_targets)
                lhs[target_idx * n_features + feature_idx] = 1
                lhs_rows.append(lhs)
                rhs_values.append(value)
            
            # Constraint 1: h' = hdot
            target_hdot0_index = derivative_state_variable_names.index('\\dot{h}_0')
            target_hdot1_index = derivative_state_variable_names.index('\\dot{h}_1')
            
            feature_hdot0_index = overall_features.index('1 \\dot{h}_0')
            feature_hdot1_index = overall_features.index('1 \\dot{h}_1')
            for j in range(n_features):
                pin(target_hdot0_index, j, 1.0 if j == feature_hdot0_index else 0.0)
                pin(target_hdot1_index, j, 1.0 if j == feature_hdot1_index else 0.0)
            
            # Constraint 2: hdot' = -4*pi^2 * h - 2*V_R^2 / (pi * m^*) OR hdot' = hddot
            # Priority: hdot' = hddot, then hdot' = -4*pi^2 * h - 2*V_R^2 / (pi * m^*)
            # ALWAYS assume hdot present as a state variable
            target_hddot0_index = derivative_state_variable_names.index('\\ddot{h}_0')
            target_hddot1_index = derivative_state_variable_names.index('\\ddot{h}_1')
            
            if '\\ddot{h}_0' in state_variable_names and '\\ddot{h}_1' in state_variable_names:
                feature_hddot0_index = overall_features.index('1 \\ddot{h}_0')
                feature_hddot1_index = overall_features.index('1 \\ddot{h}_1')
                for j in range(n_features):
                    pin(target_hddot0_index, j, 1.0 if j == feature_hddot0_index else 0.0)
                    pin(target_hddot1_index, j, 1.0 if j == feature_hddot1_index else 0.0)
            else:
                feature_h0_index = overall_features.index('1 h_0')
                feature_vr2cl0_index = overall_features.index('V_R^2 C_{L0}')
                
                feature_h1_index = overall_features.index('1 h_1')
                feature_vr2cl1_index = overall_features.index('V_R^2 C_{L1}')
                for j in range(n_features):
                    if j == feature_h0_index:
                        pin(target_hddot0_index, j, -4 * np.pi**2)
                    elif j == feature_vr2cl0_index:
                        pin(target_hddot0_index, j, +2 / (np.pi * MSTAR_TRUE))
                    else:
                        pin(target_hddot0_index, j, 0.0)
                    
                    if j == feature_h1_index:
                        pin(target_hddot1_index, j, -4 * np.pi**2)
                    elif j == feature_vr2cl1_index:
                        pin(target_hddot1_index, j, +2 / (np.pi * MSTAR_TRUE))
                    else:
                        pin(target_hddot1_index, j, 0.0)
                
            constraint_lhs = np.array(lhs_rows)
            constraint_rhs = np.array(rhs_values)
            
            optimizer_kwargs = {
                # 'reg_weight_lam': 5e-3,
                # 'regularizer': 'L0',
                # 'max_iter': 30
                'reg_weight_lam': 5e-3,
                'regularizer': 'L0',
                'relax_coeff_nu': 1.0,
                'max_iter': 100,
                'normalize_columns': False,
            }
            optimizer = ps.ConstrainedSR3(constraint_lhs=constraint_lhs, constraint_rhs=constraint_rhs, equality_constraints=True, verbose=True, **optimizer_kwargs)
        elif args.optimizer == 'SSR':
            optimizer_kwargs = {
                'max_iter': 1000,
                'alpha': 0.05,
            }
            optimizer = ps.SSR(verbose=True, **optimizer_kwargs)
        elif args.optimizer == 'FROLS':
            optimizer_kwargs = {
                'max_iter': 1000,
                'alpha': 0.05,
            }
            optimizer = ps.FROLS(verbose=True, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
        optimizer_kwargs_str = kwargs_to_string(optimizer_kwargs)
        optimizer_foldername = f"{args.optimizer}_{optimizer_kwargs_str}"
        
        print("Fitting SINDy model...")
        model = ps.SINDy(
            feature_library=overall_feature_library,
            optimizer=optimizer,
            discrete_time=False
        )
        
        # Get the current X, scaled_t0, U, and X_dot_deriv for the current LpD
        current_Xs = {key: Xs[key] for key in Xs if key[1] == lpd}
        current_scaled_t0s = {key: scaled_t0s[key] for key in scaled_t0s if key[1] == lpd}
        current_Us = {key: Us[key] for key in Us if key[1] == lpd}
        current_X_dot_derivs = {key: X_dot_derivs[key] for key in X_dot_derivs if key[1] == lpd}
        
        # Convert the dictionaries to lists for fitting
        current_Xs = list(current_Xs.values())
        current_scaled_t0s = list(current_scaled_t0s.values())
        current_Us = list(current_Us.values())
        current_X_dot_derivs = list(current_X_dot_derivs.values())
        
        model.fit(x=current_Xs, u=current_Us, t=current_scaled_t0s, feature_names=overall_variable_names, x_dot=current_X_dot_derivs)
        
        # Extract coefficients, turn into a DataFrame
        coefficients_df = pd.DataFrame(model.coefficients().T, columns=derivative_state_variable_names)
        
        # Get terms in feature library
        term_names = model.get_feature_names()
        coefficients_df.index = term_names
        
        # Save coefficients to CSV
        sindy_results_dir = os.path.join(differentiation_save_dir, sindy_case, optimizer_foldername, state_feature_library_foldername, parameter_feature_library_foldername)
        os.makedirs(sindy_results_dir, exist_ok=True)
        
        coefficients_filename = f"coefficients_two_tandem_parametric_{lpd}.csv"
        coefficients_filepath = os.path.join(sindy_results_dir, coefficients_filename)
        os.makedirs(os.path.dirname(coefficients_filepath), exist_ok=True)
        print(f"Saving coefficients to {coefficients_filepath}")
        coefficients_df.to_csv(coefficients_filepath)
    
        if not nrmse_dict_defined:
            true_nrmse_dict = {
                derivative_state_variable_name: {
                    lpd: {
                        vr: [] for vr in vr_list
                    } for lpd in lpd_list
                } for derivative_state_variable_name in X_dot_true_df.columns
            }
            
            semi_nrmse_dict = {
                derivative_state_variable_name: {
                    lpd: {
                        vr: [] for vr in vr_list
                    } for lpd in lpd_list
                } for derivative_state_variable_name in derivative_state_variable_names
            }
            
            nrmse_dict_defined = True
        
        current_X_dot_true_df = {key: X_dot_true_dfs[key] for key in X_dot_true_dfs if key[1] == lpd}
        current_X_dot_true_df = list(current_X_dot_true_df.values())
        
        for vr, scaled_t0, X, U, X_dot_deriv, X_dot_true_df in zip(vr_list, current_scaled_t0s, current_Xs, current_Us, current_X_dot_derivs, current_X_dot_true_df):
            # Change X_dot_deriv to DataFrame
            X_dot_deriv_df = pd.DataFrame(X_dot_deriv, columns=derivative_state_variable_names)
            
            # Predict derivatives using the fitted model
            X_dot_pred = model.predict(x=X, u=U)
            X_dot_pred_df = pd.DataFrame(X_dot_pred, columns=derivative_state_variable_names)
            
            # Compute NRMSE for each derivative state variable
            print(f"Computing true NRMSE for VR = {vr}, LpD = {lpd}...")
            for derivative_state_variable_name in X_dot_true_df.columns:
                true_nrmse = normalized_root_mean_squared_error(X_dot_pred_df[derivative_state_variable_name].values, X_dot_true_df[derivative_state_variable_name].values)
                true_nrmse_dict[derivative_state_variable_name][lpd][vr] = true_nrmse
                
                print(f"NRMSE for {derivative_state_variable_name} at VR = {vr}, LpD = {lpd}: {true_nrmse:.2f}%")
            
            print(f"Computing semi NRMSE for VR = {vr}, LpD = {lpd}...")
            for derivative_state_variable_name in derivative_state_variable_names:
                semi_nrmse = normalized_root_mean_squared_error(X_dot_pred_df[derivative_state_variable_name].values, X_dot_deriv_df[derivative_state_variable_name].values)
                semi_nrmse_dict[derivative_state_variable_name][lpd][vr] = semi_nrmse
                
                print(f"NRMSE for {derivative_state_variable_name} at VR = {vr}, LpD = {lpd}: {semi_nrmse:.2f}%")
            
            # Generate plots for the current VR
            fig, axs = plot_full(t0, X_dot_deriv_df, feature_names=derivative_state_variable_names, vr=vr, lpd=lpd, X_pred_df=X_dot_pred_df, xlim='auto', t_stat_start=0, nrmse_dict=semi_nrmse_dict)
            plot_filename = f"prediction_derivative_comparison_vr{vr}_lpd{lpd}.png"
            plot_filepath = os.path.join(sindy_results_dir, plot_filename)
            os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
            print(f"Saving plot to {plot_filepath}")
            fig.savefig(plot_filepath, dpi=300)
            plt.close(fig)
            
            for i in range(args.num_closeups):
                if args.closeup_xlims is not None and len(args.closeup_xlims) >= 2 * (i + 1):
                    closeup_xlim = args.closeup_xlims[2 * i: 2 * i + 2]
                    fig, axs = plot_full(t0, X_dot_deriv_df, feature_names=derivative_state_variable_names, vr=vr, lpd=lpd, X_pred_df=X_dot_pred_df, xlim=closeup_xlim, t_stat_start=closeup_xlim[0], t_stat_end=closeup_xlim[1], nrmse_dict=semi_nrmse_dict)
                    closeup_plot_filename = f"prediction_derivative_comparison_vr{vr}_lpd{lpd}_closeup{i+1}.png"
                    closeup_plot_filepath = os.path.join(sindy_results_dir, closeup_plot_filename)
                    os.makedirs(os.path.dirname(closeup_plot_filepath), exist_ok=True)
                    print(f"Saving close-up plot to {closeup_plot_filepath}")
                    fig.savefig(closeup_plot_filepath, dpi=300)
                    plt.close(fig)
                else:
                    print(f"Warning: Not enough close-up xlim values provided for close-up {i+1}. Skipping this close-up plot.")
    
    # Save NRMSE results to CSV
    true_nrmse_records = []
    semi_nrmse_records = []
    
    for vr in vr_list:
        for lpd in lpd_list:
            row = {'vr': vr, 'lpd': lpd}
            for derivative_state_variable_name in X_dot_true_df.columns:
                row[derivative_state_variable_name] = true_nrmse_dict[derivative_state_variable_name][lpd][vr]
            true_nrmse_records.append(row)
                
            row = {'vr': vr, 'lpd': lpd}
            for derivative_state_variable_name in derivative_state_variable_names:
                row[derivative_state_variable_name] = semi_nrmse_dict[derivative_state_variable_name][lpd][vr]
            semi_nrmse_records.append(row)
    
    true_nrmse_df = pd.DataFrame(true_nrmse_records)
    true_nrmse_filename = f"true_nrmse_two_tandem_parametric_vr.csv"
    true_nrmse_filepath = os.path.join(sindy_results_dir, true_nrmse_filename)
    os.makedirs(os.path.dirname(true_nrmse_filepath), exist_ok=True)
    print(f"Saving true NRMSE results to {true_nrmse_filepath}")
    true_nrmse_df.to_csv(true_nrmse_filepath)

    semi_nrmse_df = pd.DataFrame(semi_nrmse_records)
    semi_nrmse_filename = f"semi_nrmse_two_tandem_parametric_vr.csv"
    semi_nrmse_filepath = os.path.join(sindy_results_dir, semi_nrmse_filename)
    os.makedirs(os.path.dirname(semi_nrmse_filepath), exist_ok=True)
    print(f"Saving semi NRMSE results to {semi_nrmse_filepath}")
    semi_nrmse_df.to_csv(semi_nrmse_filepath)

if __name__ == "__main__":
    __main__()