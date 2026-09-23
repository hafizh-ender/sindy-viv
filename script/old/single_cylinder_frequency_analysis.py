import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal import welch

import pysindy as ps
import pysindy.optimizers as psopt

import argparse

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\providecommand{\dddot}[1]{\dot{\ddot{#1}}}",
})

DATA_DIR = "./Data/oscillating-cylinder-viv"

DRAG_FILENAME_OBJECT_0 = "Drag coefficient at object 0.csv"
LIFT_FILENAME_OBJECT_0 = "Lift coefficient at object 0.csv"
NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_0 = "Normalized displacement at object 0.csv"
NORMALIZED_VELOCITY_FILENAME_OBJECT_0 = "Normalized velocity at object 0.csv"
NORMALIZED_ACCELERATION_FILENAME_OBJECT_0 = "Normalized acceleration at object 0.csv"

MSTAR = 5.1
MSTAR_TRUE = 2 / (np.pi*0.125)
RE = 150

"""
Data Loading and Preprocessing Functions
"""
def load_data(vr, dtype=np.float64):
    drag_object_0 = pd.read_csv(f"{DATA_DIR}/re{RE}/mstar{MSTAR}/vr{vr}/{DRAG_FILENAME_OBJECT_0}", dtype=dtype)    

    lift_object_0 = pd.read_csv(f"{DATA_DIR}/re{RE}/mstar{MSTAR}/vr{vr}/{LIFT_FILENAME_OBJECT_0}", dtype=dtype)

    normalized_displacement_object_0 = pd.read_csv(f"{DATA_DIR}/re{RE}/mstar{MSTAR}/vr{vr}/{NORMALIZED_DISPLACEMENT_FILENAME_OBJECT_0}", dtype=dtype)
    
    normalized_velocity_object_0 = pd.read_csv(f"{DATA_DIR}/re{RE}/mstar{MSTAR}/vr{vr}/{NORMALIZED_VELOCITY_FILENAME_OBJECT_0}", dtype=dtype)

    normalized_acceleration_object_0 = pd.read_csv(f"{DATA_DIR}/re{RE}/mstar{MSTAR}/vr{vr}/{NORMALIZED_ACCELERATION_FILENAME_OBJECT_0}", dtype=dtype)

    return (drag_object_0, lift_object_0, normalized_displacement_object_0, normalized_velocity_object_0, normalized_acceleration_object_0)

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
def plot_full(t, X_true_df, feature_names, vr, X_pred_df=None, xlim='auto', t_stat_start=0, t_stat_end=None, nrmse_dict=None):
    if t_stat_end is None:
        t_stat_end = t[-1]
    
    
    fig, axs = plt.subplots(len(feature_names), 1, figsize=(8, 3 * len(feature_names)), sharex=True)
    
    fig.suptitle(f"$V_R = {vr}$", fontsize=14)
    
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
            nrmse_value = nrmse_dict[feature_name][vr]
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
        default=list(range(3, 9)),
        help='List of VR values to process.',
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

    args = parser.parse_args()
    
    cylinder_case = 'single'
    sindy_case = 'individual'
    
    base_save_dir = os.path.join(args.save_dir, cylinder_case)
    os.makedirs(base_save_dir, exist_ok=True)
    
    vr_list = args.vr_list

    print(f"VR List: {vr_list}")
    
    drag_data = {}
    lift_data = {}
    normalized_displacement_data = {}
    normalized_velocity_data = {}
    normalized_acceleration_data = {}
    
    for vr in vr_list:
        drag_data[(vr, 0)], lift_data[(vr, 0)], normalized_displacement_data[(vr, 0)], normalized_velocity_data[(vr, 0)], normalized_acceleration_data[(vr, 0)] = load_data(vr)
        print(f"Data loaded for VR = {vr}")

    print("Data loaded successfully.")
    
    state_variable_names = [
        'h_0', '\\dot{h}_0', '\\ddot{h}_0', 'C_{L0}', 'C_{D0}'
    ]
    
    for vr in vr_list:
        print(f"Processing VR = {vr}...")
        
        # Extract the time and state variables for the current VR
        t0 = normalized_displacement_data[(vr, 0)]['time']
        h0 = normalized_displacement_data[(vr, 0)]['error']
        h_dot0 = normalized_velocity_data[(vr, 0)]['error']
        h_ddot0 = normalized_acceleration_data[(vr, 0)]['error']
        cl0 = lift_data[(vr, 0)]['error']
        cd0 = drag_data[(vr, 0)]['error']
        
        # Solve time duplicates using the uniform strategy
        t0, h0, h_dot0, h_ddot0, cl0, cd0 = solve_time_duplicates(t0, h0, h_dot0, h_ddot0, cl0, cd0, strategy='uniform', dt=t0[1] - t0[0])
        
                # Prepare the state matrix X for the current VR
        X = np.vstack((
            h0, h_dot0, h_ddot0, cl0, cd0
        )).T
        X_df = pd.DataFrame(X, columns=state_variable_names)

        # Plot the true and derived derivatives for the current VR
        if args.generate_state_plots:
            fig, axs = plot_full(t0, X_df, feature_names=state_variable_names, vr=vr, xlim='auto', t_stat_start=0, t_stat_end=None)
            plot_filename = f"state_variables_vr{vr}.png"
            plot_filepath = os.path.join(base_save_dir, plot_filename)
            os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
            print(f"Saving plot to {plot_filepath}")
            fig.savefig(plot_filepath, dpi=300)
            plt.close(fig)

            for i in range(args.num_closeups):
                if args.closeup_xlims is not None and len(args.closeup_xlims) >= 2 * (i + 1):
                    closeup_xlim = args.closeup_xlims[2 * i: 2 * i + 2]
                    
                    fig, axs = plot_full(t0, X_df, feature_names=state_variable_names, vr=vr, xlim=closeup_xlim, t_stat_start=closeup_xlim[0], t_stat_end=closeup_xlim[1])
                    closeup_plot_filename = f"state_variables_vr{vr}_closeup{i+1}.png"
                    closeup_plot_filepath = os.path.join(base_save_dir, closeup_plot_filename)
                    os.makedirs(os.path.dirname(closeup_plot_filepath), exist_ok=True)
                    print(f"Saving close-up plot to {closeup_plot_filepath}")
                    fig.savefig(closeup_plot_filepath, dpi=300)
                    plt.close(fig)
                else:
                    print(f"Warning: Not enough close-up xlim values provided for close-up {i+1}. Skipping this close-up plot.")
                    
        N = len(t0)
        dt = t0[1] - t0[0]
        f = fftfreq(N, dt)
        fft_h0 = fft(h0)
        fft_h_dot0 = fft(h_dot0)
        fft_h_ddot0 = fft(h_ddot0)
        fft_cl0 = fft(cl0)
        fft_cd0 = fft(cd0)
        

if __name__ == "__main__":
    __main__()