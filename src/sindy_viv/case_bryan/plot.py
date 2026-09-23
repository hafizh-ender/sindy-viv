import numpy as np

import matplotlib.pyplot as plt

def plot_data(
    # Data related arguments
    time, # t^* = U_inf t / D, NOT tau = f_N t. Shape: (N,)
    data, # Shape: (N, num_state_variables)
    state_variable_names,
    parameter_values_dict,
    
    # Visualization related arguments
    xlim=None,
    
    ylim_mean_time_start=0,
    ylim_mean_time_end=None,
):
    if xlim is None:
        xlim = (time[0], time[-1])
    
    if ylim_mean_time_end is None:
        ylim_mean_time_end = time[-1]
    
    fig, axs = plt.subplots(len(state_variable_names), 1, figsize=(6, 1 * len(state_variable_names)), sharex=True, layout='constrained')
    
    suptitle_str = ',\\quad '.join([f"{key}={value}" for key, value in parameter_values_dict.items()])
    fig.suptitle(f"${suptitle_str}$", fontsize=14)
    
    for i, state_variable_name in enumerate(state_variable_names):
        ax = axs[i] if len(state_variable_names) > 1 else axs
        
        # Plot the data for the current state variable
        ax.plot(time, data[:, i], label=f"${state_variable_name}$", color='tab:blue', linewidth=1.0)
            
        # Turn on Y-label for all subplots
        ax.set_ylabel(f"${state_variable_name}$", fontsize=12)
        
        # Turn on X-label for the last subplot only
        if i == len(state_variable_names) - 1:
            ax.set_xlabel("$t^*$", fontsize=12)
        
        # Find mean value of the data in the specified time range
        mean_value = data[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)].mean()
        relative_max_value = np.max(np.abs(data[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)] - mean_value))
        
        # Set the y-axis limits based on the mean value and relative maximum value
        ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)
        
        # Set the x-axis limits
        ax.set_xlim(xlim)
        
        # Turn on grid for all subplots
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Turn off top and right spines for all subplots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make linewidth of bottom and left spines thin
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)

    return fig, axs

def plot_compare_prior_check(
    # Data related arguments
    time, # t^* = U_inf t / D, NOT tau = f_N t
    data_true,
    data_pred,
    true_state_variable_names,
    pred_state_variable_names,
    parameter_values_dict,
    
    # Visualization related arguments
    xlim=None,
    
    ylim_mean_time_start=0,
    ylim_mean_time_end=None,
    
    ylim_source='true', # 'true' or 'pred'
):
    state_variable_names = [var for var in true_state_variable_names if var in pred_state_variable_names]
    
    if len(state_variable_names) == 0:
        raise ValueError("No common state variable names found between true and predicted data.")
    
    data_true_to_compare = data_true[:, [i for i, var in enumerate(true_state_variable_names) if var in state_variable_names]]
    data_pred_to_compare = data_pred[:, [i for i, var in enumerate(pred_state_variable_names) if var in state_variable_names]]
    
    return plot_compare(
        time=time,
        data_true=data_true_to_compare,
        data_pred=data_pred_to_compare,
        state_variable_names=state_variable_names,
        parameter_values_dict=parameter_values_dict,
        xlim=xlim,
        ylim_mean_time_start=ylim_mean_time_start,
        ylim_mean_time_end=ylim_mean_time_end,
        ylim_source=ylim_source
    )

def plot_compare(
    # Data related arguments
    time, # t^* = U_inf t / D, NOT tau = f_N t
    data_true,
    data_pred,
    state_variable_names,
    parameter_values_dict,
    
    # Visualization related arguments
    xlim=None,
    
    ylim_mean_time_start=0,
    ylim_mean_time_end=None,
    
    ylim_source='true', # 'true' or 'pred'
):
    if ylim_source not in ['true', 'pred']:
        raise ValueError("ylim_source must be either 'true' or 'pred'")
    
    if xlim is None:
        xlim = (time[0], time[-1])
    
    if ylim_mean_time_end is None:
        ylim_mean_time_end = time[-1]
    
    fig, axs = plt.subplots(len(state_variable_names), 1, figsize=(7.5, 2.5 * len(state_variable_names)), sharex=True, layout='constrained')
        
    suptitle_str = ',\\quad '.join([f"{key}={value}" for key, value in parameter_values_dict.items()])
    fig.suptitle(f"${suptitle_str}$", fontsize=14)
    
    for i, state_variable_name in enumerate(state_variable_names):
        ax = axs[i] if len(state_variable_names) > 1 else axs
        
        # Plot the true data for the current state variable
        ax.plot(time, data_true[:, i], label=f"True ${state_variable_name}$", color='tab:blue', linewidth=1.0)
        
        # Plot the predicted data for the current state variable
        ax.plot(time, data_pred[:, i], label=f"Predicted ${state_variable_name}$", color='tab:red', linestyle='--', linewidth=1.0)
        
        # Turn on X-label for the last subplot only
        if i == len(state_variable_names) - 1:
            ax.set_xlabel("$t^*$", fontsize=12)
            
        # Turn on Y-label for all subplots
        ax.set_ylabel(f"${state_variable_name}$", fontsize=12)
        
        # Find mean value of the true data in the specified time range
        if ylim_source == 'true':
            mean_value = data_true[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)].mean()
            relative_max_value = np.max(np.abs(data_true[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)] - mean_value))
        elif ylim_source == 'pred':
            mean_value = data_pred[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)].mean()
            relative_max_value = np.max(np.abs(data_pred[:, i][(time >= ylim_mean_time_start) & (time <= ylim_mean_time_end)] - mean_value))

        # Set the y-axis limits based on the mean value and relative maximum value
        ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)
        
        # Set the x-axis limits
        ax.set_xlim(xlim)
        
        # Turn on grid for all subplots
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Turn off top and right spines for all subplots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make linewidth of bottom and left spines thin
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        
    return fig, axs