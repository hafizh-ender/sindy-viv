import numpy as np

import matplotlib.pyplot as plt

def _time_windows(time, xlim, start, end, crop_to_xlim):
    """Select statistics once and crop lines while retaining boundary segments."""
    time = np.asarray(time)
    
    if time.ndim != 1 or not len(time):
        raise ValueError("Time must be a nonempty 1-D array.")
    
    if xlim is None:
        xlim = (time[0], time[-1])
        
    if end is None:
        end = time[-1]
        
    # Sorted timestamps permit slice views instead of a new mask for every channel.
    if np.all(time[1:] >= time[:-1]):
        stats = slice(np.searchsorted(time, start, side='left'), np.searchsorted(time, end, side='right'))
        visible = slice(None)
        # Cropping is opt-in so default plots retain all samples for later panning.
        if crop_to_xlim:
            low, high = sorted(xlim)
            visible = slice(
                max(0, np.searchsorted(time, low, side='left') - 1),
                min(len(time), np.searchsorted(time, high, side='right') + 1),
            )
    else:
        # Preserve line ordering for callers supplying unsorted timestamps.
        stats = (time >= start) & (time <= end)
        visible = slice(None)
        
    # Reject empty statistics windows before allocating a figure.
    if not time[stats].size:
        raise ValueError("The y-axis statistics interval contains no samples.")
    
    return time, xlim, stats, visible


def _matching_columns(data, source_names, chosen_names):
    # Reuse the original array when channel names already match in order.
    if list(source_names) == list(chosen_names):
        return data
    indices = [source_names.index(name) for name in chosen_names]
    
    # A contiguous selection can also be represented by a view.
    if indices == list(range(indices[0], indices[0] + len(indices))):
        return data[:, indices[0]:indices[0] + len(indices)]
    
    return data[:, indices]

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
    
    crop_to_xlim=False,
):
    """Plot states; opt into cropping when later panning is not required."""
    time, xlim, stats, visible = _time_windows(time, xlim, ylim_mean_time_start, ylim_mean_time_end, crop_to_xlim)
    
    fig, axs = plt.subplots(len(state_variable_names), 1, figsize=(6, 1 * len(state_variable_names)), sharex=True, layout='constrained')
    
    suptitle_str = ',\\quad '.join([f"{key}={value}" for key, value in parameter_values_dict.items()])
    
    # Avoid an empty math expression when no parameter values were supplied.
    if suptitle_str:
        fig.suptitle(f"${suptitle_str}$", fontsize=14)
    
    for i, state_variable_name in enumerate(state_variable_names):
        ax = axs[i] if len(state_variable_names) > 1 else axs
        
        # Plot the data for the current state variable
        ax.plot(time[visible], data[visible, i], label=f"${state_variable_name}$", color='tab:blue', linewidth=1.0)
            
        # Turn on Y-label for all subplots
        ax.set_ylabel(f"${state_variable_name}$", fontsize=12)
        
        # Turn on X-label for the last subplot only
        if i == len(state_variable_names) - 1:
            ax.set_xlabel("$t^*$", fontsize=12)
        
        # Find mean value of the data in the specified time range
        values = data[stats, i]
        mean_value = values.mean()
        relative_max_value = np.max(np.abs(values - mean_value))
        
        # Set the y-axis limits based on the mean value and relative maximum value
        ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)
        
        # Set the x-axis limits
        ax.set_xlim(xlim)
        
        # Turn on grid for all subplots
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Turn off top and right spines for all subplots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make linewidth of bottom and left spines thin
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)

    return fig, axs

def plot_compare_prior_check(
    # Data related arguments
    time, # t^* = U_inf t / D, NOT tau = f_N t
    data_true,
    data_pred,
    true_state_variable_names,
    pred_state_variable_names,
    parameter_values_dict,
    
    error_metrics_dict=None, # Optional dictionary containing error metrics for each state variable
    
    # Visualization related arguments
    xlim=None,
    
    ylim_mean_time_start=0,
    ylim_mean_time_end=None,
    
    ylim_source='true', # 'true' or 'pred'
    
    crop_to_xlim=False,
):
    state_variable_names = [var for var in true_state_variable_names if var in pred_state_variable_names]
    
    if len(state_variable_names) == 0:
        raise ValueError("No common state variable names found between true and predicted data.")
    
    # Align both arrays to the true-variable order before comparing channel values.
    data_true_to_compare = _matching_columns(data_true, true_state_variable_names, state_variable_names)
    data_pred_to_compare = _matching_columns(data_pred, pred_state_variable_names, state_variable_names)
    
    return plot_compare(
        time=time,
        data_true=data_true_to_compare,
        data_pred=data_pred_to_compare,
        state_variable_names=state_variable_names,
        error_metrics_dict=error_metrics_dict,
        parameter_values_dict=parameter_values_dict,
        xlim=xlim,
        ylim_mean_time_start=ylim_mean_time_start,
        ylim_mean_time_end=ylim_mean_time_end,
        ylim_source=ylim_source,
        crop_to_xlim=crop_to_xlim,
    )

def plot_compare(
    # Data related arguments
    time, # t^* = U_inf t / D, NOT tau = f_N t
    data_true,
    data_pred,
    state_variable_names,
    parameter_values_dict,
    
    error_metrics_dict=None, # Optional dictionary containing error metrics for each state variable
    
    # Visualization related arguments
    xlim=None,
    
    ylim_mean_time_start=0,
    ylim_mean_time_end=None,
    
    ylim_source='true', # 'true' or 'pred'
    crop_to_xlim=False,
):
    """Compare states, optionally cropping lines to the initial x-axis limits."""
    if ylim_source not in ['true', 'pred']:
        raise ValueError("ylim_source must be either 'true' or 'pred'")
    
    time, xlim, stats, visible = _time_windows(time, xlim, ylim_mean_time_start, ylim_mean_time_end, crop_to_xlim)
    ylim_data = data_true if ylim_source == 'true' else data_pred
    
    fig, axs = plt.subplots(len(state_variable_names), 1, figsize=(6, 1 * len(state_variable_names)), sharex=True, layout='constrained')
        
    suptitle_str = ',\\quad '.join([f"{key}={value}" for key, value in parameter_values_dict.items()])
    # Avoid an empty math expression when no parameter values were supplied.
    if suptitle_str:
        fig.suptitle(f"${suptitle_str}$", fontsize=14)
    
    for i, state_variable_name in enumerate(state_variable_names):
        ax = axs[i] if len(state_variable_names) > 1 else axs
        
        # Plot the true data for the current state variable
        ax.plot(time[visible], data_true[visible, i], label=f"True ${state_variable_name}$", color='tab:blue', linewidth=1.0)
        
        # Plot the predicted data for the current state variable
        ax.plot(time[visible], data_pred[visible, i], label=f"Predicted ${state_variable_name}$", color='tab:red', linestyle='--', linewidth=1.0)
        
        # Turn on X-label for the last subplot only
        if i == len(state_variable_names) - 1:
            ax.set_xlabel("$t^*$", fontsize=12)
            
        # Turn on Y-label for all subplots
        ax.set_ylabel(f"${state_variable_name}$", fontsize=12)
        
        # Add error metrics to top right corner of the subplot if provided
        if error_metrics_dict is not None and state_variable_name in error_metrics_dict:
            error_metrics = error_metrics_dict[state_variable_name]
            
            error_metrics_str = ', '.join([f"${key}={value:.4f}$" for key, value in error_metrics.items()])
            
            ax.text(0.98, 0.95, error_metrics_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
                    )
        
        # Find mean value of the true data in the specified time range
        values = ylim_data[stats, i]
        mean_value = values.mean()
        relative_max_value = np.max(np.abs(values - mean_value))

        # Set the y-axis limits based on the mean value and relative maximum value
        ax.set_ylim(mean_value - 1.1 * relative_max_value, mean_value + 1.1 * relative_max_value)
        
        # Set the x-axis limits
        ax.set_xlim(xlim)
        
        # Turn on grid for all subplots
        ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Turn off top and right spines for all subplots
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Make linewidth of bottom and left spines thin
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_linewidth(0.8)
        
    return fig, axs
