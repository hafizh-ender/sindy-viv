import numpy as np

def _check_input_target_shapes(input, target):
    """
    Check if the input and target arrays have the same shape.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
    """
    input = np.asarray(input, dtype=float)
    target = np.asarray(target, dtype=float)
    
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target arrays must have the same shape. Got input shape {input.shape} and target shape {target.shape}."
        )
    
    return input, target

def _reduce(values, reduction):
    if reduction == 'mean':
        return np.mean(values)
    if reduction == 'sum':
        return np.sum(values)
    if reduction == 'none':
        return values
    raise ValueError(
        f"Unknown reduction '{reduction}'. "
        "Supported reductions: 'mean', 'sum', 'none'."
    )

def normalized_root_mean_squared_error(input, target, reduction='mean'):
    """
    Calculate the normalized root mean squared error (NRMSE) between input and target arrays.
    The NRMSE is computed as the RMSE divided by the standard deviation of the target and expressed as a percentage.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
        reduction (str): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.
        
    Returns:
        float or numpy.ndarray: The normalized root mean squared error (NRMSE) as a percentage.
    """
    input, target = _check_input_target_shapes(input, target)
    
    diff = input - target
    
    rmse = np.sqrt(np.mean(diff**2, axis=0))  # Collapse the first axis (time dimension) to compute RMSE for each variable
    
    denominator = np.std(target, axis=0)
    denominator = np.where(denominator > 0, denominator, np.nan)  # Avoid division by zero; set to NaN if std is zero
    
    nrmse = rmse / denominator * 100  # Express as a percentage
    
    return _reduce(nrmse, reduction)

def relative_l2_error(input, target, reduction='mean'):
    """
    Calculate the relative L2 error between input and target arrays.
    The relative L2 error is computed as the L2 norm of the difference between input and target, divided by the L2 norm of the target, and expressed as a percentage.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
        reduction (str): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.

    Returns:
        float or numpy.ndarray: The relative L2 error as a percentage.
    """
    input, target = _check_input_target_shapes(input, target)
    
    diff = input - target
    
    l2e = np.linalg.norm(diff, axis=0)  # Collapse the first axis (time dimension) to compute L2 error for each variable
    
    denominator = np.linalg.norm(target, axis=0)
    denominator = np.where(denominator > 0, denominator, np.nan)  # Avoid division by zero; set to NaN if L2 norm is zero
    
    rl2e = l2e / denominator * 100  # Express as a percentage
    
    return _reduce(rl2e, reduction)