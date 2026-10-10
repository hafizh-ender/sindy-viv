import numpy as np

def _check_input_target_shapes(input, target):
    """
    Check if the input and target arrays have the same shape.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
    """
    input = np.asarray(input)
    target = np.asarray(target)
    
    if input.shape != target.shape:
        raise ValueError(
            f"Input and target arrays must have the same shape. Got input shape {input.shape} and target shape {target.shape}."
        )
    
    # Keep numeric inputs in their existing storage. Arithmetic is converted
    # to float64 one block at a time below.
    if input.dtype.kind not in "biuf":
        input = input.astype(np.float64)
        
    if target.dtype.kind not in "biuf":
        target = target.astype(np.float64)
        
    return np.atleast_1d(input), np.atleast_1d(target)

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

def _squared_sums(input, target, block_size, centered):
    """Accumulate in float64 using one bounded scratch array.

    Centered sums use two passes to avoid cancellation for signals with
    large offsets and small fluctuations.
    """
    if not isinstance(block_size, (int, np.integer)) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    
    input, target = _check_input_target_shapes(input, target)
    
    error_sum = np.zeros(target.shape[1:], dtype=np.float64)
    target_sum = np.zeros_like(error_sum)
    
    # Center before squaring to retain variance accuracy for signals with large offsets.
    mean = np.mean(target, axis=0, dtype=np.float64) if centered else None
    
    # Bound temporary memory by block_size while retaining float64 arithmetic.
    scratch = np.empty((min(block_size, len(target)), *target.shape[1:]), dtype=np.float64)
    for start in range(0, len(target), block_size):
        stop = min(start + block_size, len(target))
        
        buffer = scratch[:stop - start]
        
        np.subtract(input[start:stop], target[start:stop], out=buffer, dtype=np.float64)
        
        np.square(buffer, out=buffer)
        
        error_sum += np.sum(buffer, axis=0)
        
        # Reuse the buffer for centered variance or the uncentered L2 denominator.
        if centered:
            np.subtract(target[start:stop], mean, out=buffer, dtype=np.float64)
        else:
            np.copyto(buffer, target[start:stop])
            
        np.square(buffer, out=buffer)
        target_sum += np.sum(buffer, axis=0)
        
    return error_sum, target_sum


def normalized_root_mean_squared_error(input, target, reduction='mean', *, block_size=10_000):
    """
    Calculate the normalized root mean squared error (NRMSE) between input and target arrays.
    The NRMSE is computed as the RMSE divided by the standard deviation of the target.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
        reduction (str): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.
        block_size (int): Maximum rows of float64 temporary storage.
        
    Returns:
        float or numpy.ndarray: The normalized root mean squared error (NRMSE).
    """
    error_sum, centered_sum = _squared_sums(input, target, block_size, centered=True)
    # Constant target channels retain the existing NaN normalization behavior.
    denominator = np.where(centered_sum > 0, centered_sum, np.nan)
    # The common sample-count factor in RMSE and std cancels.
    return _reduce(np.sqrt(error_sum / denominator), reduction)

def relative_l2_error(input, target, reduction='mean', *, block_size=10_000):
    """
    Calculate the relative L2 error between input and target arrays.
    The relative L2 error is computed as the L2 norm of the difference between input and target, divided by the L2 norm of the target.
    
    Parameters:
        input (numpy.ndarray): The input array. Shape: (N, num_variables)
        target (numpy.ndarray): The target array. Shape: (N, num_variables)
        reduction (str): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'.
        block_size (int): Maximum rows of float64 temporary storage.

    Returns:
        float or numpy.ndarray: The relative L2 error.
    """
    error_sum, target_sum = _squared_sums(input, target, block_size, centered=False)
    # A zero target norm makes relative error undefined rather than infinite.
    denominator = np.where(target_sum > 0, target_sum, np.nan)
    return _reduce(np.sqrt(error_sum) / np.sqrt(denominator), reduction)
