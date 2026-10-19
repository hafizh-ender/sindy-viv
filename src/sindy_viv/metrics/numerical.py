
from __future__ import annotations

from numpy.typing import ArrayLike
from typing import Literal
import numpy as np

def _check_input_target_shapes(input: ArrayLike, target: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return shape-aligned arrays with at least one dimension.

    Numeric storage is retained; other convertible values become float64.
    Scalar inputs become length-one arrays. Mismatched shapes raise ValueError;
    no normalization, finite filtering or NaN replacement is performed.

    Parameters:
        input (array-like): Predicted samples, shape-aligned with target.
        target (array-like): Reference samples used for normalization.
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

def _reduce(values: ArrayLike, reduction: Literal["mean", "sum", "none"]) -> float | np.ndarray:
    """Reduce per-channel metric values by mean/sum or return them unchanged.

    NaN values propagate. Any reduction other than mean, sum or none raises ValueError.

    Parameters:
        values (array-like): Per-channel metric values to reduce.
        reduction (Literal['mean', 'sum', 'none']): How to combine channel metrics: mean, sum, or none.
    """
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

def _squared_sums(
    input: ArrayLike,
    target: ArrayLike,
    block_size: int,
    centered: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate in float64 using one bounded scratch array.

    Centered sums use two passes to avoid cancellation for signals with
    large offsets and small fluctuations.

    Parameters:
        input (array-like): Predicted samples, shape-aligned with target.
        target (array-like): Reference samples used for normalization.
        block_size (int): Maximum rows per temporary feature or metric block; must be positive.
        centered (bool): Whether to center reference values before accumulating squared sums.
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

def normalized_root_mean_squared_error(
    input: ArrayLike,
    target: ArrayLike,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    *,
    block_size: int = 10000,
) -> float | np.ndarray:
    """Calculate the normalized root mean squared error (NRMSE) between input and target arrays.
    The NRMSE is computed as the RMSE divided by the standard deviation of the target.

    Parameters:
        input (array-like): The input array. Shape: (N, num_variables).
        target (array-like): The target array. Shape: (N, num_variables).
        reduction (Literal['mean', 'sum', 'none']): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'. Default: 'mean'
        block_size (int): Maximum rows of float64 temporary storage. Default: 10000

    Returns:
        float or numpy.ndarray: The normalized root mean squared error (NRMSE).


    Notes:
        Reductions first accumulate over sample axis zero in float64, then
        combine channel values. Scalar/1-D inputs and higher-dimensional arrays
        are accepted when shapes match. Undefined denominators return NaN,
        which propagates through mean/sum; reduction='none' preserves channels.
    """
    error_sum, centered_sum = _squared_sums(input, target, block_size, centered=True)

    # Constant target channels retain the existing NaN normalization behavior.
    denominator = np.where(centered_sum > 0, centered_sum, np.nan)

    # The common sample-count factor in RMSE and std cancels.
    return _reduce(np.sqrt(error_sum / denominator), reduction)

def relative_l2_error(
    input: ArrayLike,
    target: ArrayLike,
    reduction: Literal['mean', 'sum', 'none'] = 'mean',
    *,
    block_size: int = 10000,
) -> float | np.ndarray:
    """Calculate the relative L2 error between input and target arrays.
    The relative L2 error is computed as the L2 norm of the difference between input and target, divided by the L2 norm of the target.

    Parameters:
        input (array-like): The input array. Shape: (N, num_variables).
        target (array-like): The target array. Shape: (N, num_variables).
        reduction (Literal['mean', 'sum', 'none']): Specifies the reduction to apply to the output. Options are 'mean', 'sum', or 'none'. Default: 'mean'
        block_size (int): Maximum rows of float64 temporary storage. Default: 10000

    Returns:
        float or numpy.ndarray: The relative L2 error.


    Notes:
        Reductions first accumulate over sample axis zero in float64, then
        combine channel values. Scalar/1-D inputs and higher-dimensional arrays
        are accepted when shapes match. Undefined denominators return NaN,
        which propagates through mean/sum; reduction='none' preserves channels.
    """
    error_sum, target_sum = _squared_sums(input, target, block_size, centered=False)

    # A zero target norm makes relative error undefined rather than infinite.
    denominator = np.where(target_sum > 0, target_sum, np.nan)

    return _reduce(np.sqrt(error_sum) / np.sqrt(denominator), reduction)
