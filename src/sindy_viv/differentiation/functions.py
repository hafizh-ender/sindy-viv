import importlib

from sindy_viv.utils import kwargs_to_object, initiate_object
    
def perform_derivative(time, data, differentiator):
    """
    Differentiate data using a differentiator object. 
    The differentiator object should implement the `__call__` method that takes `data` and `time` as arguments.

    Parameters:
        time (numpy.ndarray or float): Time points, or a scalar time step.
        data (numpy.ndarray): Data to differentiate. Shape: (N, num_variables)
        differentiator (object): An instance of a differentiator class.

    Returns:
        numpy.ndarray: The differentiated data, same shape as `data`.
    """
    # Apply the differentiator to the data
    return differentiator(data, time)