import numpy as np

import matplotlib.pyplot as plt

import importlib
from functools import partial

"""
String formatting
"""
def add_dot_text(feature_names):
    """
    Add a dot to the feature names for LaTeX formatting.
    """
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

def dict_to_string(dict):
    """
    Convert a dictionary to a string representation.
    """
    return '-'.join([f"{key}{value}" if not isinstance(value, dict) else f"{key}[{dict_to_string(value)}]" for key, value in dict.items()])

def library_import_to_string(instance, module, kwargs):
    """
    Convert a library import to a string representation.
    """
    return f"{module}.{instance}({dict_to_string(kwargs)})"

"""
Dynamic import
"""
def kwargs_to_object(kwargs):
    """
    Check kwargs dictionary iteratively for object-like entries and initialize them as objects.
    Example:
    kwargs = {
        'feature_library': {
            'module': 'pysindy.feature_library',
            'class_name': 'PolynomialLibrary',
            'kwargs': {'degree': 3, 'include_bias': True}
        },
        'optimizer': {
            'module': 'pysindy.optimizers',
            'class_name': 'STLSQ',
            'kwargs': {'alpha': 0.1, 'threshold': 0.05}
        }
    }
    
    to
    
    kwargs = {
        'feature_library': PolynomialLibrary(degree=3, include_bias=True),
        'optimizer': STLSQ(alpha=0.1, threshold=0.05)
    }
    """
    processed_kwargs = {}    
    for key, value in kwargs.items():
        if isinstance(value, dict):
            if 'module' in value:
                # Dynamically import the module, then instantiate it with the provided arguments
                module = importlib.import_module(value['module'])
                
                if 'class_name' in value:
                    # Dynamically get the class from the module and instantiate it with the provided arguments
                    cls = getattr(module, value['class_name'])
                    
                    # Instantiate the class with the provided arguments
                    if 'kwargs' in value:
                        # Perform kwargs_to_object recursively on the kwargs to handle nested objects
                        nested_kwargs = kwargs_to_object(value['kwargs'])
                        
                        processed_kwargs[key] = cls(**nested_kwargs)
                    else:
                        # If no kwargs are provided, instantiate the class without arguments
                        processed_kwargs[key] = cls()
                elif 'function_name' in value:
                    # Dynamically get the function from the module and call it with the provided arguments
                    func = getattr(module, value['function_name'])
                    
                    # Define a partial function that calls the original function with the provided arguments
                    if 'kwargs' in value:
                        # Perform kwargs_to_object recursively on the kwargs to handle nested objects
                        nested_kwargs = kwargs_to_object(value['kwargs'])
                        
                        processed_kwargs[key] = partial(func, **nested_kwargs)
                    else:
                        processed_kwargs[key] = partial(func)
            else:
                # If 'module' is not in the dictionary, recursively process the dictionary to handle nested objects
                processed_kwargs[key] = kwargs_to_object(value)
                
        elif isinstance(value, list):
            # If the value is a list, check if any of its elements are dictionaries that need to be converted to objects
            processed_kwargs[key] = [
                kwargs_to_object(item) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            # If the value is not a dictionary or list, keep it as is
            processed_kwargs[key] = value
            
    return processed_kwargs

def initiate_object(module, class_name, kwargs):
    """
    Dynamically import a module and instantiate a class with the provided keyword arguments.
    
    Parameters:
        module (str): The module path to import the class from.
        class_name (str): The name of the class to instantiate.
        kwargs (dict): Additional keyword arguments to pass to the class constructor.
        
    Returns:
        An instance of the specified class.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as err:
        raise ImportError(f"Could not import module '{module}'.") from err

    try:
        cls = getattr(mod, class_name)
    except AttributeError as err:
        raise AttributeError(
            f"Module '{module}' has no attribute '{class_name}'."
        ) from err
    
    # Process kwargs to instantiate the class
    processed_kwargs = kwargs_to_object(kwargs)

    # Instantiate the class with the processed kwargs
    instance = cls(**processed_kwargs)
    
    return instance

"""
Reading configuration
"""
import json
import yaml

def load_config(config_path, type='json'):
    """
    Load a configuration file (JSON or YAML) and return it as a dictionary.
    
    Parameters:
        config_path (str): The path to the configuration file.
        type (str): The type of the configuration file ('json' or 'yaml').
        
    Returns:
        dict: The configuration as a dictionary.
    """
    if type == 'json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif type == 'yaml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Invalid configuration type. Please specify 'json' or 'yaml'.")
        