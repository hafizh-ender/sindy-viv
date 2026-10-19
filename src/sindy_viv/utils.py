
from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from typing import Any
import importlib
from functools import partial

# String formatting
def add_dot_text(feature_names: Sequence[str] | None) -> list[str]:
    """Return first/second/third LaTeX derivative labels for simple state names.

    Input order is retained. Names use an optional underscore subscript; this
    formatter does not differentiate arbitrary symbolic expressions.

    Parameters:
        feature_names (Sequence[str] or None): Ordered variable names to format with the next derivative label.
    """
    feature_names_with_dot = []

    for name in feature_names:
        if name.startswith(r'\dot{'):

            # Replace \dot{ with \ddot{
            new_name = name.replace(r'\dot{', r'\ddot{')
        elif name.startswith(r'\ddot{'):

            # Retain \ddot{ with \dddot{
            new_name = name.replace(r'\ddot{', r'\dddot{')
        else:

            # Get variable without subscript
            var_name = name.split('_')[0]
            subscript = name.split('_')[1] if '_' in name else ''
            new_name = f"\\dot{{{var_name}}}_{subscript}" if subscript else f"\\dot{{{var_name}}}"

        feature_names_with_dot.append(new_name)

    return feature_names_with_dot

def dict_to_string(dictionary: Mapping[str, Any]) -> str:
    """Format nested dictionaries as insertion-ordered hyphen-separated labels.

    This legacy display format is not a reversible serialization or a Python
    constructor expression. Values use their ordinary string representations.

    Parameters:
        dictionary (Mapping[str, Any]): Mapping to format using the legacy label convention.
    """
    return '-'.join([f"{key}{value}" if not isinstance(value, dict) else f"{key}[{dict_to_string(value)}]" for key, value in dictionary.items()])

def library_import_to_string(instance: str, module: str, kwargs: dict[str, Any]) -> str:
    """Return a legacy display label for a configured library constructor.

    The result is descriptive text, not executable Python; kwargs use dict_to_string.

    Parameters:
        instance (str): Configured constructor name to include in the display label.
        module (str): Importable module containing the requested class or function.
        kwargs (dict[str, Any]): Constructor keyword settings, including nested object descriptors.
    """
    return f"{module}.{instance}({dict_to_string(kwargs)})"

# Dynamic import
def _resolve_config_value(value: Any) -> Any:
    """Recursively resolve a descriptor, nested container, or unchanged literal value.

    A descriptor requires a module and exactly one class_name/function_name.
    Classes/factories are instantiated; functions become partial callables.
    Malformed descriptors raise ValueError instead of silently dropping keys.

    Parameters:
        value (object): Nested descriptor, container, or literal to resolve recursively.
    """
    if isinstance(value, dict):
        if 'module' not in value:
            return kwargs_to_object(value)

        names = [name for name in ('class_name', 'function_name') if name in value]
        if len(names) != 1:
            raise ValueError("Object descriptors require exactly one class_name or function_name.")

        name = names[0]
        if not isinstance(value['module'], str) or not value['module'] or not isinstance(value[name], str) or not value[name]:
            raise ValueError("Descriptor module and object names must be nonempty strings.")

        if set(value) - {'module', name, 'kwargs'}:
            raise ValueError("Unsupported object descriptor fields.")

        arguments = value.get('kwargs', {})
        if not isinstance(arguments, dict):
            raise ValueError("Descriptor kwargs must be a dictionary.")

        if name == 'class_name':
            return initiate_object(value['module'], value[name], arguments)

        module = importlib.import_module(value['module'])
        function = getattr(module, value[name])

        return partial(function, **kwargs_to_object(arguments))

    if isinstance(value, list):
        return [_resolve_config_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_resolve_config_value(item) for item in value)

    return value

def kwargs_to_object(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a new kwargs dictionary with nested object descriptors resolved.

    Ordinary dictionaries, lists and tuples retain their structure. Existing
    objects are reused, the input configuration is not mutated, and function
    descriptors produce partial callables rather than executing the function.

    Parameters:
        kwargs (dict[str, Any]): Constructor keyword settings, including nested object descriptors.
    """
    if not isinstance(kwargs, dict):
        raise ValueError("Object constructor kwargs must be a dictionary.")

    return {key: _resolve_config_value(value) for key, value in kwargs.items()}

def initiate_object(module: str, class_name: str, kwargs: dict[str, Any]) -> Any:
    """Import and call a configured class/factory with recursively resolved kwargs.

    module and class_name identify an importable callable. Nested object/function
    descriptors are resolved without mutating kwargs. Returns the constructed
    object; import, lookup and constructor errors retain their original causes.

    Parameters:
        module (str): Importable module containing the requested class or function.
        class_name (str): Name of the class or factory to instantiate.
        kwargs (dict[str, Any]): Constructor keyword settings, including nested object descriptors.
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

# Reading configuration
import json
import yaml

def load_config(config_path: str | PathLike[str], type: str = 'json') -> Any:
    """Load a JSON/YAML value without descriptor construction or schema validation.

    config_path identifies a text file; type selects json or yaml. The argument
    name is retained for compatibility. Returns the parsed value, usually a
    configuration dictionary, but JSON/YAML may also contain lists/scalars/null.
    Invalid formats raise ValueError; parser and file errors propagate.

    Parameters:
        config_path (str or PathLike[str]): Path to the JSON or YAML configuration file.
        type (str): Configuration format: json or yaml. Default: 'json'
    """
    if type == 'json':
        with open(config_path, 'r') as f:
            return json.load(f)
    elif type == 'yaml':
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Invalid configuration type. Please specify 'json' or 'yaml'.")
