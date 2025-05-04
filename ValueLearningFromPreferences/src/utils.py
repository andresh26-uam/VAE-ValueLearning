import types
import argparse
import json
from types import LambdaType
import numpy as np


def merge_dicts_recursive(base, update):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dicts_recursive(base[key], value)
            else:
                if key not in base:
                    base[key] = value


def train_test_split_initial_state_distributions(n_states, split_percentage=0.7):
    n = n_states

    # Step 2: Randomly select 70% of the indexes
    indices = np.arange(n)
    np.random.shuffle(indices)

    split_point = int(split_percentage * n)
    first_indices = indices[:split_point]
    second_indices = indices[split_point:]

    # Step 3: Create uniform distributions
    first_distribution = np.zeros(n)
    second_distribution = np.zeros(n)

    first_distribution[first_indices] = 1 / len(first_indices)
    second_distribution[second_indices] = 1 / len(second_indices)

    # Output the distributions
    return first_distribution, second_distribution

def load_json_config(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def filter_none_args(args):
    """Removes arguments that have a None value."""
    filtered_args = {k: v for k, v in vars(args).items() if v is not None}
    return argparse.Namespace(**filtered_args)


def serialize_lambda(func):
    """Serialize a lambda function by extracting its source code."""
    if isinstance(func, LambdaType):
        if func.__code__.co_argcount != 2:
            raise ValueError(
                "state_encoder lambda must accept exactly 2 arguments: (state_obs, info)")
        return {
            'co_code': func.__code__.co_code.hex(),
            'co_varnames': func.__code__.co_varnames,
            'co_names': func.__code__.co_names,
            'co_consts': func.__code__.co_consts,
            'co_flags': func.__code__.co_flags,
        }
    raise ValueError(
        "State encoder must be a lambda or serializable function.")


def deserialize_lambda(code_data):
    bytecode = bytes.fromhex(code_data['co_code'])
    code = types.CodeType(
        2,  # Number of arguments (co_argcount)
        0,  # Positional-only arguments (Python 3.8+)
        0,  # KW-only arguments (Python 3.8+)
        len(code_data['co_varnames']),  # nlocals
        2,  # stacksize
        code_data['co_flags'],  # flags
        bytecode,  # code
        tuple(code_data['co_consts']),  # constants
        tuple(code_data['co_names']),  # names
        tuple(code_data['co_varnames']),  # variable names
        "",  # filename (empty string)
        "<lambda>",  # name
        0,  # first line number
        b""  # lnotab (empty)
    )
    return types.FunctionType(code, globals())


def import_from_string(module_class_name):
    """Import a class from its module and name."""
    splitted = module_class_name.rsplit(".", 1)
    if len(splitted) == 1:
        return splitted[0]
    module_name, class_name = splitted
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


def serialize_policy_kwargs(policy_kwargs):
    """Serialize policy_kwargs with special handling for classes."""
    serialized_kwargs = {}
    for key, value in policy_kwargs.items():
        if isinstance(value, type):  # Handle class types
            serialized_kwargs[key] = value.__module__ + "." + value.__name__
        else:
            serialized_kwargs[key] = value
    return serialized_kwargs


def deserialize_policy_kwargs(serialized_policy_kwargs):
    """Deserialize policy_kwargs, reconstructing classes where necessary."""
    deserialized_kwargs = {}
    for key, value in serialized_policy_kwargs.items():
        if isinstance(value, str) and '.' in value:  # Potential class name
            try:
                deserialized_kwargs[key] = import_from_string(value)
            except (ImportError, AttributeError):
                deserialized_kwargs[key] = value  # Fallback to string
        else:
            deserialized_kwargs[key] = value
    return deserialized_kwargs


def print_tensor_and_grad_fn(grad_fn, level=0):
    indent = "  " * level
    if grad_fn is None:
        return
    if getattr(grad_fn, 'variable', None) is not None:
        print(f"{indent}AccumulateGrad for tensor: {grad_fn.variable}")
    else:
        print(f"{indent}Grad function: {grad_fn}")
        if hasattr(grad_fn, 'next_functions'):
            for next_fn in grad_fn.next_functions:
                if next_fn[0] is not None:
                    print_tensor_and_grad_fn(next_fn[0], level + 1)

