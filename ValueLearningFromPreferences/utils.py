import types
import argparse
import itertools
import json
import os
from types import LambdaType
from typing import List

import numpy as np

from use_cases.roadworld_env_use_case.values_and_costs import BASIC_PROFILES


MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS = os.path.join(MODULE_PATH, "checkpoints/")
TRAIN_RESULTS_PATH = os.path.join(MODULE_PATH, "train_results/")

os.makedirs(CHECKPOINTS, exist_ok=True)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            if hasattr(obj, "tolist"):  # numpy arrays have this
                return {"$array": obj.tolist()}  # Make a tagged object
        return super(NpEncoder, self).default(obj)


def convert(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    raise TypeError(x)


def deconvert(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
    return x


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


def sample_example_profiles(profile_variety, n_values=3, basic_profiles=BASIC_PROFILES) -> List:
    ratios = np.linspace(0, 1, profile_variety)

    if n_values < 1:
        raise ValueError('Need more values: n_values must be bigger than 0')
    if n_values == 1:
        profile_set = list(ratios)
    if n_values == 2:
        profile_set = [(1.0-ratio, ratio) for ratio in ratios]
    if n_values == 3:
        profile_combinations = [set(itertools.permutations((ratios[i], ratios[j], ratios[-i-j-1])))
                                for i in range(len(ratios)) for j in range(i, (len(ratios)-i+1)//2)]
    else:
        def recursFind(N, nc=3, i=0, t=0, p=[]):
            if nc == 1:
                # No need to explore, last value is N-t
                if N-t >= i:
                    yield p+[N-t]
                else:
                    # p+[N-t] is a solution, but it has already been given in another order
                    pass
            elif i*nc+t > N:
                # impossible to find nc values>=i (there are some <i. But that would yield to already given solutions)
                return
            else:
                for j in range(i, N):
                    yield from recursFind(N, nc-1, j, t+j, p+[j])

        profile_combinations = [set(itertools.permutations(
            ratios[i] for i in idx)) for idx in recursFind(len(ratios)-1, n_values)]

    if n_values >= 3:
        profile_set = list(set(tuple(
            float(f"{a_i:0.3f}") for a_i in a) for l in profile_combinations for a in l))
        [profile_set.remove(pr) for pr in basic_profiles]
        for pr in reversed(basic_profiles):
            profile_set.insert(0, pr)

    a = np.array(profile_set, dtype=np.dtype(
        [(f'{i}', float) for i in range(n_values)]))
    sortedprofiles = a[np.argsort(
        a, axis=-1, order=tuple([f'{i}' for i in range(n_values)]), )]
    profile_set = list(tuple(t) for t in sortedprofiles.tolist())

    profile_set = [tuple(round(num, 2) for num in t) for t in profile_set]

    return profile_set


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

