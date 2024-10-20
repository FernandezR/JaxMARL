# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py

Utilities for use with Synchronized Predator-prey
"""

import jax
import jax.numpy as jnp
import random
import torch
import yaml

from collections import namedtuple
from typing import Any, Optional, Tuple, Union, List, Dict

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2024, Multi-Agent Synchronized Predator-Prey'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'

def jax_to_torch(jax_array: jax.numpy.ndarray) -> torch.Tensor:
    """
    Converts jax array to torch tensor using dlpack
    """
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_array))

def torch_to_jax(torch_tensor: torch.Tensor) -> jax.numpy.ndarray:
    """
    Converts jax array to torch tensor using dlpack
    """
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(torch_tensor))

def load_yaml_config(config_file_path):
    """
    Load and parse yaml config file

    Args:
        config_file_path (str): Path to the yaml config file

    Returns:
        config_dict (dict): Dictionary with configuration parameters
    """
    # Create loader
    loader = yaml.SafeLoader

    # Custom tag handler for joining strings
    def join(loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])

    def eval_handler(loader, node):
        seq = loader.construct_sequence(node)
        return eval(seq[0])

    # Register the join tag handler
    loader.add_constructor('!join', join)

    # Register the eval tag handler
    loader.add_constructor('!eval', eval_handler)

    # Load and parse config file
    with open(config_file_path, "r") as config_file:
        try:
            config_dict = yaml.load(config_file, Loader=loader)
        except yaml.YAMLError:
            raise yaml.YAMLError("[jaxmarl.environments.pred_prey.utils.load_yaml_config()]: "
                                 f"Error with config file, {config_file_path}")
    return config_dict

def dict2namedtuple(dictionary):
    """
    Converts a dictionary to namedtuple.

    Parameters
    ----------
    dictionary : dict
                 Dictionary object to be converted.

    Returns
    -------
    Object : namedtuple
        The namedtuple conversion of the input dictionary.
    """
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)

def generate_random_predator_color() -> Tuple[float, float, float]:
    """
    Generates a random RGB color tuple, excluding specified colors.

    Black, White, and Green are excluded
    """
    # Black, White, and Green
    excluded_colors = [(0.0, 0.0, 0.0), (255.0, 255.0, 255.0), (0.0, 255.0, 0.0)]

    def is_excluded(color, excluded_color):
        return any(abs(c1 - c2) < 10 for c1, c2 in zip(color, excluded_color))

    while True:
        color = (float(random.randint(0, 255)),  # R
                 float(random.randint(0, 255)),  # G
                 float(random.randint(0, 255)))  # B

        # Check if color is excluded
        # If it is not return the color
        # Else generate a new color
        if not any(is_excluded(color, excluded_color) for excluded_color in excluded_colors):
            return color

def check_colors(color_list: list[tuple[float, float, float], ...]):
    """ Prints colors in a color list. """
    from colorama import init, Fore, Back, Style

    # Initialize colorama
    init()

    for color in color_list:
        # Use colorama to create a colored string
        r ,g, b = color

        # Color code by component
        color = Fore.RED + str(r) + Style.RESET_ALL + ", " + \
                Fore.GREEN + str(g) + Style.RESET_ALL + ", " + \
                Fore.BLUE + str(b) + Style.RESET_ALL

        # Print color code
        print(color)

        # Print the word COLOR in the given color code
        print(f"\033[38;2;{r};{g};{b}mCOLOR\033[0m")

def jax_chebyshev(x: jnp.ndarray, y: jnp.ndarray, axis: int = None, keepdims: bool = False) -> jnp.ndarray:
    """
    Compute the Chebyshev distance.

    Computes the Chebyshev distance between two 1-D arrays `x` and `y`,
    which is defined as

    Taken from scipy and converted for use with jax.

    .. math::

       \\max_i {|x_i-y_i|}.

    Parameters
    ----------
    x :        (N,) jnp.ndarray
               Input vector.
    y :        (N,) jnp.ndarray
               Input vector.
    axis :     int
               Axis along which to compute the Chebyshev distance. (Default: None)
    keepdims : bool
               Whether to keep the dimensions of the input.

    Returns
    -------
    chebyshev : double
        The Chebyshev distance between vectors `x` and `y`.
    """
    return jnp.max(jnp.abs(x - y), axis=axis, keepdims=keepdims)
