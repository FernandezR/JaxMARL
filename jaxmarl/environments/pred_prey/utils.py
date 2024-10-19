# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
utils.py

Utilities for use with Synchronized Predator-prey
"""

import jax
import torch

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
