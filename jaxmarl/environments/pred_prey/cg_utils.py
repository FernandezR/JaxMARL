# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cg_utils.py

Coordination graph utilities for use with predator-prey environments.
"""

import jax.numpy as jnp
import numpy as np

from math import factorial
from random import randrange

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2024, Multi-Agent Synchronized Predator-Prey'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'


def cg_edge_list(cg_topology, num_agents) -> np.array:
    """
    Args:
        cg_topology (str, int, or list): Topology for the coordination graph
            if cg_topology is a string:
                Specifies edges for the following topologies:
                    empty - No connections'
                    trio  - Arrange agents in groups of 3
                    pairs - Arrange agents in groups of 2
                    line  - Arrange agents in a straight line
                    cycle - Arrange agents in a circle
                    star  - Arrange agents in a star around agent 0
                    full  - Connect all agents

            else if cg_topology is an integer:
                Specifies number of random edges to create for agent connections

            else if cg_topology is a list:
                Specifies the specific set of edges for agent connections

        num_agents (int):                Number of agents in the graph

    Returns:
        edges (jnp.array): List of agent connection edges for the coordination graph
    """
    edges = []

    if isinstance(cg_topology, str):
        if cg_topology == 'empty':
            pass
        elif cg_topology == 'trio':
            assert (num_agents % 3) == 0, "'trio' cg topology, is only for an odd number of agents divisible by 3 " \
                                          f"and will not work for the given '{num_agents}' number of agents"
            trios = list(zip(*[iter(range(num_agents))]*3))
            for trio in trios:
                edges.append([(i + trio[0], i + trio[0] + 1) for i in range(3 - 1)] + [(trio[-1], trio[0])])

            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'pairs':
            assert (num_agents % 2) == 0, "'pairs' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            edges = list(zip(*[iter(range(num_agents))]*2))
        elif cg_topology == 'double_line':
            assert (num_agents % 2) == 0, "'double_line' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(i, i + 1) for i in range(half - 1)])
            edges.append([(i, i + 1) for i in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'line':
            edges = [(i, i + 1) for i in range(num_agents - 1)]
        elif cg_topology == 'double_cycle':
            assert (num_agents % 2) == 0, "'double_cycle' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(i, i + 1) for i in range(half - 1)] + [(half - 1, 0)])
            edges.append([(i, i + 1) for i in range(half, num_agents - 1)] + [(num_agents - 1, half)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'cycle':
            edges = [(i, i + 1) for i in range(num_agents - 1)] + [(num_agents - 1, 0)]
        elif cg_topology == 'double_star':
            assert (num_agents % 2) == 0, "'double_star' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([(0, i + 1) for i in range(half - 1)])
            edges.append([(half, i + 1) for i in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        elif cg_topology == 'star':
            edges = [(0, i + 1) for i in range(num_agents - 1)]
        elif cg_topology == 'double_full':
            assert (num_agents % 2) == 0, "'double_full' cg topology, is only for an even number of agents and will " \
                                          f"not work for the given '{num_agents}' number of agents"
            half = int(num_agents / 2)
            edges.append([[(j, i + j + 1) for i in range(half - j - 1)] for j in range(half - 1)])
            edges.append([[(j, i + j + 1) for i in range(num_agents - j - 1)] for j in range(half, num_agents - 1)])
            # Flatten edges list
            edges = [edge for subgraph in edges for sublist in subgraph for edge in sublist]
        elif cg_topology == 'full':
            edges = [[(j, i + j + 1) for i in range(num_agents - j - 1)] for j in range(num_agents - 1)]
            # Flatten edges list
            edges = [edge for sublist in edges for edge in sublist]
        else:
            raise ValueError("[jaxmarl.environments.pred_prey.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be one of the following when it is a string: "
                             "{'empty','trio','pairs','double_line','line','double_cycle','cycle',"
                             "'double_star','star','double_full','full'}")

    elif isinstance(cg_topology, int):
        if 0 <= cg_topology <= factorial(num_agents - 1):
            raise ValueError("[jaxmarl.environments.pred_prey.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be (<= n_agents!) when it is an integer")

        for i in range(cg_topology):
            edge_found = False
            while not edge_found:
                edge = (randrange(num_agents), randrange(num_agents))
                if (edge[0] != edge[1]) and (edge not in edges) and ((edge[1], edge[0]) not in edges):
                    edges.append(edge)
                    edge_found = True

    elif isinstance(cg_topology, list):
        # TODO: May need need to do check for duplicate edges
        if all([isinstance(edge, tuple) and (len(edge) == 2) and (all([isinstance(i, int) for i in edge]))
                for edge in cg_topology]):
            raise ValueError("[jaxmarl.environments.pred_prey.cg_utils.cg_edge_list()]: "
                             "Parameter cg_topology must be a list of int-tuples of length 2 with no duplicates "
                             "when it is a list specifying the agent connections.")
        edges = cg_topology

    else:
        raise ValueError("[jaxmarl.environments.pred_prey.cg_utils.cg_edge_list()]: "
                         f"{type(cg_topology)}, not supported for parameter cg_topology. "
                         "Parameter cg_topology must be either one of these string options "
                         "{'empty','trio','pairs','double_line','line','double_cycle','cycle',"
                         "'double_star','star','double_full','full'}, an integer for the number "
                         "of random edges that is (<= n_agents!), or a list of int-tuples for "
                         "each direct edge specification.")

    return np.array(edges)

def set_cg_adj_matrix(edges, num_agents) -> jnp.ndarray:
    """
    Takes a list of tuples [0..n_agents)^2 and constructs the internal CG edge representation as an adjacency matrix
    for use with Graph Neural Networks

    Args:
        edges (list):     List of agent connection edges for the coordination graph
        num_agents (int): Number of agents in the graph

    Returns:
        adjacency_matrix (torch.Tensor): Adjacency matrix representing the fixed coordination graph

    """
    adjacency_matrix = jnp.zeros((num_agents, num_agents), dtype=jnp.float32)
    for edge in edges:
        adjacency_matrix = adjacency_matrix.at[edge[0], edge[1]].set(1)
        adjacency_matrix = adjacency_matrix.at[edge[1], edge[0]].set(1)

    return adjacency_matrix
