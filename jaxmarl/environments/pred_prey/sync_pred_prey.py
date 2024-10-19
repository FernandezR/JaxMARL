# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sync_pred_prey.py

Synchronized Predator-prey

A version of predator-prey that meets the requirement of a
Multi-Agent Synchronization Task as defined in []()
"""

# TODO: Fix float64 errors with adjacency matrix

import chex
import jax
import jax.numpy as jnp
import math
import numpy as np
import random

from networkx.linalg.graphmatrix import adjacency_matrix
from overrides import override

from collections import namedtuple
from enum import IntEnum
from flax.struct import dataclass
from functools import partial
from typing import Any, Optional, Tuple, Union, List, Dict

from jaxmarl.environments.pred_prey import cg_utils

from jaxmarl.environments import spaces, overcooked_layouts
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from .rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2024, Multi-Agent Synchronized Predator-Prey'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'

###########################################
#                 Globals                 #
###########################################
SEEDRANGE = (1, int(1e9))

PLAYER1_COLOUR = (255.0, 127.0, 14.0)  # kinda orange
PLAYER2_COLOUR = (31.0, 119.0, 180.0)  # kinda blue
PLAYER3_COLOUR = (236.0, 64.0, 122.0)  # kinda pink
PLAYER4_COLOUR = (255.0, 235.0, 59.0)  # yellow
PLAYER5_COLOUR = (41.0, 182.0, 246.0)  # baby blue
PLAYER6_COLOUR = (171.0, 71.0, 188.0)  # purple
PLAYER7_COLOUR = (121.0, 85.0, 72.0)   # brown
PLAYER8_COLOUR = (255.0, 205.0, 210.0) # salmon
PLAYER9_COLOUR = (44.0, 160.0, 44.0)   # green
RED_COLOUR = (214.0, 39.0, 40.0)

###########################################
#             Util Functions              #
###########################################
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

###########################################
#           Env Support Classes           #
###########################################
@dataclass
class State:
    """ State class for the Synchronized Predator-prey environment. """
    agent_positions: jnp.ndarray
    prey_positions: jnp.ndarray
    agent_actives: jnp.ndarray
    prey_actives: jnp.ndarray
    shared_reward: jnp.ndarray
    shared_true_reward:jnp.ndarray
    grid: jnp.ndarray
    step: jnp.ndarray
    terminal: jnp.ndarray
    ep_info: dict[str, Any]
    adjacency_mask: jnp.ndarray

###########################################
#            Main Env Class               #
###########################################
class SyncPredPrey(MultiAgentEnv):
    """
    JAX Compatible version of the Synchronized Predator Prey environment.
    """
    ###########################################
    #            Class properties             #
    #                                         #
    # Class properties are accessed using     #
    # cls instead of self                     #
    ###########################################
    # Used for caching tiles for rendering
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    grid = None

    def __init__(self, args, seed=random.randint(*SEEDRANGE)):
        """
        Initializes the Synchronized Predator-Prey environment.

        Args:
            args (types.SimpleNamespace/dict): Parsed configuration arguments
            seed (int):                        Seed for random number generator
        """
        # Convert env args dictionary to namedtuple
        if isinstance(args, dict):
            self.args = dict2namedtuple(args)
            args = dict2namedtuple(args)

        # Initialize MultiAgentEnv variables
        super().__init__(num_agents=getattr(args, "num_predators", 8))

        # Specifies whether state and obs spaces are flattened
        self.cnn = True

        # Flag for whether to expect actions to be torch tensors
        # and if torch tensors should be returned be step and reward functions
        self.use_torch = getattr(args, "use_torch", False)

        # Create initial random generator key based on a seed
        self.prng = jax.random.key(seed)
        # self.np_prng = np.random.default_rng(seed)

        ###########################################
        #             Env Properties              #
        ###########################################
        # World
        self.num_predators = getattr(args, "num_predators", 8)
        self.num_prey = getattr(args, "num_prey", 8)
        self.num_entities = self.num_predators + self.num_prey
        self.episode_limit = getattr(args, "episode_limit", 200)
        self.episode_limit_jax = jnp.array([getattr(args, "episode_limit", 200)], dtype=jnp.int16)
        self.observe_ids = getattr(args, "observe_ids", True)
        self.world_shape = np.array(getattr(args, "world_shape", [10,10]), dtype=np.int8)
        self.world_shape_jax = jnp.array(getattr(args, "world_shape", [10, 10]), dtype=jnp.int8)
        self.truncate_episodes = getattr(args, "truncate_episodes", True)
        self.features_enum = IntEnum('Features', ['empty'] +
                                            [f'predator_{i + 1}' for i in range(self.num_predators)] +
                                            ['prey', 'wall'], start=0)
        self.num_features = len(self.features_enum) - 1

        # State
        self.state_shape = self.world_shape.tolist() + [self.num_features]
        self.state_size = np.prod(self.state_shape)

        # Agent
        self.agent_obs = np.array(getattr(args, "agent_obs", [2,2]), dtype=np.int8)
        self.agent_obs_size = self.agent_obs * 2 + 1
        self.agent_obs_len = np.prod(self.agent_obs_size) * self.num_features

        # Prey
        self.prey_rest = getattr(args, "prey_rest", 0.0)

        # Obs
        self.observe_current_timestep = getattr(args, "observe_current_timestep", False)
        self.observe_grid_pos = getattr(args, "observe_grid_pos", False)
        self.world_shape_oversized = self.world_shape + (self.agent_obs * 2)

        # Reward
        self.modified_penalty_condition = getattr(args, "modified_penalty_condition", None)
        self.miscapture_punishment = getattr(args, "miscapture_punishment", -2)
        self.reward_capture = getattr(args, "reward_capture", 10)
        self.reward_collision = getattr(args, "reward_collision", 0)
        self.reward_time = getattr(args, "reward_time", 0)

        # Shaping Reward
        self.use_shaping_reward = getattr(args, "use_shaping_reward", False)
        self.shaping_reward_condition = getattr(args, "shaping_reward_condition", 2)
        self.shaping_reward = getattr(args, "shaping_reward", 0.5)
        self.shaping_reward_disable_timestep = getattr(args, "shaping_reward_disable_timestep", None)

        # Capture
        self.capture_action = getattr(args, "capture_action", True)
        self.capture_action_conditions = getattr(args, "capture_action_conditions", 2)
        self.capture_conditions = getattr(args, "capture_conditions", 0)
        self.capture_freezes = getattr(args, "capture_freezes", True)
        self.capture_terminal = getattr(args, "capture_terminal", False)
        self.diagonal_capture = getattr(args, "diagonal_capture", False)
        self.remove_frozen = getattr(args, "remove_frozen", True)
        self.num_capture_actions = getattr(args, "num_capture_actions", 1)

        # Actions
        if self.diagonal_capture:
            self.diagonal_actions = jnp.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=jnp.int8)

        # Using a dictionary when creating the enum allows for accessing the enum
        # by name (i.e. self.actions_enum.stay or self.actions_enum['stay']
        # or by value (i.e. self.actions_enum(0))
        # adding .name (i.e. self.actions_enum(0).name)  will return the name of the item.
        actions = {'stay': 0, 'left': 1, 'right': 2, 'up': 3, 'down': 4}
        self.num_movement_actions = 4
        if self.capture_action:
            if self.num_capture_actions > 1:
                for i in range(self.num_capture_actions):
                    actions[f'capture_{i}'] = 5 + i
            else:
                actions[f'capture'] = 5

            self.actions_enum = IntEnum('Actions', actions)
            self.grid_actions = jnp.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]] +
                                          [[0, 0]] * self.num_capture_actions, dtype=jnp.int8)
        else:
            self.actions_enum = IntEnum('Actions', actions)
            self.grid_actions = jnp.array([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]], dtype=jnp.int8)

        # Core
        self.dimension_position = 2  # x,y
        self.num_positions = np.prod(self.world_shape)
        self.grid_position_indicies = jnp.array(range(self.num_positions), dtype=jnp.int8)
        # Grid positions are indexed in y,x [row, column] order
        self.grid_positions = jnp.array(np.meshgrid(np.array(range(self.world_shape[0])),
                                                    np.array(range(self.world_shape[1]))),
                                        dtype=jnp.int8).T.reshape(-1, 2)

        # Spaces
        _state_shape = (
            (self.state_shape[0], self.state_shape[1],self.state_shape[2])
            if self.cnn
            else (self.state_size,)
        )
        self.state_spaces = spaces.Box(low=0, high=1, shape=_state_shape, dtype=jnp.uint8)
        _obs_shape = (
            (self.agent_obs_size[0], self.agent_obs_size[1], self.num_features)
            if self.cnn
            else (self.agent_obs_len,)
        )
        self.observation_spaces = {
            i: spaces.Box(low=0, high=1, shape=_obs_shape, dtype=jnp.uint8) for i in range(self.num_predators)
        }
        self.action_spaces = {
            i: spaces.Discrete(len(self.actions_enum)) for i in range(self.num_predators)
        }

        # Adjacency Matrix
        self.use_adj_matrix = getattr(args, "use_adj_matrix", False)
        self.use_adj_mask = getattr(args, "use_adj_mask", False)
        self.use_self_attention = getattr(args, "cg_topology", None)
        self.fixed_graph_topology = getattr(args, "cg_topology", None)
        self.use_fixed_graph = False if self.fixed_graph_topology is None else True
        self.proximal_distance = getattr(args, "proximal_distance", 2.0)
        self.dropout = getattr(args, "dropout", False)
        self.dropout_prob = getattr(args, "dropout_prob", 0.35)

        if self.use_adj_matrix and not self.use_fixed_graph:
            pairs = [[(j, i + j + 1) for i in range(self.num_predators - j - 1)] for j in range(self.num_predators - 1)]
            pairs = [pair for sublist in pairs for pair in sublist]
            self.pairs = jnp.array(pairs, dtype=jnp.int8)

    ###########################################
    #               Properties                #
    ###########################################
    @property
    def name(self) -> str:
        """ Environment name. """
        return "Synchronized Predator-Prey"

    @property
    def num_actions(self) -> int:
        """ Number of actions possible in environment. """
        return len(self.actions_enum)

    @property
    def state_space(self) -> spaces.Box:
        """ State space of the environment. """
        return self.state_spaces

    ###########################################
    #               Utilities                 #
    ###########################################
    @partial(jax.jit, static_argnums=(0,))
    def reset_state(self, prng: chex.PRNGKey):
        """ Resets the state of the environment. """
        # Get key for random generator and update old key
        # By default this returns two keys unless you specify otherwise
        prng, key = jax.random.split(prng)

        # Create empty grid
        grid = jnp.zeros((self.world_shape[0], self.world_shape[1]), jnp.int8)

        # Randomly select the grid positions for all entities
        entity_grid_pos_indicies = jax.random.choice(key,
                                                     self.grid_position_indicies,
                                                     shape=(self.num_entities,),
                                                     replace=False)

        # Add predators to grid
        predator_positions = self.grid_positions[entity_grid_pos_indicies[:self.num_predators]]
        def add_predator_to_grid(i: int, val: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Function for adding predators to grid
            """
            grid1, predator_pos = val
            grid1 = grid1.at[predator_pos[i][0],
                             predator_pos[i][1]].set(jnp.int8(i+1))
            return grid1, predator_pos

        grid, _ = jax.lax.fori_loop(0, self.num_predators, add_predator_to_grid,
                                    (grid, predator_positions))

        # Add prey to grid
        prey_positions = self.grid_positions[entity_grid_pos_indicies[self.num_predators:]]
        grid = grid.at[prey_positions[:, 0], prey_positions[:, 1]].set(jnp.int8(self.features_enum.prey))
        # Save for use if prey become different
        # def add_prey_to_grid(i: int, val: Tuple[jnp.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #     """
        #     Function for adding prey to grid
        #     """
        #     grid1, prey_pos = val
        #     grid1 = grid1.at[prey_pos[i][0],
        #                      prey_pos[i][1]].set(jnp.int8(self.features_enum.prey))
        #     return grid1, prey_pos
        #
        # grid, _ = jax.lax.fori_loop(0, self.num_prey, add_prey_to_grid,
        #                             (grid, prey_positions))

        # Reset the State of the environment
        state = State(agent_positions=predator_positions,
                      prey_positions=prey_positions,
                      agent_actives=jnp.ones((self.num_predators,1), dtype=jnp.int8),
                      prey_actives=jnp.ones((self.num_prey, 1), dtype=jnp.int8),
                      grid=grid,
                      adjacency_mask=jnp.ones((self.num_predators, self.num_predators), dtype=jnp.float32)
                      if self.use_adj_mask else None,
                      shared_reward=jnp.array([0.0], dtype=jnp.float32),
                      shared_true_reward=jnp.array([0.0], dtype=jnp.float32),
                      step=jnp.array([0], dtype=jnp.int16),
                      terminal = jnp.array([False]),
                      ep_info={})

        return state

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> jnp.ndarray:
        """
        Returns the available actions for each agent.

        Would need to be adjusted if agents need to move
        more than one grid space at a time.
        """
        # Initialize avail_actions data object
        available_actions = jnp.zeros((self.num_predators,self.num_actions), dtype=jnp.int32)

        # Stay action is available for all agents regardless of active status
        available_actions = available_actions.at[:, self.actions_enum.stay].set(jnp.int32(1))

        def find_avail_actions(i: int, avail_acts: jnp.ndarray) -> jnp.ndarray:
            """ For loop function for computing available movement actions """

            def active_fun(avail_acts: jnp.ndarray) -> jnp.ndarray:
                """ Conditional function for handling active agents """
                # Compute possible actions
                new_pos = self.grid_actions[1:5] + state.agent_positions[i]

                # Check which actions stay within the confines of the grid
                allowed = jnp.logical_and(new_pos >= 0, new_pos < (self.world_shape_jax - 1)).all(axis=1)
                allowed = allowed.astype(jnp.int32)

                # Update avail_actions data object
                avail_acts = avail_acts.at[i, 1:5].set(allowed)

                # Check whether capture is an available action
                if self.capture_action:
                    # Check all possible positions around the agent
                    new_pos = self.grid_actions[1:5] + state.agent_positions[i]
                    new_pos = jnp.clip(new_pos, 0, (self.world_shape_jax - 1))

                    # If there is a prey at any of the positions around the predator then capture is possible
                    allowed = jnp.where(state.grid[new_pos[:, 0], new_pos[:, 1]] == self.features_enum.prey,
                                        True, False)
                    allowed = jnp.any(allowed)

                    # Check for prey on the diagonal of the predator
                    if self.diagonal_capture:
                        # Check all possible diagonal positions around the agent
                        new_pos = self.diagonal_actions + state.agent_positions[i]
                        new_pos = jnp.clip(new_pos, 0, (self.world_shape_jax - 1))

                        # If there is a prey at any of the diagonal positions
                        # around the predator then capture is possible
                        diag_allowed = jnp.where(
                            state.grid[new_pos[:, 0], new_pos[:, 1]] == self.features_enum.prey,True, False)
                        diag_allowed = jnp.any(diag_allowed)

                        # Combine normal and diagonal positions
                        allowed = jnp.logical_or(allowed, diag_allowed)

                    # Set capture actions to available if a prey was found adjacent to the predator
                    # This will also handle when self.num_capture_actions is greater than 1
                    avail_acts = jnp.where(allowed, avail_acts.at[i, 5:].set(jnp.int32(1)), avail_acts)

                return avail_acts

            def inactive_fun(avail_acts: jnp.ndarray) -> jnp.ndarray:
                """ Conditional function for handling inactive agents """
                return avail_acts

            avail_acts = jax.lax.cond(state.agent_actives[i][0] == 1, active_fun, inactive_fun, avail_acts)

            return avail_acts

        # Compute available actions
        # Movement actions are unavailable if agents are inactive/frozen or if next to a wall
        # Capture actions are unavailable if agents are inactive/frozen or if they are not adjacent to a prey
        available_actions = jax.lax.fori_loop(0, self.num_agents, find_avail_actions, available_actions)

        return available_actions

    @partial(jax.jit, static_argnums=(0,))
    def get_fixed_adjacency_matrix(self) :
        """ Creates the fixed adjacency matrix for the environment. """
        # Initialize adjacency matrix
        edges_list = cg_utils.cg_edge_list(self.fixed_graph_topology, self.num_predators)
        fixed_adjacency_matrix = cg_utils.set_cg_adj_matrix(edges_list, self.num_predators)

        # Update with self attention if specified
        if self.use_self_attention:
            fixed_adjacency_matrix = fixed_adjacency_matrix + jnp.eye(self.num_predators, dtype=jnp.float32)

        return fixed_adjacency_matrix

    @partial(jax.jit, static_argnums=(0,))
    def get_proximal_adjacency_matrix(self, state: State) -> jnp.ndarray:
        """ Returns the proximal adjacency matrix for the current State. """
        # Initialize adjacency matrix
        adj_matrix = jnp.zeros((self.num_predators, self.num_predators), dtype=jnp.float32)

        # Compute current proximal distances between all agents
        distances = jax_chebyshev(state.agent_positions[self.pairs[:,0]],
                                  state.agent_positions[self.pairs[:,1]],
                                  axis=1)

        # Check which agents are within the proximal threshold
        valid_pairs = jnp.where(distances <= self.proximal_distance, True, False)

        def create_links(i: int, adj_matrix: jnp.ndarray) -> jnp.ndarray:
            """ Fills in the links in the adjacency matrix. """
            def active_fun(adj_matrix: jnp.ndarray) -> jnp.ndarray:
                """ Conditional function for handling active agents """
                # Add links between agents
                adj_matrix = adj_matrix.at[self.pairs[i][0], self.pairs[i][1]].set(1)
                adj_matrix = adj_matrix.at[self.pairs[i][1], self.pairs[i][0]].set(1)

                # TODO: Find a way to do this without repeating
                # Add self attention links if flag is set
                if self.use_self_attention:
                    adj_matrix = adj_matrix.at[self.pairs[i][0], self.pairs[i][0]].set(1)
                    adj_matrix = adj_matrix.at[self.pairs[i][1], self.pairs[i][1]].set(1)

                return adj_matrix

            def inactive_fun(adj_matrix: jnp.ndarray) -> jnp.ndarray:
                """ Conditional function for handling inactive agents """
                return adj_matrix

            adj_matrix = jax.lax.cond(valid_pairs[i] & ((state.agent_actives[self.pairs[i][0]][0] == 1) &
                                      (state.agent_actives[self.pairs[i][1]][0] == 1)),
                                      active_fun, inactive_fun, adj_matrix)

            return adj_matrix

        adj_matrix = jax.lax.fori_loop(0, len(valid_pairs), create_links, adj_matrix)

        return adj_matrix

    @partial(jax.jit, static_argnums=(0,))
    def get_adjacency_matrix(self, state: State) -> jnp.ndarray:
        """ Returns the adjacency matrix for the environment. """
        if self.use_fixed_graph:
            # Create fixed adjacency matrix
            adjacency_matrix = self.get_fixed_adjacency_matrix()

            # Update the fixed adjacency matrix using the mask for active agents
            if self.use_adj_mask and self.capture_freezes:
                adjacency_matrix = adjacency_matrix * state.adjacency_mask

            return jnp.expand_dims(adjacency_matrix, 0)
        else:
            return jnp.expand_dims(self.get_proximal_adjacency_matrix(state), axis=0)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: State) -> jnp.ndarray:
        """
        Environment observation function
        """
        # Create oversized grid to allow for observations when at the edge of the grid
        # Padding input is ((# rows to add before the first row, # rows to add after the last row),
        #                   (# rows to add before the first column, # rows to add after the last column))
        grid = jnp.pad(state.grid,
                       ((self.agent_obs[0], self.agent_obs[0]), (self.agent_obs[1], self.agent_obs[1])),
                       constant_values=self.features_enum.wall)

        def mini_obs(agent_pos: jnp.ndarray, agent_actives: jnp.ndarray) -> jnp.ndarray:
            """
            Mapping function for creating agent observations.
            """
            # Get observation with the agent in the center
            obs = jax.lax.dynamic_slice(grid,
                                        start_indices=(agent_pos[0], agent_pos[1]),
                                        slice_sizes=(self.agent_obs_size[0], self.agent_obs_size[1]))

            # Expand to number of features
            # one-hot (drop first channel as its empty blocks)
            obs = jax.nn.one_hot(obs - 1, self.num_features, dtype=jnp.int8)

            # Reshape observation to 1-D tensor
            obs = obs.reshape(-1)

            # Append current episode step to obs
            if self.observe_current_timestep:
                timestep = state.step / self.episode_limit_jax
                obs = jnp.concatenate([timestep, obs], axis=-1)

            # Append current grid position step to obs
            if self.observe_grid_pos:
                obs = jnp.concatenate([agent_pos/self.world_shape_jax, obs], axis=-1)

            # Zero out obs for inactive/frozen agents
            obs = jnp.where(agent_actives == 0, 0, obs)

            return obs

        vmap_mini_obs = jax.vmap(mini_obs, (0, 0), 0)
        obs = vmap_mini_obs(state.agent_positions, state.agent_actives)

        return obs

    def get_reward(self):
        raise NotImplemented("The reward function is part of the env step and is returned in the state.")

    @partial(jax.jit, static_argnums=(0,))
    def env_step(self, prng: chex.PRNGKey, state: State, actions: jnp.ndarray) -> State:
        """
        Environment step function.
        """
        ###########################################
        #            Initialize step              #
        ###########################################
        # Initialize reward for the step
        shared_reward = jnp.array([1.0 * self.reward_time], dtype=jnp.float32)
        shared_true_reward = jnp.array([0.0], dtype=jnp.float32)

        # Get current grid from the state
        grid = state.grid

        ###########################################
        #              Move predators             #
        ###########################################
        # Get random generator key and update the main key
        prng, key = jax.random.split(prng)

        # Randomly iterate over the predators
        predator_random_indicies = jax.random.permutation(key, self.num_predators)

        # Not currently needed but saved for later
        # Zero out actions for inactive agents
        # def inactive_check(action, actives):
        #     action = jnp.where(actives == 0, self.actions_enum.stay, action)
        #     return action
        # vmap_inactive_check = jax.vmap(inactive_check, (0, 0), 0)
        # actions = vmap_inactive_check(actions, state.agent_actives)

        # Predators move based on actions from an algorithm
        new_predator_positions = state.agent_positions + self.grid_actions[actions]

        # Bound the new positions by the shape of the grid
        new_predator_positions = jnp.clip(new_predator_positions, 0, (self.world_shape_jax - 1))

        # Check if the agents position changed
        moved = ~jnp.all(new_predator_positions == state.agent_positions, axis=1)

        def predator_position_update(i:int, val:Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
                -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """ For loop function to update predator position in the grid. """
            grid, new_pos, moved = val
            index = predator_random_indicies[i]

            # jax.debug.print("Index {} action {} moved {} new_pos {} pos {}", index, actions[index], moved[index], new_pos[index], state.agent_positions[index])
            # jax.debug.print("Index {} grid[new_pos[index][0], new_pos[index][1]] {}", index, grid[new_pos[index][0], new_pos[index][1]])
            # jax.debug.print("Index {} new pos empty {}", index,
            #                 grid[new_pos[index][0], new_pos[index][1]]  == self.features_enum.empty)

            def moved_fun(grid: jnp.ndarray, moved: jnp.ndarray, new_pos: jnp.ndarray) \
                    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """ Conditional function for handling agents that moved """
                # Set previous position in grid to empty
                grid = grid.at[state.agent_positions[index][0],
                               state.agent_positions[index][1]].set(jnp.int8(self.features_enum.empty))

                # Set new position to predator index
                # self.features_enum[f'predator_{index + 1}']
                grid = grid.at[new_pos[index][0], new_pos[index][1]].set(jnp.int8(index+1))
                return grid, moved, new_pos

            def unmoved_fun(grid: jnp.ndarray, moved: jnp.ndarray, new_pos: jnp.ndarray) \
                    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """ Conditional function for handling unmoved agents """
                # Set moved to false and new position to the current/old position
                return grid, moved.at[index].set(False), new_pos.at[index].set(state.agent_positions[index])

            grid, moved, new_pos = (
                jax.lax.cond((state.agent_actives[index][0] == 1) &
                             moved[index] &
                             (grid[new_pos[index][0], new_pos[index][1]] == self.features_enum.empty),
                             moved_fun, unmoved_fun, grid, moved, new_pos))

            return grid, new_pos, moved

        # Update predator position in the grid
        grid, new_predator_positions, updated_moved = jax.lax.fori_loop(0, self.num_predators,
                                                                        predator_position_update,
                                                                        (grid, new_predator_positions, moved))

        ###########################################
        #                Move prey                #
        ###########################################
        # Prey agents move randomly

        # Get random generator keys for the prey
        # and update the main key
        # prng, key, *keys = jax.random.split(prng, self.num_prey+2)
        prng, key, prey_key = jax.random.split(prng, 3)

        # Randomly iterate over the predators
        prey_random_indicies = jax.random.permutation(key, self.num_prey)

        # Check for available actions for the prey
        possible_prey_positions = jnp.expand_dims(state.prey_positions, axis=1) + self.grid_actions[1:5]
        available_prey_positions = grid[possible_prey_positions[:, :, 0],
                                        possible_prey_positions[:, :, 1]]

        available_prey_positions = jnp.where(available_prey_positions == self.features_enum.empty, 1, 0)

        if self.diagonal_capture:
            # Check for possible predators on the diagonal
            possible_prey_capture_diag_positions = (jnp.expand_dims(state.prey_positions, axis=1) +
                                                    self.diagonal_actions)
            # Combine all positions around the prey to check for capture predators
            possible_prey_capture_positions = jnp.concatenate([possible_prey_positions,
                                                               possible_prey_capture_diag_positions], axis=1)
        else:
            possible_prey_capture_positions = possible_prey_positions

        # Get the entities in the positions around the prey
        # and zero out the prey entities
        possible_capture_predators = grid[possible_prey_capture_positions[:, :, 0],
                                          possible_prey_capture_positions[:, :, 1]]
        possible_capture_predators = jnp.where(possible_capture_predators == self.features_enum.prey, 0,
                                               possible_capture_predators)

        # Check if predators have capture action selected
        if self.capture_action:
            # Get actions predators in positions around the prey.
            # We must subtract 1 since the action indicies start at 0
            # and the predator indicies start at 1.
            # We append -1 to the actions so that the -1 feature index will return a -1 action.
            possible_capture_predators_actions = jnp.append(actions, -1)[possible_capture_predators - 1]

            # Create mask for pedators with capture actions
            mask_possible_capture_predators_actions = jnp.where(possible_capture_predators_actions >= 5, 1, 0)

            # Mask out predators with non-capture actions
            possible_capture_predators_actions = (possible_capture_predators_actions *
                                                  mask_possible_capture_predators_actions)
            possible_capture_predators = possible_capture_predators * mask_possible_capture_predators_actions

            if self.num_capture_actions == 1:
                num_capture_predators = jnp.where(possible_capture_predators != 0, 1, 0).sum(axis=1)

        else:
            # Count if there are enough predators around the prey
            num_capture_predators = jnp.where(possible_capture_predators != 0, 1, 0).sum(axis=1)

        # Move prey if they have not been captured
        # Else remove them and the capturing predators from the grid
        def prey_position_update(i:int, val:Tuple[chex.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                                  jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
                -> Tuple[chex.PRNGKey, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                         jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """ For loop function to update prey position in the grid. """
            (prey_key, grid, prey_positions, predator_actives, prey_actives,
             adjacency_mask, shared_reward, shared_true_reward) = val
            index = prey_random_indicies[i]
            prey_key, loop_key = jax.random.split(prey_key)

            size = len(possible_capture_predators[index])

            def active_fun(grid: jnp.ndarray, prey_positions: jnp.ndarray,
                           predator_actives: jnp.ndarray, prey_actives: jnp.ndarray,
                           adjacency_mask: jnp.ndarray, shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """ Conditional function for handling active agents """
                ###########################################
                #          Check if prey captured         #
                ###########################################
                def check_prey_captured(predator_indices: jnp.ndarray,
                                        shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                        -> Tuple[bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    """
                    Handle prey that can be captured and get capture predator indicies.
                    """
                    # Capture predator indicies
                    # Only get the number of required predators
                    capture_predator_indicies = (
                        possible_capture_predators)[index][predator_indices[:self.capture_action_conditions]]
                    capture_predator_indicies = capture_predator_indicies - 1

                    return True, capture_predator_indicies, shared_reward, shared_true_reward

                def check_prey_not_captured(predator_indices: jnp.ndarray,
                                            shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                        -> Tuple[bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    """
                    Handle prey that cannot be captured.
                    """
                    return (False, jnp.zeros((self.capture_action_conditions,), dtype=jnp.int8),
                            shared_reward, shared_true_reward)

                if self.capture_action:
                    if self.num_capture_actions == 1:
                        # Get indicies for non-zero elements where capture predators exist
                        non_zero_indices = jnp.nonzero(possible_capture_predators[index], size=size)[0]

                        # Check whether prey is captured
                        captured, capture_predator_indicies, shared_reward, shared_true_reward = (
                            jax.lax.cond(num_capture_predators[index] >= self.capture_action_conditions,
                                         check_prey_captured, check_prey_not_captured, non_zero_indices, shared_reward, shared_true_reward))

                    else:
                        # Pad zeroed out agents with non-capture actions
                        # This is needed so that unique does put 0 as the first found value
                        possible_capture_predators_actions_padded = (
                            jnp.where(possible_capture_predators_actions[index] == 0,
                                      self.num_actions,
                                      possible_capture_predators_actions[index]))

                        # Check for predators with unique capture actions
                        unique_capture_actions, unique_capture_actions_indices = (
                            jnp.unique(possible_capture_predators_actions_padded, size=size,
                                       return_index=True, fill_value=0))

                        # Unpad the zeroed out agents with non-capture actions
                        unique_capture_actions = jnp.where(unique_capture_actions == self.num_actions,
                                                           0, unique_capture_actions)

                        # Count the number of unique capture actions
                        num_unique_capture_actions = jnp.count_nonzero(unique_capture_actions)

                        # Check whether prey is captured with each capture predator using a unique capture action
                        captured, capture_predator_indicies, shared_reward, shared_true_reward = (
                            jax.lax.cond(num_unique_capture_actions == self.capture_action_conditions,
                                         check_prey_captured, check_prey_not_captured, unique_capture_actions_indices,
                                         shared_reward, shared_true_reward))

                else:
                    # Get indicies for non-zero elements where capture predators exist
                    non_zero_indices = jnp.nonzero(possible_capture_predators[index], size=size)[0]

                    # Check whether prey is captured
                    captured, capture_predator_indicies, shared_reward, shared_true_reward = (
                        jax.lax.cond(num_capture_predators[index] >= self.capture_action_conditions,
                                     check_prey_captured, check_prey_not_captured, non_zero_indices,
                                     shared_reward, shared_true_reward))

                ###########################################
                #         Update state for prey           #
                ###########################################
                def prey_captured(grid: jnp.ndarray, prey_positions: jnp.ndarray,
                                  predator_actives: jnp.ndarray, prey_actives: jnp.ndarray, adjacency_mask: jnp.ndarray,
                                  shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                 jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    """ Handle grid update for captured prey. """
                    # Update reward
                    shared_reward = shared_reward + self.reward_capture
                    shared_true_reward = shared_true_reward + self.reward_capture

                    # Zero out previous prey position
                    grid = grid.at[state.prey_positions[index][0],
                                   state.prey_positions[index][1]].set(jnp.int8(self.features_enum.empty))

                    # Zero out capture predator position
                    grid = grid.at[new_predator_positions[capture_predator_indicies][:, 0],
                                   new_predator_positions[capture_predator_indicies][:, 1]].set(
                        jnp.int8(self.features_enum.empty))

                    # Update actives for predators and prey
                    predator_actives = predator_actives.at[capture_predator_indicies].set(0)
                    prey_actives = prey_actives.at[index].set(0)

                    # Zero out capture predators in the adjacency mask so that
                    # they don't affect the graph communications
                    if self.use_adj_mask:
                        adjacency_mask = adjacency_mask.at[:, capture_predator_indicies].set(0)
                        adjacency_mask = adjacency_mask.at[capture_predator_indicies, :].set(0)

                    return (grid, prey_positions, predator_actives, prey_actives, adjacency_mask,
                            shared_reward, shared_true_reward)

                def prey_not_captured(grid: jnp.ndarray, prey_positions: jnp.ndarray,
                                      predator_actives: jnp.ndarray, prey_actives: jnp.ndarray,
                                      adjacency_mask: jnp.ndarray,
                                      shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                        -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                 jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                    """ Handle grid update for  non-captured prey. """
                    # Get check for number of available prey movement postions
                    # non_zero_indices = jnp.nonzero(available_prey_positions[index], size=self.num_movement_actions)[0]
                    num_non_zero_indices = available_prey_positions[index].sum()

                    def apply_shaping_reward(shared_reward, shared_true_reward):
                        """ Handle shaping reward function. """
                        shared_reward += self.shaping_reward

                        return shared_reward, shared_true_reward

                    def no_shaping_reward(shared_reward, shared_true_reward):
                        """ Handle non-shaping reward function. """
                        def apply_penalty(shared_reward, shared_true_reward):
                            """ Handle penalty reward function. """
                            shared_reward = shared_reward + self.miscapture_punishment
                            shared_true_reward = shared_true_reward + self.miscapture_punishment

                            return shared_reward, shared_true_reward

                        def no_penalty(shared_reward, shared_true_reward):
                            """ Handle non-penalty reward function. """
                            return shared_reward, shared_true_reward

                        # Determine whether penalty reward function should be applied
                        if self.capture_action and self.num_capture_actions > 1:
                            if self.modified_penalty_condition is None:
                                shared_reward, shared_true_reward = (
                                    jax.lax.cond(num_unique_capture_actions > 0,
                                                 apply_penalty, no_penalty, shared_reward, shared_true_reward))
                            else:
                                shared_reward, shared_true_reward = (
                                    jax.lax.cond(num_unique_capture_actions >= self.modified_penalty_condition,
                                                 apply_penalty, no_penalty, shared_reward, shared_true_reward))
                        else:
                            if self.modified_penalty_condition is None:
                                shared_reward, shared_true_reward = (
                                    jax.lax.cond(num_capture_predators[index] > 0,
                                                 apply_penalty, no_penalty, shared_reward, shared_true_reward))
                            else:
                                shared_reward, shared_true_reward = (
                                    jax.lax.cond(num_capture_predators[index] >= self.modified_penalty_condition,
                                                 apply_penalty, no_penalty, shared_reward, shared_true_reward))

                        return shared_reward, shared_true_reward

                    # Determine whether shaping or penalty reward function should be applied
                    if self.capture_action and self.num_capture_actions > 1:
                        shared_reward, shared_true_reward =  (
                            jax.lax.cond(self.use_shaping_reward &
                                         (num_unique_capture_actions == self.shaping_reward_condition),
                                         apply_shaping_reward, no_shaping_reward, shared_reward, shared_true_reward))
                    else:
                        shared_reward, shared_true_reward =  (
                            jax.lax.cond(self.use_shaping_reward &
                                         (num_capture_predators[index] == self.shaping_reward_condition),
                                         apply_shaping_reward, no_shaping_reward, shared_reward, shared_true_reward))

                    def resting(grid: jnp.ndarray, prey_positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
                        """ Handle resting prey. """
                        return grid, prey_positions

                    def not_resting(grid: jnp.ndarray, prey_positions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
                        """ Handle moving prey that are not resting. """
                        # Pad first position incase it is the 0th index
                        # This is needed because the empty spaces are filled with zeroes
                        non_zero_indices = jnp.nonzero(available_prey_positions[index],
                                                       size=self.num_movement_actions)[0]
                        non_zero_indices_padded = (
                            non_zero_indices.at[0].set(non_zero_indices[0] + self.num_movement_actions))

                        # Randomly permute the indices
                        random_permuted_indices = jax.random.permutation(loop_key, non_zero_indices_padded)

                        # Get non-zero indices for the permutation
                        permuted_non_zero_indices = jnp.nonzero(random_permuted_indices,
                                                                size=self.num_movement_actions)[0]

                        # Unpad the value that was in the first position
                        random_permuted_indices = jnp.where(random_permuted_indices > (self.num_movement_actions - 1),
                                                            random_permuted_indices - self.num_movement_actions,
                                                            random_permuted_indices)

                        # Get the random new position for the prey
                        position_index = random_permuted_indices[permuted_non_zero_indices[0]]
                        new_pos = possible_prey_positions[index][position_index]

                        # Zero out previous prey position
                        grid = grid.at[state.prey_positions[index, 0],
                                       state.prey_positions[index, 1]].set(jnp.int8(self.features_enum.empty))

                        # Move prey to new position
                        grid = grid.at[new_pos[0], new_pos[1]].set(jnp.int8(self.features_enum.prey))

                        # Update prey positions
                        prey_positions = prey_positions.at[index].set(new_pos)

                        return grid, prey_positions

                    grid, prey_positions = (
                        jax.lax.cond((num_non_zero_indices == 0) &
                                     (jax.random.uniform(loop_key) < self.prey_rest),
                                     resting, not_resting, grid, prey_positions))

                    return grid, prey_positions, predator_actives, prey_actives, adjacency_mask, shared_reward, shared_true_reward

                # Update state for prey
                (grid, prey_positions, predator_actives, prey_actives,
                 adjacency_mask, shared_reward, shared_true_reward) = (
                    jax.lax.cond(captured, prey_captured, prey_not_captured,
                                 grid, prey_positions, predator_actives, prey_actives,
                                 adjacency_mask, shared_reward, shared_true_reward))

                return (grid, prey_positions, predator_actives, prey_actives,
                        adjacency_mask, shared_reward, shared_true_reward)

            def inactive_fun(grid: jnp.ndarray, prey_positions: jnp.ndarray,
                             predator_actives: jnp.ndarray, prey_actives: jnp.ndarray, adjacency_mask: jnp.ndarray,
                             shared_reward: jnp.ndarray, shared_true_reward: jnp.ndarray) \
                    -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
                """ Conditional function for handling inactive/frozen agents """
                return (grid, prey_positions, predator_actives, prey_actives,
                        adjacency_mask, shared_reward, shared_true_reward)

            grid, prey_positions, predator_actives, prey_actives, adjacency_mask, shared_reward, shared_true_reward = (
                jax.lax.cond(state.prey_actives[index][0] == 1, active_fun, inactive_fun,
                             grid, prey_positions, predator_actives, prey_actives,
                             adjacency_mask, shared_reward, shared_true_reward))

            return (prey_key, grid, prey_positions, predator_actives, prey_actives,
                    adjacency_mask, shared_reward, shared_true_reward)

        # Update prey position in the grid
        (prey_key,
         grid,
         new_prey_positions,
         new_predator_actives,
         new_prey_actives,
         new_adjacency_mask,
         shared_reward,
         shared_true_reward) = jax.lax.fori_loop(0, self.num_prey, prey_position_update,
                                               (prey_key,
                                                grid,
                                                state.prey_positions,
                                                state.agent_actives,
                                                state.prey_actives,
                                                state.adjacency_mask,
                                                shared_reward,
                                                shared_true_reward))

        ###########################################
        #               Update state              #
        ###########################################
        state = State(agent_positions=new_predator_positions,
                      prey_positions=new_prey_positions,
                      agent_actives=new_predator_actives,
                      prey_actives=new_prey_actives,
                      grid=grid,
                      adjacency_mask=new_adjacency_mask,
                      shared_reward=shared_reward,
                      shared_true_reward=shared_true_reward,
                      step=state.step + 1,
                      terminal=jnp.array([False]),
                      ep_info={})

        def episode_terminal(state):
            """ Handle terminal env step. """
            return state.replace(terminal=jnp.array([True]))

        def episode_not_terminal(state):
            """ Handle non-terminal env step. """
            return state

        # Check whether the episode is in a terminal state
        # Terminal is reached is either the predators or prey are all inactive
        predator_alive = ~jnp.all(state.agent_actives)
        prey_alive = ~jnp.all(state.prey_actives)
        state = jax.lax.cond(jnp.logical_or(state.terminal[0], jnp.logical_or(predator_alive, prey_alive)),
                             episode_terminal, episode_not_terminal, state)

        def episode_limit_reached(state):
            """ Handle episode limit reached. """
            state = state.replace(terminal=jnp.array([True]))
            state = state.replace(ep_info={"episode_limit": self.truncate_episodes})
            return state

        def episode_limit_not_reached(state):
            """ Handle episode limit not reached. """
            return state.replace(ep_info={"episode_limit": False})

        state = jax.lax.cond(state.step[0] >= self.episode_limit,
                             episode_limit_reached, episode_limit_not_reached, state)

        return state

    ###########################################
    #              Core Functions             #
    ###########################################
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, prng: chex.PRNGKey) \
            -> Union[Tuple[State, jnp.ndarray, jnp.ndarray], Tuple[State, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Resets the environment and returns initial observations.
        """
        # Reset the state of the environment
        state = self.reset_state(prng)

        if self.use_adj_matrix:
            return state, self.get_obs(state), self.get_avail_actions(state), self.get_adjacency_matrix(state)
        else:
            return state, self.get_obs(state), self.get_avail_actions(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, prng: chex.PRNGKey, state: State, actions: jnp.ndarray) \
            -> Union[Tuple[State, jnp.ndarray, jnp.ndarray], Tuple[State, jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        Resets the environment and returns initial observations.
        """
        # Take a step in the environment given the provided actions
        state = self.env_step(prng, state, actions)

        if self.use_adj_matrix:
            return state, self.get_obs(state), self.get_avail_actions(state), self.get_adjacency_matrix(state)
        else:
            return state, self.get_obs(state), self.get_avail_actions(state)

    ###########################################
    #             Render Functions            #
    ###########################################
    # @classmethod
    # def render_tile(
    #     cls,
    #     obj: int,
    #     agent_dir: Union[int, None] = None,
    #     agent_hat: bool = False,
    #     highlight: bool = False,
    #     tile_size: int = 32,
    #     subdivs: int = 3,
    # ) -> np.ndarray:
    #     """
    #     Render a tile and cache the result
    #     """
    #
    #     # Hash map lookup key for the cache
    #     key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
    #     if obj:
    #         key = (obj, 0, 0, 0) + key if obj else key
    #
    #     if key in cls.tile_cache:
    #         return cls.tile_cache[key]
    #
    #     img = onp.zeros(
    #         shape=(tile_size * subdivs, tile_size * subdivs, 3),
    #         dtype=onp.uint8,
    #     )
    #
    #     # Draw the grid lines (top and left edges)
    #     fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    #     fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # # class Items(IntEnum):
    #
    #     if obj == Items.agent1:
    #         # Draw the agent 1
    #         agent_color = PLAYER1_COLOUR
    #     elif obj == Items.agent2:
    #         # Draw agent 2
    #         agent_color = PLAYER2_COLOUR
    #     elif obj == Items.agent3:
    #         # Draw agent 3
    #         agent_color = PLAYER3_COLOUR
    #     elif obj == Items.agent4:
    #         # Draw agent 4
    #         agent_color = PLAYER4_COLOUR
    #     elif obj == Items.agent5:
    #         # Draw agent 5
    #         agent_color = PLAYER5_COLOUR
    #     elif obj == Items.agent6:
    #         # Draw agent 6
    #         agent_color = PLAYER6_COLOUR
    #     elif obj == Items.agent7:
    #         # Draw agent 7
    #         agent_color = PLAYER7_COLOUR
    #     elif obj == Items.agent8:
    #         # Draw agent 8
    #         agent_color = PLAYER8_COLOUR
    #     elif obj == Items.defection_coin:
    #         # Draw the red coin as GREEN COOPERATE
    #         fill_coords(
    #             img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
    #         )
    #     elif obj == Items.cooperation_coin:
    #         # Draw the blue coin as DEFECT/ RED COIN
    #         fill_coords(
    #             img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
    #         )
    #     elif obj == Items.wall:
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))
    #
    #     elif obj == Items.interact:
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))
    #
    #     elif obj == 99:
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))
    #
    #     elif obj == 100:
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))
    #
    #     elif obj == 101:
    #         # white square
    #         fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))
    #
    #     # Overlay the agent on top
    #     if agent_dir is not None:
    #         if agent_hat:
    #             tri_fn = point_in_triangle(
    #                 (0.12, 0.19),
    #                 (0.87, 0.50),
    #                 (0.12, 0.81),
    #                 0.3,
    #             )
    #
    #             # Rotate the agent based on its direction
    #             tri_fn = rotate_fn(
    #                 tri_fn,
    #                 cx=0.5,
    #                 cy=0.5,
    #                 theta=0.5 * math.pi * (1 - agent_dir),
    #             )
    #             fill_coords(img, tri_fn, (255.0, 255.0, 255.0))
    #
    #         tri_fn = point_in_triangle(
    #             (0.12, 0.19),
    #             (0.87, 0.50),
    #             (0.12, 0.81),
    #             0.0,
    #         )
    #
    #         # Rotate the agent based on its direction
    #         tri_fn = rotate_fn(
    #             tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
    #         )
    #         fill_coords(img, tri_fn, agent_color)
    #
    #     # Highlight the cell if needed
    #     if highlight:
    #         highlight_img(img)
    #
    #     # Downsample the image to perform supersampling/anti-aliasing
    #     img = downsample(img, subdivs)
    #
    #     # Cache the rendered tile
    #     cls.tile_cache[key] = img
    #     return img
    #
    # def render(self, state: State) -> np.ndarray:
    #     """ Render the grid state. """
    #     # Image Dimensions
    #     tile_size = 32
    #     width_px = GRID.shape[0] * tile_size
    #     height_px = GRID.shape[0] * tile_size
    #
    #     # Create the image
    #     img = np.zeros(shape=(height_px, width_px, 3), dtype=nnp.uint8)
    #     grid = onp.array(state.grid)
    #     grid = onp.pad(
    #         grid, ((PADDING, PADDING), (PADDING, PADDING)), constant_values=Items.wall
    #     )
    #     for a in range(self.num_agents):
    #         startx, starty = self.get_obs_point(
    #             state.agent_positions[a, 0],
    #             state.agent_positions[a, 1],
    #             state.agent_positions[a, 2]
    #         )
    #         highlight_mask[
    #             startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
    #         ] = True
    #
    #     # Render the grid
    #     for j in range(0, grid.shape[1]):
    #         for i in range(0, grid.shape[0]):
    #             cell = grid[i, j]
    #             if cell == 0:
    #                 cell = None
    #             agent_here = []
    #             for a in range(1, self.num_agents+1):
    #                 agent_here.append(cell == a)
    #             # if cell in [1,2]:
    #             #     print(f'coordinates: {i},{j}')
    #             #     print(cell)
    #
    #
    #             agent_dir = None
    #             for a in range(self.num_agents):
    #                 agent_dir = (
    #                     state.agent_positions[a,2].item()
    #                     if agent_here[a]
    #                     else agent_dir
    #                 )
    #
    #             agent_hat = False
    #             for a in range(self.num_agents):
    #                 agent_hat = (
    #                     bool(state.agent_inventories[a].sum() > INTERACT_THRESHOLD)
    #                     if agent_here[a]
    #                     else agent_hat
    #                 )
    #
    #             tile_img = InTheGrid.render_tile(
    #                 cell,
    #                 agent_dir=agent_dir,
    #                 agent_hat=agent_hat,
    #                 highlight=highlight_mask[i, j],
    #                 tile_size=tile_size,
    #             )
    #
    #             ymin = j * tile_size
    #             ymax = (j + 1) * tile_size
    #             xmin = i * tile_size
    #             xmax = (i + 1) * tile_size
    #             img[ymin:ymax, xmin:xmax, :] = tile_img
    #
    #     img = onp.rot90(
    #         img[
    #             (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
    #             (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
    #             :,
    #         ],
    #         2,
    #     )
    #     # Render the inventory
    #     # agent_inv = []
    #     # for a in range(self.num_agents):
    #     #     agent_inv.append(self.render_inventory(state.agent_inventories[a], img.shape[1]))
    #
    #
    #     time = self.render_time(state, img.shape[1])
    #     # img = onp.concatenate((img, *agent_inv, time), axis=0)
    #     img = onp.concatenate((img, time), axis=0)
    #     return img
    #
    # def render_time(self, state: State, width_px: int) -> np.array:
    #     """ Render time bar representing steps at the bottom of the image. """
    #     # Tile dimensions
    #     tile_height = 32
    #     tile_width = width_px // self.episode_limit
    #
    #     # Create the image
    #     img = np.zeros(shape=(2 * tile_height, width_px, 3), dtype=np.uint8)
    #
    #     # Add the bar for each step taken
    #     j = 0
    #     for i in range(0, state.step):
    #         ymin = j * tile_height
    #         ymax = (j + 1) * tile_height
    #         xmin = i * tile_width
    #         xmax = (i + 1) * tile_width
    #         img[ymin:ymax, xmin:xmax, :] = np.int8(255)
    #
    #     return img

    ###########################################
    #          Saved Unused Functions         #
    ###########################################
    # def step_non_jit(self, prng: chex.PRNGKey, state: State, actions: jnp.ndarray):
    #     """
    #     Environment step function.
    #     """
    #     ###########################################
    #     #            Initialize step              #
    #     ###########################################
    #     # Initialize reward for the step
    #     self.shared_reward = jnp.array([1.0 * self.reward_time], dtype=jnp.float32)
    #     self.shared_true_reward = jnp.array([0.0], dtype=jnp.float32)
    #
    #     # Get current grid from the state
    #     grid = state.grid
    #
    #     ###########################################
    #     #              Move predators             #
    #     ###########################################
    #     # Get random generator key and update the main key
    #     prng, key = jax.random.split(prng)
    #
    #     # Randomly iterate over the predators
    #     predator_random_indicies = jax.random.permutation(key, self.num_predators)
    #
    #     # Not currently needed but saved for later
    #     # Zero out actions for inactive agents
    #     # def inactive_check(action, actives):
    #     #     action = jnp.where(actives == 0, self.actions_enum.stay, action)
    #     #     return action
    #     # vmap_inactive_check = jax.vmap(inactive_check, (0, 0), 0)
    #     # actions = vmap_inactive_check(actions, state.agent_actives)
    #
    #     # Predators move based on actions from an algorithm
    #     new_predator_positions = state.agent_positions + self.grid_actions[actions]
    #
    #     # Bound the new positions by the shape of the grid
    #     new_predator_positions = jnp.clip(new_predator_positions, 0, (self.world_shape_jax - 1))
    #
    #     # Check if the agents position changed
    #     moved = ~jnp.all(new_predator_positions == state.agent_positions, axis=1)
    #
    #     def predator_position_update(i: int, val: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
    #             -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #         """ For loop function to update predator position in the grid. """
    #         grid, new_pos, moved = val
    #         index = predator_random_indicies[i]
    #
    #         def moved_fun(grid: jnp.ndarray, moved: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #             """ Conditional function for handling agents that moved """
    #             # Set previous position in grid to empty
    #             grid = grid.at[state.agent_positions[index][0],
    #             state.agent_positions[index][1]].set(jnp.int8(self.features_enum.empty))
    #
    #             # Set new position to predator index
    #             # self.features_enum[f'predator_{index + 1}']
    #             grid = grid.at[new_pos[index][0], new_pos[index][1]].set(jnp.int8(index + 1))
    #             return grid, moved
    #
    #         def unmoved_fun(grid: jnp.ndarray, moved: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    #             """ Conditional function for handling unmoved agents """
    #             return grid, moved.at[index].set(False)
    #
    #         grid, moved = jax.lax.cond((state.agent_actives[index][0] == 1) &
    #                                    moved[index] &
    #                                    (grid[new_pos[index][0], new_pos[index][1]] == self.features_enum.empty),
    #                                    moved_fun, unmoved_fun, grid, moved)
    #
    #         return grid, new_pos, moved
    #
    #     # Update predator position in the grid
    #     grid, _, updated_moved = jax.lax.fori_loop(0, self.num_predators, predator_position_update,
    #                                                (grid, new_predator_positions, moved))
    #
    #     ###########################################
    #     #                Move prey                #
    #     ###########################################
    #     # Prey agents move randomly
    #
    #     # Get random generator keys for the prey
    #     # and update the main key
    #     # prng, key, *keys = jax.random.split(prng, self.num_prey+2)
    #     prng, key, prey_key = jax.random.split(prng, 3)
    #
    #     # Randomly iterate over the predators
    #     prey_random_indicies = jax.random.permutation(key, self.num_prey)
    #
    #     # Check for available actions for the prey
    #     possible_prey_positions = jnp.expand_dims(state.prey_positions, axis=1) + self.grid_actions[1:5]
    #     available_prey_positions = grid[possible_prey_positions[:, :, 0],
    #     possible_prey_positions[:, :, 1]]
    #     available_prey_positions = jnp.where(available_prey_positions == self.features_enum.empty, 1, 0)
    #
    #     if self.diagonal_capture:
    #         # Check for possible predators on the diagonal
    #         possible_prey_capture_diag_positions = (jnp.expand_dims(state.prey_positions, axis=1) +
    #                                                 self.diagonal_actions)
    #         # Combine all positions around the prey to check for capture predators
    #         possible_prey_capture_positions = jnp.concatenate([possible_prey_positions,
    #                                                            possible_prey_capture_diag_positions], axis=1)
    #     else:
    #         possible_prey_capture_positions = possible_prey_positions
    #
    #     # Get the entities in the positions around the prey
    #     # and zero out the prey entities
    #     possible_capture_predators = grid[possible_prey_capture_positions[:, :, 0],
    #     possible_prey_capture_positions[:, :, 1]]
    #     possible_capture_predators = jnp.where(possible_capture_predators == self.features_enum.prey, 0,
    #                                            possible_capture_predators)
    #
    #     # Check if predators have capture action selected
    #     if self.capture_action:
    #         # Get actions predators in positions around the prey.
    #         # We must subtract 1 since the action indicies start at 0
    #         # and the predator indicies start at 1.
    #         # We append -1 to the actions so that the -1 feature index will return a -1 action.
    #         possible_capture_predators_actions = jnp.append(actions, -1)[possible_capture_predators - 1]
    #
    #         # Create mask for pedators with capture actions
    #         mask_possible_capture_predators_actions = jnp.where(possible_capture_predators_actions >= 5, 1, 0)
    #
    #         # Mask out predators with non-capture actions
    #         possible_capture_predators_actions = (possible_capture_predators_actions *
    #                                               mask_possible_capture_predators_actions)
    #         possible_capture_predators = possible_capture_predators * mask_possible_capture_predators_actions
    #
    #         if self.num_capture_actions == 1:
    #             num_capture_predators = jnp.where(possible_capture_predators != 0, 1, 0).sum(axis=1)
    #
    #     else:
    #         # Count if there are enough predators around the prey
    #         num_capture_predators = jnp.where(possible_capture_predators != 0, 1, 0).sum(axis=1)
    #
    #     # Move prey if they have not been captured
    #     # Else remove them and the capturing predators from the grid
    #     def prey_position_update(i: int, val: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) \
    #             -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #         """ For loop function to update prey position in the grid. """
    #         prey_key, grid, prey_positions, predator_actives, prey_actives = val
    #         index = prey_random_indicies[i]
    #         prey_key, loop_key = jax.random.split(prey_key)
    #
    #         def active_fun(grid: jnp.ndarray, prey_positions: jnp.ndarray,
    #                        predator_actives: jnp.ndarray, prey_actives: jnp.ndarray) \
    #                 -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #             """ Conditional function for handling active agents """
    #
    #             ###########################################
    #             #          Check if prey captured         #
    #             ###########################################
    #             captured = False
    #             if self.capture_action:
    #                 if self.num_capture_actions == 1:
    #                     if num_capture_predators[index] >= self.capture_action_conditions:
    #                         captured = True
    #
    #                         # Capture predator indicies
    #                         # Only get the number of required predators
    #                         non_zero_indices = jnp.nonzero(possible_capture_predators[index])[0]
    #                         capture_predator_indicies = (
    #                             possible_capture_predators)[index][non_zero_indices[:self.capture_action_conditions]]
    #                         capture_predator_indicies = capture_predator_indicies - 1
    #
    #                 else:
    #                     # Check for unique capture actions and retrive index of first occurance
    #                     unique_capture_actions, unique_capture_actions_indices = (
    #                         jnp.unique(possible_capture_predators_actions[index], return_index=True))
    #
    #                     # Check for non-zero capture actions
    #                     # Required because jnp.unique also counts 0 as unique
    #                     non_zero_indices = jnp.nonzero(unique_capture_actions)[0]
    #
    #                     # Mask out 0 values found by jnp.unique
    #                     unique_capture_actions = unique_capture_actions[non_zero_indices]
    #                     unique_capture_actions_indices = unique_capture_actions_indices[non_zero_indices]
    #
    #                     # Ensure there are enough unique capture predators
    #                     if len(unique_capture_actions) >= self.capture_action_conditions:
    #                         captured = True
    #
    #                         # Capture predator indicies
    #                         capture_predator_indicies = possible_capture_predators[index][
    #                             unique_capture_actions_indices]
    #                         capture_predator_indicies = capture_predator_indicies - 1
    #
    #             else:
    #                 if num_capture_predators[index] >= self.capture_action_conditions:
    #                     captured = True
    #
    #                     # Capture predator indicies
    #                     # Only get the number of required predators
    #                     non_zero_indices = jnp.nonzero(possible_capture_predators[index])[0]
    #                     capture_predator_indicies = (
    #                         possible_capture_predators)[index][non_zero_indices[:self.capture_action_conditions]]
    #                     capture_predator_indicies = capture_predator_indicies - 1
    #
    #             ###########################################
    #             #         Update state for prey           #
    #             ###########################################
    #             # If prey is not captured compute the random movement action
    #             # or whether the prey should rest
    #             rest = False
    #             if not captured:
    #                 non_zero_indices = jnp.nonzero(available_prey_positions[index])[0]
    #                 if (len(non_zero_indices) == 0) or (jax.random.uniform(loop_key) < self.prey_rest):
    #                     rest = True
    #                 else:
    #                     position_index = jax.random.choice(loop_key, non_zero_indices)
    #                     new_pos = possible_prey_positions[index][position_index]
    #
    #             if captured:
    #                 # Zero out previous prey position
    #                 grid = grid.at[state.prey_positions[index][0],
    #                                state.prey_positions[index][1]].set(jnp.int8(self.features_enum.empty))
    #
    #                 # Zero out capture predator position
    #                 grid = grid.at[new_predator_positions[capture_predator_indicies][0],
    #                                new_predator_positions[capture_predator_indicies][1]].set(
    #                     jnp.int8(self.features_enum.empty))
    #
    #                 # Update actives for predators and prey
    #                 predator_actives = predator_actives.at[capture_predator_indicies].set(0)
    #                 prey_actives = prey_actives.at[index].set(0)
    #             else:
    #                 # Move prey when not resting
    #                 if not rest:
    #                     # Zero out previous prey position
    #                     grid = grid.at[state.prey_positions[index, 0],
    #                                    state.prey_positions[index, 1]].set(jnp.int8(self.features_enum.empty))
    #
    #                     # Move prey to new position
    #                     grid = grid.at[new_pos[0], new_pos[1]].set(jnp.int8(self.features_enum.prey))
    #
    #                     # Update prey positions
    #                     prey_positions = prey_positions.at[index].set(new_pos)
    #
    #             return grid, prey_positions, predator_actives, prey_actives
    #
    #         def inactive_fun(grid: jnp.ndarray, prey_positions: jnp.ndarray,
    #                          predator_actives: jnp.ndarray, prey_actives: jnp.ndarray) \
    #                 -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    #             """ Conditional function for handling inactive/frozen agents """
    #             return grid, prey_positions, predator_actives, prey_actives
    #
    #         grid, prey_positions, predator_actives, prey_actives = (
    #             jax.lax.cond(state.prey_actives[index][0] == 1, active_fun, inactive_fun,
    #                          grid, prey_positions, predator_actives, prey_actives))
    #
    #         return prey_key, grid, prey_positions, predator_actives, prey_actives
    #
    #     # Update prey position in the grid
    #     (prey_key,
    #      grid,
    #      new_prey_positions,
    #      new_predator_actives,
    #      new_prey_actives) = jax.lax.fori_loop(0, self.num_prey, prey_position_update,
    #                                            (prey_key,
    #                                             grid,
    #                                             state.prey_positions,
    #                                             state.agent_actives,
    #                                             state.prey_actives))