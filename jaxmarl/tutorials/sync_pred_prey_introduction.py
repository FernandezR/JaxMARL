# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sync_pred_prey_introduction.py

Synchronized Predator-Prey Introduction

An example for using the Synchronized Predator-Prey environment
"""
from types import SimpleNamespace

from PIL import Image
import os
import random
import sys

# Prevents Jax from taking all GPU resources
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp

# Import jaxmarl modules
# If ModuleNotFoundError is raised, add jaxmarl to path and try again.
try:
    import jaxmarl.environments.pred_prey.sync_pred_prey as sync_pred_prey

    from jaxmarl import make
    from jaxmarl.environments.pred_prey.utils import jax_to_torch, torch_to_jax, dict2namedtuple, load_yaml_config

except ModuleNotFoundError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    import jaxmarl.environments.pred_prey.sync_pred_prey as sync_pred_prey

    from jaxmarl import make
    from jaxmarl.environments.pred_prey.utils import jax_to_torch, torch_to_jax, dict2namedtuple, load_yaml_config

__author__ = 'Rolando Fernandez'
__copyright__ = 'Copyright 2024, Multi-Agent Synchronized Predator-Prey'
__credits__ = ['Rolando Fernandez']
__license__ = ''
__version__ = '0.0.1'
__maintainer__ = 'Rolando Fernandez'
__email__ = 'rfernandez@utexas.edu'
__status__ = 'Dev'

###########################################
#                2 Catcher                #
###########################################
# Load config
config = load_yaml_config(os.path.join(os.path.dirname(sync_pred_prey.__file__),
                                       "configs/sync_pred_prey/2_catcher/2_catchers.yaml"))

# Update env args
config['env_args']['episode_limit'] = 200
# config['env_args']['use_adj_matrix'] = True
# config['env_args']['cg_topology'] = 'full'

config = SimpleNamespace(**config)

# Set random seeds
random.seed(7)
prng = jax.random.key(7)

# Load environment
env = make('sync_pred_prey', args=config.env_args)

# Initialize environment
prng, key = jax.random.split(prng)
state, obs, avail_actions = env.reset(key)
# state, obs, avail_actions, adj_matrix = env.reset(key)

# Get initial state image
if not os.path.exists('state_pics'):
    os.makedirs('state_pics')

pics = []
img = env.render(state)
pics.append(Image.fromarray(img))
pics[0].save("state_pics/state_0.png")

for t in range(200):
    prng, *keys = jax.random.split(prng, 3)
    actions = jax.random.choice(keys[0], a=env.num_actions, shape=(env.num_predators,))

    state, obs, avail_actions = env.step(keys[1], state, actions)
    # state, obs, avail_actions, adj_matrix = env.step(keys[1], state, actions)

    img = env.render(state)
    pics.append(Image.fromarray(img))
    pics[t+1].save(f"state_pics/state_{t+1}.png")

pics[0].save(
    "state.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=1000,
    loop=0,
)

###########################################
#                3 Catcher                #
###########################################
# # Load config
# config = load_yaml_config(os.path.join(os.path.dirname(sync_pred_prey.__file__),
#                                        "configs/sync_pred_prey/3_catcher/3_catchers.yaml"))
#
# # Update env args
# config['env_args']['episode_limit'] = 200
# # config['env_args']['use_adj_matrix'] = True
# # config['env_args']['cg_topology'] = 'full'
#
# config = SimpleNamespace(**config)
#
# # Set random seeds
# random.seed(280)
# prng = jax.random.key(280)
#
# # Load environment
# env = make('sync_pred_prey', args=config.env_args)
#
# # Initialize environment
# prng, key = jax.random.split(prng)
# state, obs, avail_actions = env.reset(key)
# # state, obs, avail_actions, adj_matrix = env.reset(key)
#
# # Get initial state image
# if not os.path.exists('state_pics/3_catcher'):
#     os.makedirs('state_pics/3_catcher')
#
# pics = []
# img = env.render(state)
# pics.append(Image.fromarray(img))
# pics[0].save("state_pics/3_catcher/state_0.png")
#
# for t in range(200):
#     prng, *keys = jax.random.split(prng, 3)
#     actions = jax.random.choice(keys[0], a=env.num_actions, shape=(env.num_predators,))
#
#     state, obs, avail_actions = env.step(keys[1], state, actions)
#     # state, obs, avail_actions, adj_matrix = env.step(keys[1], state, actions)
#
#     img = env.render(state)
#     pics.append(Image.fromarray(img))
#     pics[t+1].save(f"state_pics/3_catcher/state_{t+1}.png")
#
# pics[0].save(
#     "state_3_catcher.gif",
#     format="GIF",
#     save_all=True,
#     optimize=False,
#     append_images=pics[1:],
#     duration=1000,
#     loop=0,
# )
