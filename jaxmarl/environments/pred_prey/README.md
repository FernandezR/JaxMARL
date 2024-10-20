# Predator-Prey Games

## Environments

### Synchronized Predator-Prey

Inspired by the original Predator-Prey games, the [Synchronized Predator-Prey](https://arxiv.org/abs/2404.18798) environment expands on Predator-Prey games by requiring synchronization through coordination as described in [Multi-Agent Synchronization Tasks](https://arxiv.org/abs/2404.18798). Agents must work together (synchronize their actions) to capture the prey or suffer a penalty for failed capture attempts. This creates a fully cooperative game that requires a high-degree of coordination between the agents.

Thus, Synchronized Predator-Prey can be used for studying the efficacy of different coordination strategies on a Multi-Agent Synchronization Task domain.

## Visualisation

We render each timestep and then create a gif from the collection of images. Further examples are provided [here](https://github.com/FernandezR/JaxMARL/tree/main/jaxmarl/tutorials).

```python
import jax
import jax.numpy as jnp

from PIL import Image

from jaxmarl import make

# load environment
prng = jax.random.PRNGKey(7)
env = make('sync_pred_prey')
prng, key = jax.random.split(prng)
state, obs, avail_actions = env.reset(key)

# render each timestep
pics = []
img = env.render(state)
pics.append(Image.fromarray(img))
pics[0].save("state_pics/state_0.png")

for t in range(200):
    prng, *keys = jax.random.split(prng, 3)
    actions = jax.random.choice(keys[0], a=env.num_actions, shape=(env.num_predators,))

    state, obs, avail_actions = env.step(keys[1], state, actions)

    img = env.render(state)
    pics.append(Image.fromarray(img))
    pics[t+1].save(f"state_pics/state_{t+1}.png")

# create and save gif     
pics[0].save(
    "state.gif",
    format="GIF",
    save_all=True,
    optimize=False,
    append_images=pics[1:],
    duration=1000,
    loop=0,
)
```