from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets.rvs_d4rl_dataset import RvsBatch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey



def mse_update(actor: Model, batch: RvsBatch,
               rng: PRNGKey):
    rng, key = jax.random.split(rng)

    def loss_fn(actor_params: Params):
        actions = actor.apply_fn({'params': actor_params},
                                 batch.observations,
                                 batch.outcomes,
                                 training=True,
                                 rngs={'dropout': key})
        actor_loss = ((actions - batch.actions)**2).mean()
        return actor_loss, {'actor_loss': actor_loss}

    return (rng, *actor.apply_gradient(loss_fn))

def mse_eval(actor: Model, batch: RvsBatch,
               rng: PRNGKey):
    rng, key = jax.random.split(rng)
    actions = actor.apply_fn({'params': actor.params},
                                 batch.observations,
                                 batch.outcomes,
                                 training=True,
                                 rngs={'dropout': key})
    actor_loss = ((actions - batch.actions)**2).mean()
    actor_loss = jax.lax.stop_gradient(actor_loss)
    return (rng, {'actor_loss': actor_loss})
