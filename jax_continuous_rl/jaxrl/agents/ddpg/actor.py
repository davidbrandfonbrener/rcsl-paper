from typing import Tuple
import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params


def update(actor: Model, critic: Model,
           batch: Batch, bc_alpha: float) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params):
        actions = actor.apply_fn({'params': actor_params}, batch.observations)
        q1, q2 = critic(batch.observations, actions)
        lmbda = bc_alpha / jax.lax.stop_gradient(jnp.abs(q1).mean())
        bc_loss = ((batch.actions - actions)**2).mean()
        q_loss = q1.mean()
        actor_loss = -lmbda * q_loss + bc_loss
        return actor_loss, {'actor_loss': actor_loss, 'q_loss': q_loss, 
                            'bc_loss': bc_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
