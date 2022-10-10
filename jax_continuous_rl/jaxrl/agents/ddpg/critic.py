from typing import Tuple

import jax.numpy as jnp
import jax

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update(rng: PRNGKey,
            actor: Model, critic: Model, target_critic: Model, batch: Batch,
           discount: float, 
           policy_noise: float, noise_clip: float):
    
    next_actions = actor(batch.next_observations)
    #noise = jax.random.normal(rng, next_actions.shape) * policy_noise
    #noise = jax.lax.clamp(-noise_clip, noise, noise_clip)
    next_actions = jax.lax.clamp(-1.0, next_actions, 1.0)
    
    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q
    target_q = jax.lax.stop_gradient(target_q)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
