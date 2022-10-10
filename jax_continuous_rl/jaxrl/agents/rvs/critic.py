from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets.rvs_d4rl_dataset import RvsBatch
from jaxrl.networks.common import InfoDict, Model, Params


def quantile_loss_fn(dist, target):
    num_quantiles = dist.shape[-1]
    quantiles = (jnp.arange(0, num_quantiles) + 0.5) / float(num_quantiles)

    delta = dist - target
    delta_neg = (delta < 0.).astype(jnp.float32)
    weight = jnp.abs(quantiles - delta_neg)

    loss = jnp.abs(delta) * weight
    return jnp.sum(jnp.mean(loss, axis=-1))


batch_quantile_loss_fn = jax.vmap(quantile_loss_fn, 
                            in_axes=(0,0))


def update(critic: Model, batch: RvsBatch):

    def critic_loss_fn(critic_params: Params):
        vdist = critic.apply_fn({'params': critic_params}, batch.observations)

        critic_loss = batch_quantile_loss_fn(vdist, batch.outcomes).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'v': vdist.mean(),
            'v_max': jnp.max(vdist, axis=-1).mean(),
            'v_last': vdist[:, -1].mean(),
            'v_min': jnp.min(vdist, axis=-1).mean(),
            'v_first': vdist[:, 0].mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
