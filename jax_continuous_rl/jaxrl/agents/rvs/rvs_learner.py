"""Implementations of algorithms for continuous control."""
from pickletools import optimize
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.rvs import actor
from jaxrl.datasets.rvs_d4rl_dataset import RvsBatch
from jaxrl.networks import rvs_policies
from jaxrl.networks.common import InfoDict, Model


_mse_update_jit = jax.jit(actor.mse_update)
_mse_eval_jit = jax.jit(actor.mse_eval)


class RvsLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 outcomes: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 1e-3,
                 num_steps: int = int(1e6),
                 hidden_dims: Sequence[int] = (256, 256),
                 distribution: str = 'det',
                 **kwargs):

        self.distribution = distribution

        rng = jax.random.PRNGKey(seed)
        rng, actor_key = jax.random.split(rng)

        action_dim = actions.shape[-1]
        if distribution == 'det':
            actor_def = rvs_policies.MSEPolicy(hidden_dims,
                                           action_dim,
                                           dropout_rate=0.0)
        else:
            raise NotImplemented

        optimizer = optax.adam(actor_lr)

        self.actor = Model.create(actor_def,
                                  inputs=[actor_key, observations, outcomes],
                                  tx=optimizer)
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       outcomes: np.ndarray,
                       temperature: float = 1.0):
        self.rng, actions = rvs_policies.sample_actions(self.rng,
                                                    self.actor.apply_fn,
                                                    self.actor.params,
                                                    observations,
                                                    outcomes,
                                                    temperature,
                                                    self.distribution)

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: RvsBatch):
        if self.distribution == 'det':
            self.rng, self.actor, info = _mse_update_jit(
                self.actor, batch, self.rng)
        return info


    def eval(self, batch: RvsBatch):
        if self.distribution == 'det':
            self.rng, info = _mse_eval_jit(
                self.actor, batch, self.rng)
        return info