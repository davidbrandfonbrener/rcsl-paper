"""Implementations of algorithms for continuous control."""

from typing import Callable, Sequence, Tuple
import functools

import jax
import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP


class QRValue(nn.Module):
    hidden_dims: Sequence[int]
    num_quantiles: int

    @nn.compact
    def __call__(self, observations: jnp.ndarray):
        critic = MLP((*self.hidden_dims, self.num_quantiles))(observations)
        return critic

@functools.partial(jax.jit, static_argnames=('apply_fn'))
def get_values(apply_fn, params, observations):
    return apply_fn({'params': params}, observations)