__all__ = ['value_loss']

import jax
import jax.numpy as jnp
import rlax
import chex
from functools import partial

Array = chex.Array
Scalar = chex.Scalar

def value_loss(g_t: Array, v_tm1: Array) -> Array:
    """MSE of returns (GAE or discounted) and values"""
    return 0.5 * jnp.mean((g_t - v_tm1) ** 2)