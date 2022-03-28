__all__ = ['Squeeze']

from typing import Optional
import haiku as hk
import jax.numpy as jnp


class Squeeze(hk.Module):
    """Squeeze trailing dimension"""
    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def __call__(self, inputs):
        return jnp.squeeze(inputs, -1)
