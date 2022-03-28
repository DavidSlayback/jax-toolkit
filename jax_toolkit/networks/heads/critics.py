__all__ = ['Q', 'V']

from typing import Sequence, Iterable, Optional, Union, Any, Callable

from functools import partial
import haiku as hk
import jax.numpy as jnp
from ..initializers import get_orthogonal_activation

Q = partial(hk.Linear, w_init=get_orthogonal_activation('linear'), b_init=jnp.zeros)  # 1. initialized critic
V = partial(Q, 1)  # Single-value critic without squeeze
# V = partial(hk.Sequential, [Q(1), jnp.squeeze])  # Single-value critic with squeeze
