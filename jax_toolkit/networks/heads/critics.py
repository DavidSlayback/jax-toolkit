__all__ = ['Q', 'V']

from typing import Sequence, Iterable, Optional, Union, Any, Callable

from functools import partial
import haiku as hk
import jax.numpy as jnp
from ..initializers import get_orthogonal_activation
from ..util.squeeze import Squeeze

Q = partial(hk.Linear, w_init=get_orthogonal_activation('linear'), b_init=jnp.zeros)  # 1. initialized critic
V = hk.Sequential([Q(1), Squeeze()])  # Single-value critic with squeeze
