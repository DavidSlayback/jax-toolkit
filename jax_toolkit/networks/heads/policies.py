__all__ = ['CategoricalHead', 'GaussianHeadDependentLogStd', 'GaussianHeadIndependentLogStd', 'BetaHead']

from typing import Sequence, Iterable, Optional, Union, Any, Callable

import haiku as hk
import jax.nn
import jax.numpy as jnp
from ..initializers import get_orthogonal_activation
from .distributions import *


class CategoricalHead(hk.Module):
    """Categorical (or multi-categorical) output head

    Args:
        num_outputs: Either int (num_actions) or sequence of ints (last is num_actions, a la option-critic)
        name: duh
    """
    def __init__(self,
                 num_outputs: Union[int, Sequence[int]],
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._shape = num_outputs
        self._linear = hk.Linear(jnp.prod(num_outputs), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray) -> Categorical:
        logits = self._linear(inputs)
        if not isinstance(self._shape, int): logits = hk.Reshape(self._shape)(logits)
        return Categorical(logits=logits)


class GaussianHeadDependentLogStd(hk.Module):
    """Gaussian (Normal) distribution with state-dependent std. Outputs [-inf, inf]

    Args:
        num_outputs: Either int (size of action vector) or sequence of ints (last is size of action vector)
        name: duh
    """
    def __init__(self,
                 num_outputs: Union[int, Sequence[int]],
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._shape = num_outputs
        self._linear = hk.Linear(2 * jnp.prod(num_outputs), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray) -> Gaussian:
        mean_std = self._linear(inputs)
        if not isinstance(self._shape, int): mean_std = hk.Reshape(self._shape)(mean_std)
        mean, std = jnp.split(mean_std, 2, -1)
        return Gaussian(mean, jnp.exp(std))


class GaussianHeadIndependentLogStd(hk.Module):
    """Gaussian (Normal) distribution with state-independent std. Outputs [-inf, inf]

    Args:
        num_outputs: Either int (size of action vector) or sequence of ints (last is size of action vector)
        name: duh
    """
    def __init__(self,
                 num_outputs: Union[int, Sequence[int]],
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._shape = num_outputs
        self._linear = hk.Linear(jnp.prod(num_outputs), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)
        self._log_std = hk.get_parameter('log_std', (jnp.prod(num_outputs),), init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray) -> Gaussian:
        mean = self._linear(inputs)
        if not isinstance(self._shape, int): mean = hk.Reshape(self._shape)(mean)
        mean, std = jnp.broadcast_arrays(mean, self._log_std)
        return Gaussian(mean, jnp.exp(std))


class BetaHead(hk.Module):
    """Beta distribution (outputs [0,1])

    Linear layer representing alpha and beta vectors, followed by softplus and then +1
    Args:
        num_outputs: Either int (size of action vector) or sequence of ints (last is size of action vector)
        name: duh
    """
    def __init__(self,
                 num_outputs: Union[int, Sequence[int]],
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._shape = num_outputs
        self._linear = hk.Sequential([hk.Linear(2 * jnp.prod(num_outputs), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros),
                                      jax.nn.softplus])

    def __call__(self, inputs: jnp.ndarray) -> Beta:
        alpha_beta = self._linear(inputs) + 1
        if not isinstance(self._shape, int): alpha_beta = hk.Reshape(self._shape)(alpha_beta)
        alpha, beta = jnp.split(alpha_beta, 2, -1)
        return Beta(alpha, beta)


class TerminationHead(hk.Module):
    """Option termination head (outputs [0,1])

    Args:
        num_outputs: Either int (size of action vector) or sequence of ints (last is size of action vector)
        name: duh
    """
    def __init__(self,
                 num_outputs: Union[int, Sequence[int]],
                 temperature: float = 1.,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._shape = num_outputs
        self._temp = temperature
        self._linear = hk.Linear(jnp.prod(num_outputs), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray) -> Bernoulli:
        logits = self._linear(inputs) / self._temp
        if not isinstance(self._shape, int): logits = hk.Reshape(self._shape)(logits)
        return Bernoulli(logits=logits)






