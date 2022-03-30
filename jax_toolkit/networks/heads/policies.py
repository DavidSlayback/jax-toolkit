__all__ = ['CategoricalHead', 'GaussianHeadDependentLogStd', 'GaussianHeadIndependentLogStd', 'BetaHead', 'TerminationHead']

from typing import Sequence, Iterable, Optional, Union, Any, Callable

import math
import haiku as hk
import jax.nn
import jax.numpy as jnp
from ..initializers import get_orthogonal_activation
from .distributions import *
import numpy as onp


def subselect(logits: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    """Subselect some logits according to indices

    Indices have same time and batch dimensions as logits

    Args:
        logits: Some action output (logits, mean, alpha_beta, etc) of size [T?, B?, n_index_options, n_act]
        indices: Indices with which to sub-select of size [T?, B?[
    Returns:
        logits: [T?, B?, n_act]

    Option critic input for 4-option network might be [200,], logits [200, 4, 30], output is [200, 30]
    """
    shape_diff = len(logits.shape) - len(indices.shape)
    indices = jnp.reshape(indices, indices.shape + (1,) * shape_diff)
    return jnp.take_along_axis(logits, indices, -2)


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
        self._linear = hk.Linear(int(onp.prod(num_outputs)), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray, indices: Optional[jnp.ndarray] = None) -> Categorical:
        """Sub-select using indices (e.g., options)"""
        logits = self._linear(inputs)
        if not isinstance(self._shape, int):
            logits = hk.Reshape(self._shape)(logits)  # [T?, B?, n_opt, n_act], indices are [T?, B?]
            if indices is not None: logits = subselect(logits, indices)
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
        self._linear = hk.Linear(int(2 * onp.prod(num_outputs)), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray, indices: Optional[jnp.ndarray] = None) -> Gaussian:
        """Sub-select using indices (e.g., options)"""
        mean_std = self._linear(inputs)
        if not isinstance(self._shape, int):
            mean_std = hk.Reshape(self._shape)(mean_std)
            if indices is not None: mean_std = subselect(mean_std, indices)
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
        self._linear = hk.Linear(int(onp.prod(num_outputs)), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)
        self._log_std = hk.get_parameter('log_std', (int(onp.prod(num_outputs)),), init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray, indices: Optional[jnp.ndarray] = None) -> Gaussian:
        """Sub-select using indices (e.g., options)"""
        mean = self._linear(inputs)
        if not isinstance(self._shape, int):
            mean = hk.Reshape(self._shape)(mean)
            if indices is not None: mean = subselect(mean, indices)
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
        self._linear = hk.Sequential([hk.Linear(int(2 * onp.prod(num_outputs)), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros),
                                      jax.nn.softplus])

    def __call__(self, inputs: jnp.ndarray, indices: Optional[jnp.ndarray] = None) -> Beta:
        """Sub-select using indices (e.g., options)"""
        alpha_beta = self._linear(inputs) + 1
        if not isinstance(self._shape, int):
            alpha_beta = hk.Reshape(self._shape)(alpha_beta)
            if indices is not None: alpha_beta = subselect(alpha_beta, indices)
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
        self._linear = hk.Linear(int(onp.prod(num_outputs)), w_init=get_orthogonal_activation('pi'), b_init=jnp.zeros)

    def __call__(self, inputs: jnp.ndarray, indices: Optional[jnp.ndarray] = None) -> Bernoulli:
        """Sub-select using indices (e.g., options)"""
        logits = self._linear(inputs) / self._temp
        if not isinstance(self._shape, int):
            logits = hk.Reshape(self._shape)(logits)
            if indices is not None: logits = subselect(logits, indices)
        return Bernoulli(logits=logits)






