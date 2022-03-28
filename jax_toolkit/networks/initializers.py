__all__ = ['ORTHOGONAL_INIT_VALUES', 'get_orthogonal_activation_std', 'get_orthogonal_activation']

from typing import Callable, Dict, Union
import jax.numpy as jnp
import haiku.initializers as ini

ORTHOGONAL_INIT_VALUES: Dict[str, float] = {
    'relu': 2. ** 0.5,  # He et. al 2015
    'elu': 1.55 ** 0.5,  # https://stats.stackexchange.com/a/320443
    'selu': 3. / 4,
    'tanh': 5. / 3,
    'sigmoid': 1.,
    'linear': 1.,
    'pi': 1e-2
}


def get_orthogonal_activation_std(activation_fn: Union[str, Callable]) -> float:
    """Get appropriate initialization value for given activation function (or string name of one)"""
    name = activation_fn.__name__ if isinstance(activation_fn, Callable) else activation_fn
    if name in ORTHOGONAL_INIT_VALUES.keys(): return ORTHOGONAL_INIT_VALUES[activation_fn.__name__]
    else: return ORTHOGONAL_INIT_VALUES['relu']  # ReLU default


def get_orthogonal_activation(activation_fn: Union[str, Callable]) -> ini.Initializer:
    return ini.Orthogonal(get_orthogonal_activation_std(activation_fn))