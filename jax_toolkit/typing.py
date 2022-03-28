# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
__all__ = ['Nest', 'NestedArray', 'NestedTensor', 'NestedSpec', 'TensorTransformation', 'TensorValuedCallable', 'Batches', 'Transition',
           'PRNGKey', 'Networks', 'PolicyNetwork', 'Sample', 'TrainingState', 'TrainingMetrics', 'TrainingStepOutput', 'Variables',
           'Seed', 'EnvironmentFactory', 'ModelToSnapshot']
"""Common type definitions. (acme base and jax types)"""

import dataclasses
from typing import Any, Callable, Dict, Generic, Mapping, TypeVar

import chex
import dm_env
import jax.numpy as jnp

"""Common types used throughout Acme."""

from typing import Any, Callable, Iterable, Mapping, NamedTuple, Union
from dm_env import specs

# Define types for nested arrays and tensors.
# TODO(b/144758674): Replace these with recursive type definitions.
NestedArray = Any
NestedTensor = Any

NestedSpec = Union[
    specs.Array,
    Iterable['NestedSpec'],
    Mapping[Any, 'NestedSpec'],
]

Nest = Union[NestedArray, NestedTensor, NestedSpec]

TensorTransformation = Callable[[NestedTensor], NestedTensor]
TensorValuedCallable = Callable[..., NestedTensor]


class Batches(int):
    """Helper class for specification of quantities in units of batches.
    Example usage:
        # Configure the batch size and replay size in units of batches.
        config.batch_size = 32
        config.replay_size = Batches(4)
        # ...
        # Convert the replay size at runtime.
        if isinstance(config.replay_size, Batches):
          config.replay_size = config.replay_size * config.batch_size  # int: 128
    """


class Transition(NamedTuple):
    """Container for a transition."""
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()

"""Below is JAX specific"""
PRNGKey = jnp.ndarray
Networks = TypeVar('Networks')
PolicyNetwork = TypeVar('PolicyNetwork')
Sample = TypeVar('Sample')
TrainingState = TypeVar('TrainingState')

TrainingMetrics = Mapping[str, jnp.ndarray]
"""Metrics returned by the training step.
Typically these are logged, so the values are expected to be scalars.
"""

Variables = Mapping[str, NestedArray]
"""Mapping of variable collections.
A mapping of variable collections, as defined by Learner.get_variables.
The keys are the collection names, the values are nested arrays representing
the values of the corresponding collection variables.
"""


@chex.dataclass(frozen=True, mappable_dataclass=False)
class TrainingStepOutput(Generic[TrainingState]):
    state: TrainingState
    metrics: TrainingMetrics


Seed = int
EnvironmentFactory = Callable[[Seed], dm_env.Environment]


@dataclasses.dataclass
class ModelToSnapshot:
    """Stores all necessary info to be able to save a model.
    Attributes:
      model: a jax function to be saved.
      params: fixed params to be passed to the function.
      dummy_kwargs: arguments to be passed to the function.
    """
    model: Any  # Callable[params, **dummy_kwargs]
    params: Any
    dummy_kwargs: Dict[str, Any]
