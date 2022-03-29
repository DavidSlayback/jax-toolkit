# Copyright 2022 The Brax Authors.
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

"""Input normalization utils. Removed pmap stuff"""
__all__ = ['create_observation_normalizer', 'NormParams']
from typing import Optional, NamedTuple, Tuple, Callable
import jax.numpy as jnp


class NormParams(NamedTuple):
    steps: jnp.ndarray
    mean: jnp.ndarray
    var: jnp.ndarray


def create_observation_normalizer(obs_size: Optional[int] = None,
                                  normalize_observations: bool = True,
                                  mean_shift: bool = True,
                                  num_leading_batch_dims: int = 1,
                                  apply_clipping: bool = True) -> Tuple[NormParams, Callable[..., NormParams], Callable[[NormParams, jnp.ndarray], jnp.ndarray]]:
    """Observation normalization based on running statistics.

    Args:
        obs_size: Size of observations. If none, scalar (rewards)
        normalize_observations: If False, just update number of steps
        mean_shift: If False, only divide by variance, don't shift to 0 mean (typical for reward normalization by discounted returns)
        num_leading_batch_dims: How many dimensions of observations? Apply doesn't care, update does
        apply_clipping: If true, clip variance within min/max, clip output within -5, 5
    Return:
        data: Initialized NormParams object (zeros) "NormParams(0)"
        update_fn: Normalization parameter update function "update(params, obs, mask) -> u_params"
        apply_fn: Observation normalization function "apply(params, obs) -> n_obs"
    """
    assert num_leading_batch_dims == 1 or num_leading_batch_dims == 2  # B or [T, B]
    if normalize_observations:
        def update_fn(params: NormParams, obs: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> NormParams:
            """Update params using obs where not masked out"""
            normalization_steps, running_mean, running_variance = params

            if mask is not None:
                step_increment = jnp.sum(mask)
                mask = jnp.broadcast_arrays(mask, obs)[0]  # for shape matching during multiplication
            else: step_increment = obs.shape[0] * (obs.shape[1] if num_leading_batch_dims == 2 else 1)
            total_new_steps = normalization_steps + step_increment

            # Compute the incremental update and divide by the number of new steps.
            input_to_old_mean = obs - running_mean
            if mask is not None: input_to_old_mean = input_to_old_mean * mask
            mean_diff = jnp.sum(input_to_old_mean / total_new_steps,
                                axis=((0, 1) if num_leading_batch_dims == 2 else 0))
            new_mean = running_mean + mean_diff

            # Compute difference of input to the new mean for Welford update.
            input_to_new_mean = obs - new_mean
            if mask is not None: input_to_new_mean = input_to_new_mean * mask
            var_diff = jnp.sum(input_to_new_mean * input_to_old_mean,
                               axis=((0, 1) if num_leading_batch_dims == 2 else 0))
            return NormParams(total_new_steps, new_mean, running_variance + var_diff)

    else:
        def update_fn(params: NormParams, obs: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> NormParams:
            if mask is not None: step_increment = jnp.sum(mask)
            else: step_increment = obs.shape[0] * (obs.shape[1] if num_leading_batch_dims == 2 else 1)
            return params._replace(steps=params.steps + step_increment)
    data, apply_fn = make_data_and_apply_fn(obs_size, normalize_observations, mean_shift, apply_clipping)
    return data, update_fn, apply_fn


def make_data_and_apply_fn(obs_size: Optional[int] = None,
                           normalize_observations: bool = True,
                           mean_shift: bool = True,
                           apply_clipping: bool = True):
    """Creates data (init) and an apply function for the normalizer."""
    obs_shape = () if obs_size is None else (obs_size,)  # None means scalar, as in reward
    if normalize_observations:
        data = NormParams(jnp.zeros(()), jnp.zeros(obs_shape), jnp.ones(obs_shape))

        def apply_fn(params: NormParams, obs: jnp.ndarray, std_min_value: float = 1e-6, std_max_value: float = 1e6):
            normalization_steps, running_mean, running_variance = params
            variance = running_variance / (normalization_steps + 1.0)
            if mean_shift: obs = obs - running_mean
            # We clip because the running_variance can become negative,
            # presumably because of numerical instabilities.
            if apply_clipping:
                variance = jnp.clip(variance, std_min_value, std_max_value)
                return jnp.clip(obs / jnp.sqrt(variance), -5, 5)
            else:
                return obs / jnp.sqrt(variance)
    else:
        data = NormParams(jnp.zeros(()), jnp.zeros(()), jnp.zeros(()))
        def apply_fn(params: NormParams, obs: jnp.ndarray):
            del params
            return obs
    return data, apply_fn
