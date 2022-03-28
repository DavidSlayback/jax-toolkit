# python3
# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""bsuite adapter for OpenAI gym run-loops. Added some more general modifications"""
__all__ = ['GymFromDMEnv', 'DMEnvFromGym', 'space2spec', 'spec2space']
from typing import Any, Dict, Optional, Tuple, Union

import dm_env
from dm_env import specs
import gym
from gym import spaces
import numpy as np

# OpenAI gym step format = obs, reward, is_finished, other_info
_GymTimestep = Tuple[np.ndarray, float, bool, Dict[str, Any]]


class GymFromDMEnv(gym.Env):
    """A wrapper that converts a dm_env.Environment to an OpenAI gym.Env."""
    metadata = {'render.modes': ['human', 'rgb_array'], "video.frames_per_second": 30}

    def __init__(self, env: dm_env.Environment):
        self._env = env
        self._last_observation = None
        self.viewer = None

    def step(self, action: Union[float, int, np.ndarray]) -> _GymTimestep:
        timestep = self._env.step(action)
        self._last_observation = timestep.observation
        reward = timestep.reward or 0.
        return timestep.observation, reward, timestep.last(), {}

    def reset(self) -> np.ndarray:
        timestep = self._env.reset()
        self._last_observation = timestep.observation
        return timestep.observation

    def render(self, mode: str = 'rgb_array') -> Union[np.ndarray, bool]:
        if self._last_observation is None:
            raise ValueError('Environment not ready to render. Call reset() first.')

        if mode == 'rgb_array':
            return self._last_observation

        if mode == 'human':
            if self.viewer is None:
                # pylint: disable=import-outside-toplevel
                # pylint: disable=g-import-not-at-top
                from gym.envs.classic_control import rendering
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self._last_observation)
            return self.viewer.isopen

    @property
    def action_space(self) -> gym.Space:
        return spec2space(self._env.action_spec())

    @property
    def observation_space(self) -> gym.Space:
        return spec2space(self._env.observation_spec())

    @property
    def reward_range(self) -> Tuple[float, float]:
        reward_spec = self._env.reward_spec()
        if isinstance(reward_spec, specs.BoundedArray):
            return reward_spec.minimum, reward_spec.maximum
        return -float('inf'), float('inf')

    def __getattr__(self, attr):
        """Delegate attribute access to underlying environment."""
        return getattr(self._env, attr)


def spec2space(spec) -> gym.Space:
    """Convert dm_env spec to OpenAI Space.
    Cannot handle MultiDiscrete currently"""
    if isinstance(spec, specs.DiscreteArray):  # Discrete
        return gym.spaces.Discrete(spec.num_values)
    elif isinstance(spec, specs.BoundedArray):  # Bounded arrays
        if isinstance(spec.minimum, float) and isinstance(spec.maximum, float) and spec.minimum == 0. and spec.maximum == 1. and spec.dtype == np.uint8:
            return spaces.MultiBinary(spec.shape)
        return spaces.Box(spec.minimum, spec.maximum, spec.shape, spec.dtype)
    elif isinstance(spec, specs.Array):  # Unbounded arrays
        return gym.spaces.Box(-np.inf, np.inf, spec.shape, spec.dtype)
    elif isinstance(spec, tuple):  # Unpack tuple
        return gym.spaces.Tuple(spec2space(s) for s in spec)
    elif isinstance(spec, Dict):  # Unpack dict
        return gym.spaces.Dict({k: spec2space(v) for k, v in spec.items()})
    else:
        raise ValueError(f'Unexpected dm_env spec: {spec}')


def space2spec(space: gym.Space, name: str = None):
    """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.
    Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
    specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
    Dict spaces are recursively converted to tuples and dictionaries of specs.
    Args:
      space: The Gym space to convert.
      name: Optional name to apply to all return spec(s).
    Returns:
      A dm_env spec or nested structure of specs, corresponding to the input
      space.
    """
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

    elif isinstance(space, spaces.Box):
        if np.all(space.low == -np.inf) and np.all(space.high == np.inf):
            return specs.Array(space.shape, space.dtype, name=name)  # Special case, no bounds
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                  minimum=space.low, maximum=space.high, name=name)

    elif isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                                  maximum=1.0, name=name)

    elif isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                                  minimum=np.zeros(space.shape),
                                  maximum=space.nvec, name=name)

    elif isinstance(space, spaces.Tuple):
        return tuple(space2spec(s, name) for s in space.spaces)

    elif isinstance(space, spaces.Dict):
        return {key: space2spec(value, name) for key, value in space.spaces.items()}

    else:
        raise ValueError('Unexpected gym space: {}'.format(space))


class DMEnvFromGym(dm_env.Environment):
    """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

    def __init__(self, gym_env: gym.Env):
        self.gym_env = gym_env
        # Convert gym action and observation spaces to dm_env specs.
        self._observation_spec = space2spec(self.gym_env.observation_space,
                                            name='observations')
        self._action_spec = space2spec(self.gym_env.action_space, name='actions')
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        observation = self.gym_env.reset()
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        # Convert the gym step result to a dm_env TimeStep.
        observation, reward, done, info = self.gym_env.step(action)
        self._reset_next_step = done

        if done:
            is_truncated = info.get('TimeLimit.truncated', False)
            if is_truncated:
                return dm_env.truncation(reward, observation)
            else:
                return dm_env.termination(reward, observation)
        else:
            return dm_env.transition(reward, observation)

    def close(self):
        self.gym_env.close()

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec