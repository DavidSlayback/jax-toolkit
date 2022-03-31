__all__ = ['ClipToSpec', 'RescaleToSpec', 'TanhToSpec', 'BetaToSpec', 'get_action_scale_fns']

from typing import Union, Callable, Tuple

"""Rescaling layers (e.g. to match action specs)."""
import dataclasses

from jax import lax
import jax.numpy as jnp
from dm_env import specs
from gym.spaces import Box
from ...utils.dm_gym_conversion import space2spec, spec2space


def scale11_to_01(inputs: jnp.ndarray) -> jnp.ndarray: return 0.5 * (inputs + 1.)
def unscale01_to_11(inputs: jnp.ndarray) -> jnp.ndarray: return 2 * inputs - 1.


def get_action_scale_fns(space_or_spec: Union[specs.BoundedArray, Box],
                        apply_tanh: bool = False,
                        apply_clip: bool = False,
                        from_beta: bool = False) -> Tuple[Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]]:
    """Return a compilable function to scale action to range (and inversion function)

    Args:
        space_or_spec: Gym Box Space or dm_env BoundedArray to scale to
        apply_tanh: If True, squash inputs to [-1, 1] before rescaling
        apply_clip: If True, clipping
        from_beta: If True, assume input is from beta distribution [0,1]
    """
    assert not (apply_clip and apply_tanh)  # Only one of these
    if isinstance(space_or_spec, Box): space_or_spec = space2spec(space_or_spec)
    range = space_or_spec.maximum - space_or_spec.minimum
    offset = space_or_spec.minimum
    def scale_from_01(inputs: jnp.ndarray) -> jnp.ndarray: return inputs * range + offset
    def unscale_to_01(inputs: jnp.ndarray) -> jnp.ndarray: return (inputs - offset) / range
    if apply_clip:
        def scale(inputs: jnp.ndarray) -> jnp.ndarray: return jnp.clip(inputs, space_or_spec.minimum, space_or_spec.maximum)
        def inv_scale(inputs: jnp.ndarray) -> jnp.ndarray: return inputs  # Can't unclip
    elif from_beta:
        scale = scale_from_01
        inv_scale = unscale_to_01
    elif apply_tanh:
        def scale(inputs: jnp.ndarray) -> jnp.ndarray:
            return scale_from_01(scale11_to_01(jnp.tanh(inputs)))
        def inv_scale(inputs: jnp.ndarray) -> jnp.ndarray:
            return jnp.arctanh(unscale01_to_11(unscale_to_01(inputs)))
    else:
        def scale(inputs: jnp.ndarray) -> jnp.ndarray:
            return scale_from_01(scale11_to_01(inputs))
        def inv_scale(inputs: jnp.ndarray) -> jnp.ndarray:
            return unscale01_to_11(unscale_to_01(inputs))
    return scale, inv_scale




@dataclasses.dataclass
class ClipToSpec:
    """Clips inputs to within a BoundedArraySpec."""
    spec: specs.BoundedArray

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.clip(inputs, self.spec.minimum, self.spec.maximum)


@dataclasses.dataclass
class RescaleToSpec:
    """Rescales inputs in [-1, 1] to match a BoundedArraySpec."""
    spec: specs.BoundedArray

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scale = self.spec.maximum - self.spec.minimum
        offset = self.spec.minimum
        inputs = 0.5 * (inputs + 1.0)  # [0, 1]
        output = inputs * scale + offset  # [minimum, maximum]
        return output


@dataclasses.dataclass
class BetaToSpec:
    """Rescales inputs in [0, 1] to match a BoundedArraySpec."""
    spec: specs.BoundedArray

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scale = self.spec.maximum - self.spec.minimum
        offset = self.spec.minimum
        output = inputs * scale + offset  # [minimum, maximum]
        return output


@dataclasses.dataclass
class TanhToSpec:
    """Squashes real-valued inputs to match a BoundedArraySpec."""
    spec: specs.BoundedArray

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scale = self.spec.maximum - self.spec.minimum
        offset = self.spec.minimum
        inputs = lax.tanh(inputs)  # [-1, 1]
        inputs = 0.5 * (inputs + 1.0)  # [0, 1]
        output = inputs * scale + offset  # [minimum, maximum]
        return output
