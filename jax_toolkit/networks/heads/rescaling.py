__all__ = ['ClipToSpec', 'RescaleToSpec', 'TanhToSpec', 'BetaToSpec']

"""Rescaling layers (e.g. to match action specs)."""
import dataclasses

from jax import lax
import jax.numpy as jnp
from dm_env import specs


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
