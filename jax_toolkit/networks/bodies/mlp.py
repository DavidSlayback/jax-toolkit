__all__ = ['MLP', 'LayerNormMLP']

from typing import Sequence, Iterable, Optional, Union, Any, Callable

from functools import partial
import haiku as hk
import jax.numpy as jnp
import jax
from ..initializers import get_orthogonal_activation


class MLP(hk.nets.MLP):
    """A multi-layer perceptron module. Automatically infer orthogonal initialization from activation function"""

    def __init__(
            self,
            output_sizes: Iterable[int],
            w_init: Optional[hk.initializers.Initializer] = None,
            b_init: Optional[hk.initializers.Initializer] = None,
            with_bias: bool = True,
            activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
            activate_final: bool = False,
            name: Optional[str] = None,
            orthogonal_initialization: bool = False
    ):
        w_init = get_orthogonal_activation(activation) if orthogonal_initialization else w_init
        b_init = jnp.zeros if (with_bias and orthogonal_initialization) else b_init
        super().__init__(output_sizes, w_init, b_init, with_bias, activation, activate_final, name)


class LayerNormMLP(hk.Module):
    """MLP where first layer is always followed by a LayerNorm and tanh nonlinearity"""

    def __init__(self,
                 output_sizes: Sequence[int],
                 with_bias: bool = True,
                 w_init: Optional[hk.initializers.Initializer] = None,
                 b_init: Optional[hk.initializers.Initializer] = None,
                 activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.elu,
                 activate_final: bool = False,
                 name: str = 'feedforward_mlp_torso',
                 orthogonal_initialization: bool = False
                 ):
        """Construct the MLP.
        Args:
          output_sizes: a sequence of ints specifying the size of each layer.
          w_init: initializer for Linear layers.
          activation: nonlinearity to use in the MLP, defaults to elu.
            Note! The default activation differs from the usual MLP default of ReLU
            for legacy reasons.
          activate_final: whether or not to use the activation function on the final
            layer of the neural network.
          name: a name for the module.
        """
        super().__init__(name=name)
        w_init = get_orthogonal_activation(activation) if orthogonal_initialization else w_init
        b_init = jnp.zeros if (with_bias and orthogonal_initialization) else b_init
        self._network = hk.Sequential([
            hk.Linear(output_sizes[0], w_init=get_orthogonal_activation('tanh') if orthogonal_initialization else w_init),
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            jax.nn.tanh,
            hk.nets.MLP(
                output_sizes[1:],
                w_init=get_orthogonal_activation(activation) if orthogonal_initialization else w_init,
                b_init=jnp.zeros if orthogonal_initialization else b_init,
                with_bias=with_bias,
                activation=activation,
                activate_final=activate_final),
        ])

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Forwards the policy network."""
        return self._network(inputs)
