__all__ = ['LearnableGRU', 'LearnableLayerNormGRU', 'LayerNormGRU', 'GRU', 'get_gru_cls']

from functools import partial
from typing import Optional, Callable
import haiku as hk
from haiku._src.recurrent import add_batch
import jax.numpy as jnp
import jax
from ..initializers import *


GRU = hk.GRU


class LearnableGRU(GRU):
    """GRU with learnable initial state"""
    def initial_state(self, batch_size: Optional[int]):
        state = hk.get_parameter("init_state", [self.hidden_size], init=jnp.zeros)
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state


class LayerNormGRU(GRU):
    """Layer-normalized GRU. LN for inputs, hiddens"""
    def __call__(self, inputs, state):
        if inputs.ndim not in (1, 2):
            raise ValueError("GRU input must be rank-1 or rank-2.")

        input_size = inputs.shape[-1]
        hidden_size = self.hidden_size
        w_i = hk.get_parameter("w_i", [input_size, 3 * hidden_size], inputs.dtype,
                               init=self.w_i_init)  # Input weights
        w_h = hk.get_parameter("w_h", [hidden_size, 3 * hidden_size], inputs.dtype,
                               init=self.w_h_init)  # Hidden weights
        b = hk.get_parameter("b", [3 * hidden_size], inputs.dtype, init=self.b_init)
        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * hidden_size], axis=0)  # (bias_ih, bias_hh), bias_ah


        gates_x = hk.LayerNorm(-1, True, True)(jnp.matmul(inputs, w_i))  # W_iz, W_ir, W_ia x
        zr_x, a_x = jnp.split(
            gates_x, indices_or_sections=[2 * hidden_size], axis=-1)
        zr_h = hk.LayerNorm(-1, True, True)(jnp.matmul(state, w_h_z))  # W_hz, W_hr, W_ha x
        zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(jax.nn.sigmoid(zr), indices_or_sections=2, axis=-1)  #z_t, r_t

        a_h = jnp.matmul(r * state, w_h_a)
        a = jnp.tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a
        return next_state, next_state


class LearnableLayerNormGRU(LearnableGRU, LayerNormGRU): pass


def get_gru_cls(ln: bool = False, learnable: bool = False, orthogonal_init: bool = True) -> Callable:
    """Get appropriate GRU class. LayerNorm, Learnable state, orthogonal initialization"""
    if ln: cls = LearnableLayerNormGRU if learnable else LayerNormGRU
    else: cls = LearnableGRU if learnable else hk.GRU
    if orthogonal_init: cls = partial(cls, w_i_init=get_orthogonal_activation('sigmoid'), w_h_init=get_orthogonal_activation('sigmoid'), b_init=jnp.zeros)
    return cls