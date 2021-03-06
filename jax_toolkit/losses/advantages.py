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
__all__ = ['compute_gae', 'termination_advantage']
"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""
from typing import Tuple
import jax
import jax.numpy as jnp
import chex
import rlax

Array = chex.Array
Numeric = chex.Numeric
Scalar = chex.Scalar


def compute_gae(
        v_tm1: Array,
        r_t: Array,
        discount_t: Array,
        v_t: Array,
        lambda_: Numeric,
        stop_target_gradients: bool = True,
) -> Tuple[Array, Array]:
    """Calculates the TD(lambda) temporal difference error.
    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node74.html).
    Args:
        v_tm1: sequence of state values at time t-1.
        r_t: sequence of rewards at time t.
        discount_t: sequence of discounts at time t.
        v_t: sequence of state values at time t.
        lambda_: mixing parameter lambda, either a scalar or a sequence.
        stop_target_gradients: bool indicating whether or not to apply stop gradient
            to targets.
    Returns:
        TD(lambda) temporal difference error (GAE) and returns
    """
    chex.assert_rank([v_tm1, r_t, discount_t, v_t, lambda_], [1, 1, 1, 1, {0, 1}])
    chex.assert_type([v_tm1, r_t, discount_t, v_t, lambda_], float)

    target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_)
    target_tm1 = jax.lax.select(stop_target_gradients,
                                jax.lax.stop_gradient(target_tm1), target_tm1)
    return target_tm1 - v_tm1, target_tm1


def termination_advantage(
        q_t_w_tm1: Array,
        v_t: Array,
        discount_t: Array,
        deliberation_cost: Scalar
) -> Array:
    """Calculates termination advantage

    Advantage of terminating option w_tm1 at state s_t
    Q(s', w) - v(s') + eta

    Args:
        q_t_w_tm1: Q-value of previous option in new state
        v_t: Value of overall policy in new state
        discount_t: Mask applied
        deliberation_cost: Extra cost incurred for switching options
    Returns:
         Termination advantage
    """
    chex.assert_rank([q_t_w_tm1, v_t, discount_t], [1, 1, 1])
    chex.assert_type([q_t_w_tm1, v_t, discount_t, deliberation_cost], float)
    return (q_t_w_tm1 - v_t + deliberation_cost) * discount_t
    ...

# def compute_gae(truncation: jnp.ndarray,
#                 termination: jnp.ndarray,
#                 rewards: jnp.ndarray,
#                 values: jnp.ndarray,
#                 bootstrap_value: jnp.ndarray,
#                 lambda_: float = 1.0,
#                 discount: float = 0.99):
#     r"""Calculates the Generalized Advantage Estimation (GAE).
#     Args:
#       truncation: A float32 tensor of shape [T, B] with truncation signal.
#       termination: A float32 tensor of shape [T, B] with termination signal.
#       rewards: A float32 tensor of shape [T, B] containing rewards generated by
#         following the behaviour policy.
#       values: A float32 tensor of shape [T, B] with the value function estimates
#         wrt. the target policy.
#       bootstrap_value: A float32 of shape [B] with the value function estimate at
#         time T.
#       lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
#         lambda_=1.
#       discount: TD discount.
#     Returns:
#       A float32 tensor of shape [T, B]. Can be used as target to
#         train a baseline (V(x_t) - vs_t)^2.
#       A float32 tensor of shape [T, B] of advantages.
#     """
#
#     truncation_mask = 1 - truncation
#     # Append bootstrapped value to get [v1, ..., v_t+1]
#     values_t_plus_1 = jnp.concatenate(
#         [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
#     deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
#     deltas *= truncation_mask
#
#     acc = jnp.zeros_like(bootstrap_value)
#     vs_minus_v_xs = []
#
#     def compute_vs_minus_v_xs(carry, target_t):
#         lambda_, acc = carry
#         truncation_mask, delta, termination = target_t
#         acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
#         return (lambda_, acc), (acc)
#
#     (_, _), (vs_minus_v_xs) = jax.lax.scan(compute_vs_minus_v_xs, (lambda_, acc),
#                                            (truncation_mask, deltas, termination),
#                                            length=int(truncation_mask.shape[0]),
#                                            reverse=True)
#     # Add V(x_s) to get v_s.
#     vs = jnp.add(vs_minus_v_xs, values)
#
#     vs_t_plus_1 = jnp.concatenate(
#         [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
#     advantages = (rewards + discount *
#                   (1 - termination) * vs_t_plus_1 - values) * truncation_mask
#     return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)
