__all__ = ['kl_loss', 'clipped_ppo_loss', 'unclipped_ppo_loss', 'entropy_loss', 'pg_loss']

import jax
import jax.numpy as jnp
import rlax
import chex
from functools import partial

Array = chex.Array
Scalar = chex.Scalar

clipped_ppo_loss = rlax.clipped_surrogate_pg_loss
unclipped_ppo_loss = partial(clipped_ppo_loss, epsilon=1e6)  # Super high value to avoid clipping
pg_loss = rlax.policy_gradient_loss


def kl_loss(
    prob_ratios_t: Array,
    reverse: bool = False,
) -> Array:
    """Compute approximate KL divergence (i.e., not the closed-form)

    https://github.com/chloechsu/revisiting-ppo/blob/ed21a2aadb8328abd5f73e211de77f87edb69c13/src/policy_gradients/steps.py#L2
    Args:
        prob_ratios_t: Ratio of action probabilities for actions a_t:
            rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ)
        reverse: If true, KL divergence new to old, otherwise old to new
    """
    return jax.lax.select(reverse, jnp.mean(-jnp.log(prob_ratios_t)), jnp.mean(prob_ratios_t * jnp.log(prob_ratios_t)))


def entropy_loss(entropy: Array) -> Array: return -jnp.mean(entropy)