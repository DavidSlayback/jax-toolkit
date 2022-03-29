__all__ = ['tfp', 'tfd', 'Distribution', 'Categorical', 'Beta', 'Gaussian', 'Bernoulli']

import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp
tfd = tfp.distributions
tfb = tfp.bijectors


# Most common distributions
Distribution = tfd.Distribution
Categorical = tfd.Categorical
Beta = tfd.Beta
Gaussian = tfd.MultivariateNormalDiag
SquashedGaussian = lambda *args, **kwargs: tfd.TransformedDistribution(distribution=Gaussian(*args, **kwargs), bijector=tfb.Tanh())  # Tanh squashed gaussian
Bernoulli = tfd.Bernoulli