from typing import Optional, Sequence
from functools import partial
import gym
B = 16
T = 5
updates = 20

from jax_toolkit.networks.bodies.mlp import MLP
from jax_toolkit.networks.heads.critics import V
from jax_toolkit.networks.heads.policies import CategoricalHead
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import optax

class Agent(hk.Module):
    def __init__(self,
                 envs: gym.Env,
                 layer_sizes: Sequence[int] = (64,64),
                 name: Optional[str] = None):
        super().__init__(name=name)
        a_space = getattr(envs, 'single_action_space', envs.action_space)
        self.n_act = a_space.shape or a_space.n
        o_space = getattr(envs, 'single_observation_space', envs.observation_space)
        n_obs = o_space.shape[0]
        self.actor = hk.Sequential([MLP(layer_sizes, activation=jax.nn.tanh), CategoricalHead(int(self.n_act))])
        self.critic = hk.Sequential([MLP(layer_sizes, activation=jax.nn.tanh), V(), jnp.squeeze])
        # self.critic = MLP(layer_sizes, activation=jax.nn.tanh)

    def __call__(self, inputs: jnp.ndarray):
        return self.actor(inputs), self.critic(inputs)

if __name__ == "__main__":
    """https://github.com/deepmind/bsuite/blob/master/bsuite/baselines/jax/actor_critic/agent.py"""
    e = gym.vector.SyncVectorEnv([partial(gym.make, 'CartPole-v1') for _ in range(B)])
    print(e.action_space)
    print(e.observation_space)
    o = e.reset()
    rng = hk.PRNGSequence(0)
    net = hk.without_apply_rng(hk.transform(lambda x: Agent(e)(x)))
    params = jax.jit(net.init)(next(rng), o)
    optimizer = optax.adam(3e-4)
    initial_opt_state = optimizer.init(params)
    def loss(params: hk.Params, ):
        ...
    @jax.jit
    def sgd_step():
        grads = jax.grad(loss)()
        ...
    for u in range(updates):
        for t in range(T):
            pi, v = jax.jit(net.apply)(params, o)
            a = pi.sample(seed=next(rng))
            o, r, d, info = e.step(onp.array(a))