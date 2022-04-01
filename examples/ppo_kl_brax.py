import argparse
import json
import os
import pathlib
import random
import sys
import time
import warnings
from distutils.util import strtobool
from typing import Optional, Tuple, Dict, Sequence, NamedTuple, Union, Callable

import brax
import psutil
import rlax
import wandb
from wandb.util import generate_id

warnings.filterwarnings('ignore', r'Mean of empty slice')

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from threadpoolctl import ThreadpoolController
from po_brax.envs import create_gym_env

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from jax_toolkit.networks.rnn.gru import get_gru_cls_or_fake, get_gru_cls
from jax_toolkit.networks.bodies.mlp import MLP
from jax_toolkit.networks.heads import *
from jax_toolkit.utils.normalization import create_observation_normalizer, NormParams
from jax_toolkit.losses.advantages import compute_gae
from jax_toolkit.losses import entropy_loss, kl_loss, value_loss, unclipped_ppo_loss
import chex

BASE_PATH = pathlib.Path(__file__).parent
H_DIM = 128
DIMS = [64]

# Types
GRUState = Union[jnp.ndarray, Tuple]
class ACObs(NamedTuple):
    o: chex.Array  # obs
    p_a: chex.Array  # previous action
    r: chex.Array  # reset


class ACOutput(NamedTuple):
    policy: Distribution
    value: jnp.ndarray


def _core_input(inputs: ACObs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.concatenate([inputs.o, inputs.p_a], -1), inputs.r


class ActorCritic(hk.RNNCore):
    """Actor critic class"""
    def __init__(self, envs, args):
        super().__init__(name='AC')
        action_size = envs.single_action_space.shape[0]
        self._recurrent = args.recurrent
        self.shared = hk.ResetCore(get_gru_cls_or_fake(args.recurrent, args.ln, args.learn_state)(H_DIM))
        self.actor = hk.Sequential([
            MLP(DIMS, activation=jax.nn.elu),
            BetaHead(action_size) if args.policy_param == 'beta' else GaussianHeadDependentLogStd(action_size)
        ])
        self.critic = hk.Sequential([
            MLP(DIMS, activation=jax.nn.elu),
            V(),
            partial(jnp.squeeze, axis=-1)
        ])

    def _head(self, f: jnp.ndarray):
        return self.actor(f), self.critic(f)

    def __call__(self, inputs: ACObs, state: GRUState) -> Tuple[ACOutput, GRUState]:
        """One step, B dim"""
        inputs = jax.tree_map(lambda t: t[None, ...], inputs)  # Add T dimension
        out, state = self.unroll(jax.tree_map(lambda t: t[None, ...], inputs), state)  # Add T dimension, state doesn't need it
        return jax.tree_map(lambda t: jnp.squeeze(t, 0), out), state
        # f, state = self.shared(_core_input(inputs), state)
        # pi, v = self._head(f)
        # return ACOutput(pi, v), state

    def step(self, inputs: ACObs, state: GRUState):
        """One step, just actor"""
        out, state = self.shared(_core_input(inputs), state)
        return self.actor(out), state

    def unroll(self, inputs: ACObs, state: GRUState):
        """Sequence of steps, T,B dim"""
        if self._recurrent: f, state = hk.dynamic_unroll(self.shared, _core_input(inputs), state)
        else: f, state = hk.BatchApply(self.shared)(_core_input(inputs), state)
        pi = hk.BatchApply(self.actor)(f[:-1])
        v = hk.BatchApply(self.critic)(f)
        # pi, v = hk.BatchApply(self._head)(f)
        return ACOutput(pi, v), state

    def initial_state(self, batch_size: Optional[int]):
        return self.shared.initial_state(batch_size)


def make_models(envs, args):
    # Initial state may require parameters if learnable
    initial_state_init, initial_state = hk.without_apply_rng(hk.transform(lambda b: ActorCritic(envs, args).initial_state(b)))
    _, step = hk.without_apply_rng(hk.transform(lambda i, state: ActorCritic(envs, args).step(i, state)))
    init, unroll = hk.without_apply_rng(hk.transform(lambda i, state: ActorCritic(envs, args).unroll(i, state)))
    # init, step = hk.without_apply_rng(hk.transform(lambda i, state: ActorCritic(envs, args)(i, state)))
    # _, unroll = hk.without_apply_rng(hk.transform(lambda i, state, a: ActorCritic(envs, args).unroll(i, state, a)))
    return init, (initial_state_init, initial_state), (step, unroll)


def make_envs(args, run_name: str):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    # use the gym environment for convenience
    env = create_gym_env(args.gym_id, args.num_envs, args.seed, 'gpu' if args.cuda else 'cpu', action_repeat=args.frame_skip, episode_length=args.time_limit)
    eval_env = create_gym_env(args.gym_id, args.num_eval_envs, args.seed+1, 'gpu' if args.cuda else 'cpu', action_repeat=args.frame_skip, episode_length=args.time_limit, eval_metrics=True, discount=args.gamma)
    return env, eval_env, # o


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo-name', type=str, default='PPO')
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')

    # env
    parser.add_argument('--gym-id', type=str, default="ant",
        help='Environment from brax (or po_brax)')
    parser.add_argument('--frame-skip', type=int, default=6,
        help='Number of environment steps to repeat action')
    parser.add_argument('--time-limit', type=int, default=1000,
        help='Max timesteps of episode')
    parser.add_argument('--reward-scale', type=float, default=0.1,
        help='If not 1, scale raw reward to agent by this factor')

    # Env eval
    parser.add_argument('--log-frequency', type=int, default=20,
        help='Frequency of evaluation and stat logging')
    parser.add_argument('--num-eval-envs', type=int, default=128,
        help='the number of parallel evaluation environments')

    # Base
    parser.add_argument('--learning-rate', type=float, default=3e-4,
        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=int(1e7),
        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--torch-benchmark', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.benchmark=True`')
    parser.add_argument('--torch-script', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, attempt to use torchscript for agent')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, cuda will be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='if toggled, this experiment will be tracked with Weights and Biases')
    parser.add_argument('--wandb-project-name', type=str, default="cleanoc_debug2",
        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default='davidslayback',
        help="the entity (team) of wandb's project")
    parser.add_argument('--wandb-mode', type=str, default='offline',
                        help='Online or offline')
    parser.add_argument('--tb', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='use tensorboard')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='whether to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--capture-video-episode-frequency', type=int, default=20,
        help='Capture a full episode once every (this many) learning iterations')
    parser.add_argument('-v', '--verbose', type=int, default=1, help="Once logs to tensorboard, twice prints")

    # Model specific arguments
    parser.add_argument('--normalize-obs', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Normalize observations with running statistics')
    parser.add_argument('--ln', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='Use layer-normalization before MLP (common practice for some envs)')
    parser.add_argument('--recurrent', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='Use GRU')
    parser.add_argument('--learn-state', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help='If using recurrent mode, learn the initial state used on reset')
    parser.add_argument('--policy-param', type=str, default='beta',
                        help="Either 'beta' or 'gaussian'. Use either beta policy parameterization or gaussian with state-dependent log_std")

    # Algorithm specific arguments
    parser.add_argument('--num-envs', type=int, default=2048,
        help='the number of parallel game environments')
    parser.add_argument('--num-steps', type=int, default=20,
        help='the number of steps to run in each environment per policy rollout')
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gamma', type=float, default=0.99,
        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='the lambda for the general advantage estimation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
        help="Toggles advantages normalization")
    parser.add_argument('--ent-coef', type=float, default=1e-3,
        help="coefficient of the entropy")
    parser.add_argument('--ent-coef-steps', type=float, default=0.,
        help="Step where final entropy reaches final fraction of original. If 0, hold entropy constant. 0 < x <= 1 means fraction of total steps")
    parser.add_argument('--ent-coef-final-fraction', type=float, default=0.1,
        help="Fraction of initial entropy to decay to. Default to 1/10")
    parser.add_argument('--vf-coef', type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=1.,
        help='the maximum norm for the gradient clipping')

    # PPO specific arguments
    parser.add_argument('--num-minibatches', type=int, default=32,
        help='the number of mini-batches')
    parser.add_argument('--update-epochs', type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument('--kl-coef', type=float, default=3.,
        help="the surrogate clipping coefficient")


    args, unknown = parser.parse_known_args()
    print(f"Unknown {unknown}")
    args.batch_size = int(args.num_envs * args.num_steps)
    if args.ent_coef_steps and args.ent_coef_steps <= 1: args.ent_coef_steps = args.total_timesteps * args.ent_coef_steps
    args.ent_coef_steps = int(args.ent_coef_steps)
    args.minibatch_size = int(args.num_envs // args.num_minibatches)
    # fmt: on
    return args


if __name__ == "__main__":
    controller = ThreadpoolController()
    args = parse_args()
    with controller.limit(limits=12):
        # Filter out runs with an entropy schedule and non-default fraction
        if not (args.ent_coef_steps) and (args.ent_coef_final_fraction != 0.1): sys.exit(0)
        if args.learn_state and (not args.recurrent): sys.exit(0)  # Learnable state makes no sense for reactive
        if (not args.gae_lambda) and (args.gae_lambda < 1): sys.exit(0)
        if not args.track: args.tb = True  # If not tracking with wandb, only thing we can do is tb
        run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time_ns())}"
        if args.track:
            tags = [args.algo_name, 'Recurrent' if args.recurrent else 'Reactive', f'Env_{args.gym_id}']
            tags.extend([f'Brax-{args.gym_id}'])
            run = wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=args.tb,
                config=vars(args),
                name=run_name,
                save_code=True,
                group=f"{'_'.join(tags)}",
                mode=args.wandb_mode,
                tags=tags,
                id=generate_id(16),
                monitor_gym=True
            )
            rdir = os.path.split(run.dir)[0]
        if args.tb:
            writer = SummaryWriter(f"runs/{run_name}")
            writer.add_text(
                "hyperparameters",
                "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )

        # Verify GPU
        backend = 'gpu' if (args.cuda and jax.default_backend() == 'gpu') else 'cpu'
        args.cuda = True if backend == 'gpu' else False

        # Create envs
        envs, eval_envs = make_envs(args, run_name)
        next_obs = envs.reset()
        next_done = envs._state.done
        action = envs.action_space.sample()
        rng = hk.PRNGSequence(args.seed)

        # Create model
        # agent = Agent(envs, args)
        init, (state_init, state_create), (act, unroll) = make_models(envs, args)
        state_params = state_init(None, args.num_envs)  # Only really need these once...could I just do a jnp.zeros?
        state = state_create(state_params, args.num_envs)

        # params = init(next(rng), ACObs(next_obs, a, next_done), state)
        tobs = ACObs(next_obs[None, ...], action[None, ...], next_done[None,...])
        params = jax.jit(init, backend=backend)(next(rng), tobs, state)
        # act = jax.jit(act, backend=backend)
        unroll = jax.jit(unroll, backend=backend)
        # test = jax.jit(step, backend=backend)(params, ACObs(next_obs, a, next_done), state)
        # test2 = jax.jit(unroll, backend=backend)(params, ACObs(next_obs[None, ...], a[None, ...], next_done[None, ...]), state)

        # Create optimizer
        num_updates = args.total_timesteps // args.batch_size
        schedule = optax.linear_schedule(1., 0., num_updates) if args.anneal_lr else optax.constant_schedule(1.)
        optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.learning_rate), optax.scale_by_schedule(schedule))
        opt_params = optimizer.init(params)

        # Entropy
        final_ent = args.ent_coef * args.ent_coef_final_fraction if args.ent_coef_steps else args.ent_coef
        entropy_schedule = optax.linear_schedule(args.ent_coef, final_ent, num_updates)

        # Observation normalization
        norm_params, norm_update_fn, norm_apply_fn = create_observation_normalizer(next_obs.shape[-1], num_leading_batch_dims=2)

        # Action scaling
        a_scale, inv_a_scale = get_action_scale_fns(envs.single_action_space, from_beta=args.policy_param=='beta', apply_tanh=not args.policy_param=='beta')


        @partial(jax.jit, backend=backend)
        def sample_step(o, a, d, state, n_params, p_params, key):
            """One step, get scaled action, actual sample, and state"""
            obs = ACObs(norm_apply_fn(n_params, o), a, d)
            pi, state = act(p_params, obs, state)
            a = pi.sample(seed=key)
            return a_scale(a), a, state

        @partial(jax.jit, backend=backend)
        def update_norm_and_unroll(o, p_a, a, d, state, n_params, p_params):
            """Multi-step, get old log probs, update normalization parameters

            Called after rollout
            Returns:
                old_log_prob
                normalized obs
                normalizer parameters
            """
            n_params = norm_update_fn(n_params, o[:-1])
            o = norm_apply_fn(n_params, o)
            obs = ACObs(norm_apply_fn(n_params, o), p_a, d)
            pi, v = unroll(p_params, obs, state)[0]
            return (pi.log_prob(a).sum(-1), v), o, n_params


        # @partial(jax.jit, backend=backend)
        def loss(p_params: hk.Params, o, p_a, a, d, initial_state, adv, ret, lp_old, vf_coef: float, ent_coef: float, kl_coef: float):
            """Loss on one minibatch of one epoch"""
            if args.norm_adv: adv = jax.nn.normalize(adv)
            obs = ACObs(o, p_a, d)
            pi, v = unroll(p_params, obs, initial_state)[0]
            lp = pi.log_prob(a).sum(-1)
            logratio = lp - lp_old
            ratio = jnp.exp(logratio)
            pi_loss = jnp.mean(jax.vmap(unclipped_ppo_loss)(ratio, adv))
            kl_penalty = kl_loss(ratio)
            ent = pi.entropy().sum(-1)
            ent_loss = entropy_loss(ent)
            vf_loss = value_loss(ret, v[:-1])
            total_loss = pi_loss + (vf_loss * vf_coef) + (ent_loss * ent_coef) + (kl_penalty * kl_coef)
            return total_loss, {'total_loss': total_loss,
                                'vf_loss': vf_loss,
                                'pi_loss': pi_loss,
                                'ent_loss': ent_loss,
                                'kl_loss': kl_penalty}

        @partial(jax.jit, backend=backend, static_argnames=('vf_coef', 'kl_coef'))
        def update_one_epoch(p_params: hk.Params, opt_params: optax.OptState, b_idxes, o, p_a, a, d, initial_state, adv, ret, lp_old, vf_coef: float, ent_coef: float, kl_coef: float):
            """Update using randomly sampled indices"""
            total_loss = ()
            for b_ind in b_idxes:
                total_loss, grads = jax.value_and_grad(loss, has_aux=True)(
                    p_params,
                    o=o[:, b_ind],
                    p_a=p_a[:, b_ind],
                    a=a[:, b_ind],
                    d=d[:, b_ind],
                    initial_state=initial_state[b_ind],
                    adv=adv[:, b_ind],
                    ret=ret[:, b_ind],
                    lp_old=lp_old[:, b_ind],
                    vf_coef=vf_coef,
                    ent_coef=ent_coef,
                    kl_coef=kl_coef
                )
                updates, opt_params = optimizer.update(grads, opt_params, p_params)
                p_params = optax.apply_updates(p_params, updates)
            return p_params, opt_params, total_loss

        # Generalized advantage estimation
        gae = jax.jit(jax.vmap(partial(compute_gae, lambda_=args.gae_lambda)), backend=backend)


        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        for update in range(1, num_updates + 1):
            initial_state = state
            obs = [next_obs]; prev_actions = [action]; rewards = []; dones = [next_done]; actions = []
            ent_coef = entropy_schedule(global_step)  # Entropy schedule
            keys = jax.random.split(next(rng), args.num_steps)  # Reserve random keys for action sampling
            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                scaled_action, action, state = sample_step(obs[-1], prev_actions[-1], dones[-1], state, norm_params, params, keys[step])
                next_obs, r, next_done, info = envs.step(scaled_action)
                prev_actions.append(scaled_action); actions.append(action); obs.append(next_obs); rewards.append(r * args.reward_scale); dones.append(next_done)

            # bootstrap value if not done
            o, p_a, a, r, d = (jnp.stack(_) for _ in (obs, prev_actions, actions, rewards, dones))
            (old_lp, v), o, norm_params = update_norm_and_unroll(o, p_a, a, d, initial_state, norm_params, params)  # o is now normalized
            discount_mask = (1. - d[1:]) * args.gamma
            advantages, returns = gae(v[:-1], r, discount_mask, v[1:])

            # Policy loss. Shuffle trajectories only (not timesteps)
            b_inds = np.arange(args.num_envs)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                mb_inds = np.split(b_inds, args.num_minibatches)
                params, opt_params, total_loss = update_one_epoch(params, opt_params, mb_inds, o, p_a, a, d, initial_state, advantages, returns, old_lp, args.vf_coef, entropy_schedule(update), args.kl_coef)

            base_logs = {'global_step': global_step}
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if ((update-1) % args.log_frequency) == 0:
                o_e, a_e, d_e = eval_envs.reset(), jnp.zeros_like(action), jnp.zeros_like(next_done)
                e_state = state_create(params, args.num_eval_envs)
                e_keys = jax.random.split(next(rng), args.time_limit)
                for t in range(args.time_limit):
                    a_e, _, state = sample_step(o_e, a_e, d_e, e_state, norm_params, params, keys[t])
                    o_e, _, d_e, _ = eval_envs.step(a_e)
                base_logs |= eval_envs.get_stats()
            if args.verbose:
                base_logs |= {
                    f"charts/{l}": np.array(v) for l, v in total_loss[-1].items()
                }
                y_pred, y_true = np.array(v[:-1]), np.array(returns)
                var_y = np.var(y_true)
                explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                base_logs |= {
                    "charts/learning_rate": schedule(update),
                    "losses/explained_variance": explained_var,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                }
                if args.verbose > 1:
                    print(f"Update {update} of {num_updates}")
                    print(f"SPS: {base_logs['charts/SPS']}")
                    try:
                        print(f"DiscountedReturnMean: {base_logs['charts/mean_discounted_episodic_return']}")
                    except KeyError: pass
            if args.tb:
                for k, v in base_logs.items(): writer.add_scalar(k, v, global_step)
            else: wandb.log(base_logs, commit=True)

    print(f"PPO {vars(args)} finished")
    if args.tb: writer.close()
    if args.track:
        wandb.finish()
        if args.wandb_mode == 'offline':
            import subprocess
            subprocess.run(['wandb', 'sync', rdir])  # Sync afterward if we're running offline
