@chex.dataclass
class TrainingState:
    """Convenience class for parameter state"""
    step: int
    opt_state: optax.OptState
    params: hk.Params
    norm_params: NormParams
    unroll: Callable
    tx: optax.GradientTransformation

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, unroll, params, norm_params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            unroll=unroll,
            params=params,
            norm_params=norm_params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class Agent:
    def __init__(self, envs, args):
        self.args = args
        self.backend = 'gpu' if (args.cuda and jax.default_backend() == 'gpu') else 'cpu'
        self._b_size = envs.num_envs
        self._init, (self._state_param, self._get_state), (self._step, self._unroll) = make_models(envs, args)
        self._dummy_x = ACObs(envs.observation_space.sample(), envs.action_space.sample(), jnp.zeros((args.num_envs,)))
        self._n_params, self._norm_update, self._norm_apply = create_observation_normalizer(self._dummy_x.o.shape[-1], args.normalize_obs, num_leading_batch_dims=2)
        # Learning parameters
        self._gamma = args.gamma  # Discount
        self._gae = jax.vmap(partial(compute_gae, lambda_=args.gae_lambda))  # Lambda-returns and advantages
        num_updates = args.total_timesteps // args.batch_size
        schedule = optax.linear_schedule(1., 0., num_updates) if args.anneal_lr else optax.constant_schedule(1.)
        self._opt = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.learning_rate), optax.scale_by_schedule(schedule))

        def _loss(params: hk.Params, trajectories):


    @partial(jax.jit, static_argnums=(0, 1), backend=self.backend)
    def initial_state(self, batch_size: Optional[int]) -> GRUState:
        """Get starting hidden state"""
        p = self._state_param(None)
        return self._get_state(p, batch_size)

    @partial(jax.jit, static_argnums=0)
    def initial_network_params(self, rng: jnp.ndarray, state: GRUState):
        """Get initial model parameters"""
        return self._init(rng, jax.tree_map(lambda x: x[None, ...], self._dummy_x), state)

    def initial_norm_params(self):
        """Initial observation normalizer parameters"""
        return self._n_params

    def initial_training_state(self, rng: jnp.ndarray) -> TrainingState:
        return TrainingState.create(unroll=self._unroll,
                                    params=self.initial_network_params(rng, self._dummy_x, jnp.zeros(self._b_size)),
                                    norm_params=self._n_params,
                                    tx=self._opt)


    @partial(jax.jit, static_argnums=0)
    def step(self, rng: jnp.ndarray, params: hk.Params, n_params: NormParams, obs: ACObs, state: GRUState):
        """Take one step, get action and state"""
        obs = obs._replace(o=self._norm_apply(n_params, obs.o))
        pi, state = self._step(params, obs, state)
        action = pi.sample(seed=rng)
        return action, state

    @partial(jax.jit, static_argnums=0)
    def update(self):
        ...