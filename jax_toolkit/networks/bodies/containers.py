from typing import Sequence, Optional, Union, Any, Callable

import haiku as hk
import jax.numpy as jnp
import jax
hk.Sequential

class SequentialFromRNN(hk.Module):
    def __init__(self):
        super().__init__()