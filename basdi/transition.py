from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as onp

from typing import Tuple

from .typing import *

__all__ = [
    "gauss_transition_model",
]

def gauss_transition_model(key:RNG, cur_drift: ArrayLike, step:int|None=None, *, sigmas:ArrayLike=1, multiply:int=1)->Tuple[Array, Array]:
    """ 
    """
    cur_drift = jnp.asarray(cur_drift)
    sigmas = jnp.asarray(sigmas)

    d = jax.random.normal(key, shape=(multiply,) + cur_drift.shape) * sigmas

    next_drift = (cur_drift + d).reshape((-1,) + cur_drift.shape[1:])
    next_idx = jnp.tile(jnp.arange(cur_drift.shape[0], dtype="int16"), multiply)

    return next_drift, next_idx

