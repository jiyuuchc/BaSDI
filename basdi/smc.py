from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as onp

from .lmmodel import LMModel
from .driftmodel import DriftModel
from .typing import *


__all__ = [
    "smc_history_reduce",
    "smc_history_draw_sample",
    "smc_resample",
    "smc_run_drift_inference",
    "smc_apply_drift",
]

def smc_history_reduce(history:SMCHistory, *, axis=0)->onp.ndarray:
    """ Compute sample average of a history object from SMC

    Args:
        history: history object with N elements. Each element is a tuple of (record, index).
        record is an array of shape [n_samples, ...], index is an int array [n_samples] indicating
        the index of parent from previous element

    Returns:
        Array [N, ...]
    """
    cm = []
    rs = onp.arange(history[-1][0].shape[0])
    for d0, idx in history[::-1]:
        cm.insert(0, d0[rs].mean(axis=axis))
        rs = idx[rs]
    return onp.stack(cm, axis=0)


def smc_history_draw_sample(history: SMCHistory, index:int)->Sequence[onp.ndarray]:
    """ obtain sample from history object

    Args:
        history: history object
        index: index number
    
    Returns:
        one sample of N states
    """
    s = []
    for d0, parent_idx in history[::-1]:
        s.insert(0, d0[index])
        index = parent_idx[index]
    return onp.stack(s, axis=0)


def smc_resample(key:RNG, history:SMCHistory, p:ArrayLike, *, n_samples:int=-1)->SMCHistory:
    """ weighted resampling for SMC

    Args:
        key: jax RNG
        history: history object with N elements. Each element is a tuple of (record, index).
        p: weight array of shape [n_samples] 
    
    Keyward Args:
        n_samples: negative means don't change the number
    
    Returns:
        resampled history object
    """
    if n_samples < 0:
        n_samples = p.shape[0] 

    rs = jax.random.choice(key, p.shape[0], [n_samples], p=p)
    rs = onp.asarray(rs)
    last_drift, last_idx = history[-1]
    history[-1] = onp.asarray(last_drift)[rs], onp.asarray(last_idx)[rs]

    return history


def smc_run_drift_inference(key: RNG, lm_model:LMModel, drift_model: DriftModel, trans_model: TransModel, data:Iterable, init_state: ArrayLike):
    """ Run sequential monte carlo (smc) to draw drift samples

    Args:
        key: jax RNG
        lm_model: a built LMModel to fit dato onto
        drift_model: a drift model subclassed from DriftModel
        trans_model: a callable of signiture f(rng_key, cur_drift_state, step)->next_drift_state
        data: iterable returning (localizations, errors, masks)
        init_state: initial drift state of shape [n_samples, state_dim]
    
    Returns:
        a history object. A list of (states, parent_idx). states are array of shape [n_samples, state_dim], parent_idx of [n_sample]
    """
    history = [(init_state, onp.arange(len(init_state)))]
    n_samples = init_state.shape[0]

    for step, obs in enumerate(data):
        key, k1, k2 = jax.random.split(key, 3)

        d0, di = trans_model(k1, history[-1][0], step=step)
        history.append((d0, di))

        p = drift_model.compute_likelihoods(lm_model, obs, d0)

        p = p - p.max()
        p = jnp.exp(p)
        p = p / p.sum()

        history = smc_resample(k2, history, p, n_samples=n_samples)

    return history[1:]


def smc_apply_drift(drift_model:DriftModel, data: Iterable, drifts: ArrayLike)->onp.ndarray:
    """ apply drift to data 

    Args:
        drift_model: a drift model of type DriftModel
        data: iterable of (localizations, errors, masks)
        drifts: a sequence of drift states according to drift_model
    
    Returns:
        new localizations.
    """
    results = []
    for (loc, err, mask), drift in zip(data, drifts):
        loc_new = onp.asarray(drift_model.apply(loc, drift))
        if mask is not None:
            loc_new = loc_new[mask]
        results.append(loc_new)

    return onp.concatenate(results)
