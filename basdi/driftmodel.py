from __future__ import annotations

from typing import Any, Iterable

import jax.numpy as jnp
import numpy as onp

from .lmmodel import LMModel
from .typing import *

__all__ = [
    "DriftModel", 
    "SimpleDrift", 
    "AnisotropicExpansion",
    "IsotropicExpansion",
]

class DriftModel:
    @classmethod
    def apply(cls, locs:ArrayLike, states:ArrayLike)->ArrayLike:
        raise NotImplementedError()
    
    @classmethod
    def compute_likelihood(cls, model:LMModel, obs:Any, states: ArrayLike)->Array:
        raise NotImplementedError()


class SimpleDrift(DriftModel):
    @classmethod
    def apply(cls, locs: Any, states:ArrayLike)->ArrayLike:
        _, dim = locs.shape

        if states.shape[-1] != dim:
            raise ValueError(f"Expected state dim == {dim}, got {states.shape[-1]}")

        locs_ = locs + states[..., None, :dim]

        return locs_

    @classmethod
    def compute_likelihoods(cls, model: LMModel, obs: tuple, states: ArrayLike)->Array:
        locs, err, mask = obs
        locs_ = cls.apply(locs, states)
        p = model.e_log_ll(locs_, err, mask)

        return p


class AnisotropicExpansion(DriftModel):
    norm_axis = None

    @classmethod
    def apply(cls, locs: ArrayLike, states:ArrayLike)->ArrayLike:
        _, dim = locs.shape

        if states.shape[-1] != dim * 3:
            raise ValueError(f"Expected state dim == {dim * 3}, got {states.shape[-1]}")

        drift = states[..., None, :dim]
        exp_center = states[..., None, dim:dim*2]
        scale = jnp.exp(states[..., None, dim*2:])

        locs_ = locs + drift
        locs_ = (locs_ - exp_center) * scale + exp_center

        return locs_

    @classmethod
    def compute_likelihoods(cls, model: LMModel, obs: tuple, states: ArrayLike)->Array:
        locs, err, mask = obs
        _, dim = locs.shape

        locs_ = cls.apply(locs, states)
        p = model.e_log_ll(locs_, err, mask)

        if cls.norm_axis is None:
            p += states[..., -dim:].sum(axis=-1)
        else:
            p += states[..., -dim:][..., cls.norm_axis].sum(axis=-1)

        return p


class IsotropicExpansion(AnisotropicExpansion):

    @classmethod
    def apply(cls, locs: ArrayLike, states:ArrayLike)->ArrayLike:
        _, dim = locs.shape

        if states.shape[-1] != dim * 2 + 1:
            raise ValueError(f"Expected state dim == {dim * 2 + 1}, got {states.shape[-1]}")
        
        expanded_states = jnp.concatenate([states, states[..., -1:], states[..., -1:]], axis=-1)

        return AnisotropicExpansion.apply(locs, expanded_states)
