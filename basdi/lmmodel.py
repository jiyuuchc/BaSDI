from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as onp

from .typing import *
from .utils import gauss_filter, sub_pixel_samples

PixelSize = float|Sequence[float]

__all__ = ["LMModel"]

class LMModel:
    """ Localization microscopy model
    """

    def __init__(
            self,
            ps:PixelSize,
            mode:str="simple", 
            norm_axis:Sequence(int)|None = None,
            sparsity:float=0.5, 
            rng:RNG=jax.random.PRNGKey(42),
            block_size: int = -1,
        ):
        self.mode = mode
        self.ps = onp.asarray(ps)
        self.sparsity = float(sparsity)
        self.norm_axis = norm_axis
        self.rng = rng
        self.block_size = block_size

        self.border_width = 50
        self._built = False
    

    def _render(self, locs):
        dim = locs.shape[1]
        ps = onp.broadcast_to(self.ps, dim)
        bins = [jnp.arange(self.mins[k], self.maxs[k], ps[k]) for k in range(dim)]
        h, _ = jnp.histogramdd(locs, bins=bins)

        return h


    def _get_bbox(self, locs):
        self.mins = onp.floor(locs.min(axis=0) - self.ps * self.border_width)
        self.maxs = onp.ceil(locs.max(axis=0) + self.ps * self.border_width)


    def _build_minimal(self, locs, errs, n_samples):
        h = self._render(locs) + self.sparsity
        hm = jnp.zeros_like(h, dtype="float32")
        mean_sigma = tuple((errs.mean(axis=0) / self.ps).tolist())

        def _build(carry, x):
            key, hm = carry
            key, k1 = jax.random.split(key)
            hh = jax.random.gamma(k1, h)
            hh = gauss_filter(hh, mean_sigma)
            hh = hh / hh.sum(axis=self.norm_axis, keepdims=True)
            hm += jnp.log(hh)
            return (key, hm), None

        (self.rng, hm), _ = jax.lax.scan(_build, (self.rng, hm), None, length=n_samples)

        return hm / n_samples


    def _build_simple(self, locs, errs, n_samples):
        hm = jnp.zeros_like(self._render(locs), dtype='float32')
        mean_sigma = tuple((errs.mean(axis=0) / self.ps).tolist())

        def _build(carry, x):
            key, hm = carry
            key, k1, k2 = jax.random.split(key, 3)
            locs_ = locs + jax.random.normal(k1, shape=locs.shape) * errs
            h = self._render(locs_)
            h = jax.random.gamma(k2, h + self.sparsity)
            h = gauss_filter(h, mean_sigma)
            h = h / h.sum(axis=self.norm_axis, keepdims=True)
            hm += jnp.log(h)
            return (key, hm), None

        (self.rng, hm), _ = jax.lax.scan(_build, (self.rng, hm), None, length=n_samples)

        return hm / n_samples


    def build(self, locs:ArrayLike, errs:ArrayLike, n_samples:int)->None:
        locs = jnp.asarray(locs)
        errs = jnp.asarray(errs)

        self._get_bbox(locs)
        errs = jnp.broadcast_to(errs, locs.shape)

        if self.mode == "minimal":
            hm = self._build_minimal(locs, errs, n_samples)

        elif self.mode == "simple":
            hm = self._build_simple(locs, errs, n_samples)

        else: 
            raise ValueError(f"unkonw model {self.mode}")


        self._hm = hm
        self._built = True


    def e_log_ll(self, locs:ArrayLike, errs:ArrayLike|None=None, mask:ArrayLike|None=None)->float:
        """ Compute expectation of loglikelihood

        Args:
            obs: [..., n, 3] represent n observed locs. "..." are batch dims
            errs: localization errors. same (broadcast) dim as obs. Not needed for simple model
            mask: [..., n] boolean array masking valid entries of obs and errs

        returns:
            p: [...] loglokelihoods
        """
        if not self._built:
            raise RuntimeError("LMModel not built yet.")

        locs_ = (locs - self.mins) / self.ps
        p = sub_pixel_samples(
            self._hm, 
            locs_, 
            oob_mode="copy", 
            edge_indexing=True,
            blocksize=self.block_size,
        ).mean(axis=-1, where=mask)

        return p


    def render(self, locs:ArrayLike)->Array:
        """ Create a histogram image from localizations
        """
        if not self._built:
            self._get_bbox(locs)
        
        return self._render(locs)
