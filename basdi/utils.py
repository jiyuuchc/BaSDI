from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp
import numpy as onp

from jax.scipy.signal import convolve
from typing import Sequence, Tuple

from .typing import *

__all__ = [
    "sub_pixel_samples",
    "gauss_filter",
    "forback",
    "frc",
]

@partial(jax.jit, static_argnames="oob_mode")
def _retrieve_value_at(img, loc, *, oob_mode="constant", oob_constant=0):
    dim = loc.shape[-1]
    iloc = jnp.floor(loc).astype(int) # [n, d]
    res = loc - iloc # [n, d]

    offsets = jnp.asarray(
        [[(i >> j) % 2 for j in range(dim)] for i in range(2 ** dim)]
    ) # [2**d, d]
    offsets = offsets[:, None, :] # [2**d, 1, d]
    ilocs = iloc + offsets # [2**d, n, d]

    weight = jnp.prod(res * (offsets == 1) + (1 - res) * (offsets == 0), axis=-1) # [2**d, n]
    max_indices = jnp.asarray(img.shape)[:dim]

    if oob_mode == "constant":
        sel = (ilocs >= 0).all(axis=-1, keepdims=True) & (ilocs < max_indices).all(axis=-1, keepdims=True)

        values = jnp.where(
            sel,
            img[tuple(jnp.moveaxis(ilocs, -1, 0))],
            oob_constant,
        ) #[2**d, n, 1]

    elif oob_mode == "copy":
        ilocs = jnp.clip(ilocs, 0, max_indices-1)
        values = img[tuple(jnp.moveaxis(ilocs, -1, 0))]

    else:
        raise ValueError(f"Unknown out-of-bound mode {oob_mode}")

    value = (values.squeeze(-1) * weight).sum(axis=0)

    return value

def sub_pixel_samples(
        img:ArrayLike, 
        locs:ArrayLike, 
        *, 
        oob_mode:str="constant", 
        oob_constant:float=0, 
        edge_indexing:bool=False, 
        blocksize:int=-1
    )->Array:
    """Retrieve image values as non-integer locations by linear interpolation

    Args:
        img: array of shape [D1,D2,..,Dk, ...]
        locs: array of shape [d1,d2,..,dn, k]
    
    Keyword Args:
        oob_mode: out-of-bound behavior, "constant" - use oob_constant, "copy" - copy values at border
        oob_constant: optional float constant, defualt 0.
        edge_indexing: if True, the index for the first value in img is 0.5, otherwise 0. Default is False
        blocksize: if positive, pad the index array "loc" to blocksize to reduce jax recompilation

    Returns:
        values: array of shape [d1,d2,..,dn, ...]
    """

    # BLOCKSIZE=1024*1024

    loc_shape = locs.shape
    img_shape = img.shape
    d_loc = loc_shape[-1]

    if edge_indexing:
        locs = locs - 0.5

    img = img.reshape(img_shape[:d_loc] + (-1,))
    locs = locs.reshape(-1, d_loc)
    n_locs = locs.shape[0]
    if blocksize > 0:
        BLOCKSIZE = blocksize
        padding = (n_locs-1)//BLOCKSIZE*BLOCKSIZE+BLOCKSIZE - n_locs
        locs = jnp.pad(locs, [[0, padding], [0,0]])
    values = _retrieve_value_at(img, locs, oob_mode=oob_mode, oob_constant=oob_constant)
    values = values[:n_locs]

    out_shape = loc_shape[:-1] + img_shape[d_loc:]
    values = values.reshape(out_shape)

    return values


def _get_gauss_filter(sigma):
    filter_size = int(sigma * 4 + 1)
    filter = jnp.exp(-jnp.arange(-filter_size, filter_size+1) ** 2 / 2 / sigma / sigma)
    filter = filter / filter.sum()
    return filter


@partial(jax.jit, static_argnums=(1,))
def gauss_filter(data:ArrayLike, sigma: float|Sequence[float])->Array:
    """ ND Gaussian blur.

    Args:
        data: ndarray
        sigma: filter sizes. 
    
    Returns:
        ndarray
    """
    try:
        iter(sigma)
    except:
        sigma = (sigma,)
    
    dim = len(data.shape)
    if len(sigma) == 1:
        sigma = sigma * dim

    if len(sigma) != dim:
        raise ValueError(f"Cannot broadcast sigma={sigma} to shape [{dim}]")
    
    filters = (_get_gauss_filter(s) for s in sigma)
    
    for k, f in enumerate(filters):
        filter_shape = [1,] * dim
        filter_shape[k] = -1
        f = f.reshape(filter_shape)
        data = convolve(data, f, mode="same")

    return data


def _ofs_filter(f, data, eps):
    data = convolve(
        data, f,
        mode="same",
    )
    data += data.sum() * eps
    return data


def forback(E, T, eps=0):
    """ Forward-backward integration of marginal prob 

    Args:
        E: [N, ...] expectation distribution at N time steps
        T: transfer filter
        eps: additional background prob for the transfer filter
    
    Returns:
        Array of [N, ...] marginal prob
    """
    # f, ps, _, ss = e_xys.shape
    # T0 = jnp.asarray([
    #     [p_xy ** 2, p_xy, p_xy ** 2],
    #     [p_xy, 1, p_xy],
    #     [p_xy ** 2, p_xy, p_xy ** 2],
    # ])
    # T = jnp.stack(
    #     [T0 * p_s, T0, T0 * p_s],
    #     axis=-1,
    # )
    T = T / T.sum()
    f = E.shape[0]

    # forward computation
    a = [E[0]]
    for i in range(1, f):
        a_t = _ofs_filter(T, a[i-1], eps) * E[i]
        a.append(a_t / a_t.max()) # prevent overflow

    # backward computation
    b = [jnp.ones_like(E[-1])]
    for i in range(1, f):
        b_t = _ofs_filter(T, b[0] * E[f-1-i], eps)
        b.insert(0, b_t / b_t.max())

    # calculate the probability
    g = jnp.stack(a) * jnp.stack(b)
    g = g / g.sum(axis=(1,2,3), keepdims=True)

    return g


def frc(render_a:ArrayLike, render_b:ArrayLike, *, nn:int=-1)->tuple[Array, Array]:
    fftimg1 = jnp.fft.fft2(render_a - render_a.mean())
    fftimg2 = jnp.fft.fft2(render_b - render_b.mean())

    fftc = fftimg1 * jnp.conjugate(fftimg2)
    ffta1 = jnp.abs(fftimg1) ** 2
    ffta2 = jnp.abs(fftimg2) ** 2

    h, w = fftc.shape[-2:]
    x = jnp.fft.fftfreq(w)
    y = jnp.fft.fftfreq(h)
    xx, yy = jnp.meshgrid(x, y)
    R = jnp.sqrt(yy ** 2 + xx ** 2)
    if nn < 0:
        nn = min(h, w)
    N, bins = jnp.histogram(R.flatten(), nn)

    Rd = jnp.digitize(R, bins)

    frc_v = []
    for k in range(nn):
        a = fftc.sum(axis=(-1,-2), where=Rd==k)
        b = ffta1.sum(axis=(-1,-2), where=Rd==k) * ffta2.sum(axis=(-1,-2), where=Rd==k)
        b = jnp.sqrt(b)

        frc_v.append(jnp.real(a)/jnp.real(b))

    frc_v = jnp.asarray(frc_v)
    bins = (bins[1:] + bins[:-1]) /2

    return bins, frc_v


