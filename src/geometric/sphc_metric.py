import jax
import jax.numpy as jnp
import numpy as np

from typing import Sequence, Union

from src.masking.mask import safe_mask
from src.sph_ops.base import make_degree_norm_fn

Array = Union[jnp.ndarray, np.ndarray]


def euclidean_degree_constructor(degrees: Sequence):
    norm_per_degree_fn = make_degree_norm_fn(degrees=degrees)
    norm_per_degree_fn = jax.vmap(norm_per_degree_fn)

    def euclidean_degree(x: Array, idx_i: Array, idx_j: Array):
        x_ij = jax.vmap(lambda i, j: x[j] - x[i])(idx_i, idx_j)  # shape: (n_pairs,m_tot)
        return norm_per_degree_fn(x_ij)

    return euclidean_degree


def l2_norm(x: Array, axis: int = -1):
    """
    Take the l2 norm.

    Args:
        x (Array): Coordinates, shape: (...,c)
        axis (int):

    Returns:

    """
    y = jnp.sum(x ** 2, axis=axis, keepdims=True)  # shape: (...)
    return jnp.squeeze(safe_mask(y > 0., jnp.sqrt, y), axis=axis)  # shape: (...)


def euclidean(x: Array, idx_i: Array, idx_j: Array):
    """

    Args:
        x (Array): sphc, shape: (n,m_tot)
        idx_i (Array):
        idx_j (Array):

    Returns:

    """
    x_ij = jax.vmap(lambda i, j: x[j] - x[i])(idx_i, idx_j)  # shape: (n_pairs,m_tot)
    return l2_norm(x_ij, axis=-1)[:, None]  # shape: (n_pairs)


def euclidean_constructor(*args, **kwargs):
    return euclidean


def cosine_degree_constructor(degrees: Sequence[int]):
    raise NotImplementedError


def cosine_constructor(*args, **kwargs):
    return cosine


def cosine(x: Array, idx_i: Array, idx_j: Array):
    """

    Args:
        x (Array): sphc, shape: (n,m_tot)
        idx_i (Array):
        idx_j (Array):

    Returns:

    """
    x_norm = l2_norm(x, axis=-1)[..., None]  # shape: (n,1)
    x_ = safe_mask(x_norm != 0, lambda y: y / x_norm, x, 0)  # shape: (n,m_tot)
    x_ij = jax.vmap(lambda i, j: x_[j] * x_[i])(idx_i, idx_j).sum(axis=-1)  # shape: (n_pairs)
    return safe_mask(x_ij != 0, lambda z: z, x_ij)[:, None]  # shape: (n_pairs)