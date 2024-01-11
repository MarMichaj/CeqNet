import jax.numpy as jnp
from typing import (Any, Callable, Tuple)


Array = Any


def pbc_diff(r_ij: jnp.ndarray, lat: jnp.ndarray):
    """
    Clamp differences of vectors to super cell.

    Args:
        r_ij (Array): distance vectors, shape: (...,3)
        lat (Array): matrix containing lattice vectors as columns, shape: (3,3)

    Returns: clamped distance vectors, shape: (...,3)

    """
    lat_inv = jnp.linalg.inv(lat)  # shape: (3,3)
    c = jnp.einsum('ij, ...j -> ...i', lat_inv, r_ij)  # shape: (...,3)
    delta = r_ij - jnp.einsum('ij, ...j -> ...i', lat, jnp.rint(c))  # shape: (...,3)
    return delta


def add_cell_offsets(r_ij: jnp.ndarray, cell: jnp.ndarray, cell_offsets: jnp.ndarray):
    """
    Add offsets to distance vectors given a cell and cell offsets. Cell is assumed to be
    Args:
        r_ij (Array): Distance vectors, shape: (n_pairs,3)
        cell (Array): Unit cell matrix, shape: (3,3). Unit cell vectors are assumed to be row-wise.
        cell_offsets (Array): Offsets for each pairwise distance, shape: (n_pairs,3).
    Returns:
    """
    offsets = jnp.einsum('...i, ij -> ...j', cell_offsets, cell)
    return r_ij + offsets


# def get_pbc_fn(lat_and_inv: Tuple[Array, Array]) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#
#     Args:
#         lat_and_inv (Array): Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
#
#     Returns:
#
#     """
#     def pbc_diff_fn(r_ij: jnp.ndarray) -> jnp.ndarray:
#         """
#         Clamp differences of vectors to super cell.
#
#         Args:
#             r_ij (Array): distance vectors, shape: (...,3)
#
#         Returns: clamped distance vectors, shape: (...,3)
#
#         """
#
#         lat, lat_inv = lat_and_inv  # shape: (3,3) // shape: (3,3)
#         c = jnp.einsum('ij, ...j -> ...i', lat_inv, r_ij)  # shape: (...,3)
#         delta = r_ij - jnp.einsum('ij, ...j -> ...i', lat, jnp.rint(c))  # shape: (...,3)
#         return delta
#
#     return pbc_diff_fn
