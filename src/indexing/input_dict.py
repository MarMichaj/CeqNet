import jax
import depr.grand.src.indexing.structure as structure
import jax.numpy as jnp


def input_dict(x, atomic_numbers, idx_i, idx_j):
    return {
        structure.Z: atomic_numbers,
        structure.Rij: jax.vmap(lambda i, j: x[i] - x[j])(idx_i, idx_j),
        structure.idx_i: idx_i,
        structure.idx_j: idx_j
    }


def sph_input_dict(x, atomic_numbers, idx_i, idx_j, l):

    def sph_h_l(R):
        return real_sph_harm_deg(l, R)

    sph_h_l_all = jax.vmap(sph_h_l, in_axes=0)

    def normalize(x):
        return x / jnp.linalg.norm(x)

    norm_all = jax.vmap(normalize, in_axes=0)

    Rij = jax.vmap(lambda i, j: x[i] - x[j])(idx_i, idx_j)
    r_ij_norm = norm_all(Rij)  # normed molecule_inspect, input for spherical harmonics, shape : (len(idx_i),3)
    Y_ij = sph_h_l_all(r_ij_norm)

    return {
        structure.Z: atomic_numbers,
        structure.Rij: jax.vmap(lambda i, j: x[i] - x[j])(idx_i, idx_j),
        structure.idx_i: idx_i,
        structure.idx_j: idx_j,
        structure.Y_ij: Y_ij
    }
