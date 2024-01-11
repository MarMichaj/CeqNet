import jax
import jax.numpy as jnp

def max_distances(R: jnp.ndarray):
    """
    Computes the maximum distance between the atoms in a molecule.
    :param R: atom positions, shape: (n,3)
    :return: distance, shape: (1)
    """
    distances = jnp.array(R[:, None] - R[None, :])
    return jnp.max(distances)

