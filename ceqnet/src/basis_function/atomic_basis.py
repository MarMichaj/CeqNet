import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import (Any, Callable, Dict, Tuple)


def get_basis_fn(x):
    if x == "rbf":
        return rbf
    elif x == "sto":
        return sto
    elif x == "cos":
        return cos
    else:
        raise ValueError(
            "{} currently not implemented as atomic basis function. Try 'rbf' or 'sto' instead.".format(x)
        )


class AtomicBasisFn(nn.Module):
    """ Class representation for the atomic basis functions.
    """
    n_abf: int
    basis_fn: Callable
    domain: Tuple[float, float]
    n_discretization_points: int
    init_coefficient_fn: Callable = nn.initializers.normal(stddev=.25)
    init_exponent_fn: Callable = nn.initializers.normal(stddev=.025)
    max_z = 100

    def setup(self):
        self.discretization_points = jnp.linspace(self.domain[0], self.domain[1], self.n_discretization_points)

    @nn.compact
    def __call__(self, r, z):
        """

        Args:
            r (Array): inter atomic molecule_inspect, shape: (...,n,n,1)
            z (Array): atomic types, shape: (...,n)

        Returns:

        """
        coefficient = nn.Embed(self.max_z, self.n_abf, embedding_init=self.init_coefficient_fn)(z)  # shape: (...,n,n_abf)
        exponent = nn.Embed(self.max_z, self.n_abf, embedding_init=self.init_exponent_fn)(z)  # shape: (...,n,n_abf)

        # expand column dimension for broadcasting
        coefficient = jnp.expand_dims(coefficient, -3)  # shape: (...,1,n,n_abf)
        exponent = jnp.expand_dims(exponent, -3)  # shape: (...,1,n,n_abf)
        return self.basis_fn(r,
                             coefficient=coefficient,
                             exponent=exponent,
                             l_k=self.discretization_points)  # shape: (...,n,n,L)

    @property
    def discretization_interval(self):
        return jnp.abs(self.discretization_points[0] - self.discretization_points[1])


def rbf_(r, coefficient, exponent, l_k):
    """
    Helper RBF function which is usually called to vmap over.
    Args:
        r (): molecule_inspect, shape: (...,n,n,1)
        coefficient (): shape: (...,1,n)
        exponent (): shape: (...,1,n)
        l_k (): shape: (L)

    Returns:

    """
    c = jnp.expand_dims(coefficient, -1)  # shape: (...,1,n,1)
    e = jnp.expand_dims(exponent, -1)  # shape: (...,1,n,1)
    return jnp.abs(c) * jnp.exp(-abs(e)*(r - l_k)**2)


@jax.jit
def rbf(r, coefficient, exponent, l_k):
    """
    Discretized atomic basis function, which is a linear superposition of n_b radial basis functions with different
    coefficients and exponents.

    Args:
        r (array_like): Distances, shape: (...,n,n,1)
        coefficient (array_like): Coefficients, shape: (...,1,n,n_abf)
        exponent (array_like): Exponents, shape: (...,1,n,n_b)
        l_k (array_like): Discretization points, shape: (L)

    Returns: Discretized atomic basis function.

    """
    return jax.vmap(rbf_, in_axes=(None, -1, -1, None), out_axes=-1)(r, coefficient, exponent, l_k).sum(-1)


def sto_(r, coefficient, exponent, l_k):
    """
    Helper RBF function which is usually called to vmap over.
    Args:
        r (): molecule_inspect, shape: (...,n,n,1)
        coefficient (): shape: (...,1,n)
        exponent (): shape: (...,1,n)
        l_k (): shape: (L)

    Returns:

    """
    c = jnp.expand_dims(coefficient, -1)  # shape: (...,1,n,1)
    e = jnp.expand_dims(exponent, -1)  # shape: (...,1,n,1)
    return jnp.abs(c) * jnp.exp(-abs(e)*abs(r - l_k))


@jax.jit
def sto(r, coefficient, exponent, l_k):
    """
    Discretized atomic basis function, which is a linear superposition of n_b radial basis functions with different
    coefficients and exponents.

    Args:
        r (array_like): Distances, shape: (...,n,n,1)
        coefficient (array_like): Coefficients, shape: (...,1,n,n_abf)
        exponent (array_like): Exponents, shape: (...,1,n,n_b)
        l_k (array_like): Discretization points, shape: (L)

    Returns: Discretized atomic basis function.

    """
    return jax.vmap(sto_, in_axes=(None, -1, -1, None), out_axes=-1)(r, coefficient, exponent, l_k).sum(-1)


def cos_(r, coefficient, exponent, l_k):
    """
    Helper cos function which is usually called to vmap over.
    Args:
        r (): molecule_inspect, shape: (...,n,n,1)
        coefficient (): shape: (...,1,n)
        exponent (): shape: (...,1,n)
        l_k (): shape: (L)

    Returns:

    """
    c = jnp.expand_dims(coefficient, -1)  # shape: (...,1,n,1)
    e = jnp.expand_dims(exponent, -1)  # shape: (...,1,n,1)
    return jnp.abs(c) * jnp.cos(10*e * jnp.abs(r - l_k))


@jax.jit
def cos(r, coefficient, exponent, l_k):
    """
    Discretized atomic basis function, which is a linear superposition of n_b radial basis functions with different
    coefficients and exponents.

    Args:
        r (array_like): Distances, shape: (...,n,n,1)
        coefficient (array_like): Coefficients, shape: (...,1,n,n_abf)
        exponent (array_like): Exponents, shape: (...,1,n,n_b)
        l_k (array_like): Discretization points, shape: (L)

    Returns: Discretized atomic basis function.

    """
    return jax.vmap(cos_, in_axes=(None, -1, -1, None), out_axes=-1)(r, coefficient, exponent, l_k).sum(-1)


@jax.jit
def rbf_normalized(r, alpha, gamma, l_k):
    """
    Discretized atomic basis function, which is a linear superposition of n_b radial basis functions with different
    coefficients and exponents. This version of normalized by sqrt(n_b).

    Args:
        r (array_like): Distances, shape: (...,1)
        alpha (array_like): Coefficients, shape: (n_b)
        gamma (array_like): Exponents, shape: (n_b)
        l_k (array_like): Discretization points, shape: (L)

    Returns: Discretized atomic basis function.

    """
    return rbf(r=r, alpha=alpha, gamma=gamma, l_k=l_k) / jnp.sqrt(alpha.shape[-1])


@jax.jit
def rbf_grid(r, alpha, gamma, l_k):
    # r.shape = [...,1]
    # alpha.shape = [D]
    # gamma.shape = [D]
    # l_k.shape = [D]

    return jnp.exp(-abs(gamma) * (r - l_k) ** 2)
    # return.shape = [...,D]
