# TODO: Does create_A work properly for J and R padded?

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import erf

from typing import (Dict, Any)

from src.nn.base.sub_module import BaseSubModule
from src.nn.activation_function.activation_function import shifted_softplus
from src.nn.activation_function.activation_function import get_activation_fn
from src.geometric.metric import coordinates_to_distance_matrix
from src.masking.mask import safe_mask
from src.padding import pad_ceq
from src.solver import get_solver
import logging
from functools import partial


class CeqLayer(BaseSubModule):
    solver: str  # name of solver
    activation_name: str = "shifted_softplus"  # name of activation function
    module_name: str = 'charge_equilibrium_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus

    @nn.compact
    def __call__(self,
                 R: jnp.ndarray,
                 chi: jnp.ndarray,
                 J: jnp.ndarray,
                 sigma: jnp.ndarray,
                 Q_tot: float,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [1]:
        Ko, Tsz Wai, et al. "A fourth-generation high-dimensional neural network potential with accurate electrostatics
        including non-local charge transfer." Nature communications 12.1 (2021): 1-11.

        Args:
            R (Array): vector of atom positions
            chi (Array): vector of electronegativities
            J (Array): vector of hardnesses
            sigma (Array): vector of charge density widths of covalent
            Q_tot (float): total charge of the molecule
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        A = self.create_A(R, J, sigma)  # shape: (n_atoms,n_atoms)
        A_ = pad_ceq(A)  # shape: (n_atoms+1,n_atoms+1)
        chi_ = jnp.pad(-chi, ((0, 1)), mode='constant', constant_values=((0, Q_tot)))
        params = A_
        x = chi_
        q = self.fixed_point_layer(self.ceq_fp_function, params, x)
        return {'q': q}

    def create_A(self, R, J, sigma):

        """
        takes molecule_inspect, hardnesses and density widths to create the matrix A as in [1] used for charge equilibration
        :return: A
        """
        r_ij = coordinates_to_distance_matrix(R)  # shape: (n_atoms,n_atoms)
        gamma_ij_ = jnp.sqrt(jnp.expand_dims(sigma, axis=0)**2 + jnp.expand_dims(sigma, axis=1)**2)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_mask(gamma_ij_ > 0., jnp.sqrt, gamma_ij_)  # shape: (n_atoms,n_atoms)
        A_ = safe_mask(gamma_ij != 0, lambda x: x/jnp.sqrt(2) * gamma_ij, r_ij)  # shape: (n_atoms,n_atoms)
        A_ = erf(A_)  # shape: (n_atoms,n_atoms)
        A_ = safe_mask(r_ij != 0, lambda x: x/r_ij, A_)  # shape: (n_atoms,n_atoms)
        A__ = jnp.diag(J) + 1 / (jnp.diag(sigma) * jnp.pi)  # shape: (n_atoms,n_atoms)
        A = A_ + A__  # shape: (n_atoms,n_atoms)
        A = pad_ceq(A)  # shape: (n_atoms,n_atoms)
        return A_ + A__

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1))
    def fixed_point_layer(self, f, params, x):
        z_star = get_solver(self.solver, lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
        return z_star

    def fixed_point_layer_fwd(self, f, params, x):
        z_star = self.fixed_point_layer(f, params, x)
        return z_star, (params, x, z_star)

    def fixed_point_layer_bwd(self, f, res, z_star_bar):
        params, x, z_star = res
        _, vjp_a = jax.vjp(lambda params, x: f(params, x, z_star), params, x)
        _, vjp_z = jax.vjp(lambda z: f(params, x, z), z_star)
        return vjp_a(get_solver(self.solver, lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(z_star)))

    def ceq_fp_function(self, params, x, z):
        """
        Function of which the charge equilibrium returns a fixed point
        :param params:
        :param x:
        :param z:
        :return:
        """
        A_ = params
        chi_ = x
        return jnp.dot(A_, z) + z - chi_

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'activation_name': self.activation_name,
                                   }
                }
