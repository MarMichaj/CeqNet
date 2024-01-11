from typing import (Any)
Array = Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.scipy.special import erf
import logging
from functools import partial
from typing import (Dict, Any, Sequence)

from ceqnet.src.masking.mask import safe_scale
from ceqnet.src.nn.base.sub_module import BaseSubModule
from ceqnet.src.nn.activation_function.activation_function import shifted_softplus
from ceqnet.src.nn.activation_function.activation_function import get_activation_fn
from ceqnet.src.geometric.metric import coordinates_to_distance_matrix
from ceqnet.src.masking.mask import safe_mask
from ceqnet.src.padding import pad_ceq
from ceqnet.src.solver import get_solver
from ceqnet.src.nn.mlp import MLP
from ceqnet.src.nn.activation_function.activation_function import silu



class _CeqPartialCharge(BaseSubModule):
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    activation_name: str = "shifted_softplus"  # name of activation function
    module_name: str = 'charge_equilibrium_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [1]:
        Ko, Tsz Wai, et al. "A fourth-generation high-dimensional neural network potential with accurate electrostatics
        including non-local charge transfer." Nature communications 12.1 (2021): 1-11.

        Args: dict 'inputs' containing:
            R (Array): vector of atom positions, shape: (n_atoms,3)
            chi (Array): vector of electronegativities features, shape: (n_atoms,F)
            J (Array): vector of hardnesses, shape: (n_atoms,1)
            sigma (Array): vector of charge density widths of covalent, shape: (n_atoms,1)
            Q_tot (float): total charge of the molecule, shape: (1)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        R = inputs['R']
        chi = inputs['x']
        J = inputs['J']
        sigma = inputs['sigma']
        Q_tot = inputs['Q']
        point_mask = inputs['point_mask']

        A_ = self.create_A(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi = MLP(features=[chi.shape[-1], 1], use_bias=False, activation_fn=silu)(chi).squeeze(axis=-1)  # shape: (n)
        chi_ = jnp.pad(-chi, ((0, 1)), mode='constant',
                       constant_values=((0, jnp.squeeze(Q_tot))))  # shape: (n)
        params = A_
        x = chi_
        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(A_,chi_)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, params, x)
        q = q[:-1]
        q = safe_scale(q, point_mask)
        return {'q': q}

    def  create_A(self, R, J, sigma, point_mask):

        """
        takes molecule_inspect, hardnesses and density widths to create the matrix A as in [1] used for charge equilibration
        :return: A
        """
        r_ij = jnp.squeeze(coordinates_to_distance_matrix(R))  # shape: (n_atoms,n_atoms)
        gamma_ij = jnp.sqrt(jnp.expand_dims(sigma, axis=0)**2 + jnp.expand_dims(sigma, axis=1)**2)  # shape: (n_atoms,n_atoms)
        A_ = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1/(x*jnp.sqrt(2)), operand=gamma_ij)  # shape: (n_atoms,n_atoms)
        A_ = erf(A_ * r_ij)  # shape: (n_atoms,n_atoms)
        A_ = safe_mask(mask=(r_ij != 0), fn=lambda x: x/r_ij, operand=A_, intermediate_placeholder=1)  # shape: (n_atoms,n_atoms)
        # besides on the diagonal, A_ now contains the correct entries of the matrix A
        J = safe_scale(J, point_mask)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x:1/x, operand=sigma)
        A__ = jnp.diag(J) + jnp.diag(sigma_inv) /jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)
        A = A_ + A__  # shape: (n_atoms,n_atoms)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis =1)
        A = safe_scale(A, pair_mask, 0)
        A = pad_ceq(A, point_mask)  # shape: (n_atoms,n_atoms)

        # we need to regularize the Matrix A, so we just add ones on the diagonal in all zero lines
        point_mask_inv = jnp.ones(len(point_mask)) - point_mask
        point_mask_inv = jnp.pad(point_mask_inv, ((0,1)), mode='constant', constant_values=((0,0)))
        A = A + jnp.diag(point_mask_inv)

        return A

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

    fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

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
        return jnp.dot(A_, jnp.squeeze(z)) + z - chi_

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'activation_name': self.activation_name,
                                   'solver': self.solver,
                                   'prop_keys': self.prop_keys
                                   }
                }
