from typing import (Any)

Array = Any
import jax
import jax.numpy as jnp
import numpy as np
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


def get_observable_module(name, h):
    if name == 'charge_equilibrium_layer':
        return CeqPartialCharge(**h)
    elif name == 'partial_charge_trivial':
        return PartialChargeTrivial(**h)
    elif name == 'redist_charge_layer':
        return RedistPartialCharge(**h)
    elif name == 'energy':
        return Energy(**h)
    elif name == 'trivial_partial_charge_dipole':
        return TrivialPartialChargeDipole(**h)
    elif name == 'ceq_dipole_layer':
        return CeqDipole(**h)
    elif name == 'redist_dipole_layer':
        return RedistDipole(**h)
    elif name == 'E_elec_E_loc':
        return E_elec_E_loc(**h)
    elif name == 'E_elec':
        return E_elec(**h)
    elif name == 'quadrupole_layer':
        return Quadrupole(**h)
    elif name == 'mixed_dipole_redis_layer':
        return MixedDipoleRedis(**h)
    elif name == 'mixed_dipole_ceq_layer':
        return MixedDipoleCeq(**h)
    elif name == 'atomic_dipole_layer':
        return AtomicDipole(**h)
    elif name == 'atomic_quadrupole_layer':
        return AtomicQuadrupole(**h)
    elif name == 'ceq_quadrupole_layer':
        return CeqQuadrupole(**h)
    elif name == 'redis_quadrupole_layer':
        return RedisQuadrupole(**h)
    elif name == 'ceq_atomic_quadrupole_layer':
        return CeqAtomicQuadrupole(**h)
    elif name == 'redis_atomic_quadrupole_layer':
        return RedisAtomicQuadrupole(**h)
    elif name == 'trivial_dipole_layer_zi':
        return TrivialDipole_zi(**h)
    elif name == 'redis_dipole_layer_zi':
        return RedisDipole_zi(**h)
    elif name == 'trivial_dipole_layer_zi_w':
        return TrivialDipole_zi_w(**h)
    elif name == 'redis_dipole_layer_zi_w':
        return RedisDipole_zi_w(**h)
    elif name == 'redis_atomic_quadrupole_layer_zi_w':
        return RedisAtomicQuadrupole_zi_w(**h)
    elif name == 'mixed_dipole_redis_layer_zi_w':
        return MixedDipoleRedis_zi_w(**h)
    elif name == 'redis_quadrupole_layer_zi_w':
        return RedisQuadrupole_zi_w(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)


class RedistPartialCharge(BaseSubModule):
    module_name: str = 'redist_charge_layer'  # name of module

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        q = inputs['x']  # shape: (n,F)
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False, activation_fn=silu)(q).squeeze(axis=-1)  # shape: (n)
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        return {'q': q}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
        }
        }


class CeqPartialCharge(BaseSubModule):
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    activation_name: str = None  # name of activation function
    module_name: str = 'charge_equilibrium_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function " + str(self.activation_name) + " not known.")
            self.activation_fn = silu
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [42]:
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

        H = self.create_H(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi = MLP(features=[chi.shape[-1], 1], use_bias=False, activation_fn=silu)(chi).squeeze(axis=-1)  # shape: (n)
        chi = safe_scale(chi, point_mask)
        chi_neg = jnp.pad(-chi, (0, 1), mode='constant', constant_values=(0, jnp.squeeze(Q_tot)))  # shape: (n+1)

        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(H, chi_neg)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, H, chi_neg)
        # slicing off the Lagrange multiplier
        q = q[:-1]  # shape: (n)
        q = safe_scale(q, point_mask)
        return {'q': q, 'chi_': chi}

    def create_H(self, R, J, sigma, point_mask):

        """
        Creates the matrix in [42] equation (6) (we call it H).
        Returns: H
        """
        # defining the non-diagonal entries of A in the matrix A_ndiag (with 0's on diagonal)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_scale(jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1 / (x * jnp.sqrt(2)), operand=gamma_ij, placeholder=0)
        # shape: (n_atoms,n_atoms)
        A_ndiag = erf(A_ndiag * r_ij)  # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(r_ij != 0), fn=lambda x: x / r_ij, operand=A_ndiag, placeholder=0)
        # shape: (n_atoms,n_atoms)

        # defining the diagonal entries of A in the matrix A_diag (with 0's off diagonal)
        J = safe_scale(J, point_mask)  # shape: (n_atoms)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x: 1 / x, operand=sigma)  # shape: (n_atoms)
        A_diag = jnp.diag(J) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)

        # defining A_padded, the matrix A from [42] padded with zeros to get the shape (len(point_mask),len(point_mask))
        # = (n,n)
        A_padded = A_diag + A_ndiag  # shape: (n_atoms,n_atoms)
        A_padded = safe_scale(A_padded, pair_mask, 0)

        # creating the matrix H, a padded analogous to the quadratic matrix in eq.(6) in [42]
        H = pad_ceq(A_padded, point_mask)  # shape: (n_atoms,n_atoms)
        return H

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


class PartialChargeTrivial(BaseSubModule):
    prop_keys: Dict
    module_name: str = 'partial_charge_trivial'

    def setup(self):
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `z` and the total charge of the system `Q`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                Q (Array): Total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs (): Q - Total charge, shape: (1)

        Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        partial_charges = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)

        return {self.partial_charge_key: safe_scale(partial_charges, scale=point_mask)}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys}}


class Energy(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence[float]
    per_atom_shift: Sequence[float]
    num_embeddings: int = 100
    module_name: str = 'energy'

    def setup(self):
        self.energy_key = self.prop_keys.get('energy')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_scale), y)
        else:
            self.get_per_atom_scale = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_shift), y)
        else:
            self.get_per_atom_shift = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        """

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']
        point_mask = inputs['point_mask']
        z = inputs[self.atomic_type_key]

        e_loc = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n)
        e_loc = safe_scale(e_loc, scale=point_mask)  # shape: (n)
        return {self.energy_key: e_loc.sum(axis=-1)}  # shape: (1)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}
                }


class E_elec_E_loc(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence[float]
    per_atom_shift: Sequence[float]
    num_embeddings: int = 100
    module_name: str = 'E_elec_E_loc'

    def setup(self):
        self.energy_key = self.prop_keys.get('energy')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_scale), y)
        else:
            self.get_per_atom_scale = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_shift), y)
        else:
            self.get_per_atom_shift = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        """
        Computing electrostatic energy from partial charges and local energy from partial charges and summing them up
        like in [39], eq. (5), E[rho] - E_0.

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # shape: (n)
        q = inputs[self.prop_keys['partial_charge']]  # shape: (n)
        point_mask = inputs['point_mask']  # shape: (n)
        chi_ = inputs['chi_']  # shape: (n)
        J = inputs['J']  # shape: (n)
        R = inputs['R']
        sigma = inputs['sigma']  # shape: (n)
        z = inputs[self.atomic_type_key]  # shape: (n)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        gamma_ij = safe_scale(1 / jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        qi_qj = safe_scale(jnp.expand_dims(q, axis=0) * jnp.expand_dims(q, axis=1), scale=pair_mask)

        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)

        e_loc = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n)
        e_loc = safe_scale(e_loc, scale=point_mask).sum(axis=-1)  # shape: (1)

        e_elec = jnp.dot(q, chi_) + \
                 0.5 * jnp.dot(safe_scale(J + 2 * jnp.diag(gamma_ij) / jnp.sqrt(jnp.pi), scale=point_mask), q ** 2) + \
                 0.5 * safe_mask(operand=qi_qj * erf(r_ij), fn=lambda x: 1 / r_ij, mask=(r_ij != 0)).sum()

        return {self.energy_key: e_loc + e_elec}  # shape: (1)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}
                }


class E_elec(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence[float]
    per_atom_shift: Sequence[float]
    num_embeddings: int = 100
    module_name: str = 'E_elec'

    def setup(self):
        self.energy_key = self.prop_keys.get('energy')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_scale), y)
        else:
            self.get_per_atom_scale = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_shift), y)
        else:
            self.get_per_atom_shift = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        """
        Computing electrostatic energy from partial charges and local energy from partial charges and summing them up
        like in [39], eq. (5), E[rho] - E_0.

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # shape: (n)
        q = inputs[self.prop_keys['partial_charge']]  # shape: (n)
        point_mask = inputs['point_mask']  # shape: (n)
        chi_ = inputs['chi_']  # shape: (n)
        J = inputs['J']  # shape: (n)
        R = inputs['R']
        sigma = inputs['sigma']  # shape: (n)
        z = inputs[self.atomic_type_key]  # shape: (n)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        gamma_ij = safe_scale(1 / jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        qi_qj = safe_scale(jnp.expand_dims(q, axis=0) * jnp.expand_dims(q, axis=1), scale=pair_mask)

        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)

        e_elec = jnp.dot(q, chi_) + \
                 0.5 * jnp.dot(safe_scale(J + 2 * jnp.diag(gamma_ij) / jnp.sqrt(jnp.pi), scale=point_mask), q ** 2) + \
                 0.5 * safe_mask(operand=qi_qj * erf(r_ij), fn=lambda x: 1 / r_ij, mask=(r_ij != 0)).sum()

        return {self.energy_key: e_elec}  # shape: (1)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}
                }


class RedistDipole(BaseSubModule):
    module_name: str = 'redist_dipole_layer'  # name of module

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        q = inputs['x']  # shape: (n,F)
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False, activation_fn=silu)(q).squeeze(axis=-1)  # shape: (n)
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        return {'q': q, 'mu': mu}
        # return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
        }
    }


class TrivialPartialChargeDipole(BaseSubModule):
    prop_keys: Dict
    module_name: str = 'trivial_partial_charge_dipole'

    def setup(self):
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `z` and the total charge of the system `Q`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                Q (Array): Total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs (): Q - Total charge, shape: (1)

        Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        q = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        # return {self.partial_charge_key: safe_scale(q, scale=point_mask), 'mu': mu}
        return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys}}


class CeqDipole(BaseSubModule):
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    activation_name: str = None  # name of activation function
    module_name: str = 'ceq_dipole_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function " + str(self.activation_name) + " not known.")
            self.activation_fn = silu
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [42]:
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

        H = self.create_H(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi = MLP(features=[chi.shape[-1], 1], use_bias=False, activation_fn=silu)(chi).squeeze(axis=-1)  # shape: (n)
        chi = safe_scale(chi, point_mask)
        chi_neg = jnp.pad(-chi, (0, 1), mode='constant', constant_values=(0, jnp.squeeze(Q_tot)))  # shape: (n+1)

        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(H, chi_neg)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, H, chi_neg)
        # slicing off the Lagrange multiplier
        q = q[:-1]  # shape: (n)
        q = safe_scale(q, point_mask)
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        return {'q': q, 'mu': mu}
        # return {'mu': mu}

    def create_H(self, R, J, sigma, point_mask):

        """
        Creates the matrix in [42] equation (6) (we call it H).
        Returns: H
        """
        # defining the non-diagonal entries of A in the matrix A_ndiag (with 0's on diagonal)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_scale(jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1 / (x * jnp.sqrt(2)), operand=gamma_ij, placeholder=0)
        # shape: (n_atoms,n_atoms)
        A_ndiag = erf(A_ndiag * r_ij)  # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(r_ij != 0), fn=lambda x: x / r_ij, operand=A_ndiag, placeholder=0)
        # shape: (n_atoms,n_atoms)

        # defining the diagonal entries of A in the matrix A_diag (with 0's off diagonal)
        J = safe_scale(J, point_mask)  # shape: (n_atoms)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x: 1 / x, operand=sigma)  # shape: (n_atoms)
        A_diag = jnp.diag(J) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)

        # defining A_padded, the matrix A from [42] padded with zeros to get the shape (len(point_mask),len(point_mask))
        # = (n,n)
        A_padded = A_diag + A_ndiag  # shape: (n_atoms,n_atoms)
        A_padded = safe_scale(A_padded, pair_mask, 0)

        # creating the matrix H, a padded analogous to the quadratic matrix in eq.(6) in [42]
        H = pad_ceq(A_padded, point_mask)  # shape: (n_atoms,n_atoms)
        return H

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


class Quadrupole(BaseSubModule):
    module_name: str = 'quadrupole_layer'  # name of module

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """
        q = inputs['q']  # shape: (n,F)
        point_mask = inputs['point_mask']
        q = safe_scale(q, point_mask)
        R_quad = jax.vmap(lambda x: jnp.outer(x, x), in_axes=0, out_axes=0)(inputs['R'])  # shape:(n,3,3)
        R_quad = safe_scale(R_quad, point_mask[:, None, None])  # shape: (n,3,3)
        dyad_len = jax.vmap(lambda x: jnp.inner(x, x) * jnp.eye(3), in_axes=0, out_axes=0)(
            inputs['R'])  # shape: (n,3,3)
        dyad_len = safe_scale(dyad_len, point_mask[:, None, None])  # shape: (n,3,3)
        quad = jnp.sum((R_quad - 1 / 3 * dyad_len) * q[:, None, None], axis=0)  # shape: (3,3)
        qu_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad = quad[qu_ind]  # shape: (6)

        quad_ntl = jnp.sum(R_quad * q[:, None, None], axis=0)
        quad_ntl = quad_ntl[qu_ind]  # shape: (6)
        return {'quad': quad, 'quad_ntl': quad_ntl} # quad_ntl is the non-traceless quadrupole as (3,3)-matrix

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
        }
        }


class AtomicDipole(BaseSubModule):
    degrees: Sequence[int]
    module_name: str = 'atomic_dipole_layer'  # name of module

    def setup(self) -> None:
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """
        point_mask = inputs['point_mask']
        x = inputs['x']  # shape: (n,F)

        R = inputs['R']  # shape: (n,3)
        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])
        amu = atomic_dipole.sum(axis=-2)  # shape: (3)

        # contribution of atomic dipole to quadrupole, ntl
        mu_quad_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer,
                                                                                           in_axes=(-2, -2))(R,
                                                                                                             atomic_dipole)  # shape: (n, 3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        mu_quad_contr = safe_scale(mu_quad_contr, scale=point_mask[:, None, None])
        mu_quad_contr = mu_quad_contr.sum(axis=-3)  # shape: (3,3)
        mu_quad_ntl = mu_quad_contr[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        q_quad_tl = detrace(mu_quad_ntl)  # shape: (6)


        return {'mu': amu, 'quad': q_quad_tl}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees
        }
        }


class MixedDipoleRedis(BaseSubModule):
    alpha: float  # float between 0 and 1, determines how much of the dipole is computed from the atomic dipole
    degrees: Sequence[int]  # degrees of the spherical harmonics
    alpha_bool: bool = True # if True, use alpha for weighting of atomic and redis dipole, else just add them
    module_name: str = 'mixed_dipole_redis_layer'  # name of module

    def setup(self) -> None:
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """
        point_mask = inputs['point_mask']
        R = inputs['R']  # shape: (n,3)
        x = inputs['x']  # shape: (n,F)

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])
        amu = atomic_dipole.sum(axis=-2)  # shape: (3)

        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[x.shape[-1], 1], use_bias=False, activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res
        q = safe_scale(q, point_mask)
        mu_c = jnp.sum(q[:, None] * R, axis=0)  # shape: (3)
        mu = None
        if self.alpha_bool:
            mu = self.alpha * mu_c + (1 - self.alpha) * amu  # shape: (3)
        else:
            mu = mu_c + amu

        # create quadrupole from charges, ntl
        q_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        q_quad = q_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        q_quad_ntl = q_quad[quad_ind]  # shape: (6), non-traceless form

        # contribution of atomic dipole to quadrupole, ntl
        mu_quad_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer,
                                                                                               in_axes=(-2, -2))(R,
                                                                                                                 atomic_dipole)  # shape: (n, 3,3)
        mu_quad_contr = safe_scale(mu_quad_contr, scale=point_mask[:, None, None])
        mu_quad_contr = mu_quad_contr.sum(axis=-3)  # shape: (3,3)
        mu_quad_ntl = mu_quad_contr[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        if self.alpha_bool:
            q_quad_tl = detrace(self.alpha * q_quad_ntl + (1-self.alpha) * mu_quad_ntl)  # shape: (6)
        else:
            q_quad_tl = detrace(q_quad_ntl + mu_quad_ntl)

        return {'q': q, 'mu': mu, 'amu': amu, 'quad': q_quad_tl}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'alpha': self.alpha,
                                   'degrees': self.degrees
                                   }
                }


class MixedDipoleCeq(BaseSubModule):
    alpha: float  # float between 0 and 1, determines how much of the dipole is computed from the atomic dipole
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    degrees: Sequence[int]  # degrees of spherical harmonics to be used
    activation_name: str = None  # name of activation function
    alpha_bool: bool = True  # if True, use alpha for weighting of atomic and redis dipole, else just add them
    module_name: str = 'mixed_dipole_ceq_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function " + str(self.activation_name) + " not known.")
            self.activation_fn = silu
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [42]:
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
        point_mask = inputs['point_mask']
        R = inputs['R']  # shape: (n,3)
        x = inputs['x']  # shape: (n,F)

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])
        amu = atomic_dipole.sum(axis=-2)  # shape: (3)

        chi_qeq = inputs['x']
        J = inputs['J']
        sigma = inputs['sigma']
        Q_tot = inputs['Q']
        point_mask = inputs['point_mask']

        H = self.create_H(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi_qeq = MLP(features=[chi_qeq.shape[-1], 1], use_bias=False, activation_fn=silu)(chi_qeq).squeeze(
            axis=-1)  # shape: (n)
        chi_qeq = safe_scale(chi_qeq, point_mask)
        chi_qeq_neg = jnp.pad(-chi_qeq, (0, 1), mode='constant',
                              constant_values=(0, jnp.squeeze(Q_tot)))  # shape: (n+1)

        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(H, chi_qeq_neg)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, H, chi_qeq_neg)
        # slicing off the Lagrange multiplier
        q = q[:-1]  # shape: (n)
        q = safe_scale(q, point_mask)

        mu_c = jnp.sum(q[:, None] * inputs['R'], axis=0)  # shape: (3)
        mu = None
        if self.alpha_bool:
            mu = self.alpha * mu_c + (1 - self.alpha) * amu  # shape: (3)
        else:
            mu = mu_c + amu  # shape: (3)

        # create quadrupole from charges, ntl
        q_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        q_quad = q_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        q_quad_ntl = q_quad[quad_ind]  # shape: (6), non-traceless form

        # contribution of atomic dipole to quadrupole, ntl
        mu_quad_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer,
                                                                                           in_axes=(-2, -2))(R,
                                                                                                             atomic_dipole)  # shape: (n, 3,3)
        mu_quad_contr = safe_scale(mu_quad_contr, scale=point_mask[:, None, None])
        mu_quad_contr = mu_quad_contr.sum(axis=-3)  # shape: (3,3)
        mu_quad_ntl = mu_quad_contr[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        if self.alpha_bool:
            q_quad_tl = detrace(self.alpha * q_quad_ntl + (1 - self.alpha) * mu_quad_ntl)  # shape: (6)
        else:
            q_quad_tl = detrace(q_quad_ntl + mu_quad_ntl)

        return {'q': q, 'mu': mu, 'amu': amu, 'quad': q_quad_tl}

    def create_H(self, R, J, sigma, point_mask):

        """
        Creates the matrix in [42] equation (6) (we call it H).
        Returns: H
        """
        # defining the non-diagonal entries of A in the matrix A_ndiag (with 0's on diagonal)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)),
                          scale=pair_mask)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_scale(jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1 / (x * jnp.sqrt(2)), operand=gamma_ij,
                            placeholder=0)
        # shape: (n_atoms,n_atoms)
        A_ndiag = erf(A_ndiag * r_ij)  # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(r_ij != 0), fn=lambda x: x / r_ij, operand=A_ndiag, placeholder=0)
        # shape: (n_atoms,n_atoms)

        # defining the diagonal entries of A in the matrix A_diag (with 0's off diagonal)
        J = safe_scale(J, point_mask)  # shape: (n_atoms)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x: 1 / x, operand=sigma)  # shape: (n_atoms)
        A_diag = jnp.diag(J) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)

        # defining A_padded, the matrix A from [42] padded with zeros to get the shape (len(point_mask),len(point_mask))
        # = (n,n)
        A_padded = A_diag + A_ndiag  # shape: (n_atoms,n_atoms)
        A_padded = safe_scale(A_padded, pair_mask, 0)

        # creating the matrix H, a padded analogous to the quadratic matrix in eq.(6) in [42]
        H = pad_ceq(A_padded, point_mask)  # shape: (n_atoms,n_atoms)
        return H

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
        return {self.module_name: {'alpha': self.alpha,
                                   'activation_name': self.activation_name,
                                   'solver': self.solver,
                                   'prop_keys': self.prop_keys,
                                   'degrees': self.degrees
                                   }
                }


class AtomicQuadrupole(BaseSubModule):
    degrees: Sequence[int]
    module_name: str = 'atomic_quadrupole_layer'  # name of module

    def setup(self) -> None:
        if (0 in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([4, 5, 6, 7, 8])
        elif (0 not in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([3, 4, 5, 6, 7])
        else:
            self.l2_idxs = jnp.array([0, 1, 2, 3, 4])
        self.Q = jnp.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [1, -1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0]], dtype=jnp.float32)  # shape: (3,3)
        self.D = jnp.diag(jnp.array([2*np.sqrt(np.pi/15), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/5), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/15), 1]))
        self.M = jnp.linalg.inv(self.Q) @ self.D
        # atomic dipole properties
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """

        point_mask = inputs['point_mask']
        x = inputs['x']  # shape: (n,F)
        R = inputs['R']

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l2 = chi[:, self.l2_idxs] # shape (n,5)
        chi_l2 = jnp.pad(chi_l2, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=((0, 0), (0, 1)))  # shape: (n,6)
        v2 = jnp.einsum('ij, ...j -> ...i', self.M, chi_l2)  # shape: (n,6)
        # v1 has cartesian order [x,y,z]
        a_v2 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_quad_ntl = a_v2 * v2  # shape: (n,6)

        # Computation of atomic dipoles
        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])

        # contribution of atomic dipole to quadrupole
        quad_dipole_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer, in_axes=(-2, -2))(R, atomic_dipole)# shape: (n, 3,3)
        quad_dipole_contr = safe_scale(quad_dipole_contr, scale=point_mask[:, None, None])
        quad_dipole_contr = quad_dipole_contr.sum(axis=-3)  # shape: (3,3)
        qu_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_dipole_ntl = quad_dipole_contr[qu_ind] # shape: (6), non-traceless form


        atomic_quad_ntl = safe_scale(atomic_quad_ntl, scale=point_mask[:, None])
        atomic_quad_ntl = atomic_quad_ntl.sum(axis=-2)  # shape: (6)
        atomic_quad_ntl = atomic_quad_ntl + quad_dipole_ntl  # shape: (6), traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        atomic_quad_tl = detrace(atomic_quad_ntl)  # shape: (6)
        atomic_dipole = atomic_dipole.sum(axis=-2)  # shape: (3)

        return {'quad': atomic_quad_tl, 'quad_ntl': atomic_quad_ntl, 'mu': atomic_dipole}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees
        }
        }


class CeqQuadrupole(BaseSubModule):
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    activation_name: str = None  # name of activation function
    module_name: str = 'ceq_quadrupole_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function " + str(self.activation_name) + " not known.")
            self.activation_fn = silu
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Implicit layer to compute partial charges from given electronegativities and hardnesses as in [42]:
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

        H = self.create_H(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi = MLP(features=[chi.shape[-1], 1], use_bias=False, activation_fn=silu)(chi).squeeze(axis=-1)  # shape: (n)
        chi = safe_scale(chi, point_mask)
        chi_neg = jnp.pad(-chi, (0, 1), mode='constant', constant_values=(0, jnp.squeeze(Q_tot)))  # shape: (n+1)

        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(H, chi_neg)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, H, chi_neg)
        # slicing off the Lagrange multiplier
        q = q[:-1]  # shape: (n)
        q = safe_scale(q, point_mask)
        quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        quad = quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_ntl = quad[quad_ind]  # shape: (6), non-traceless form
        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])
        quad_tl = detrace(quad_ntl)  # shape: (6)

        # determine dipole from partial charges
        dipole = jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'q': q, 'quad': quad_tl, 'quad_ntl': quad_ntl, 'mu': dipole}

    def create_H(self, R, J, sigma, point_mask):

        """
        Creates the matrix in [42] equation (6) (we call it H).
        Returns: H
        """
        # defining the non-diagonal entries of A in the matrix A_ndiag (with 0's on diagonal)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_scale(jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1 / (x * jnp.sqrt(2)), operand=gamma_ij, placeholder=0)
        # shape: (n_atoms,n_atoms)
        A_ndiag = erf(A_ndiag * r_ij)  # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(r_ij != 0), fn=lambda x: x / r_ij, operand=A_ndiag, placeholder=0)
        # shape: (n_atoms,n_atoms)

        # defining the diagonal entries of A in the matrix A_diag (with 0's off diagonal)
        J = safe_scale(J, point_mask)  # shape: (n_atoms)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x: 1 / x, operand=sigma)  # shape: (n_atoms)
        A_diag = jnp.diag(J) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)

        # defining A_padded, the matrix A from [42] padded with zeros to get the shape (len(point_mask),len(point_mask))
        # = (n,n)
        A_padded = A_diag + A_ndiag  # shape: (n_atoms,n_atoms)
        A_padded = safe_scale(A_padded, pair_mask, 0)

        # creating the matrix H, a padded analogous to the quadratic matrix in eq.(6) in [42]
        H = pad_ceq(A_padded, point_mask)  # shape: (n_atoms,n_atoms)
        return H

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


class RedisQuadrupole(BaseSubModule):
    module_name: str = 'redis_quadrupole_layer'  # name of module

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        q = inputs['x']  # shape: (n,F)
        R = inputs['R']
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False, activation_fn=silu)(q).squeeze(axis=-1)  # shape: (n)
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        quad = quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_ntl = quad[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        quad_tl = detrace(quad_ntl)  # shape: (6)

        # determine dipole from partial charges
        dipole = jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'q': q, 'quad': quad_tl, 'quad_ntl': quad_ntl, 'mu': dipole}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
        }
    }


class CeqAtomicQuadrupole(BaseSubModule):
    solver: str  # name of solver
    prop_keys: Dict  # Dict of prop_keys
    degrees: Sequence[int]
    activation_name: str = None  # name of activation function
    module_name: str = 'ceq_atomic_quadrupole_layer'  # name of module

    def setup(self) -> None:
        if (0 in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([4, 5, 6, 7, 8])
        elif (0 not in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([3, 4, 5, 6, 7])
        else:
            self.l2_idxs = jnp.array([0, 1, 2, 3, 4])
        self.Q = jnp.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [1, -1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0]], dtype=jnp.float32)  # shape: (3,3)
        self.D = jnp.diag(jnp.array([2*np.sqrt(np.pi/15), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/5), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/15), 1]))
        self.M = jnp.linalg.inv(self.Q) @ self.D
        # atomic dipole properties
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """

        point_mask = inputs['point_mask']
        x = inputs['x']  # shape: (n,F)
        R = inputs['R']

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l2 = chi[:, self.l2_idxs] # shape (n,5)
        chi_l2 = jnp.pad(chi_l2, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=((0, 0), (0, 1)))  # shape: (n,6)
        v2 = jnp.einsum('ij, ...j -> ...i', self.M, chi_l2)  # shape: (n,6)
        # v1 has cartesian order [x,y,z]
        a_v2 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_quad_ntl = a_v2 * v2  # shape: (n,6)

        # Computation of atomic dipoles
        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])

        # contribution of atomic dipole to quadrupole
        quad_dipole_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer, in_axes=(-2, -2))(R, atomic_dipole)# shape: (n, 3,3)
        quad_dipole_contr = safe_scale(quad_dipole_contr, scale=point_mask[:, None, None])
        quad_dipole_contr = quad_dipole_contr.sum(axis=-3)  # shape: (3,3)
        qu_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_dipole_ntl = quad_dipole_contr[qu_ind] # shape: (6), non-traceless form


        atomic_quad_ntl = safe_scale(atomic_quad_ntl, scale=point_mask[:, None])
        atomic_quad_ntl = atomic_quad_ntl.sum(axis=-2)  # shape: (6)
        atomic_quad_ntl = atomic_quad_ntl + quad_dipole_ntl  # shape: (6), traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        atomic_quad_tl = detrace(atomic_quad_ntl)  # shape: (6)
        atomic_dipole = atomic_dipole.sum(axis=-2)  # shape: (3)


        chi = inputs['x']
        J = inputs['J']
        sigma = inputs['sigma']
        Q_tot = inputs['Q']
        point_mask = inputs['point_mask']

        H = self.create_H(R, J, sigma, point_mask)  # shape: (n_atoms+1,n_atoms+1)
        chi = MLP(features=[chi.shape[-1], 1], use_bias=False, activation_fn=silu)(chi).squeeze(axis=-1)  # shape: (n)
        chi = safe_scale(chi, point_mask)
        chi_neg = jnp.pad(-chi, (0, 1), mode='constant', constant_values=(0, jnp.squeeze(Q_tot)))  # shape: (n+1)

        if self.solver == 'scipy.linalg.solve':
            q = jax.scipy.linalg.solve(H, chi_neg)
        else:
            q = self.fixed_point_layer(self.ceq_fp_function, H, chi_neg)
        # slicing off the Lagrange multiplier
        q = q[:-1]  # shape: (n)
        q = safe_scale(q, point_mask)
        ceq_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        ceq_quad = ceq_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        ceq_quad_ntl = ceq_quad[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        ceq_quad_tl = detrace(ceq_quad_ntl)  # shape: (6)

        # determine dipole as sum of atomic dipole and partial charge dipole
        dipole = atomic_dipole + jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'q': q, 'quad': atomic_quad_tl + ceq_quad_tl, 'quad_ntl': atomic_quad_ntl + ceq_quad_ntl, 'mu': dipole}

    def create_H(self, R, J, sigma, point_mask):

        """
        Creates the matrix in [42] equation (6) (we call it H).
        Returns: H
        """
        # defining the non-diagonal entries of A in the matrix A_ndiag (with 0's on diagonal)
        pair_mask = jnp.expand_dims(point_mask, axis=0) * jnp.expand_dims(point_mask, axis=1)
        r_ij = safe_scale(jnp.squeeze(coordinates_to_distance_matrix(R)), scale=pair_mask)  # shape: (n_atoms,n_atoms)
        gamma_ij = safe_scale(jnp.sqrt(jnp.expand_dims(sigma, axis=0) ** 2 + jnp.expand_dims(sigma, axis=1) ** 2),
                              scale=pair_mask)
        # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(gamma_ij != 0), fn=lambda x: 1 / (x * jnp.sqrt(2)), operand=gamma_ij, placeholder=0)
        # shape: (n_atoms,n_atoms)
        A_ndiag = erf(A_ndiag * r_ij)  # shape: (n_atoms,n_atoms)
        A_ndiag = safe_mask(mask=(r_ij != 0), fn=lambda x: x / r_ij, operand=A_ndiag, placeholder=0)
        # shape: (n_atoms,n_atoms)

        # defining the diagonal entries of A in the matrix A_diag (with 0's off diagonal)
        J = safe_scale(J, point_mask)  # shape: (n_atoms)
        sigma_inv = safe_mask(mask=point_mask, fn=lambda x: 1 / x, operand=sigma)  # shape: (n_atoms)
        A_diag = jnp.diag(J) + jnp.diag(sigma_inv) / jnp.sqrt(jnp.pi)  # shape: (n_atoms,n_atoms)

        # defining A_padded, the matrix A from [42] padded with zeros to get the shape (len(point_mask),len(point_mask))
        # = (n,n)
        A_padded = A_diag + A_ndiag  # shape: (n_atoms,n_atoms)
        A_padded = safe_scale(A_padded, pair_mask, 0)

        # creating the matrix H, a padded analogous to the quadratic matrix in eq.(6) in [42]
        H = pad_ceq(A_padded, point_mask)  # shape: (n_atoms,n_atoms)
        return H

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
                                   'prop_keys': self.prop_keys,
                                   'degrees': self.degrees
                                   }
                }


class RedisAtomicQuadrupole(BaseSubModule):
    degrees: Sequence[int]
    module_name: str = 'redis_atomic_quadrupole_layer'  # name of module

    def setup(self) -> None:
        if (0 in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([4, 5, 6, 7, 8])
        elif (0 not in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([3, 4, 5, 6, 7])
        else:
            self.l2_idxs = jnp.array([0, 1, 2, 3, 4])
        self.Q = jnp.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [1, -1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0]], dtype=jnp.float32)  # shape: (3,3)
        self.D = jnp.diag(jnp.array([2*np.sqrt(np.pi/15), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/5), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/15), 1]))
        self.M = jnp.linalg.inv(self.Q) @ self.D
        # atomic dipole properties
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """

        point_mask = inputs['point_mask']
        x = inputs['x']  # shape: (n,F)
        R = inputs['R']

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l2 = chi[:, self.l2_idxs] # shape (n,5)
        chi_l2 = jnp.pad(chi_l2, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=((0, 0), (0, 1)))  # shape: (n,6)
        v2 = jnp.einsum('ij, ...j -> ...i', self.M, chi_l2)  # shape: (n,6)
        # v1 has cartesian order [x,y,z]
        a_v2 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_quad_ntl = a_v2 * v2  # shape: (n,6)

        # Computation of atomic dipoles
        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])

        # contribution of atomic dipole to quadrupole
        quad_dipole_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer, in_axes=(-2, -2))(R, atomic_dipole)# shape: (n, 3,3)
        quad_dipole_contr = safe_scale(quad_dipole_contr, scale=point_mask[:, None, None])
        quad_dipole_contr = quad_dipole_contr.sum(axis=-3)  # shape: (3,3)
        qu_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_dipole_ntl = quad_dipole_contr[qu_ind] # shape: (6), non-traceless form


        atomic_quad_ntl = safe_scale(atomic_quad_ntl, scale=point_mask[:, None])
        atomic_quad_ntl = atomic_quad_ntl.sum(axis=-2)  # shape: (6)
        atomic_quad_ntl = atomic_quad_ntl + quad_dipole_ntl  # shape: (6), traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        atomic_quad_tl = detrace(atomic_quad_ntl)  # shape: (6)
        atomic_dipole = atomic_dipole.sum(axis=-2)  # shape: (3)

        # Redis contribution
        q = inputs['x']  # shape: (n,F)
        R = inputs['R']
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False, activation_fn=silu)(q).squeeze(axis=-1)  # shape: (n)
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        redis_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        redis_quad = redis_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        redis_quad_ntl = redis_quad[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        redis_quad_tl = detrace(redis_quad_ntl)  # shape: (6)

        # determine dipole as sum of atomic dipole and partial charge dipole
        dipole = atomic_dipole + jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'quad': atomic_quad_tl + redis_quad_tl, 'quad_ntl': atomic_quad_ntl + redis_quad_ntl, 'mu': dipole, 'q': q}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees
        }
        }


class RedisDipole_zi(BaseSubModule):
    num_embeddings: int
    module_name: str = 'redis_dipole_layer_zi'  # name of module


    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        q = inputs['x']  # shape: (n,F)
        z = inputs['z'] # shape: (n)
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False, activation_fn=silu)(q).squeeze(axis=-1)  # shape: (n)

        point_mask = inputs['point_mask']
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                          scale=point_mask[:, None]))  # shape: (n)

        q = q + q_zi

        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        return {'q': q, 'mu': mu}
        # return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
                                   'num_embeddings': self.num_embeddings
                                   }
                }


class TrivialDipole_zi(BaseSubModule):
    prop_keys: Dict
    num_embeddings: int
    module_name: str = 'trivial_dipole_layer_zi'

    def setup(self):
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `z` and the total charge of the system `Q`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                Q (Array): Total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs (): Q - Total charge, shape: (1)

        Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        z = inputs['z']
        q = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                          scale=point_mask[:, None]))  # shape: (n)
        q = q + q_zi
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        # return {self.partial_charge_key: safe_scale(q, scale=point_mask), 'mu': mu}
        return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys,
                                   'num_embeddings': self.num_embeddings
                                   }
                }

class RedisDipole_zi_w(BaseSubModule):
    num_embeddings: int
    module_name: str = 'redis_dipole_layer_zi_w'  # name of module


    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        q = inputs['x']  # shape: (n,F)
        z = inputs['z'] # shape: (n)
        Q_tot = inputs['Q']  # shape: (1)
        q = MLP(features=[q.shape[-1], 1], use_bias=False)(q).squeeze(axis=-1)  # shape: (n)

        point_mask = inputs['point_mask']
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                          scale=point_mask[:, None]))  # shape: (n)

        q = q + q_zi

        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        return {'q': q, 'mu': mu}
        # return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {
                                   'num_embeddings': self.num_embeddings
                                   }
                }


class TrivialDipole_zi_w(BaseSubModule):
    num_embeddings: int
    prop_keys: Dict
    num_embeddings: int
    module_name: str = 'trivial_dipole_layer_zi_w'

    def setup(self):
        self.partial_charge_key = self.prop_keys.get('partial_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Predict partial charges, from atom-wise features `x`, atomic types `z` and the total charge of the system `Q`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
                Q (Array): Total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs (): Q - Total charge, shape: (1)

        Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n)

        """
        point_mask = inputs['point_mask']
        x = inputs['x']
        z = inputs['z']
        q = MLP(features=[x.shape[-1], 1])(x).squeeze(axis=-1)  # shape: (n)
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                          scale=point_mask[:, None]))  # shape: (n)
        q = q + q_zi
        mu = jnp.sum(q[:, None] * inputs['R'], axis=0)
        # return {self.partial_charge_key: safe_scale(q, scale=point_mask), 'mu': mu}
        return {'mu': mu}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'prop_keys': self.prop_keys,
                                   'num_embeddings': self.num_embeddings
                                   }
                }


class MixedDipoleRedis_zi_w(BaseSubModule):
    num_embeddings: int
    alpha: float  # float between 0 and 1, determines how much of the dipole is computed from the atomic dipole
    degrees: Sequence[int]  # degrees of the spherical harmonics
    alpha_bool: bool = True # if True, use alpha for weighting of atomic and redis dipole, else just add them
    module_name: str = 'mixed_dipole_redis_layer_zi_w'  # name of module

    def setup(self) -> None:
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """
        point_mask = inputs['point_mask']
        R = inputs['R']  # shape: (n,3)
        x = inputs['x']  # shape: (n,F)
        z = inputs['z']  # shape: (n)

        chi = inputs['chi']  # shape: (n,m_tot)
        safe_scale(chi, scale=point_mask[:, None])
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        v1 = safe_scale(v1, scale=point_mask[:, None])
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)
        a_v1 = safe_scale(a_v1, scale=point_mask[:, None])
        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])
        amu = atomic_dipole.sum(axis=-2)  # shape: (3)

        Q_tot = inputs['Q']  # shape: (1)

        q = MLP(features=[x.shape[-1], 1], use_bias=False)(x).squeeze(axis=-1)  # shape: (n)

        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                                      scale=point_mask[:, None]))  # shape: (n)

        q = q + q_zi
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res
        q = safe_scale(q, point_mask)
        mu_c = jnp.sum(q[:, None] * R, axis=0)  # shape: (3)
        mu = None
        if self.alpha_bool:
            mu = self.alpha * mu_c + (1 - self.alpha) * amu  # shape: (3)
        else:
            mu = mu_c + amu

        # create quadrupole from charges, ntl
        q_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        q_quad = q_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        q_quad_ntl = q_quad[quad_ind]  # shape: (6), non-traceless form

        # contribution of atomic dipole to quadrupole, ntl
        mu_quad_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer,
                                                                                               in_axes=(-2, -2))(R,
                                                                                                                 atomic_dipole)  # shape: (n, 3,3)
        mu_quad_contr = safe_scale(mu_quad_contr, scale=point_mask[:, None, None])
        mu_quad_contr = mu_quad_contr.sum(axis=-3)  # shape: (3,3)
        mu_quad_ntl = mu_quad_contr[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        if self.alpha_bool:
            q_quad_tl = detrace(self.alpha * q_quad_ntl + (1-self.alpha) * mu_quad_ntl)  # shape: (6)
        else:
            q_quad_tl = detrace(q_quad_ntl + mu_quad_ntl)

        return {'q': q, 'mu': mu, 'amu': amu, 'quad': q_quad_tl}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'alpha': self.alpha,
                                   'degrees': self.degrees,
                                   'num_embeddings': self.num_embeddings
                                   }
                }

class RedisAtomicQuadrupole_zi_w(BaseSubModule):
    num_embeddings: int
    degrees: Sequence[int]
    module_name: str = 'redis_atomic_quadrupole_layer_zi_w'  # name of module

    def setup(self) -> None:
        if (0 in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([4, 5, 6, 7, 8])
        elif (0 not in self.degrees) and (1 in self.degrees):
            self.l2_idxs = jnp.array([3, 4, 5, 6, 7])
        else:
            self.l2_idxs = jnp.array([0, 1, 2, 3, 4])
        self.Q = jnp.array([[0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1],
              [0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [1, -1, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0]], dtype=jnp.float32)  # shape: (3,3)
        self.D = jnp.diag(jnp.array([2*np.sqrt(np.pi/15), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/5), 2*np.sqrt(np.pi/15), 4*np.sqrt(np.pi/15), 1]))
        self.M = jnp.linalg.inv(self.Q) @ self.D
        # atomic dipole properties
        self.l1_idxs = jnp.array([1, 2, 3]) if 0 in self.degrees else jnp.array([0, 1, 2])
        self.P1 = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)  # shape: (3,3)

        c1_1 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c10 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        c11 = 1 / jnp.sqrt(3 / (4 * jnp.pi))
        self.inv_coeffs_l1 = jnp.array([c1_1, c10, c11])

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes Quadrupole from (already computed) partial charges and atomic postions.

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:
            quad: Quadrupole
            quad2: Quadrupole, non-diag. entries scaled with sqrt(2)
        """

        point_mask = inputs['point_mask']
        x = inputs['x']  # shape: (n,F)
        R = inputs['R']

        chi = inputs['chi']  # shape: (n,m_tot)
        chi_norm = safe_mask(mask=point_mask[:, None],
                             fn=partial(jnp.linalg.norm, axis=-1, keepdims=True),
                             operand=chi,
                             placeholder=0)  # shape: (n,1)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)

        chi_l2 = chi[:, self.l2_idxs] # shape (n,5)
        chi_l2 = jnp.pad(chi_l2, pad_width=((0, 0), (0, 1)), mode='constant', constant_values=((0, 0), (0, 1)))  # shape: (n,6)
        v2 = jnp.einsum('ij, ...j -> ...i', self.M, chi_l2)  # shape: (n,6)
        # v1 has cartesian order [x,y,z]
        a_v2 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_quad_ntl = a_v2 * v2  # shape: (n,6)

        # Computation of atomic dipoles
        chi_l1 = chi[:, self.l1_idxs] * self.inv_coeffs_l1[None, :]
        v1 = jnp.einsum('ij, ...j -> ...i', self.P1, chi_l1)  # shape: (n,3)
        # v1 has cartesian order [x,y,z]
        a_v1 = MLP(features=[x.shape[-1], 1],
                   activation_fn=silu)(x)  # shape: (n,1)

        atomic_dipole = a_v1 * v1  # shape: (n,3)
        atomic_dipole = safe_scale(atomic_dipole, scale=point_mask[:, None])

        # contribution of atomic dipole to quadrupole
        quad_dipole_contr = jax.vmap(jnp.outer, in_axes=(-2, -2))(atomic_dipole, R) + jax.vmap(jnp.outer, in_axes=(-2, -2))(R, atomic_dipole)# shape: (n, 3,3)
        quad_dipole_contr = safe_scale(quad_dipole_contr, scale=point_mask[:, None, None])
        quad_dipole_contr = quad_dipole_contr.sum(axis=-3)  # shape: (3,3)
        qu_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_dipole_ntl = quad_dipole_contr[qu_ind] # shape: (6), non-traceless form


        atomic_quad_ntl = safe_scale(atomic_quad_ntl, scale=point_mask[:, None])
        atomic_quad_ntl = atomic_quad_ntl.sum(axis=-2)  # shape: (6)
        atomic_quad_ntl = atomic_quad_ntl + quad_dipole_ntl  # shape: (6), traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        atomic_quad_tl = detrace(atomic_quad_ntl)  # shape: (6)
        atomic_dipole = atomic_dipole.sum(axis=-2)  # shape: (3)

        # Redis contribution
        R = inputs['R']
        Q_tot = inputs['Q']  # shape: (1)
        z = inputs['z']
        q = MLP(features=[x.shape[-1], 1], use_bias=False)(x).squeeze(axis=-1)  # shape: (n)

        point_mask = inputs['point_mask']
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                                      scale=point_mask[:, None]))  # shape: (n)

        q = q + q_zi
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        redis_quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        redis_quad = redis_quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        redis_quad_ntl = redis_quad[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        redis_quad_tl = detrace(redis_quad_ntl)  # shape: (6)

        # determine dipole as sum of atomic dipole and partial charge dipole
        dipole = atomic_dipole + jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'quad': atomic_quad_tl + redis_quad_tl, 'quad_ntl': atomic_quad_ntl + redis_quad_ntl, 'mu': dipole, 'q': q}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'num_embeddings': self.num_embeddings
        }
        }


class RedisQuadrupole_zi_w(BaseSubModule):
    num_embeddings: int
    module_name: str = 'redis_quadrupole_layer_zi_w'  # name of module

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Computes partial charges from general features via charge redistribution. (n,F)->(n,1)->(n,1)

        Args: dict 'inputs' containing:
            q (Array): array of atom charge features: (n_atoms,F)
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # shape: (n,F)
        R = inputs['R']
        Q_tot = inputs['Q']  # shape: (1)
        z = inputs['z']
        q = MLP(features=[x.shape[-1], 1], use_bias=False)(x).squeeze(axis=-1)  # shape: (n)

        point_mask = inputs['point_mask']
        # atom-type dependent charge contributions
        q_zi = jnp.squeeze(safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                                      scale=point_mask[:, None]))  # shape: (n)

        q = q + q_zi
        # We enforce charge conservation by redistribution of residual charges
        point_mask = inputs['point_mask']
        n_atoms = jnp.sum(point_mask)
        q_res = jnp.ones(q.shape) * (1 / n_atoms * (jnp.sum(q) - Q_tot))
        q = q - q_res

        q = safe_scale(q, point_mask)
        quad = jax.vmap(jnp.outer, in_axes=(-2, -2))(R, R) * q[:, None, None]  # shape: (n,3,3)
        quad = quad.sum(axis=-3)  # shape: (3,3)
        quad_ind = ([0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2])
        quad_ntl = quad[quad_ind]  # shape: (6), non-traceless form

        def detrace(quad_ntl):
            trace = quad_ntl[0] + quad_ntl[1] + quad_ntl[2]
            return quad_ntl - (trace / 3) * jnp.array([1, 1, 1, 0, 0, 0])

        quad_tl = detrace(quad_ntl)  # shape: (6)

        # determine dipole from partial charges
        dipole = jnp.sum(q[:, None] * inputs['R'], axis=0)

        return {'q': q, 'quad': quad_tl, 'quad_ntl': quad_ntl, 'mu': dipole}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'num_embeddings': self.num_embeddings
        }
    }
