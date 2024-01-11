import jax
import jax.numpy as jnp
from jax.nn import silu
import flax.linen as nn

from typing import (Any, Dict, Sequence)

import numpy as np
import logging

from ceqnet.src.nn.base.sub_module import BaseSubModule
from ceqnet.src.masking.mask import safe_scale
from ceqnet.src.masking.mask import safe_mask
from ceqnet.src.basis_function.radial import get_rbf_fn
from ceqnet.src.cutoff_function.pbc import add_cell_offsets
from ceqnet.src.cutoff_function.radial import get_cutoff_fn
from functools import partial
from jax.ops import segment_sum
from ase.data import covalent_radii

from ceqnet.src.nn.mlp import MLP
from ceqnet.src.sph_ops.spherical_harmonics import init_sph_fn

# TODO: write init_from_dict methods in order to improve backward compatibility. E.g. AtomTypeEmbed(**h)
# will only work as long as the properties of the class are exactly the ones equal to the ones in h. As soon
# as additional arguments appear in h. Maybe use something like kwargs to allow for extensions?


class AtomTypeEmbed(BaseSubModule):
    num_embeddings: int
    features: int
    prop_keys: Dict
    module_name: str = 'atom_type_embed'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                z (Array): atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (n,F)

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)  # shape: (n)
        return safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z),
                          scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}



class kQeqHardnessEmbed(BaseSubModule):
    num_embeddings: int
    prop_keys: Dict
    module_name: str = 'kQeqHardness_embed'

    """
    Embedding for hardnesses (set to 0 !) and SOAP standard deviations taken from covalent radii from ase package as in 
    [39] 
    """

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        sigma = jnp.take(covalent_radii, z)
        z = z.astype(jnp.int32)  # shape: (n)
        J = jnp.zeros_like(z)  # shape: (n)
        return {'J': J, 'sigma': sigma}

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}}


# uff_radius_qeq = {'H': 0.371, 'He': 1.3, 'Li': 1.557, 'Be': 1.24, 'B': 0.822, 'C': 0.759, 'N': 0.715, 'O': 0.669, 'F': 0.706, 'Ne': 1.768, 'Na': 2.085, 'Mg': 1.5, 'Al': 1.201, 'Si': 1.176, 'P': 1.102, 'S': 1.047, 'Cl': 0.994, 'Ar': 2.108, 'K': 2.586, 'Ca': 2.0, 'Sc': 1.75, 'Ti': 1.607, 'V': 1.47, 'Cr': 1.402, 'Mn': 1.533, 'Fe': 1.393, 'Co': 1.406, 'Ni': 1.398, 'Cu': 1.434, 'Zn': 1.4, 'Ga': 1.211, 'Ge': 1.189, 'As': 1.204, 'Se': 1.224, 'Br': 1.141, 'Kr': 2.27, 'Rb': 2.77, 'Sr': 2.415, 'Y': 1.998, 'Zr': 1.758, 'Nb': 1.603, 'Mo': 1.53, 'Tc': 1.5, 'Ru': 1.5, 'Rh': 1.509, 'Pd': 1.544, 'Ag': 1.622, 'Cd': 1.6, 'In': 1.404, 'Sn': 1.354, 'Sb': 1.404, 'Te': 1.38, 'I': 1.333, 'Xe': 2.459, 'Cs': 2.984, 'Ba': 2.442, 'La': 2.071, 'Ce': 1.925, 'Pr': 2.007, 'Nd': 2.007, 'Pm': 2.0, 'Sm': 1.978, 'Eu': 2.227, 'Gd': 1.968, 'Tb': 1.954, 'Dy': 1.934, 'Ho': 1.925, 'Er': 1.915, 'Tm': 2.0, 'Yb': 2.158, 'Lu': 1.896, 'Hf': 1.759, 'Ta': 1.605, 'W': 1.538, 'Re': 1.6, 'Os': 1.7, 'Ir': 1.866, 'Pt': 1.557, 'Au': 1.618, 'Hg': 1.6, 'Tl': 1.53, 'Pb': 1.444, 'Bi': 1.514, 'Po': 1.48, 'At': 1.47, 'Rn': 2.2, 'Fr': 2.3, 'Ra': 2.2, 'Ac': 2.108, 'Th': 2.018, 'Pa': 1.8, 'U': 1.713, 'Np': 1.8, 'Pu': 1.84, 'Am': 1.942, 'Cm': 1.9, 'Bk': 1.9, 'Cf': 1.9, 'Es': 1.9, 'Fm': 1.9, 'Md': 1.9, 'No': 1.9, 'Lr': 1.9}
missing = 0.2
qeq_radii = np.array([np.inf, 0.371, 1.3, 1.557, 1.24, 0.822, 0.759, 0.715, 0.669, 0.706, 1.768, 2.085, 1.5, 1.201, 1.176, 1.102, 1.047])

class QeqEmbed(BaseSubModule):
    num_embeddings: int
    prop_keys: Dict
    hardness_mode: str  # 'learnable' or 'zero
    radii_mode: str  # mode for determining the SOAP Gaussian variances. 'learnable', 'ase', 'qeq', 'ase_scaled',
    # 'qeq_scaled'
    module_name: str = 'qeq_embed'

    """
    Embedding for charge equilibration values (Hardnesses and SOAP Gaussian variances). Opposed to kQeqHardnessEmbed and 
    HardnessEmbed, it has modularity
    [39] 
    """

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        z = z.astype(jnp.int32)  # shape: (n)
        point_mask = inputs['point_mask']
        J = None
        sigma = None
        # assigning hardness
        if self.hardness_mode == 'learnable':
            J = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            J = jnp.squeeze(J)
        elif self.hardness_mode == 'zero':
            J = jnp.zeros_like(z)  # shape: (n)
        elif self.hardness_mode == 'exp':
            J = safe_scale(jnp.exp(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            J = jnp.squeeze(J)
        elif self.hardness_mode == '20sgm':
            J = safe_scale(20 * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            J = jnp.squeeze(J)
        elif self.hardness_mode == '15sgm':
            J = safe_scale(20 * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            J = jnp.squeeze(J)
        elif self.hardness_mode == 'a_sgm':
            J = safe_scale(jnp.abs(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z))
                               * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            J = jnp.squeeze(J)
        elif self.hardness_mode == 'a_abs':
            J = safe_scale(jnp.abs(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            J = jnp.squeeze(J)
        elif self.hardness_mode == 'ln1exp':
            J = safe_scale(jnp.log(1 + jnp.exp(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z))),
                               scale=point_mask[:, None])
            J = jnp.squeeze(J)


        # assigning SOAP Gaussian variances
        if self.radii_mode == 'ase':
            sigma = safe_scale(jnp.take(covalent_radii, z), scale=point_mask)  # shape: (n)
        elif self.radii_mode == 'qeq':
            sigma = safe_scale(jnp.take(qeq_radii, z), scale=point_mask)  # shape: (n)
        elif self.radii_mode == 'learnable':
            sigma = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == 'ase_scaled':
            sigma = safe_scale(jnp.take(covalent_radii, z), scale=point_mask)  # shape: (n)
            factors = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            factors = jnp.squeeze(factors)
            sigma = factors * sigma
        elif self.radii_mode == 'qeq_scaled':
            sigma = safe_scale(jnp.take(qeq_radii, z), scale=point_mask)  # shape: (n)
            factors = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            factors = jnp.squeeze(factors)
            sigma = factors * sigma
        elif self.radii_mode == 'exp':
            sigma = safe_scale(jnp.exp(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                           scale=point_mask[:, None])  # shape: (n_atoms,1)
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == '20sgm':
            sigma = safe_scale(20 * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                           scale=point_mask[:, None])
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == '15sgm':
            sigma = safe_scale(20 * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                           scale=point_mask[:, None])
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == 'a_sgm':
            sigma = safe_scale(jnp.abs(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z))
                * jax.nn.sigmoid(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == 'a_abs':
            sigma = safe_scale(jnp.abs(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z)),
                               scale=point_mask[:, None])
            sigma = jnp.squeeze(sigma)
        elif self.radii_mode == 'ln1exp':
            sigma = safe_scale(jnp.log(1 + jnp.exp(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z))),
                               scale=point_mask[:, None])
            sigma = jnp.squeeze(sigma)

        return {'J': J, 'sigma': sigma}

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys,
                                   'hardness_mode': self.hardness_mode,
                                   'radii_mode': self.radii_mode}}

class HardnessEmbed(BaseSubModule):
    num_embeddings: int
    prop_keys: Dict
    module_name: str = 'hardness_embed'

    """
    Returns atomic hardness as in "Ko, T.W., Finkler, J.A., Goedecker, S. et al. A fourth-generation high-dimensional 
    neural network potential with accurate electrostatics including non-local charge transfer. Nat Commun 12, 398
    (2021). https://doi.org/10.1038/s41467-020-20427-2"
    """

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']
        # Hardcoded covalent radii standard deviations:
        #cov_r_std = {'0': np.inf,'1': 5, '6': 1, '7': 1, '8': 2, '9': 3}
        #sigma = np.array([cov_r_std[str(z[i])] for i in range(len(z))])
        cov_r_std = jnp.array([0, 5, 0, 7, 3, 3, 1, 1, 2, 3, 0, 9, 7])
        sigma = safe_scale(jnp.take(cov_r_std, z), scale=point_mask)
        z = z.astype(jnp.int32)  # shape: (n)
        J = safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=1)(z), scale=point_mask[:, None])  # shape: (n_atoms,1)
        J = jnp.squeeze(J)
        return {'J': J, 'sigma': sigma}

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}}


class GeometryEmbed(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_function: str
    n_rbf: int
    radial_cutoff_fn: str
    r_cut: float
    sphc: bool
    sphc_normalization: float = None
    mic: bool = False
    solid_harmonic: bool = False
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed'

    def setup(self):
        if self.input_convention == 'positions':
            self.atomic_position_key = self.prop_keys.get('atomic_position')
            if self.mic == 'bins':
                logging.warning(f'mic={self.mic} is deprecated in favor of mic=True.')
            if self.mic == 'naive':
                raise DeprecationWarning(f'mic={self.mic} is not longer supported.')
            if self.mic:
                self.unit_cell_key = self.prop_keys.get('unit_cell')
                self.cell_offset_key = self.prop_keys.get('cell_offset')

        elif self.input_convention == 'displacements':
            self.displacement_vector_key = self.prop_keys.get('displacement_vector')
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        self.atomic_type_key = self.prop_keys.get('atomic_type')

        self.sph_fns = [init_sph_fn(y) for y in self.degrees]

        _rbf_fn = get_rbf_fn(self.radial_basis_function)
        self.rbf_fn = _rbf_fn(n_rbf=self.n_rbf, r_cut=self.r_cut)

        _cut_fn = get_cutoff_fn(self.radial_cutoff_fn)
        self.cut_fn = partial(_cut_fn, r_cut=self.r_cut)
        self._lambda = jnp.float32(self.sphc_normalization)

    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.
        Args:
            inputs (Dict):
                R (Array): atomic positions, shape: (n,3)
                idx_i (Array): index centering atom, shape: (n_pairs)
                idx_j (Array): index neighboring atom, shape: (n_pairs)
                pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:
        """
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = inputs['pair_mask']  # shape: (n_pairs)

        # depending on the input convention, calculate the displacement vectors or load them from input
        if self.input_convention == 'positions':
            R = inputs[self.atomic_position_key]  # shape: (n,3)
            # Calculate pairwise distance vectors
            r_ij = safe_scale(jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j), scale=pair_mask[:, None])
            # shape: (n_pairs,3)

            # Apply minimal image convention if needed
            if self.mic:
                cell = inputs[self.unit_cell_key]  # shape: (3,3)
                cell_offsets = inputs[self.cell_offset_key]  # shape: (n_pairs,3)
                r_ij = add_cell_offsets(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets)  # shape: (n_pairs,3)

        elif self.input_convention == 'displacements':
            R = None
            r_ij = inputs[self.displacement_vector_key]
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Scale pairwise distance vectors with pairwise mask
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])

        # Calculate pairwise distances
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        # Gaussian basis expansion of distances
        rbf_ij = safe_scale(self.rbf_fn(d_ij[:, None]), scale=pair_mask[:, None])  # shape: (n_pairs,K)
        phi_r_cut = safe_scale(self.cut_fn(d_ij), scale=pair_mask)  # shape: (n_pairs)

        # Normalized distance vectors
        unit_r_ij = safe_mask(mask=d_ij[:, None] != 0,
                              operand=r_ij,
                              fn=lambda y: y / d_ij[:, None],
                              placeholder=0
                              )  # shape: (n_pairs, 3)
        unit_r_ij = safe_scale(unit_r_ij, scale=pair_mask[:, None])  # shape: (n_pairs, 3)

        # Spherical harmonics
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            sph_ij = safe_scale(sph_fn(unit_r_ij), scale=pair_mask[:, None])  # shape: (n_pairs,2l+1)
            sph_harms_ij += [sph_ij]  # len: |L| / shape: (n_pairs,2l+1)

        sph_harms_ij = jnp.concatenate(sph_harms_ij, axis=-1) if len(self.degrees) > 0 else None
        # shape: (n_pairs,m_tot)

        geometric_data = {'R': R,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'rbf_ij': rbf_ij,
                          'phi_r_cut': phi_r_cut,
                          'sph_ij': sph_harms_ij,
                          }

        # Spherical harmonic coordinates (SPHCs)
        if self.sphc:
            z = inputs[self.atomic_type_key]
            point_mask = inputs['point_mask']
            if self.sphc_normalization is None:
                # Initialize SPHCs to zero
                geometric_data.update(_init_sphc_zeros(z=z,
                                                       sph_ij=sph_harms_ij,
                                                       phi_r_cut=phi_r_cut,
                                                       idx_i=idx_i,
                                                       point_mask=point_mask,
                                                       mp_normalization=self._lambda)
                                      )
            else:
                # Initialize SPHCs with a neighborhood dependent embedding
                geometric_data.update(_init_sphc(z=z,
                                                 sph_ij=sph_harms_ij,
                                                 phi_r_cut=phi_r_cut,
                                                 idx_i=idx_i,
                                                 point_mask=point_mask,
                                                 mp_normalization=self._lambda)
                                      )

        # Solid harmonics (Spherical harmonics + radial part)
        if self.solid_harmonic:
            rbf_ij = safe_scale(rbf_ij, scale=phi_r_cut[:, None])  # shape: (n_pairs,K)
            g_ij = sph_harms_ij[:, :, None] * rbf_ij[:, None, :]  # shape: (n_pair,m_tot,K)
            g_ij = safe_scale(g_ij, scale=pair_mask[:, None, None], placeholder=0)  # shape: (n_pair,m_tot,K)
            geometric_data.update({'g_ij': g_ij})

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_function': self.radial_basis_function,
                                   'n_rbf': self.n_rbf,
                                   'radial_cutoff_fn': self.radial_cutoff_fn,
                                   'r_cut': self.r_cut,
                                   'sphc': self.sphc,
                                   'sphc_normalization': self.sphc_normalization,
                                   'solid_harmonic': self.solid_harmonic,
                                   'mic': self.mic,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, point_mask, mp_normalization, *args, **kwargs):
    _sph_harms_ij = safe_scale(sph_ij, phi_r_cut[:, None])  # shape: (n_pairs,m_tot)
    chi = segment_sum(_sph_harms_ij, segment_ids=idx_i, num_segments=len(z))
    chi = safe_scale(chi, scale=point_mask[:, None])  # shape: (n,m_tot)
    return {'chi': chi / mp_normalization}


def _init_sphc_zeros(z, sph_ij, *args, **kwargs):
    return {'chi': jnp.zeros((z.shape[-1], sph_ij.shape[-1]), dtype=sph_ij.dtype)}


class ChargeEmbed(BaseSubModule):
    features: int
    prop_keys: Dict
    num_embeddings: int = 100
    module_name: str = 'tot_charge_embed'

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():
        Returns:
        """
        z = inputs[self.atomic_type_key]
        Q = inputs[self.total_charge_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)
        q = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)  # shape: (n,F)
        Q_ = Q // jnp.inf  # -1 if Q < 0 and 0 otherwise
        Q_ = Q_.astype(jnp.int32)  # shape: (1)
        k = nn.Embed(num_embeddings=2, features=self.features)(Q_)  # shape: (1,F)
        v = nn.Embed(num_embeddings=2, features=self.features)(Q_)  # shape: (1,F)
        q_x_k = (q * k).sum(axis=-1) / jnp.sqrt(self.features)  # shape: (n)
        q_x_k = safe_scale(q_x_k,
                           scale=point_mask,
                           placeholder=-1e10)  # shape: (n)

        def calculate_numerator(u):
            w = jnp.log(1 + jnp.exp(u))
            return w

        numerator = safe_mask(mask=point_mask, fn=calculate_numerator, operand=q_x_k)
        a = safe_mask(mask=numerator != 0, fn=lambda x: Q * x / x.sum(axis=-1), operand=numerator, placeholder=0)
        e_Q = MLP(features=[self.features, self.features],
                    activation_fn=silu,
                    use_bias=False)(a[:, None] * v)  # shape: (n,F)
        Q_mask = jnp.ones_like(e_Q) * (Q!=0)
        e_Q = jnp.where(Q_mask, e_Q, jnp.zeros_like(e_Q))
        return safe_scale(e_Q, scale=point_mask[:, None])  # shape: (n,F)

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}

class ChargeSpinEmbed(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self,
                 z: jnp.ndarray,
                 psi: jnp.ndarray,
                 point_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the total charge or the number of unpaired spins in the system, following the
        embedding procedure introduced in SpookyNet. Returns per atom embeddings of dimension F.
        Args:
            z (Array): Atomic types, shape: (n)
            psi (Array): Total charge or number of unpaired spins, shape: (1)
            point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():
        Returns: Per atom embedding, shape: (n,F)
        """
        z = z.astype(jnp.int32)  # shape: (n)
        q = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)  # shape: (n,F)
        psi_ = psi // jnp.inf  # -1 if psi < 0 and 0 otherwise
        psi_ = psi_.astype(jnp.int32)  # shape: (1)
        k = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        v = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        q_x_k = (q*k).sum(axis=-1) / jnp.sqrt(self.features)  # shape: (n)
        q_x_k = safe_scale(q_x_k,
                           scale=point_mask,
                           placeholder=-1e10)  # shape: (n)

        numerator = jnp.log(1 + jnp.exp(q_x_k))  # shape: (n)
        a = psi * numerator / numerator.sum(axis=-1)  # shape: (n)

        e_psi = MLP(features=[self.features, self.features],
                    activation_fn=silu,
                    use_bias=False)(a[:, None] * v)  # shape: (n,F)

        e_psi = jnp.where(psi != 0, e_psi, jnp.zeros_like(e_psi))
        return safe_scale(e_psi, scale=point_mask[:, None])  # shape: (n,F)


class _ChargeEmbed(BaseSubModule):
    features: int
    prop_keys: Dict
    num_embeddings: int = 100
    module_name: str = 'tot_charge_embed'

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():
        Returns:
        """
        z = inputs[self.atomic_type_key]
        Q = inputs[self.total_charge_key]
        point_mask = inputs['point_mask']

        return ChargeSpinEmbed(num_embeddings=self.num_embeddings,
                               features=self.features)(z=z, psi=Q, point_mask=point_mask)

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}