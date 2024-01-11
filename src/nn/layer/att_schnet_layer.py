# In this version of the sph_schnet_layer, compared to the previous one, I undertook some restructuring of the
# nn.modules
# TODO: cosmetics: nn.vmap, compactify multiple vmaps within AttAggreg2, add multihead attention to first layer

import jax.numpy as jnp
import flax.linen as nn
import jax
from jax.ops import segment_sum
from functools import partial
from typing import (Sequence, Dict, Any, Callable)

from src.nn.base.sub_module import BaseSubModule
from src.nn.activation_function.activation_function import shifted_softplus
from src.nn.activation_function.activation_function import get_activation_fn
from src.nn.mlp import MLP
from src.sph_ops.contract import init_clebsch_gordan_matrix
from src.masking.mask import safe_scale
import logging

Array = Any


class AttentionSchNetLayer(BaseSubModule):
    F: int  # feature dimensions for each atom in a sample
    in2f_dims: Sequence[int]  # list of dimensions of layers of in2f-MLP
    filter_dims: Sequence[int]  # list of dimensions of layers for MLP to generate attention filter weight matrix.
    # The last entry need to be F*n_deg.
    gat_filter_dims: Sequence[int]  # For gat-attenion, we need a further set of filter dimensions, the last entry of
    # which needs to be 2*F*n_deg. Preferably, one would use the same dimensions as in filter_dims, with the last entry
    # doubled
    f2out_dims: Sequence[int]  # list of dimensions of layers of f2out-MLP
    degrees: Sequence[int]  # list of spherical degrees
    layer_position: str  # position of layer in sequence. Possible values: 'first', 'middle', 'last', 'first_and_last'
    n_head: int
    att_scale: bool  # boolean to determine whether the attention coefficients alpha_ij^h get scaled by 1/sqrt(F/n_head)
    att_type: str  # possible values: 'dot_product_attention', 'gat_attention'
    activation_name: str = "shifted_softplus"  # name of activation function used to generate the attention filter
    # matrix
    module_name: str = 'attention_schnet_layer'  # name of module
    l_0_skip = True  # depending on whether l=0 is spherical degrees the first layer makes a skip connection at l
    multi_head = True  # boolean determining whether to use multi-head attention or not

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        if self.F % self.n_head != 0:
            logging.warning("F needs to be 0 modulo n_head")
        if (self.gat_filter_dims != None) and  (self.gat_filter_dims[-1] != 2 *(len(self.degrees) * self.F)):
            logging.warning("for multi-head attention, the last filter-dim needs to be 2*n_deg*n_head")

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """
        # the first layer has to.... #TODO: whats difference between first and consecutive layers?
        if self.layer_position == 'first':
            x = FirstSelfAttention(degrees=self.degrees,
                                   n_head= self.n_head,
                                   F=self.F,
                                   in2f_dims=self.in2f_dims,
                                   filter_dims=self.filter_dims,
                                   gat_filter_dims=self.gat_filter_dims,
                                   f2out_dims=self.f2out_dims,
                                   l_0_skip=self.l_0_skip,
                                   att_type=self.att_type,
                                   att_scale=self.att_scale,
                                   activation_name=self.activation_name,
                                   name="message_passing_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            w = SelfMixLayer(degrees=self.degrees,
                             F=self.F,
                             in2f_dims=self.in2f_dims,
                             filter_dims=self.filter_dims,
                             f2out_dims=self.f2out_dims,
                             name="selfmix_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            x = x + w  # shape: (n,m_tot,F)
            return {'x': x}

        elif self.layer_position == 'middle':
            w = SelfAttention(degrees=self.degrees,
                              n_head=self.n_head,
                              F=self.F,
                              in2f_dims=self.in2f_dims,
                              filter_dims=self.filter_dims,
                              gat_filter_dims=self.gat_filter_dims,
                              f2out_dims=self.f2out_dims,
                              att_type=self.att_type,
                              att_scale=self.att_scale,
                              activation_name=self.activation_name,
                              name="message_passing_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            x = x + w  # shape: (n,m_tot,F)
            w = SelfMixLayer(degrees=self.degrees,
                             F=self.F,
                             in2f_dims=self.in2f_dims,
                             filter_dims=self.filter_dims,
                             f2out_dims=self.f2out_dims,
                             name="selfmix_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            x = x + w  # shape: (n,m_tot,F)
            return {'x': x}

        elif self.layer_position == 'last':
            w = SelfAttention(degrees=self.degrees,
                              n_head=self.n_head,
                              F=self.F,
                              in2f_dims=self.in2f_dims,
                              filter_dims=self.filter_dims,
                              gat_filter_dims=self.gat_filter_dims,
                              f2out_dims=self.f2out_dims,
                              att_type=self.att_type,
                              att_scale=self.att_scale,
                              activation_name=self.activation_name,
                              name="message_passing_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j, pair_mask=pair_mask)
            x = x + w  # shape: (n,m_tot,F)
            w = SelfMixLayer(degrees=self.degrees,
                             F=self.F,
                             in2f_dims=self.in2f_dims,
                             filter_dims=self.filter_dims,
                             f2out_dims=self.f2out_dims,
                             name="selfmix_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            x = x + w  # shape: (n,m_tot,F)
            # scalar product flattening along spherical harmonics dimension
            x = jax.vmap(jax.vmap(lambda x: jnp.dot(x, x), in_axes=1), in_axes=0)(x)  # shape: (n_atoms,F)
            return {'x': x}

        elif self.layer_position == 'first_and_last':
            x = FirstSelfAttention(degrees=self.degrees,
                                   n_head=self.n_head,
                                   F=self.F,
                                   in2f_dims=self.in2f_dims,
                                   filter_dims=self.filter_dims,
                                   gat_filter_dims=self.gat_filter_dims,
                                   f2out_dims=self.f2out_dims,
                                   l_0_skip=self.l_0_skip,
                                   att_type=self.att_type,
                                   att_scale=self.att_scale,
                                   activation_name=self.activation_name,
                                   name="message_passing_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            w = w = SelfMixLayer(degrees=self.degrees,
                             F=self.F,
                             in2f_dims=self.in2f_dims,
                             filter_dims=self.filter_dims,
                             f2out_dims=self.f2out_dims,
                             name="selfmix_layer") \
                (x=x, rbf_ij=rbf_ij, sph_ij=sph_ij, phi_r_cut=phi_r_cut, idx_i=idx_i, idx_j=idx_j,
                 pair_mask=pair_mask)  # shape: (n,m_tot,F)
            x = x + w  # shape: (n,m_tot,F)
            # scalar product flattening along spherical harmonics dimension
            x = jax.vmap(jax.vmap(lambda x: jnp.dot(x, x), in_axes=1), in_axes=0)(x)  # shape: (n_atoms,F)
            return {'x': x}

        else:
            logging.warning("non-valid layer-position ", self.layer_position,
                            ". Please choose from 'first', 'middle', 'last' or 'first_and_last'.")

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'F': self.F,
                                   'in2f_dims': self.in2f_dims,
                                   'f2out_dims': self.f2out_dims,
                                   'filter_dims': self.filter_dims,
                                   'gat_filter_dims': self.gat_filter_dims,
                                   'activation_name': self.activation_name,
                                   'module_name': self.module_name,
                                   'degrees': self.degrees,
                                   'layer_position': self.layer_position,
                                   'n_head': self.n_head,
                                   'att_type': self.att_type,
                                   'att_scale': self.att_scale
                                   }
                }


class FirstSelfAttention(nn.Module):
    F: int  # feature dimensions for each atom in a sample
    in2f_dims: Sequence[int]  # list of dimensions of layers of in2f-MLP
    filter_dims: Sequence[int]  # list of dimensions of layers for filter-MLP
    gat_filter_dims: Sequence[int]  # For gat-attenion, we need a further set of filter dimensions, the last entry of
    # which needs to be 2*F*n_deg. Preferably, one would use the same dimensions as in filter_dims, with the last entry
    # doubled
    f2out_dims: Sequence[int]  # list of dimensions of layers of f2out-MLP
    l_0_skip: bool  # depending on whether 0 is spherical degrees the first layer makes a skip connection at spherical degree l=0
    degrees: Sequence[int]  # spherical harmonics degrees
    n_head: int  # number of attention heads
    att_type: str  # possible values: 'dot_product_attention', 'gat_attention'
    att_scale: bool
    activation_name: str = "shifted_softplus"  # name of activation function

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        repeats = [2 * i + 1 for i in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(repeats), axis=1, total_repeat_length=sum(repeats))
        if self.in2f_dims[-1] != self.F:
            logging.warning("Last in2f feature has to be F")
        if self.filter_dims[-1] != len(self.degrees) * self.F:
            logging.warning("Last filter feature has to be n_degrees * F")
        if self.f2out_dims[-1] != self.F:
            logging.warning("Last f2out feature has to be F")

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,m_tot,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """
        if self.l_0_skip == True:
            x = MLP(self.in2f_dims, use_bias=False, name="in2f")(x)  # shape: (n_atom,m_tot,F)
            repeats = [2 * i + 1 for i in self.degrees]
            m_tot = sum(repeats)
            x_0 = jnp.pad(array=x[:, None, :], pad_width=((0, 0), (0, m_tot - 1), (0, 0)), mode='constant')
            if self.att_type == 'dot_product_attention':
                x = FirstDotProdAttAggr(self.degrees, self.n_head, self.F, self.filter_dims, self.activation_fn,
                                    self.repeat_fn, self.att_scale, name='self_attention')\
                    (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atoms,m_tot,F)
            elif self.att_type == 'gat_attention':
                x = FirstGatAttAggr(self.degrees, self.n_head, self.F, self.gat_filter_dims, self.activation_fn,
                                    self.repeat_fn, self.att_scale, name='self_attention')\
                    (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atoms,m_tot,F)
            x = x_0 + x # shape: (n_atoms,m_tot,F)
            x = MLP(self.f2out_dims, use_bias=False, name="f2out")(x)  # shape: (n_atom,m_tot,F)
            return x
        else:
            x = MLP(self.in2f_dims, use_bias=False, name="in2f")(x)  # shape: (n,m_tot,F)
            if self.att_type == 'dot_product_attention':
                x = FirstDotProdAttAggr(self.degrees, self.n_head, self.F, self.filter_dims, self.activation_fn,
                                    self.repeat_fn, self.att_scale, name='self_attention') \
                    (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atoms,m_tot,F)
            elif self.att_type == 'gat_attention':
                x = FirstGatAttAggr(self.degrees, self.n_head, self.F, self.gat_filter_dims, self.activation_fn,
                                    self.repeat_fn, self.att_scale, name='self_attention') \
                    (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atoms,m_tot,F)
            x = MLP(self.f2out_dims, use_bias=False, name="f2out")(x)  # shape: (n,m_tot,F)
            return x


class SelfAttention(nn.Module):
    degrees: Sequence[int]  # spherical harmonics degrees
    n_head: int
    F: int  # feature dimensions for each atom in a sample
    in2f_dims: Sequence[int]  # list of dimensions of layers of in2f-MLP
    filter_dims: Sequence[int]  # list of dimensions of layers for filter-MLP
    gat_filter_dims: Sequence[int]  # For gat-attenion, we need a further set of filter dimensions, the last entry of
    # which needs to be 2*F*n_deg. Preferably, one would use the same dimensions as in filter_dims, with the last entry
    # doubled
    f2out_dims: Sequence[int]  # list of dimensions of layers of f2out-MLP
    att_type: str  # possible values: 'dot_product_attention', 'gat_attention'
    att_scale: bool
    activation_name: str = "shifted_softplus"  # name of activation function



    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        repeats = [2 * i + 1 for i in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(repeats), axis=1, total_repeat_length=sum(repeats))
        if self.in2f_dims[-1] != self.F:
            logging.warning("Last in2f feature has to be F")
        if self.filter_dims[-1] != len(self.degrees) * self.F:
            logging.warning("Last filter feature has to be n_degrees * F")
        if self.f2out_dims[-1] != self.F:
            logging.warning("Last f2out feature has to be F")

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,m_tot,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """
        x = MLP(self.in2f_dims, use_bias=False, name="in2f")(x)  # shape: (n_atom,m_tot,F)
        if self.att_type == 'dot_product_attention':
            x = DotProdAttAggr(self.degrees, self.n_head, self.F, self.filter_dims, self.activation_fn,
                               self.repeat_fn, self.att_scale, name='self_attention') \
                (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atom,m_tot,F)
        elif self.att_type == 'gat_attention':
            x = GatAttAggr(self.degrees, self.n_head, self.F, self.gat_filter_dims, self.activation_fn,
                               self.repeat_fn, self.att_scale, name='self_attention') \
                (x, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask)  # shape: (n_atom,m_tot,F)
        x = MLP(self.f2out_dims, use_bias=False, name="f2out")(x)  # shape: (n_atom,m_tot,F)
        return x


class FirstDotProdAttAggr(nn.Module):
    degrees: Sequence[int]
    n_head: int
    F: int
    filter_dims: jnp.ndarray
    activation_fn: Callable
    repeat_fn: Callable
    att_scale: bool

    def setup(self):
        self.W_Q = self.param('W_Q', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.W_K = self.param('W_K', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray):
        """

         Args:
             x (Array): Atomic features, shape: (n,m_tot,F)
             rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
             sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
             phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
             idx_i (Array): index centering atom, shape: (n_pair)
             idx_j (Array): index neighboring atom, shape: (n_pair)
             pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
             *args ():
             **kwargs ():

         Returns:

         """
        x_head = x.reshape(*x.shape[:-1], self.n_head, -1)  # shape: (n_atom,n_head,F/n_head)
        Q = jax.vmap(
            lambda x__: jax.vmap(lambda w__: jax.vmap(lambda w_, x_: jnp.matmul(w_, x_), in_axes=(0, 0))(w__, x__),
                                 in_axes=0)(self.W_Q), in_axes=0)(x_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        K = jax.vmap(
            lambda x__: jax.vmap(lambda k__: jax.vmap(lambda k_, x_: jnp.matmul(k_, x_), in_axes=(0, 0))(k__, x__),
                                 in_axes=0)(self.W_K), in_axes=0)(x_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        Q_i = safe_scale(Q[idx_i], pair_mask[:, None, None, None], len(x))  # shape: (n_pair,n_deg,n_head,F/n_head)
        K_j = safe_scale(K[idx_j], pair_mask[:, None, None, None], len(x))  # shape: (n_pair,n_deg,n_head,F/n_head)
        W_ij = RadialFilter(len(self.degrees), self.n_head, self.F, self.filter_dims, self.activation_fn, name="radial_filter") (rbf_ij, phi_r_cut)
        alpha_ij = jnp.sum((Q_i * K_j) * W_ij, axis=-1)  # shape: (n_pair,n_deg,n_head)
        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pair, m_tot, n_head)
        if self.att_scale:
            alpha_ij = alpha_ij / jnp.sqrt(self.F/self.n_head)  # shape: (n_pair, m_tot, n_head)
        x_j_head = safe_scale(x_head[idx_j], pair_mask[:, None, None], len(x))  # shape: (n_pair,n_head,F/n_head)
        z_ij = jax.vmap(lambda x__, y__: jax.vmap(lambda x_: jnp.outer(y__,x_), in_axes=0)(x__), in_axes=(0,0))(x_j_head, sph_ij)  # shape: (n_pair,n_head,m_tot,F/n_head)
        z_ij = jnp.transpose(z_ij, axes=[0,2,1,3])  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = alpha_ij[:, :, :, None] * z_ij  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = z_ij.reshape(*z_ij.shape[:-2], -1)  # shape: (n_pair,m_tot,F)
        return segment_sum(z_ij, idx_i, len(x))  # shape: (n_atoms,m_tot,F)


class DotProdAttAggr(nn.Module):
    degrees: Sequence[int]
    n_head: int
    F: int
    filter_dims: jnp.ndarray
    activation_fn: Callable
    repeat_fn: Callable
    att_scale: bool

    def setup(self):
        self.W_Q = self.param('W_Q', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.W_K = self.param('W_K', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)


    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray):
        """
        version of attention aggregation that uses query and key matrices Q and K
        Args:
            x (Array): Atomic features, shape: (n,m_tot,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """
        x_inv = self.sph_deg_flattening(x, self.degrees)  # shape: (n_atoms,n_deg,F)
        x_inv_head = x_inv.reshape(*x_inv.shape[:-1], self.n_head, -1)  # shape: (n_atoms,n_deg,n_head,F/n_head)
        # Compute the multi-head Query and Key tensors Q and W from the spherically flatted and head-wise splitted 
        # input x_inv_head. We compute one Query and Key matrix per spherical degree and head, sharing wheigts across
        # spherical orders and atoms
        Q = jax.vmap(lambda x___: jax.vmap(
            lambda W__, x__: jax.vmap(lambda W_, x_: jnp.matmul(W_, x_), in_axes=(0, 0))(W__, x__), in_axes=(0, 0))(
            self.W_Q, x___), in_axes=0) \
            (x_inv_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        K = jax.vmap(lambda x___: jax.vmap(
            lambda W__, x__: jax.vmap(lambda W_, x_: jnp.matmul(W_, x_), in_axes=(0, 0))(W__, x__), in_axes=(0, 0))(
            self.W_K, x___), in_axes=0) \
            (x_inv_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        Q_i = safe_scale(Q[idx_i], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        K_j = safe_scale(K[idx_j], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        W_ij = RadialFilter(len(self.degrees), self.n_head, self.F, self.filter_dims, self.activation_fn, name="radial_filter") (rbf_ij, phi_r_cut)
        alpha_ij = jnp.sum((Q_i * K_j) * W_ij, axis=-1)  # shape: (n_pair,n_deg,n_head)
        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pair,m_tot,n_head)
        if self.att_scale:
            alpha_ij = alpha_ij / jnp.sqrt(self.F / self.n_head)  # shape: (n_pair, m_tot, n_head)
        x_j = safe_scale(x[idx_j], pair_mask[:, None, None])  # shape: (n_pair,m_tot,F)
        x_j_head = x_j.reshape(*x_j.shape[:-1], self.n_head, -1)  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = alpha_ij[:, :, :, None] * x_j_head  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = z_ij.reshape(*z_ij.shape[:-2], self.F)  # shape: (n_pair,m_tot,F)
        return segment_sum(z_ij, idx_i, len(x))  # shape: (n_atom,m_tot,F)

    def sph_deg_flattening(self, x, degrees: Sequence[int]):
        """
        flattens given harmonic atom vector along harmonic dimension via degree-wise self-scalar product
        Args:
            x: shape: (n_atoms,m_tot,F)
        Returns:
            x_flattened: shape: (n_atoms,n_degrees,F)
        """
        _repeats = [2 * y + 1 for y in degrees]  # shape: (n_degrees,)
        seg_indices = jnp.repeat(jnp.arange(0, len(degrees)), jnp.array(_repeats),
                                 total_repeat_length=sum(_repeats))  # shape: (m_tot,)
        y = x * x  # shape: (n,m_tot,F)

        def seg_sum(y):
            return jax.ops.segment_sum(y, seg_indices, len(degrees))

        seg_sum = jax.vmap(jax.vmap(seg_sum, in_axes=-1, out_axes=-1), in_axes=0, out_axes=0)
        x_flattened = seg_sum(y)  # shape: (n_atoms,n_degrees,F)
        return x_flattened  # shape: (n_atoms,n_degrees,F)

class FirstGatAttAggr(nn.Module):
    degrees: Sequence[int]
    n_head: int
    F: int
    gat_filter_dims: Sequence[int]
    activation_fn: Callable
    repeat_fn: Callable
    att_scale: bool

    def setup(self):
        self.W_Q = self.param('W_Q', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.W_K = self.param('W_K', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.a = self.param('a', nn.initializers.lecun_normal(),
                            (len(self.degrees), self.n_head, int(2 * (self.F / self.n_head))))
        # shape: (n_deg,n_head,2*F/n_head)


    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray):
        """

         Args:
             x (Array): Atomic features, shape: (n,m_tot,F)
             rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
             sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
             phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
             idx_i (Array): index centering atom, shape: (n_pair)
             idx_j (Array): index neighboring atom, shape: (n_pair)
             pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
             *args ():
             **kwargs ():

         Returns:

         """
        x_head = x.reshape(*x.shape[:-1], self.n_head, -1)  # shape: (n_atom,n_head,F/n_head)
        Q = jax.vmap(
            lambda x__: jax.vmap(lambda w__: jax.vmap(lambda w_, x_: jnp.matmul(w_, x_), in_axes=(0, 0))(w__, x__),
                                 in_axes=0)(self.W_Q), in_axes=0)(x_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        K = jax.vmap(
            lambda x__: jax.vmap(lambda k__: jax.vmap(lambda k_, x_: jnp.matmul(k_, x_), in_axes=(0, 0))(k__, x__),
                                 in_axes=0)(self.W_K), in_axes=0)(x_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        Q_i = safe_scale(Q[idx_i], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        K_j = safe_scale(K[idx_j], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        # concatenating the query and key pairs
        Q_i_K_j = jnp.concatenate(arrays=(Q_i, K_j), axis=-1)  # shape: (n_pair,n_deg,n_head,2F/n_head)
        # scaling with rbf-molecule_inspect
        W_ij = GatRadialFilter(len(self.degrees), self.n_head, self.F, self.gat_filter_dims, self.activation_fn,
                               name="gat_radial_filter")(rbf_ij, phi_r_cut)  # shape: (n_pair,n_deg,n_head,2*F/n_head)
        Q_i_K_j = Q_i_K_j * W_ij
        alpha_ij = jnp.sum(Q_i_K_j * self.a[None, :, :, :], axis=-1)  # shape: (n_pair,n_deg,n_head)
        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pair, m_tot, n_head)
        if self.att_scale:
            alpha_ij = alpha_ij / jnp.sqrt(self.F/self.n_head)  # shape: (n_pair, m_tot, n_head)
        x_j_head = safe_scale(x_head[idx_j], pair_mask[:, None, None], len(x))  # shape: (n_pair,n_head,F/n_head)
        z_ij = jax.vmap(lambda x__, y__: jax.vmap(lambda x_: jnp.outer(y__,x_), in_axes=0)(x__), in_axes=(0,0))(x_j_head, sph_ij)  # shape: (n_pair,n_head,m_tot,F/n_head)
        z_ij = jnp.transpose(z_ij, axes=[0,2,1,3])  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = alpha_ij[:, :, :, None] * z_ij  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = z_ij.reshape(*z_ij.shape[:-2], -1)  # shape: (n_pair,m_tot,F)
        return segment_sum(z_ij, idx_i, len(x))  # shape: (n_atoms,m_tot,F)

class GatAttAggr(nn.Module):
    degrees: Sequence[int]
    n_head: int
    F: int
    gat_filter_dims: Sequence[int]
    activation_fn: Callable
    repeat_fn: Callable
    att_scale: bool

    def setup(self):
        self.W_Q = self.param('W_Q', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.W_K = self.param('W_K', nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(self.F / self.n_head), int(self.F / self.n_head)))
        # shape: (n_deg,n_head,F/n_head,F/n_head)
        self.a = self.param('a',nn.initializers.lecun_normal(),
                              (len(self.degrees), self.n_head, int(2* (self.F / self.n_head))))
        # shape: (n_deg,n_head,2*F/n_head)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray):
        """
        version of attention aggregation that uses query and key matrices Q and K
        Args:
            x (Array): Atomic features, shape: (n,m_tot,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """
        x_inv = self.sph_deg_flattening(x, self.degrees)  # shape: (n_atoms,n_deg,F)
        x_inv_head = x_inv.reshape(*x_inv.shape[:-1], self.n_head, -1)  # shape: (n_atoms,n_deg,n_head,F/n_head)
        # Compute the multi-head Query and Key tensors Q and W from the spherically flatted and head-wise splitted
        # input x_inv_head. We compute one Query and Key matrix per spherical degree and head, sharing wheights across
        # spherical orders and atoms
        Q = jax.vmap(lambda x___: jax.vmap(
            lambda W__, x__: jax.vmap(lambda W_, x_: jnp.matmul(W_, x_), in_axes=(0, 0))(W__, x__), in_axes=(0, 0))(
            self.W_Q, x___), in_axes=0) \
            (x_inv_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        K = jax.vmap(lambda x___: jax.vmap(
            lambda W__, x__: jax.vmap(lambda W_, x_: jnp.matmul(W_, x_), in_axes=(0, 0))(W__, x__), in_axes=(0, 0))(
            self.W_K, x___), in_axes=0) \
            (x_inv_head)  # shape: (n_atom,n_deg,n_head,F/n_head)
        Q_i = safe_scale(Q[idx_i], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        K_j = safe_scale(K[idx_j], pair_mask[:, None, None, None])  # shape: (n_pair,n_deg,n_head,F/n_head)
        # concatenating the query and key pairs
        Q_i_K_j = jnp.concatenate(arrays=(Q_i,K_j), axis=-1)  # shape: (n_pair,n_deg,n_head,2F/n_head)
        # scaling with rbf-molecule_inspect
        W_ij = GatRadialFilter(len(self.degrees), self.n_head, self.F, self.gat_filter_dims, self.activation_fn,
                            name="gat_radial_filter")(rbf_ij, phi_r_cut)  # shape: (n_pair,n_deg,n_head,2*F/n_head)
        Q_i_K_j = Q_i_K_j * W_ij
        alpha_ij = jnp.sum(Q_i_K_j * self.a[None, :, :, :], axis=-1)  # shape: (n_pair,n_deg,n_head)
        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pair,m_tot,n_head)
        if self.att_scale:
            alpha_ij = alpha_ij / jnp.sqrt(self.F / self.n_head)  # shape: (n_pair, m_tot, n_head)
        x_j = safe_scale(x[idx_j], pair_mask[:, None, None])  # shape: (n_pair,m_tot,F)
        x_j_head = x_j.reshape(*x_j.shape[:-1], self.n_head, -1)  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = alpha_ij[:, :, :, None] * x_j_head  # shape: (n_pair,m_tot,n_head,F/n_head)
        z_ij = z_ij.reshape(*z_ij.shape[:-2], self.F)  # shape: (n_pair,m_tot,F)
        return segment_sum(z_ij, idx_i, len(x))  # shape: (n_atom,m_tot,F)

    def sph_deg_flattening(self, x, degrees: Sequence[int]):
        """
        flattens given harmonic atom vector along harmonic dimension via degree-wise self-scalar product
        Args:
            x: shape: (n_atoms,m_tot,F)
        Returns:
            x_flattened: shape: (n_atoms,n_degrees,F)
        """
        _repeats = [2 * y + 1 for y in degrees]  # shape: (n_degrees,)
        seg_indices = jnp.repeat(jnp.arange(0, len(degrees)), jnp.array(_repeats),
                                 total_repeat_length=sum(_repeats))  # shape: (m_tot,)
        y = x * x  # shape: (n,m_tot,F)

        def seg_sum(y):
            return jax.ops.segment_sum(y, seg_indices, len(degrees))

        seg_sum = jax.vmap(jax.vmap(seg_sum, in_axes=-1, out_axes=-1), in_axes=0, out_axes=0)
        x_flattened = seg_sum(y)  # shape: (n_atoms,n_degrees,F)
        return x_flattened  # shape: (n_atoms,n_degrees,F)


class SelfMixLayer(nn.Module):
    degrees: Sequence[int]  # list of spherical degrees
    F: int  # feature dimensions for each atom in a sample
    in2f_dims: Sequence[int]  # list of dimensions of layers of in2f-MLP
    filter_dims: Sequence[int]  # list of dimensions of layers for filter-MLP
    f2out_dims: Sequence[int]  # list of dimensions of layers of f2out-MLP
    k_l3_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        self.s_l3 = self.param('s_l3', self.k_l3_init,
                               (len(self.degrees), int((len(self.degrees) - 1) * len(self.degrees) / 2), self.F))
        self.k_l3 = self.param('k_l3', self.k_l3_init, (len(self.degrees), self.F))
        repeats = [2 * i + 1 for i in self.degrees]
        self.repeat_fn_axis_0 = partial(jnp.repeat, repeats=jnp.array(repeats), axis=0,
                                        total_repeat_length=sum(repeats))
        self.repeat_fn_axis_minus3 = partial(jnp.repeat, repeats=jnp.array(repeats), axis=-3,
                                             total_repeat_length=sum(repeats))
        self.repeat_fn_axis_minus2 = partial(jnp.repeat, repeats=jnp.array(repeats), axis=-2,
                                             total_repeat_length=sum(repeats))
        # initialize Clebsch-Gordan coefficients
        self.cg = init_clebsch_gordan_matrix(self.degrees, self.degrees[-1])  # shape: (m_tot,m_tot,m_tot)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """
        Coupling layer, implementing selfmix as in paper "SE(3)-equivariant prediction of molecular wavefunctions and electronic densities".

        Args:
            x (Array): Atomic features, shape: (n,F)
            rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
            sph_ij (Array): Per degree molecule_inspect of SPHCs, shape: (n_all_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
            idx_i (Array): index centering atom, shape: (n_pair)
            idx_j (Array): index neighboring atom, shape: (n_pair)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pair)
            *args ():
            **kwargs ():

        Returns:

        """

        tensor_prod_contract = self.tensor_prod_contr(x)  # shape: (n,m_tot,(n_degrees-1)*n_deg/2,F)
        s_coeff = self.repeat_fn_axis_minus3(self.s_l3)  # shape: (m_tot,(n_degrees-1)*n_degrees/2,F)
        k_coeff = self.repeat_fn_axis_minus2(self.k_l3)
        return k_coeff * x + (s_coeff * tensor_prod_contract).sum(axis=-2)

    def tensor_prod_contr(self, x: jnp.array):
        """
        provides a tensor that contains for each atom and each spherical degree l_3 and order m_3 (along n-dim 1) the terms
        C^{l_3 l_2 l_1}_{m_3 m_2 m_1} x^{l_2}_{m_2} x^{l_1}_{m_1}, l_1\in {0,...,l_max}, l_2\in {l_1+1,...,_max} which
        need to summed over in the tensor product contraction
        """

        outer_prod_along_sph_dim = jax.vmap(jax.vmap(jnp.outer, in_axes=(-1, -1), out_axes=-1), in_axes=0, out_axes=0)
        x_x = outer_prod_along_sph_dim(x, x)  # shape: (n_atoms,m_tot,m_tot,F)
        cg_mult_vmapped = jax.vmap(jax.vmap(lambda y: self.cg * y, in_axes=-1, out_axes=-1), in_axes=0, out_axes=0)
        C_x_x = cg_mult_vmapped(x_x)  # shape: (n,m_tot,m_tot,m_tot,F)
        idx = jnp.array(self.degrees) - self.degrees[0]  # shape: (n_degrees)
        # sum degree-wise for l1 and l2
        segment_ids = self.repeat_fn_axis_0(idx)  # (m_tot)

        def seg_sum(y):
            return segment_sum(y, segment_ids, len(self.degrees))

        seg_sum_1 = jax.vmap(jax.vmap(jax.vmap(jax.vmap(seg_sum, in_axes=-1, out_axes=-1), in_axes=0), in_axes=0),
                             in_axes=0, out_axes=0)  # segmentation sum over second last axis
        seg_sum_2 = jax.vmap(jax.vmap(jax.vmap(seg_sum, in_axes=-1, out_axes=-1), in_axes=0), in_axes=0, out_axes=0)
        seg_sum = seg_sum_2(seg_sum_1(C_x_x))  # shape: (n,m_tot,n_degrees,n_degrees,F)
        lower_tri_indices = jnp.tril_indices(len(self.degrees),
                                             k=-1)  # shape: ((n_degrees-1)*n_degrees/2,), ((n_degrees-1)*n_degrees/2,)

        def get_lower_triang(_x):
            return _x[lower_tri_indices]

        lower_entries = jax.vmap(jax.vmap(jax.vmap(get_lower_triang, in_axes=-1, out_axes=-1), in_axes=0), in_axes=0,
                                 out_axes=0)
        return lower_entries(seg_sum)  # shape: (n,m_tot,(n_degrees-1)*n_degrees/2,F)

class RadialFilter(nn.Module):
    n_deg: int
    n_head: int
    F: int
    filter_dims: jnp.ndarray
    activation_fn: Callable

    def setup(self):
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        if self.filter_dims[-1] != self.n_deg * self.F:
            logging.warning("for dot-product-attention, the last filter-dim needs to be n_deg*F")

    @nn.compact
    def __call__(self,
                 rbf_ij: jnp.ndarray,
                 phi_r_cut: jnp.array):
        """

                Args:
                    rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
                    phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
                Returns:
                    W_ij: filter network for atom molecule_inspect
                """
        W_ij = MLP(self.filter_dims, self.activation_fn, name="radial_filter_MLP")(
            rbf_ij)  # shape: (n_pair,n_degrees*F)
        W_ij = jnp.reshape(W_ij, (-1, self.n_deg, self.n_head, int(self.F/self.n_head)))
        # shape: (n_pair,n_degrees,n_head,F/n_head)
        return (W_ij * phi_r_cut[:, None, None, None])  # shape:  (n_pair,n_degrees,n_head,F/n_head),
        # scaling filter weights with atom-pairwise cut-off'ed molecule_inspect by broadcasting. (some cut-off'ed molecule_inspect are 0)

class GatRadialFilter(nn.Module):
    n_deg: int
    n_head: int
    F: int
    gat_filter_dims: jnp.ndarray
    activation_fn: Callable

    def setup(self):
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus
        if self.gat_filter_dims[-1] != 2 *(self.n_deg * self.F):
            logging.warning("for multi-head attention, the last filter-dim needs to be 2*n_deg*n_head")

    @nn.compact
    def __call__(self,
                 rbf_ij: jnp.ndarray,
                 phi_r_cut: jnp.array):
        """

                Args:
                    rbf_ij (Array): RBF expanded molecule_inspect, shape: (n_pair,K)
                    phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pair)
                Returns:
                    W_ij: filter network for atom molecule_inspect
                """
        W_ij = MLP(self.gat_filter_dims, self.activation_fn, name="gat_radial_filter_MLP")(
            rbf_ij)  # shape: (n_pair,2*n_degrees*F)
        W_ij = jnp.reshape(W_ij, (-1, self.n_deg, self.n_head, int(2*(self.F/self.n_head))))
        # shape: (n_pair,n_degrees,n_head,F/n_head)
        return (W_ij * phi_r_cut[:, None, None, None])  # shape:  (n_pair,n_degrees,n_head,2*F/n_head),
        # scaling filter weights with atom-pairwise cut-off'ed molecule_inspect by broadcasting. (some cut-off'ed molecule_inspect are 0)

