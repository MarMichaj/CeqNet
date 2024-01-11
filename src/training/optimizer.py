import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze, unfreeze
from optax import constant_schedule
from dataclasses import dataclass

import logging
from typing import (Dict)
from optax import exponential_decay


@dataclass
class Optimizer:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    transition_steps: int = None
    decay_rate: float = None
    weight_decay: float = None

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:

        """
        self.learning_rate = learning_rate
        weight_decay = 0. if self.weight_decay is None else self.weight_decay
        mask = None if self.weight_decay is None else flattened_traversal(lambda path, _: path[-1] != 'bias')

        if self.transition_steps is None or self.decay_rate is None:
            step_size_fn = None
        else:
            step_size_fn = exponential_decay(1.,
                                             transition_steps=self.transition_steps,
                                             decay_rate=self.decay_rate
                                             )

        return optimizer(learning_rate=self.learning_rate,
                         b1=self.b1,
                         b2=self.b2,
                         eps=self.eps,
                         eps_root=self.eps_root,
                         weight_decay=weight_decay,
                         mask=mask,
                         step_size_fn=step_size_fn)

    def __dict_repr__(self):
        return {'optimizer': {'learning_rate': self.learning_rate,
                             'transition_steps': self.transition_steps,
                             'decay_rate': self.decay_rate,
                             'weight_decay': self.weight_decay}}

@dataclass
class Optimizer_amsgrad:

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:
        """

        self.learning_rate = learning_rate
        return optimizer_amsgrad(learning_rate=self.learning_rate)

    def __dict_repr__(self):
        return {'optimizer_amsgrad': {'learning_rate': self.learning_rate}}


def optimizer_amsgrad(learning_rate):
    return optax.chain(
        optax.scale_by_amsgrad(),
        optax.scale(-learning_rate),
        optax.scale_by_schedule(optax.constant_schedule(1)),
    )


def optimizer(learning_rate,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8,
              eps_root: float = 0.0,
              weight_decay: float = 0.0,
              mask=None,
              step_size_fn=None):

    if step_size_fn is None:
        step_size_fn = constant_schedule(1.)

    return optax.chain(
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        optax.scale(-learning_rate),
        optax.scale_by_schedule(step_size_fn),
    )


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(unfreeze(data))
        return freeze(traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()}))
    return mask


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return freeze(traverse_util.unflatten_dict(flat_mask))


# TODO: legacy, take care of secure removal

#                                                                                                                      #
#                                     Initialize optimizer from dictionary/json                                        #
#                                                                                                                      #


OPTIMIZER_hyperparameters = {'learning_rate': None,
                             'weight_decay': None,
                             'transition_steps': None,
                             'decay_rate': None
                             }


def hyper_params_to_properties(hyper_params):
    lr = hyper_params['learning_rate']
    if lr is None:
        raise ValueError("Learning rate must be specified in the optimizer hyperparameters"
                         "since no default value has been set")

    weight_decay = 0. if hyper_params['weight_decay'] is None else hyper_params['weight_decay']
    mask = None if hyper_params['weight_decay'] is None else flattened_traversal(lambda path, _: path[-1] != 'bias')
    if hyper_params['transition_steps'] is None or hyper_params['decay_rate'] is None:
        step_size_fn = None
    else:
        step_size_fn = exponential_decay(1.,
                                         transition_steps=hyper_params['transition_steps'],
                                         decay_rate=hyper_params['decay_rate']
                                         )

    return {'learning_rate': lr,
            'weight_decay': weight_decay,
            'mask': mask,
            'step_size_fn': step_size_fn
            }


def optimizer_from_hyper_params(hyper_params: Dict):
    d = {}
    for k, v_default in OPTIMIZER_hyperparameters.items():
        try:
            v = hyper_params[k]
        except KeyError:
            v = v_default
            logging.warning('The argument {} is missing in the optimizer hyperparameters. Set default '
                            '{}={}.'.format(k, k, v_default))

        d[k] = v

    d = hyper_params_to_properties(d)
    return optimizer(**d)

