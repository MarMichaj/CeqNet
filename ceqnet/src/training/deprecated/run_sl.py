import jax.numpy as jnp
import numpy as np
import jax
import logging
import time
import wandb

from functools import partial
from pprint import pformat
from typing import (Any, Callable, Dict, Sequence, Tuple, Union)
import flax
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, unfreeze
from flax.training import checkpoints

from ceqnet.src.io import save_json
from ceqnet.src.training.optimizer import Optimizer_amsgrad

logging.basicConfig(level=logging.INFO)

Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]



def run_training_sl(state: TrainState,
                 loss_fn: LossFn,
                 train_ds: DataTupleT,
                 valid_ds: DataTupleT,
                 train_bs: int,
                 valid_bs: int,
                    learning_rate: float,
                 ckpt_dir: str = None,
                 save_every_t: int = None,
                 eval_every_t: int = 1,
                 print_every_t: int = 1,
                 save_best: Sequence[str] = None,
                 ckpt_overwrite: bool = False,
                 seed: int = 0,
                 use_wandb: bool = True
                 ):
    """
    Run training like in spookynet paper for a NN.
    Args:
        state (TrainState): Flax train state.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        train_ds (Tuple): Tuple of training data. First entry is input, second is expected output.
        valid_ds (Tuple): Tuple of validation data. First entry is input, second is expected output.
        train_bs (int): Training batch size.
        valid_bs (int): Validation batch size.
        metric_fn (Callable): Dictionary of functions, which are evaluated on the validation set and logged.
        epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        save_every_t (int): Save the model every t-th epoch
        eval_every_t (int): Evaluate the metrics every t-th epoch
        print_every_t (int): Print the training loss every t-th epoch
        save_best (List): Save the model based on evaluation metric. Each entry must one key in the metric_fns
        ckpt_overwrite (bool): Whether overwriting of existing checkpoints is allowed.
        seed (int): Random seed.
    Returns:
    """
    # initializing helper variables
    rng = jax.random.PRNGKey(seed)  # jax random key
    tot_time = 0   # training time
    valid_metrics = {}  # current validation metrics
    best_valid_metrics = {}  # best validation metrics


    avg_valid_error = 0  # error to examine if performance of averaged-parameters model over time (every 1000 training
    # steps for 25 steps)
    avg_params = jax.tree_map(lambda x: x, state.params)  # We initialize the exponentially averaged model params by
    # copying the initial params
    check_avg_valid: bool  # boolean that determines whether we are within the 25 training step interval

    i_epoch = 0  # epoch counter
    step_counter = 0  # training steps counter
    check_phase = False  # boolean to determine whether we are currently in a validation error checking phase

    inputs, targets = train_ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])
    steps_per_epoch = n_data // train_bs

    while True:
        logging.info('epoch ' + str(i_epoch) + ':')
        epoch_start = time.time()  # start epoch timer
        rng, input_rng = jax.random.split(rng)  # initialize random keys
        train_start = time.time()  # set training start time
        # perform training epoch
        state, train_metrics, step_counter, avg_params, avg_valid_error, check_phase, learning_rate = train_epoch_sl(
            state, train_ds, loss_fn, train_bs, input_rng, step_counter, avg_params, avg_valid_error, valid_ds, valid_bs,
            check_phase, learning_rate)
        train_end = time.time()  # set training end time

        # check for NaNs in training metrics
        train_metrics_np = jax.device_get(train_metrics)

        if (np.isnan(train_metrics_np['loss']) or
            np.isinf(train_metrics_np['loss']) or
            (np.abs(train_metrics_np['loss']) > 2 and i_epoch > 10)
        ):

            if np.isnan(train_metrics_np['loss']):
                logging.warning(f'NaN detected during training in step {i_epoch} in the loss function value. Reload the '
                                'last checkpoint.')
            elif np.isinf(train_metrics_np['loss']):
                logging.warning(f'Inf detected during training in step {i_epoch} in the loss function value. Reload the '
                                'last checkpoint.')
            elif (np.abs(train_metrics_np['loss']) > 2):
                logging.warning(f'Value greater than 2 detected during training in step {i_epoch} in the loss function value. Reload the '
                                'last checkpoint.')


            def reset_records():
                return jax.tree_map(lambda x: jnp.zeros(x.shape), state.params['record'])

            state_dict = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, step=None,
                                                        prefix='checkpoint_loss_')
            try:
                state_dict['params']['record'] = reset_records()
            except KeyError:
                pass
            state = state.replace(params=FrozenDict(state_dict['params']))
            opt_state = state.txt.init(state.params)
            state = state.replace(opt_state=opt_state)
            #TODO: restore last checkpoint

        # Conditional setting of checkpoint per observable (based on validation error)
        valid_start, valid_end = (0., 0.)
        if i_epoch % eval_every_t == 0:
            valid_start = time.time()
            valid_metrics.update(valid_epoch_sl(state, valid_ds, loss_fn, bs=valid_bs))
            valid_end = time.time()
            if i_epoch == 0:
                best_valid_metrics.update(valid_metrics)
            # loop over all metrics and compare
            for _k, _v in best_valid_metrics.items():
                if valid_metrics[_k] < _v:
                    best_valid_metrics[_k] = valid_metrics[_k]
                    checkpoints.save_checkpoint(ckpt_dir,
                                                state,
                                                i_epoch,
                                                keep=1,
                                                prefix='checkpoint_{}_'.format(_k),
                                                overwrite=ckpt_overwrite)

        # measure run times
        epoch_end = time.time()
        e_time = epoch_end - epoch_start
        t_time = train_end - train_start
        v_time = valid_end - valid_start
        tot_time += e_time
        times = {'Epoch time (s)': e_time,
                 'Training epoch time (s)': t_time,
                 'Validation epoch time (s)': v_time,
                 'Total time (s)': tot_time}

        # local logging of metrics and times
        if i_epoch % print_every_t == 0:
            logging.info('Epoch: {}'.format(i_epoch))
            logging.info('Times: Epoch: {} s, Training: {} s, Validation: {}'.format(e_time, t_time, v_time))
            logging.info('Training metrics: {}'.format(pformat(train_metrics)))
            logging.info('Evaluation metrics: {}'.format(pformat(valid_metrics)))

        # wandb logging of metrics, times and learning_rate
        if use_wandb:
            if i_epoch > 0:
                wandb.log(times, step=i_epoch)
                log_train_metrics = {key + '_train': train_metrics[key] for (key, item) in train_metrics.items()}
                log_valid_metrics = {key + '_valid': valid_metrics[key] for (key, item) in valid_metrics.items()}
                lr = {'learning rate': learning_rate}
                wandb.log(log_train_metrics, step=i_epoch)
                wandb.log(log_valid_metrics, step=i_epoch)
                wandb.log(lr, step=i_epoch)

        i_epoch += 1

        # We stop training, when the learning rate drops below 0.5
        if learning_rate < 1e-5:  # decaying learning rate by 0.5
            break




def train_epoch_sl(state: TrainState,
                   ds: DataTupleT,
                   loss_fn: LossFn,
                   bs: int,
                   rng,
                   step_counter: int,
                   avg_params,
                   avg_valid_error,
                   valid_ds,
                   valid_bs,
                   check_phase: bool,
                   learning_rate: float):
        #-> list[Union[Union[TrainState, dict, int, float, bool], Any]]:
    """
        Training epoch via mini-batch gradient descent. Also, every 1000 training steps, and evaluation is done the check
        if validation error decreases for 25 consecutive steps (decreasing learning rate by factor 0.5 otherwise)
        Args:
            state (TrainState): Flax train state.
            ds (Tuple): Tuple of data, where first entry is input and second the expected output
            loss_fn (Callable): Loss function. Gradient is computed wrt it
            bs (int): Batch size
            rng (RandomKey): JAX PRNGKey
            step_counter:
            avg_params: parameters of exponentially averaged state
            avg_valid_error
            valid_ds
            valid_bs
            check_phase: boolean which determines if we are in a phase of checking the validation error
        Returns: Updated optimizer state and the training loss.
        """

    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    perms = jax.random.permutation(rng, n_data)
    perms = perms[:steps_per_epoch * bs]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, bs))
    batch_metrics = []
    for perm in perms:
        batch = jax.tree_map(lambda y: y[perm, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        state, metrics = train_step_fn_sl(state, batch, loss_fn)
        avg_params = update_avg_params(avg_params, state)  # update of state tracking exponential average params
        batch_metrics.append(metrics)

        # Every 1000 training steps, we adapt the learning rate based on the models performance on the validation set.

        # DEBUGGING
        logging.info('step: ' + str(step_counter))
        logging.info('phase: ' + str(check_phase))

        # entering the checking phase of the validation error of the avg_params model
        if check_phase is False and step_counter == 1000:
            avg_valid_error = valid_epoch_sl_avg(avg_params, valid_ds, loss_fn, bs=valid_bs)['loss']
            check_phase = True

        elif check_phase is True and (step_counter > 1000 and step_counter <= 1024):
            new_avg_valid_error = valid_epoch_sl_avg(avg_params, valid_ds, loss_fn, bs=valid_bs)['loss']
            # If the validation loss does not decrease for 25 consecutive steps, we decay the learning rate by a factor
            # 0.5
            if new_avg_valid_error >= avg_valid_error:
                learning_rate = learning_rate * 0.5  # decaying learning rate by 0.5
                state = state.replace(step=state.step, params=state.params, opt_state=state.opt_state,
                                        tx=Optimizer_amsgrad().get(learning_rate)) # create TrainState with decayed
                # learning rate
                check_phase = False  # we finish the checking phase, in case we decreased the learning rate
                step_counter = 0
                avg_valid_error = 0
            else:
                avg_valid_error = new_avg_valid_error
                if step_counter < 1024:
                    check_phase = True
                elif step_counter >= 1024:  # disabling the checking phase after 25 steps
                    check_phase = False
                    step_counter =-1
        step_counter += 1

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}

    return [state, epoch_metrics_np, step_counter, avg_params, avg_valid_error, check_phase, learning_rate]

def update_avg_params(avg_params, state):
    return jax.tree_map(lambda p, g: 0.999 * p + 0.001 * g, avg_params, state.params)

@partial(jax.jit, static_argnums=2)
def train_step_fn_sl(state: TrainState,
                  batch: Dict,
                  loss_fn: Callable) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    """
    Training step.
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data
        loss_fn (Callable): Loss function
    Returns: Updated optimizer state and loss for current batch.
    """
    (loss, train_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, train_metrics


def valid_epoch_sl(state: TrainState,
                ds: DataTupleT,
                metric_fn: LossFn,
                bs: int) -> Dict[str, float]:
    """
    Validation epoch for NN training.
    Args:
        state (TrainState): Flax train state.
        ds (Tuple): Validation data. First entry is input, second is expected output
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
        bs (int): Batch size.
    Returns: Validation metric.
    """
    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    batch_metrics = []
    idxs = jnp.arange(n_data)
    idxs = idxs[:steps_per_epoch * bs]  # skip incomplete batch
    idxs = idxs.reshape((steps_per_epoch, bs))
    for idx in idxs:
        batch = jax.tree_map(lambda y: y[idx, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        metrics = valid_step_fn_sl(state, batch, metric_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return epoch_metrics_np


@partial(jax.jit, static_argnums=2)
def valid_step_fn_sl(state: TrainState,
                  batch: DataTupleT,
                  metric_fn: Callable) -> Dict[str, jnp.ndarray]:
    """
    Validation step.
    Args:
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data.
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
    Returns: Validation metrics on the batch.
    """
    _, metrics = metric_fn(state.params, batch)
    return metrics

def valid_epoch_sl_avg(params: dict,
                ds: DataTupleT,
                metric_fn: LossFn,
                bs: int) -> Dict[str, float]:
    """
    Validation epoch for NN training.
    Args:
        state (TrainState): Flax train state.
        ds (Tuple): Validation data. First entry is input, second is expected output
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
        bs (int): Batch size.
    Returns: Validation metric.
    """
    inputs, targets = ds
    _k = list(inputs.keys())
    n_data = len(inputs[_k[0]])

    steps_per_epoch = n_data // bs
    batch_metrics = []
    idxs = jnp.arange(n_data)
    idxs = idxs[:steps_per_epoch * bs]  # skip incomplete batch
    idxs = idxs.reshape((steps_per_epoch, bs))
    for idx in idxs:
        batch = jax.tree_map(lambda y: y[idx, ...], ds)
        # batch = (Dict[str, Array[perm, ...]], Dict[str, Array[perm, ...]])
        metrics = valid_step_fn_sl_avg(params, batch, metric_fn)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return epoch_metrics_np


@partial(jax.jit, static_argnums=2)
def valid_step_fn_sl_avg(params: dict,
                  batch: DataTupleT,
                  metric_fn: Callable) -> Dict[str, jnp.ndarray]:
    """
    Validation step.
    Args:
        state (TrainState): Flax train state.
        batch (Tuple): Batch of validation data.
        metric_fn (Callable): Function that evaluates the model wrt some metric fn
    Returns: Validation metrics on the batch.
    """
    _, metrics = metric_fn(params, batch)
    return metrics
