# This is an exemplary training of ceq net

import logging


logging.basicConfig(level=logging.INFO)

import jax
from ceqnet.src.io import last_module, read_json, load_fg_test_data
from ceqnet.src.logging.log_metrics import log_metrics
from ceqnet.src.inference import evaluate_model
from ceqnet.src.nn.ceqnet import init_ceq_net, get_obs_fn, get_obs_and_force_fn
from ceqnet.src.inference import mae_metric, mse_metric, rmse_metric, get_n_params
from ceqnet.src.indexing import get_indices
from flax.training import checkpoints
from functools import partial
import os

print(os.getcwd())

# Set data and save paths
data_path = '/Users/martin/Master/data/fourth_gen/carbon_chain.npz'
#save_path = '/Users/martin/Master/projects/qFE/carbon_qFE_low_weights/redisrcut7.0l2'
#ckpt_dir = last_module(save_path+'/module')

# We use the current working directory as checkpoint directory
ckpt_dir = os.getcwd()
#ckpt_dir = save_path + '/module_2'



# Define prop_keys
prop_keys = {'energy': 'E',
             'force': 'F',
             'atomic_type': 'z',
             'atomic_position': 'R',
             'hirshfeld_volume': None,
             'total_charge': 'Q',
             'total_spin': None,
             'partial_charge': 'q',
             'properties': 'p'
             }

h = read_json(ckpt_dir+'/hyperparameters.json')

train_split = read_json(ckpt_dir + '/training_split.json')

ckpt = checkpoints.restore_checkpoint(ckpt_dir, prefix='checkpoint_q_', target=None)

net = init_ceq_net(h)


obs_fn = get_obs_fn(net, prop_keys['partial_charge'])
#obs_fn = get_obs_and_force_fn(net)
obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))

test_data = load_fg_test_data(data_path, train_split, prop_keys)

idx = get_indices(test_data['R'], test_data['z'], h['ceq_net']['geometry_embeddings'][0]['geometry_embed']['r_cut'])
test_input = ({'Q': test_data['Q'], 'R': test_data['R'], 'z': test_data['z'], 'idx_i': idx['idx_i'], 'idx_j': idx['idx_j']}, {'q': test_data['q']})
batch_size = 10
metric_fn = {'mae': partial(mae_metric, pad_value=0), 'mse': partial(mse_metric, pad_value=0), 'rmse': partial(rmse_metric, pad_value=0)}

n_params = get_n_params(ckpt['params'])

metrics = evaluate_model(ckpt['params'], obs_fn, test_input, batch_size, metric_fn)

from pprint import pprint
pprint(metrics)

log_metrics(metrics, ckpt_dir, h, n_params=n_params)