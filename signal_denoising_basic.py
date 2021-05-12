import os
import time
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')

from neigh_gf_src.model import Model, ADAM

from neigh_gf_src import datasets
from neigh_gf_src.arch import GCNN

# TODO: Optimize for GPU

# Parameters

VERB = True
ARCH_INFO = True

signals = {}
signals['N_samples'] = 2000
signals['N_graphs'] = 20
signals['L_filter'] = 6
signals['noise'] = 0
signals['test_only'] = False

signals['perm'] = True
signals['pct'] = True
if signals['pct']:
    signals['eps1'] = 10
    signals['eps2'] = 10
else:
    signals['eps1'] = 0.1
    signals['eps2'] = 0.3

signals['median'] = True

# Graph parameters
G_params = {}
G_params['type'] = datasets.SBM
G_params['N'] = N = 128
G_params['k'] = k = 4
G_params['p'] = 0.3
G_params['q'] = [[0, 0.0075, 0, 0.0],
                 [0.0075, 0, 0.004, 0.0025],
                 [0, 0.004, 0, 0.005],
                 [0, 0.0025, 0.005, 0]]
G_params['type_z'] = datasets.RAND
signals['g_params'] = G_params

# NN Parameters
nn_params = {}
nn_params['F'] = [1, 2, 4, 8, 4, 2, 1]
nn_params['K'] = 3
nn_params['M'] = [128, 256, N]
nonlin_s = "tanh"
if nonlin_s == "relu":
    nn_params['nonlin'] = nn.ReLU
elif nonlin_s == "tanh":
    nn_params['nonlin'] = nn.Tanh
elif nonlin_s == "sigmoid":
    nn_params['nonlin'] = nn.Sigmoid
else:
    nn_params['nonlin'] = None
nn_params['last_act_fn'] = nn.Tanh
nn_params['batch_norm'] = True
nn_params['arch_info'] = ARCH_INFO

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB


if __name__ == '__main__':

    G = datasets.create_graph(G_params)

    # Define the data model
    data = datasets.DenoisingSparse(G,
                                    signals['N_samples'],
                                    signals['L_filter'], signals['g_params']['k'],  # k is n_delts
                                    0.025,
                                    median=signals['median'])
    #data.to_unit_norm()
    data.to_tensor()

    G.compute_laplacian('normalized')
    archit = GCNN(G.L.todense(), nn_params['F'], nn_params['K'], nn_params['M'], nn_params['nonlin'], ARCH_INFO)

    model_params['arch'] = archit

    model = Model(**model_params)
    t_init = time.time()
    epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    t_conv = time.time() - t_init
    mean_err, med_err, mse = model.test(data.test_X, data.test_Y)

    print("DONE: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
        mse, mean_err, med_err, model.count_params(), round(t_conv, 4), epochs
    ), flush=True)
