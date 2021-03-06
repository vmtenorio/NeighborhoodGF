import time
import torch.nn as nn
import numpy as np
import torch
import sys
sys.path.append('..')

from neigh_gf_src.model import Model, ADAM

from neigh_gf_src import datasets
from neigh_gf_src.arch import GCNN_GF

# Parameters

VERB = True
ARCH_INFO = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

signals = {}
signals['N_samples'] = 2000
signals['min_l'] = 10
signals['max_l'] = 25
signals['median'] = True

# Graph parameters
G_params = {}
G_params['type'] = datasets.SBM
G_params['N'] = N = 256
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
nn_params['gf_type'] = "NeighborhoodGF"
nn_params['F'] = [1, 2, 4, 8, 16, 16]
nn_params['K'] = 3
nn_params['M'] = [128, 64, 32, k]
nn_params['bias_mlp'] = True
nonlin_s = "tanh"
if nonlin_s == "relu":
    nn_params['nonlin'] = nn.ReLU
elif nonlin_s == "tanh":
    nn_params['nonlin'] = nn.Tanh
elif nonlin_s == "sigmoid":
    nn_params['nonlin'] = nn.Sigmoid
else:
    nn_params['nonlin'] = None
nn_params['last_act_fn'] = nn.Softmax
nn_params['batch_norm'] = True
nn_params['arch_info'] = ARCH_INFO

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.CrossEntropyLoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 20
model_params['verbose'] = VERB


if __name__ == '__main__':

    G = datasets.create_graph(G_params)

    # Define the data model
    data = datasets.SourcelocSynthetic(G,
                                        signals['N_samples'],
                                        min_l=signals['min_l'],
                                        max_l=signals['max_l'],
                                        median=signals['median'])
    #data.to_unit_norm()
    data.to_tensor()
    data.to(device)

    G.compute_laplacian('normalized')
    archit = GCNN_GF(datasets.norm_graph(G.W.todense()),
                    nn_params['gf_type'],
                    nn_params['F'],
                    nn_params['K'],
                    nn_params['M'],
                    nn_params['bias_mlp'],
                    nn_params['nonlin'],
                    ARCH_INFO
                )

    model_params['arch'] = archit

    archit.to(device)

    model = Model(**model_params)
    t_init = time.time()
    epochs, _, _ = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
    t_conv = time.time() - t_init
    loss, acc = model.test(data.test_X, data.test_Y, regression=False)

    print("DONE: CELoss={} - Accuracy={}% - Params={} - t_conv={} - epochs={}".format(
        loss, acc*100, model.count_params(), round(t_conv, 4), epochs
    ), flush=True)
