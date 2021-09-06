import time
import torch.nn as nn
import numpy as np
import os
import json
import torch
from scipy.sparse.csgraph import shortest_path
import sys
sys.path.append('..')

from neigh_gf_src.model import Model, ADAM

from neigh_gf_src import datasets
from neigh_gf_src.arch import GCNN

# Parameters

VERB = True
ARCH_INFO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

signals = {}
signals['N_samples'] = 2000
signals['N_graphs'] = 10
signals['L_filter'] = 6
signals['noise'] = 0
signals['test_only'] = False

# Graph parameters
G_params = {}
G_params['type'] = datasets.SBM
G_params['N'] = N = 256
G_params['k'] = k = 4
G_params['p'] = 0.8
G_params['q'] = 0.1
G_params['type_z'] = datasets.RAND
signals['g_params'] = G_params

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.MSELoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 200
model_params['es_loss_type'] = "train"
model_params['verbose'] = VERB

EXPS = [
    {
        'name': "BasicGNN-ClassicGF",
        'gf_type': "BasicGNN",
        'F': [1, 32, 128, 32, 1],
        'K': 3,
        'bias_gf': False,
        'M': [],
        'bias_mlp': True,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "BasicGNN-NeighborhoodGF",
        'gf_type': "BasicGNN",
        'F': [1, 32, 128, 32, 1],
        'K': 3,
        'bias_gf': False,
        'M': [],
        'bias_mlp': True,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "NeighborhoodGF",
        'gf_type': "NeighborhoodGF",
        'F': [1, 4, 8, 4, 1],
        'K': 3,
        'bias_gf': False,
        'M': [],
        'bias_mlp': True,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "ClassicGF",
        'gf_type': "ClassicGF",
        'F': [1, 4, 8, 4, 1],
        'K': 3,
        'bias_gf': False,
        'M': [],
        'bias_mlp': True,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "BasicMLP",
        'M': [N, 128, 64, 64, 32, k],
        'bias_mlp': False,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    }
]
"""
{
    'name': "GraphDecoder",
    'gf_type': "Classic",
    'F': 150,
    'K': 3,
    'nonlin': nn.Tanh,
    'nonlin_s': "tanh", # For logging purposes
},
{
    'name': "GraphDeepDecoder",
    'gf_type': "Classic",
    'F': [1, 32, 128, 32, 1],
    'K': 3,
    'nonlin': nn.Tanh,
    'nonlin_s': "tanh", # For logging purposes
},
{
    'name': "GraphDecoder",
    'gf_type': "Neighborhood",
    'F': 150,
    'K': 3,
    'nonlin': nn.Tanh,
    'nonlin_s': "tanh", # For logging purposes
},
{
    'name': "GraphDeepDecoder",
    'gf_type': "Neighborhood",
    'F': [1, 32, 128, 32, 1],
    'K': 3,
    'nonlin': nn.Tanh,
    'nonlin_s': "tanh", # For logging purposes
},
"""

def build_filter(S, K, ftype="classic"):
    H = np.zeros(N,N)
    if ftype == "classic":
        S = norm_graph(self.G.W.todense())
        for l in range(K):
            H += np.linalg.matrix_power(S, l)
    else:
        distances = shortest_path(self.G.W.todense(), directed=False, unweighted=True)
        for i in range(K):
            H += (distances == i).astype(int)

p_n_list = [0, .025, .05, 0.075, .1]

def test_arch(signals, nn_params, model_params, p_n, device, input_ftype):

    mse = np.zeros(signals['N_graphs'])
    mean_err = np.zeros(signals['N_graphs'])
    med_err = np.zeros(signals['N_graphs'])
    epochs = np.zeros(signals['N_graphs'])
    t_train = np.zeros(signals['N_graphs'])
    train_err = np.zeros((signals['N_graphs'], model_params['epochs']))
    val_err = np.zeros((signals['N_graphs'], model_params['epochs']))

    for i in range(signals['N_graphs']):

        G = datasets.create_graph(signals['g_params'])

        # Define the data model
        data = datasets.DenoisingSparse(G,
                                        signals['N_samples'],
                                        signals['L_filter'], signals['g_params']['k'],  # k is n_delts
                                        p_n,
                                        median=signals['median'],
                                        ftype=input_ftype)
        #data.to_unit_norm()
        data.to_tensor()
        data.to(device)

        #G.compute_laplacian('normalized')
        if "MLP" in nn_params['name']:
            archit = MLP(nn_params['M'],
                        nn_params['bias_mlp'],
                        nn_params['nonlin'],
                        ARCH_INFO
                        )
        else:
            archit = GCNN(datasets.norm_graph(G.W.todense()),
                        nn_params['gf_type'],
                        nn_params['F'],
                        nn_params['K'],
                        nn_params['bias_gf'],
                        nn_params['M'],
                        nn_params['bias_mlp'],
                        nn_params['nonlin'],
                        ARCH_INFO
                        )

        archit.to(device)

        model_params['arch'] = archit

        model = Model(**model_params)
        t_init = time.time()
        epochs[i], train_err[i,:], val_err[i,:] = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
        t_train[i] = time.time() - t_init
        mean_err[i], med_err[i], mse[i] = model.test(data.test_X, data.test_Y)

        print("DONE {}: MSE={} - Mean Err={} - Median Err={} - Params={} - t_conv={} - epochs={}".format(
            i+1, mse[i], mean_err[i], med_err[i], model.count_params(), round(t_train[i], 4), epochs[i]
        ), flush=True)

    results = {
            "n_params": model.count_params(),
            "train_time": np.mean(t_train),
            "mse": np.mean(mse),
            "mean_err": np.mean(mean_err),
            "median_err": np.median(med_err),
            "epochs": np.mean(epochs),
            "train_err": np.mean(train_err, axis=0).tolist(),
            "val_error": np.mean(val_err, axis=0).tolist()
        }
    return results

def save_results(path, results):
    if not os.path.exists('results'):
        os.mkdir('results')
    with open('results/' + path, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':

    results = {}
    for exp in EXPS:

        print("***************************")
        print("Starting " + exp['name'])
        results[exp['name']] = exp.copy()
        del results[exp['name']]['nonlin'] # Not possible to be saved in json format
        for iftype in ["classic", "neighbours"]:
            results_exp = {}
            for p_n in p_n_list:
                print("***************************")
                print("Starting with p_n: ", str(p_n))
                results_exp[p_n] = test_arch(signals, exp, model_params, p_n, device, iftype)

            results[exp['name']][iftype] = results_exp

    save_results(time.strftime("%Y%m%d-%H%M") + ".json", results)









