import time
import torch.nn as nn
import numpy as np
import os
import json
import torch
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
signals['N_graphs'] = 1
signals['L_filter'] = 10
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
G_params['q'] = np.array([[0, 0.0075, 0, 0.0],
                          [0.0075, 0, 0.004, 0.0025],
                          [0, 0.004, 0, 0.005],
                          [0, 0.0025, 0.005, 0]])
# For the smaller graph
G_params['q'] = np.array([[0, 0.075, 0, 0.0],
                          [0.075, 0, 0.04, 0.025],
                          [0, 0.04, 0, 0.05],
                          [0, 0.025, 0.05, 0]])
G_params['type_z'] = datasets.RAND
signals['g_params'] = G_params

# Model parameters
model_params = {}
model_params['opt'] = ADAM
model_params['learning_rate'] = 0.001
model_params['decay_rate'] = 0.99
model_params['loss_func'] = nn.CrossEntropyLoss()
model_params['epochs'] = 200
model_params['batch_size'] = 50
model_params['eval_freq'] = 4
model_params['max_non_dec'] = 10
model_params['verbose'] = VERB

EXPS = [
    {
        "name": "NeighborhoodGF",
        "gf_type": "NeighborhoodGF",
        'F': [1, 2, 4, 8, 4, 2, 1],
        'K': 3,
        'M': [128, 256, k],
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'batch_norm': True,
        'arch_info': ARCH_INFO
    },
    {
        "name": "NeighborhoodGF-Binarization",
        "gf_type": "NeighborhoodGFType2",
        'F': [1, 2, 4, 8, 4, 2, 1],
        'K': 3,
        'M': [128, 256, k],
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'batch_norm': True,
        'arch_info': ARCH_INFO
    },
    {
        "name": "ClassicGF",
        "gf_type": "ClassicGF",
        'F': [1, 2, 4, 8, 4, 2, 1],
        'K': 3,
        'M': [128, 256, k],
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'batch_norm': True,
        'arch_info': ARCH_INFO
    }
]

k_list = [4, 8, 16, 32]

def test_arch(signals, nn_params, model_params, k, device):

    loss = np.zeros(signals['N_graphs'])
    acc = np.zeros(signals['N_graphs'])
    epochs = np.zeros(signals['N_graphs'])
    t_train = np.zeros(signals['N_graphs'])
    train_err = np.zeros((signals['N_graphs'], model_params['epochs']))
    val_err = np.zeros((signals['N_graphs'], model_params['epochs']))

    g_params = signals['g_params'].copy()

    g_params['k'] = k
    repeat_times = k // g_params['q'].shape[0]
    g_params['q'] = np.tile(g_params['q'], (repeat_times, repeat_times))
    nn_params['M'][-1] = k

    for i in range(signals['N_graphs']):

        G = datasets.create_graph(g_params)

        # Define the data model
        data = datasets.SourcelocSynthetic(G,
                                            signals['N_samples'],
                                            min_l=10, max_l=25,
                                            median=signals['median'])
        #data.to_unit_norm()
        data.to_tensor()
        data.to(device)

        G.compute_laplacian('normalized')
        archit = GCNN(G.L.todense(),
                      nn_params['gf_type'],
                      nn_params['F'],
                      nn_params['K'],
                      nn_params['M'],
                      nn_params['nonlin'],
                      ARCH_INFO
                    )

        archit.to(device)

        model_params['arch'] = archit
        print(data.train_X.size(), data.train_Y.max(), nn_params['M'])

        model = Model(**model_params)
        t_init = time.time()
        epochs[i], train_err[i,:], val_err[i,:] = model.fit(data.train_X, data.train_Y, data.val_X, data.val_Y)
        t_train[i] = time.time() - t_init
        loss[i], acc[i] = model.test(data.test_X, data.test_Y, regression=False)

        print("DONE {}: CELoss={} - Accuracy={} - Params={} - t_conv={} - epochs={}".format(
            i, loss[i], 100*acc[i], model.count_params(), round(t_train[i], 4), epochs[i]
        ), flush=True)

    results = {
            "n_params": model.count_params(),
            "train_time": np.mean(t_train),
            "loss": np.mean(loss),
            "acc": np.mean(acc),
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
        results_exp = {}
        for k in k_list:
            print("***************************")
            print("Starting with k: ", str(k))
            results_exp[k] = test_arch(signals, exp, model_params, k, device)

        results[exp['name']]["results"] = results_exp

    save_results(time.strftime("%Y%m%d-%H%M") + ".json", results)
