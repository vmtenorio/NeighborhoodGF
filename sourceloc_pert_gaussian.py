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
from neigh_gf_src.arch import GCNN_GF, MLP

# Parameters

VERB = True
ARCH_INFO = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

signals = {}
signals['N_samples'] = 4000
signals['N_graphs'] = 10
signals['pos_value'] = .15
signals['neg_value'] = -0.15
signals['diffusion'] = 5
signals['median'] = True
signals['perm'] = True

# Graph parameters
G_params = {}
G_params['type'] = datasets.SBM
G_params['N'] = N = 256
G_params['k'] = k = 4
G_params['p'] = 0.8
G_params['q'] = [[0, 0.0075, 0, 0.0],
                 [0.0075, 0, 0.004, 0.0025],
                 [0, 0.004, 0, 0.005],
                 [0, 0.0025, 0.005, 0]]
G_params['q'] = 0.2
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
model_params['max_non_dec'] = 15
model_params['min_es'] = 40
model_params['verbose'] = VERB

EXPS = [
    {
        'name': "NeighborhoodGF",
        'gf_type': "NeighborhoodGF",
        'F': [1, 32, 32],
        'K': 3,
        'bias_gf': True,
        'M': [16, k],
        'bias_mlp': False,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "NeighborhoodGF-Binarization",
        'gf_type': "NeighborhoodGFType2",
        'F': [1, 32, 32],
        'K': 3,
        'bias_gf': True,
        'M': [16, k],
        'bias_mlp': False,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "ClassicGF",
        'gf_type': "ClassicGF",
        'F': [1, 32, 32],
        'K': 3,
        'bias_gf': True,
        'M': [16, k],
        'bias_mlp': False,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    },
    {
        'name': "BasicMLP",
        'M': [N, 256, 128, 64, 32, k],
        'bias_mlp': True,
        'nonlin': nn.Tanh,
        'nonlin_s': "tanh", # For logging purposes
        'arch_info': ARCH_INFO
    }
]

prob_list = [0, 5, 10, 15, 20]

def test_arch(signals, nn_params, model_params, prob, device):

    loss = np.zeros(signals['N_graphs'])
    acc = np.zeros(signals['N_graphs'])
    epochs = np.zeros(signals['N_graphs'])
    t_train = np.zeros(signals['N_graphs'])
    train_err = np.zeros((signals['N_graphs'], model_params['epochs']))
    val_err = np.zeros((signals['N_graphs'], model_params['epochs']))

    for i in range(signals['N_graphs']):

        Ga, Gb = datasets.perturbated_graphs(signals['g_params'],
                                             creat=prob, dest=prob,
                                             pct=True, perm=signals['perm']
                                            )

        # Define the data model
        data = datasets.SourcelocSyntheticGaussian(Ga, # Using the first graph to create the signal
                                            signals['N_samples'],
                                            pos_value=signals['pos_value'],
                                            neg_value=signals['neg_value'],
                                            diffusion=signals['diffusion'],
                                            median=signals['median']
                                          )
        #data.to_unit_norm()
        #data.add_noise(p_n, test_only=True)
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
            archit = GCNN_GF(datasets.norm_graph(Gb.W.todense()),
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
        loss[i], acc[i] = model.test(data.test_X, data.test_Y, regression=False)

        print("DONE {}: CELoss={} - Accuracy={} - Params={} - t_conv={} - epochs={}".format(
            i+1, loss[i], 100*acc[i], model.count_params(), round(t_train[i], 4), epochs[i]
        ), flush=True)

    results = {
            "n_params": model.count_params(),
            "train_time": np.mean(t_train),
            "loss": np.mean(loss),
            "loss_std": np.std(loss),
            "loss_full": loss.tolist(),
            "acc": np.mean(acc),
            "acc_std": np.std(acc),
            "acc_full": acc.tolist(),
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
        for prob in prob_list:
            print("***************************")
            print("Starting with prob: ", str(prob))
            results_exp[prob] = test_arch(signals, exp, model_params, prob, device)

        results[exp['name']]["results"] = results_exp

    save_results(time.strftime("%Y%m%d-%H%M") + ".json", results)

