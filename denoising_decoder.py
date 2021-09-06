import copy

import numpy as np
import time
import torch
from torch import nn
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import shortest_path

from neigh_gf_src import datasets as ds
from neigh_gf_src.arch import GraphDecoder, GraphDeepDecoder
from neigh_gf_src.model import ModelSamuel

def create_filter(S, ps, x=None):
    if ps['type'] == 'BLH':
        _, V = ordered_eig(S)
        V = np.real(V)
        eigvalues = np.ones(V.shape[0])*0.001
        bl_k = int(S.shape[0]*ps['k'])
        if ps['firsts']:
            eigvalues[:bl_k] = 1
        else:
            x_freq = V.T.dot(x)
            idx = np.flip(np.abs(x_freq).argsort(), axis=0)[:bl_k]
            eigvalues[idx] = 1

        H = V.dot(np.diag(eigvalues).dot(V.T))
    elif ps['type'] == 'RandH':
        hs = np.random.rand(ps['K'])
        hs /= np.sum(hs)
    elif ps['type'] == 'FixedH':
        hs = ps['hs']
    else:
        print('Unkwown filter type')
        return None

    if ps['neigh']:
        H = np.zeros((S.shape))
        distances = shortest_path(S, directed=False, unweighted=True)
        for l, h in enumerate(hs):
            H += h*(distances == l).astype(int)
    elif ps['type'] != 'BLH':
        H = np.zeros((S.shape))
        for l, h in enumerate(hs):
            H += h*np.linalg.matrix_power(S, l)

    if ps['H_norm']:
        H /= np.linalg.norm(H)

    return H

SEED = 10
torch.manual_seed(SEED)
np.random.seed(SEED)

# Graph parameters
G_params = {}
G_params['type'] = ds.SBM
G_params['k'] = 8
G_params['type_z'] = ds.CONT
G_params['N'] = 256
G_params['p'] = 0.75
G_params['q'] = 0.0075

G = ds.create_graph(G_params, SEED)
S = np.asarray(G.W.todense())

hs = [0.3, 0.5, 0.2]

h_params = {'type': 'FixedH', 'hs': hs, 'H_norm': False, 'neigh': False}
H = create_filter(S, h_params)

h_params_n = {'type': 'FixedH', 'hs': hs, 'H_norm': False, 'neigh': True}
Hn = create_filter(S, h_params_n)

Exps = [{'name': "DecoderH", 'H': H, 'fts': 150, 'std': 0.1, 'legend': '2L-DecH'},
        {'name': "DecoderHn", 'H': Hn, 'fts': 150, 'std': 0.1, 'legend': '2L-DecHn'},
        {'name': "DeepDecoderH", 'H': H, 'fts': [97]*5 + [1], 'nodes':[256]*6, 'in_std': 0.1, 'w_std': 0.1, 'legend': 'DeepDecH'},
        {'name': "DeepDecoderHn", 'H': Hn, 'fts': [97]*5 + [1], 'nodes':[256]*6, 'in_std': 0.1, 'w_std': 0.1, 'legend': 'DeepDecHn'}
        ]

n_signals = 20
epochs = 500
lr = 0.001

p_n = [0., .025, .05, .075, .1]

err = np.zeros((len(p_n), 2, len(Exps), n_signals, epochs))
perf = np.zeros((len(p_n), 2, len(Exps), n_signals))
for k, pn in enumerate(p_n):
    for i in range(n_signals):
        w = np.random.randn(G.N)
        x1 = H@w
        x2 = Hn@w
        x1_p = np.linalg.norm(x1)**2
        x2_p = np.linalg.norm(x2)**2

        noise = np.random.randn(G.N)
        x1_n = x1 + noise*np.sqrt(x1_p*pn/x1.size)
        x2_n = x2 + noise*np.sqrt(x2_p*pn/x2.size)

        for j, exp in enumerate(Exps):
            if "Deep" in exp['name']:
                dec = GraphDeepDecoder(exp['fts'], exp['nodes'], exp['H'], act_fn=nn.ReLU(),
                               last_act_fn=None, input_std=exp['in_std'],
                               w_std=exp['w_std'])
            else:
                dec = GraphDecoder(exp['fts'], exp['H'], exp['std'])

            model1 = ModelSamuel(copy.deepcopy(dec), epochs=epochs, learning_rate=lr, verbose=True)
            model2 = ModelSamuel(copy.deepcopy(dec), epochs=epochs, learning_rate=lr, verbose=True)

            _, err[k, 0, j, i, :], _ = model1.fit(x1_n, x1)/x1_p
            _, err[k, 1, j, i, :], _ = model2.fit(x2_n, x2)/x2_p

            _, perf[k, 0, j, i] = model1.test(x1)
            _, perf[k, 1, j, i] = model2.test(x2)

        print(i, end=' ')
    print("Ended {}".format(pn))

timestr = time.strftime("%Y%m%d-%H%M")
np.save("results/" + timestr + "-err", err)
np.save("results/" + timestr + "-perf", perf)
with open(timestr + "-leg.txt", 'w') as f:
    f.write("p_n x H or Hn in the input x H or Hn in the filter x n_signal (x epoch)\n")

