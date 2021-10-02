import dgl.data
from neigh_gf_src import graphutils as gu

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

N_SIGNALS = 25
N_EPOCHS = 100
LR = 0.01
K_MAX = 10

def train(g, model, verb=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    features = features.to(device)
    labels = labels.to(device)

    losses = np.zeros(N_EPOCHS)
    for e in range(N_EPOCHS):
        # Forward
        logits = model(features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])

        losses[e] = loss

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0 and verb:
            print('Epoch {}/{} - Loss: {:.3f} - Val acc: {:.3f} (best {:.3f})\
                     - Test acc: {:.3f} (best {:.3f})'.format(
                    e, N_EPOCHS, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    
    return losses, best_val_acc, best_test_acc

class GNN(nn.Module):
    def __init__(self,
                H,
                #Spow,
                F,          # Features in each graph filter layer (list)
                nonlin,     # Non linearity function
                arch_info): # Print architecture information
        super(GNN, self).__init__()

        #self.Spow = Spow
        self.H = H
        self.N = H.shape[1]
        self.F = F
        self.nonlin = nonlin

        # Define the layer
        # Grahp Filter Layers
        gfl = []
        for l in range(len(self.F)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gfl.append(GNNStep(self.H, self.F[l], self.F[l+1]))
            if l < len(self.F)-2:
                gfl.append(self.nonlin())

        self.GFL = nn.Sequential(*gfl)

        if arch_info:
            print("Architecture:")
            print("Graph N_nodes: {}".format(self.N))
            print("F: {}".format(self.F))
            print("Non lin: " + str(self.nonlin))

    def forward(self, x):

        Fin = x.shape[1]
        xN = x.shape[0]
        assert Fin == self.F[0]

        assert xN == self.N

        # Define the forward pass
        # Graph filter layers
        # Goes from TxF[0]xN to TxF[-1]xN with GFL
        y = self.GFL(x)

        return y

class GNNStep(nn.Module):
    def __init__(self, H, Fin, Fout):

        super(GNNStep, self).__init__()

        #self.Spow = nn.Parameter(Spow, requires_grad=False)
        self.H = nn.Parameter(H, requires_grad=False)
        self.N = H.shape[1]
        self.Fin = Fin
        self.Fout = Fout

        #self.h = nn.Parameter(torch.Tensor(self.K))
        #stdh = 1. / np.sqrt(self.K)
        #self.h.data.uniform_(-stdh, stdh)
        #self.h.data.fill_(1.)

        self.weights = nn.Parameter(torch.Tensor(self.Fin, self.Fout))
        stdv = 1. / np.sqrt(self.Fin * self.Fout)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        xN, xFin = x.shape
        assert xFin == self.Fin
        assert xN == self.N

        #H = torch.zeros((self.N, self.N), device=self.device)
        #for k in range(self.K):
        #    H += self.h[k] * self.Spow[k,:,:]

        return self.H @ x @ self.weights

def results_to_dict(losses, val_acc, test_acc):
    rdict = {}
    rdict['mean_loss'] = np.mean(losses, 1).tolist()
    rdict['median_loss'] = np.median(losses, 1).tolist()
    rdict['mean_val_acc'] = np.mean(val_acc)
    rdict['median_val_acc'] = np.median(val_acc)
    rdict['mean_test_acc'] = np.mean(test_acc)
    rdict['median_test_acc'] = np.median(test_acc)
    return rdict

def test_arch(arch_name):
    dataset = getattr(dgl.data, arch_name)()
    g = dataset[0]

    n_classes = dataset.num_classes
    n_feat = g.ndata['feat'].shape[1]

    A = g.adjacency_matrix().to_dense()#.to('cuda')
    d = torch.real(torch.linalg.eigvals(A))
    dmax = torch.max(d)
    S = A / dmax
    N = S.shape[0]

    nfilter = gu.NeighborhoodGF(S.numpy(), K_MAX, arch_name, device)

    Spow = gu.calc_powers(S, K_MAX, device)
    Spown = nfilter.H

    results_exp = {}
    for j, exp in enumerate(EXPS):
        print("Experiment {}/{} - ".format(j+1, len(EXPS)), end='')
        results_exp[j] = exp.copy()
        del results_exp[j]['nonlin']

        F = exp['F']
        F[0] = n_feat
        F[-1] = n_classes
        K = exp['K']
        nonlin= exp['nonlin']

        # For simplicity, assuming constant and all 1s filter coefficients
        H = torch.sum(Spow[:K,:,:], axis=0)
        Hn = torch.sum(Spown[:K,:,:], axis=0)

        losses_H = np.zeros((N_EPOCHS, N_SIGNALS))
        val_acc_H = np.zeros((N_SIGNALS))
        test_acc_H = np.zeros((N_SIGNALS))
        losses_Hn = np.zeros((N_EPOCHS, N_SIGNALS))
        val_acc_Hn = np.zeros((N_SIGNALS))
        test_acc_Hn = np.zeros((N_SIGNALS))
        for i in range(N_SIGNALS):
            model1 = GNN(H, F, nonlin, False)
            model2 = GNN(Hn, F, nonlin, False)
            model1.to(device)
            model2.to(device)

            losses_H[:,i], val_acc_H[i], test_acc_H[i] = train(g, model1, False)
            losses_Hn[:,i], val_acc_Hn[i], test_acc_Hn[i] = train(g, model2, False)
            print(i+1, end=' ', flush=True)

        results_exp[j]['H'] = results_to_dict(losses_H, val_acc_H, test_acc_H)
        results_exp[j]['Hn'] = results_to_dict(losses_Hn, val_acc_Hn, test_acc_Hn)
        print("DONE", flush=True)
        
    return results_exp


EXPS = [
    {
        'F': [None, 512, None],
        'K': 2,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 3,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 4,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 5,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 6,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 7,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 8,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 9,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    },
    {
        'F': [None, 512, None],
        'K': 10,
        'nonlin': nn.ReLU,
        'nonlin_str': "relu"
    }
]

def save_results(path, results):
    if not os.path.exists('results'):
        os.mkdir('results')
    with open('results/' + path, 'w') as f:
        json.dump(results, f)


arch_names = ["Cora", "Citeseer", "Pubmed"]
results = {}
for i, name in enumerate(arch_names):
    print("****************************")
    print("Starting {} - ".format(name))
    results[name] = test_arch(name+"GraphDataset")

save_results(time.strftime("%Y%m%d-%H%M") + ".json", results)

