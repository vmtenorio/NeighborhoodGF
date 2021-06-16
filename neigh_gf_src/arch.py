import math
import numpy as np

import torch.nn as nn
import torch
from torch import Tensor, eye, manual_seed, nn, no_grad, optim
from . import layers


class GCNN(nn.Module):
    def __init__(self,
                S,
                gf_type,    # Type of graph filter to use
                F,          # Features in each graph filter layer (list)
                K,          # Filter taps in each graph filter layer
                M,          # Neurons in each fully connected layer (list)
                bias_mlp,   # Whether or not to use Bias in the FC layers
                nonlin,     # Non linearity function
                arch_info): # Print architecture information
        super(GCNN, self).__init__()
        # In python 3
        # super()

        # Define parameters
        #if type(S) != torch.FloatTensor:
        #    self.S = torch.FloatTensor(S)
        #else:
        #    self.S = S
        self.S = S
        self.N = S.shape[0]
        self.F = F
        self.K = K
        self.M = M
        self.nonlin = nonlin

        self.gf = getattr(layers, gf_type)

        # Define the layer
        # Grahp Filter Layers
        gfl = []
        for l in range(len(self.F)-1):
            # print("Graph filter layer: " + str(l))
            # print(str(self.F[l]) + ' x ' + str(self.F[l+1]))
            gfl.append(self.gf(self.S, self.F[l], self.F[l+1], self.K))
            gfl.append(self.nonlin())

        self.GFL = nn.Sequential(*gfl)

        # Fully connected Layers
        fcl = []
        firstLayerIn = self.N*self.F[-1]
        if len(self.M) > 0:
            # As last layer has no nonlin (if its softmax is done later, etc.)
            # define here the first layer before loop
            fcl.append(nn.Linear(firstLayerIn, self.M[0], bias=bias_mlp))
            for m in range(1,len(self.M)):
                # print("FC layer: " + str(m))
                # print(str(self.M[m-1]) + ' x ' + str(self.M[m]))
                fcl.append(self.nonlin())
                fcl.append(nn.Linear(self.M[m-1], self.M[m], bias=bias_mlp))

        self.FCL = nn.Sequential(*fcl)

        if arch_info:
            print("Architecture:")
            print("Graph N_nodes: {}".format(self.N))
            print("F: {}, K: {}, M: {}".format(self.F, self.K, self.M))
            print("Non lin: " + str(self.nonlin))

    def forward(self, x):

        # Check type
        # if type(x) != torch.FloatTensor:
        #     x = torch.FloatTensor(x)

        # Params
        T = x.shape[0]
        try:
            Fin = x.shape[1]
            xN = x.shape[2]
            assert Fin == self.F[0]
        except IndexError:
            xN = x.shape[1]
            Fin = 1
            x = x.unsqueeze(1)
            assert self.F[0] == 1

        assert xN == self.N

        # Define the forward pass
        # Graph filter layers
        # Goes from TxF[0]xN to TxF[-1]xN with GFL
        y = self.GFL(x)

        # return y.squeeze(2)

        # y = y.reshape([T, 1, self.N*self.F[-1]]) # Por que esta ahi el 1
        y = y.reshape([T, self.N*self.F[-1]])

        return self.FCL(y)


class GraphDecoder(nn.Module):
    """
    This class represent a basic graph decoder with only one layer following
    the model G(C) = ReLU(HC)v, where instead of upsampling a fixed
    low pass graph filter is used.
    """
    def __init__(self, features, H, scale_std=0.01):
        """
        The arguments are:
        - features: the number of features
        - H: fixed graph filter
        - scale_std: scale the std of the learnable weights initialization
        """
        super().__init__()
        N = H.shape[0]
        self.input = Tensor(H).view([1, N, N])
        self.v = torch.ones(features)/math.sqrt(features)
        self.v[math.ceil(features/2):] *= -1
        self.conv = nn.Conv1d(N, features, kernel_size=1,
                              bias=False)
        std = scale_std/math.sqrt(N)
        self.conv.weight.data.normal_(0, std)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.conv(input)).squeeze().t().mv(self.v)

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)
    
    
class GraphDeepDecoder(nn.Module):
    def __init__(self,
                 # Decoder args
                 features, nodes, H,
                 # Activation functions
                 act_fn=nn.ReLU(), last_act_fn=None,
                 input_std=0.01, w_std=0.01):
        assert len(features) == len(nodes), ERR_DIFF_N_LAYERS

        super(GraphDeepDecoder, self).__init__()
        self.model = nn.Sequential()
        self.fts = features
        self.nodes = nodes
        self.H = H
        self.kernel_size = 1
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.w_std = w_std/np.sqrt(features)
        self.build_network()

        shape = [1, self.fts[0], self.nodes[0]]
        std = input_std/np.sqrt(shape[2])
        self.input = Tensor(torch.zeros(shape)).data.normal_(0, std)

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def build_network(self):
        for l in range(len(self.fts)-1):
            conv = nn.Conv1d(self.fts[l], self.fts[l+1], self.kernel_size,
                             bias=False)
            conv.weight.data.normal_(0, self.w_std[l])
            self.add_layer(conv)
            self.add_layer(layers.FixedFilter(self.H))

            if l < (len(self.fts)-2):
                # Not the last layer
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
            else:
                # Last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)
        return self.model

    def forward(self, x):
        return self.model(x).squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)
