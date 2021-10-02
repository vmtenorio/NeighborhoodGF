import math
import numpy as np
from enum import Enum

import torch.nn as nn
import torch
from torch import Tensor, nn
from . import layers

from scipy import sparse

from torch_geometric.nn import GCNConv, SGConv, GATConv
from torch_geometric.utils import from_scipy_sparse_matrix


class GCNN_GF(nn.Module):
    """
    In this type of network, each layer in the output is calculated as

    X^{(\ell+1)}_j = \sigma(\sum_{i=1}^F^{(\ell)} H_{ij} X^{(\ell)}_i)

    Where $H_{ij}$ is a graph filter with learnable coefficients.
    As such, there are F^{(\ell)}*F^{(\ell+1)}*K learnable parameters
    per layer, where K is the order of the filter used.
    """
    def __init__(self,
                S,
                gf_type,    # Type of graph filter to use
                F,          # Features in each graph filter layer (list)
                K,          # Filter taps in each graph filter layer
                bias_gf,    # Whether or not to use Bias in the GF layers
                M,          # Neurons in each fully connected layer (list)
                bias_mlp,   # Whether or not to use Bias in the FC layers
                nonlin,     # Non linear function
                arch_info): # Print architecture information
        super(GCNN_GF, self).__init__()
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
            gfl.append(self.gf(self.S, self.F[l], self.F[l+1], self.K, bias_gf))
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
        Arguments
        ---------
            - features: the number of features
            - H: fixed graph filter
            - scale_std: scale the std of the learnable weights initialization
        """
        super().__init__()
        N = H.shape[0]
        self.input = nn.Parameter(Tensor(H).view([1, N, N]), requires_grad=False)
        v = torch.ones(features)/math.sqrt(features)
        v[math.ceil(features/2):] *= -1
        self.v = nn.Parameter(v, requires_grad=False)
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
    
class GCNN(nn.Module):
    """
    Message Passing Graph Neural Network that, instead of using the 
    adjacency matrix, utilizes a graph filter to aggregate the information of
    a bigger neighborhood in each layer.

    X^{(\ell+1)} = \sigma(H X^{(\ell)} \Theta^{(\ell)})
    """
    def __init__(self,
                 # Network args
                 features, H,
                 # Activation functions
                 act_fn=nn.ReLU(), last_act_fn=None,
                 input_std=0.01, w_std=0.01):

        super(GCNN, self).__init__()
        self.model = nn.Sequential()
        self.fts = features
        self.H = H
        self.kernel_size = 1
        self.act_fn = act_fn
        self.last_act_fn = last_act_fn
        self.w_std = w_std/np.sqrt(features)
        self.build_network()

        # Declare the input for the denoising problem
        shape = [1, self.fts[0], self.H.shape[0]]
        std = input_std/np.sqrt(shape[2])
        self.input = nn.Parameter(Tensor(torch.zeros(shape)).data.normal_(0, std), requires_grad=False)

    def add_layer(self, module):
        self.model.add_module(str(len(self.model) + 1), module)

    def build_network(self):
        for l in range(len(self.fts)-1):
            conv = nn.Conv1d(self.fts[l], self.fts[l+1], self.kernel_size,
                             bias=False)
            conv.weight.data.normal_(0, self.w_std[l])
            self.add_layer(conv)

            if l < (len(self.fts)-2):
                # Not the last layer
                self.add_layer(layers.FixedFilter(self.H))
                if self.act_fn is not None:
                    self.add_layer(self.act_fn)
            else:
                # Last layer
                if self.last_act_fn is not None:
                    self.add_layer(self.last_act_fn)
        return self.model

    def forward(self, x):

        n_samp, f_in, xN = x.shape
        assert f_in == self.fts[0]
        assert xN == self.H.shape[0]

        return self.model(x).squeeze()

    def count_params(self):
        return sum(p.numel() for p in self.model.parameters()
                   if p.requires_grad)

class MLP(nn.Module):
    """
    A simple torch Module for a Multilayer Perceptron network.

    Parameters
    ----------
        - M: list with the number of neurons of each FC layer
        - bias_mlp: whether or not to use bias
        - nonlin: element-wise non-linear function to be applied after each layer
        - arch_info: whether or not to print the information of the architecture
    """

    def __init__(self,
                M,          # Neurons in each fully connected layer (list)
                bias_mlp,   # Whether or not to use Bias in the FC layers
                nonlin,     # Non linearity function
                arch_info): # Print architecture information
        super(MLP, self).__init__()
        # In python 3
        # super()

        self.M = M
        self.nonlin = nonlin
        self.bias_mlp = bias_mlp

        # Define the layer

        # Fully connected Layers
        fcl = []
        if len(self.M) > 0:
            # As last layer has no nonlin (if its softmax is done later, etc.)
            # define here the first layer before loop
            fcl.append(nn.Linear(self.M[0], self.M[1], bias=bias_mlp))
            for m in range(2,len(self.M)):
                # print("FC layer: " + str(m))
                # print(str(self.M[m-1]) + ' x ' + str(self.M[m]))
                fcl.append(self.nonlin())
                fcl.append(nn.Linear(self.M[m-1], self.M[m], bias=bias_mlp))

        self.FCL = nn.Sequential(*fcl)

        if arch_info:
            print("Architecture:")
            print("M: {}, Bias: {}".format(self.M, self.bias_mlp))
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
            assert Fin*xN == self.M[0]
            x = x.reshape([T, Fin*xN])
        except IndexError:
            xN = x.shape[1]
            assert xN == self.M[0]

        # Define the forward pass
        return self.FCL(x)


class SOTAGraphNN(nn.Module):
    """
    Class implementing a GCNN using layers from SOTA architectures provided by
    the library torch_geometric. Currently, there are:
    - GCNConv - Kipf 2017
    - SGCConv - Wu 2019. layer_params should contain the parameter K
    - GATConv - Veličković 2017. layer_params should contain a list with the
        number of attention heads for each layer

    Arguments
    ---------
        - S: GSO to be used in the networks. It is expected as a numpy array, as it
            is transformed to the appropiate format
        - layer_params: the parameters of the layers to be implemented. It should
            contain the name of the layer and further attributes depending on the
            layer
        - features: list with the number of features at each layer.
        - nonlin: element-wise non-linear function to be applied after each layer-
    """

    def __init__(self, S, layer_params, features, nonlin=nn.ReLU()):
        super(SOTAGraphNN, self).__init__()
        self.S = S
        edge_index, _ = from_scipy_sparse_matrix(sparse.csr_matrix(S))
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)

        conv_layers = []
        for i in range(1, len(features)):
            conv_layers.append(self.conv_layer(features[i-1], features[i], layer_params, i))
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.n_layers = len(conv_layers)
        
        self.nonlin = nonlin
        
        shape = [S.shape[0], features[0]]
        std = .1/np.sqrt(shape[0])
        input = torch.Tensor(torch.zeros(shape)).data.normal_(0, std)
        self.input = nn.Parameter(input, requires_grad=False)

    def conv_layer(self, in_feat, out_feat, params, i=None):
        """
        This method is used to create a new layer from the parameters inside layer_params
        dictionary and the input and output features
        """

        layer = None
        if params['name'] == "GCNConv":
            layer = GCNConv(in_feat, out_feat)
        elif params['name'] == "SGConv":
            layer = SGConv(in_feat, out_feat, K=params['K'])
        elif params['name'] == "GATConv":
            if i > 0:
                in_feat = in_feat*params['heads'][i-1]
            layer = GATConv(in_feat, out_feat, heads=params['heads'][i])
        else:
            raise NotImplementedError("Layer type {} not found".format(params['name']))
        return layer

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, self.edge_index)
            if i < self.n_layers-1:
                x = self.nonlin(x)
        return x.squeeze()
        