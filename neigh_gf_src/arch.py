import torch.nn as nn
import torch
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

