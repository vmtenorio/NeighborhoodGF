import torch
import torch.nn as nn
import math

from scipy.sparse.csgraph import shortest_path

import numpy as np


DEBUG = False

# Error messages
ERR_WRONG_INPUT_SIZE = 'Number of input samples must be 1'


class GraphFilter(nn.Module):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K
                 ):
        super(GraphFilter, self).__init__()
        self.S = S
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        self.Spow = torch.zeros(K, self.N, self.N)
        self.Spow[0, :, :] = torch.eye(self.N)
        # mat_power = np.eye(self.N)
        self.distances = shortest_path(self.S, directed=False, unweighted=True)
        for k in range(1, K):
            self.Spow[k, :, :] = torch.from_numpy((self.distances == k).astype(int))
            # mat_power = mat_power @ self.S
            # self.Spow[k, :, :] = torch.from_numpy(((self.Spow[i-1]@self.S) > 0).astype(int))
        self.Spow = self.Spow.repeat(max(self.Fout, self.Fin), 1, 1, 1)

    def calc_filter(self, fout):
        weights = self.weights[fout*self.K:fout*self.K+self.K]
        return (weights.view([self.K, 1, 1])*self.Spow[0, :, :, :]).sum(0)

    def calc_all_filters(self, F):
        Hs = self.weights.view([F, self.K, 1, 1])*self.Spow
        return Hs.sum(1)


class GraphFilterUp(GraphFilter):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K
                 ):
        super(GraphFilterUp, self).__init__(S, Fin, Fout, K)

        assert Fout >= Fin
        self.mult = Fout // Fin
        self.weights = nn.Parameter(torch.Tensor(self.K*self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape T x Fin x N
        # T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        Hs = self.calc_all_filters(self.Fout)
        xF2 = x.repeat(1, self.mult + 1, 1)[:, :self.Fout, :].permute([1, 0, 2])
        y = torch.bmm(xF2, Hs).permute([1, 0, 2])

        return y


class GraphFilterDown(GraphFilter):
    def __init__(self,
                 # GSO
                 S,
                 # Size of the filter
                 Fin, Fout, K):
        super(GraphFilterDown, self).__init__(S, Fin, Fout, K)

        assert Fin % Fout == 0
        self.mult = Fin // Fout
        self.weights = nn.Parameter(torch.Tensor(self.K*self.Fin))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x shape T x Fin x N
        T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        xF2 = x.permute([1, 0, 2])
        Hs = self.calc_all_filters(self.Fin)
        Y_aux = torch.bmm(xF2, Hs).permute([1, 0, 2])
        # OPT1: FASTER
        y = Y_aux.reshape(T, self.Fout, self.mult, self.N).sum(2)/self.mult

        return y


class BaseGF(nn.Module):

    def __init__(self,
                # GSO
                S,
                # Size of the filter
                Fin, Fout, K,
                bias_gf
                ):
        super(BaseGF, self).__init__()
        self.S = S
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K
        self.bias_gf = bias_gf

        # Fernando's weight Initialization
        self.weights = nn.parameter.Parameter(torch.Tensor(self.Fin*self.K, self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.K)
        self.weights.data.uniform_(-stdv, stdv)
        if self.bias_gf:
            self.bias = nn.parameter.Parameter(torch.Tensor(1, self.Fout, self.N))
            self.bias.data.uniform_(-stdv, stdv)

        self.build_filter()
        #torch.set_printoptions(threshold=100)

    def forward(self, x):
        # x shape T x N x Fin
        # Graph filter
        T = x.shape[0]
        xFin = x.shape[1]
        xN = x.shape[2]

        assert xN == self.N
        assert xFin == self.Fin

        x = x.permute(2, 1, 0)      # N x Fin x T
        x = x.reshape([self.N, self.Fin*T])
        x_list = []

        for k in range(self.K):
            x1 = torch.matmul(self.Spow[k,:,:], x)
            x_list.append(x1)
        # x shape after loop: K x N x Fin*T
        x = torch.stack(x_list)

        x = x.reshape([self.K, self.N, self.Fin, T])
        x = x.permute(3,1,2,0)
        x = x.reshape([T*self.N, self.K*self.Fin])

        # Apply weights
        y = torch.matmul(x, self.weights)       # y shape: T*N x Fout

        y = y.reshape([T, self.N, self.Fout])
        y = y.permute(0, 2, 1)

        if self.bias_gf:
            y = y + self.bias

        return y


class NeighborhoodGF(BaseGF):

    def build_filter(self):
        self.Spow = torch.zeros((self.K, self.N, self.N))
        self.Spow[0,:,:] = torch.eye(self.N)
        self.distances = shortest_path(self.S, directed=False, unweighted=True)
        for k in range(1, self.K):
            self.Spow[k, :, :] = torch.from_numpy((self.distances == k).astype(int))
        self.Spow = nn.Parameter(self.Spow, requires_grad=False)


class NeighborhoodGFType2(BaseGF):

    def build_filter(self):
        self.Spow = torch.zeros((self.K, self.N, self.N))
        self.Spow[0,:,:] = torch.eye(self.N)
        mat_power = np.eye(self.N)
        for k in range(1, self.K):
            mat_power = mat_power @ self.S
            self.Spow[k, :, :] = torch.from_numpy((mat_power > 0).astype(int))
        self.Spow = nn.Parameter(self.Spow, requires_grad=False)


class ClassicGF(BaseGF):

    def build_filter(self):
        self.Spow = torch.zeros((self.K, self.N, self.N))
        self.Spow[0,:,:] = torch.eye(self.N)

        for k in range(1, self.K):
            self.Spow[k, :, :] = self.Spow[k-1,:,:] @ self.S
        self.Spow = nn.Parameter(self.Spow, requires_grad=False)


class BasicGNN(nn.Module):

    def __init__(self, S, Fin, Fout, K=1, bias_gf=False):
        super(BasicGNN, self).__init__()
        self.S = S
        self.N = S.shape[0]
        self.Fin = Fin
        self.Fout = Fout
        self.K = K

        self.weights = nn.Parameter(torch.Tensor(self.Fin, self.Fout))
        stdv = 1. / math.sqrt(self.Fin * self.Fout)
        self.weights.data.uniform_(-stdv, stdv)

        if K > 1:
            self.h = nn.Parameter(torch.Tensor(self.K), requires_grad=False)
            self.H = nn.Parameter(self.build_filter(), requires_grad=False)
        else:
            self.H = nn.Parameter(self.S, requires_grad=False)

    def build_filter(self):
        H = self.h[0] * torch.eye(self.N)
        S = torch.from_numpy(self.S)
        matpower = torch.from_numpy(self.S)

        for k in range(1, self.K):
            matpower = matpower @ S
            H += self.h[k] * matpower
        return H

    def forward(self, x):
        T, xF, xN = x.shape
        assert xN == self.N
        assert xF == self.Fin

        return self.weights.T @ x @ self.H.T

class NeighborhoodGFGNN(BasicGNN):
    def build_filter(self):
        H = self.h[0] * torch.eye(self.N)
        distances = shortest_path(self.S, directed=False, unweighted=True)
        for k in range(1, self.K):
            H += self.h[k] * torch.from_numpy(norm_graph((self.distances == k).astype(int)))
        return H


class FixedFilter(nn.Module):
    def __init__(self, H):
        nn.Module.__init__(self)
        self.H_T = torch.Tensor(H).t()

    def forward(self, input):
        assert input.shape[0] == 1, ERR_WRONG_INPUT_SIZE
        return input.matmul(self.H_T)

