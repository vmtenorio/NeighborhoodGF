import numpy as np
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi, BarabasiAlbert
from scipy.sparse.csgraph import shortest_path

import os
import torch


# Graph Type Constants
SBM = 1
ER = 2
BA = 3

MAX_RETRIES = 50

# Comm Node Assignment Constants
CONT = 1    # Contiguous nodes
ALT = 2    # Alternated nodes
RAND = 3    # Random nodes


def assign_nodes_to_comms(N, k):
    """
    Distribute contiguous nodes in the same community while assuring that all
    communities have (approximately) the same number of nodes.
    """
    z = np.zeros(N, dtype=np.int)
    leftover_nodes = N % k
    grouped_nodes = 0
    for i in range(k):
        if leftover_nodes > 0:
            n_nodes = np.ceil(N/k).astype(np.int)
            leftover_nodes -= 1
        else:
            n_nodes = np.floor(N/k).astype(np.int)
        z[grouped_nodes:(grouped_nodes+n_nodes)] = i
        grouped_nodes += n_nodes
    return z

def create_graph(ps, seed=None):
    """
    Create a random graph using the parameters specified in the dictionary ps.
    The keys that this dictionary should nclude are:
        - type: model for the graph. Options are SBM (Stochastic Block Model),
          ER (Erdos-Renyi) or BA (BarabasiAlbert)
        - N: number of nodes
        - k: number of communities (for SBM only)
        - p: edge probability for nodes in the same community
        - q: edge probability for nodes in different communities (for SBM only)
        - type_z: specify how to assigns nodes to communities (for SBM only).
          Options are CONT (continous), ALT (alternating) and RAND (random)
    """
    if ps['type'] == SBM:
        if ps['type_z'] == CONT:
            z = assign_nodes_to_comms(ps['N'], ps['k'])
        elif ps['type_z'] == ALT:
            z = np.array(list(range(ps['k']))*int(ps['N']/ps['k']) +
                         list(range(ps['N'] % ps['k'])))
        elif ps['type_z'] == RAND:
            z = assign_nodes_to_comms(ps['N'], ps['k'])
            np.random.shuffle(z)
        else:
            z = None
        G = StochasticBlockModel(N=ps['N'], k=ps['k'], p=ps['p'], z=z,
                                 q=ps['q'], connected=True, seed=seed,
                                 max_iter=MAX_RETRIES)
    elif ps['type'] == ER:
        G = ErdosRenyi(N=ps['N'], p=ps['p'], connected=True, seed=seed,
                       max_iter=MAX_RETRIES)
    elif ps['type'] == BA:
        G = BarabasiAlbert(N=ps['N'], m=ps['m'], m0=ps['m0'], seed=seed)
        G.info = {'comm_sizes': np.array([ps['N']]),
                  'node_com': np.zeros((ps['N'],), dtype=int)}
    else:
        raise RuntimeError('Unknown graph type')

    assert G.is_connected(), 'Graph is not connected'

    G.set_coordinates('spring')
    G.compute_fourier_basis()
    return G

def my_eig(S):
    d,V = np.linalg.eig(S)
    order = np.argsort(-d)
    d = d[order]
    V = V[:,order]
    D = np.diag(d)
    try:
        VV = np.linalg.inv(V)
        SS = V.dot(D.dot(VV))
        diff = np.absolute(S-SS)
        if diff.max() > 1e-6:
            print("Eigendecomposition not good enough")
    except np.linalg.LinAlgError:
        print("V matrix is not invertible")
    return V,D

def norm_graph(A):
    """Receives adjacency matrix and returns normalized (divided by biggest eigenvalue) """
    V,D = my_eig(A)
    if np.max(np.abs(np.imag(V))) < 1e-6:
        V = np.real(V)
    if np.max(np.abs(np.imag(D))) < 1e-6:
        D = np.real(D)
    d = np.diag(D)
    dmax = d[0]
    return (A/dmax).astype(np.float32)

def perturbate_probability(Gx, eps_c, eps_d):
    A_x = Gx.W.todense()
    no_link_ind = np.where(A_x == 0)
    link_ind = np.where(A_x == 1)

    mask_c = np.random.choice([0, 1], p=[1-eps_c, eps_c],
                              size=no_link_ind[0].shape)
    mask_d = np.random.choice([0, 1], p=[1-eps_d, eps_d],
                              size=link_ind[0].shape)

    A_x[link_ind] = np.logical_xor(A_x[link_ind], mask_d).astype(int)
    A_x[no_link_ind] = np.logical_xor(A_x[no_link_ind], mask_c).astype(int)
    A_x = np.triu(A_x, 1)
    A_y = A_x + A_x.T
    return A_y

def perturbate_percentage(Gx, creat, destr):
    A_x_triu = Gx.W.todense()
    A_x_triu[np.tril_indices(Gx.N)] = -1

    # Create links
    no_link_i = np.where(A_x_triu == 0)
    links_c = np.random.choice(no_link_i[0].size, int(Gx.Ne * creat/100),
                               replace=False)
    idx_c = (no_link_i[0][links_c], no_link_i[1][links_c])

    # Destroy links
    link_i = np.where(A_x_triu == 1)
    links_d = np.random.choice(link_i[0].size, int(Gx.Ne * destr/100),
                               replace=False)
    idx_d = (link_i[0][links_d], link_i[1][links_d])

    A_x_triu[np.tril_indices(Gx.N)] = 0
    A_x_triu[idx_c] = 1
    A_x_triu[idx_d] = 0
    A_y = A_x_triu + A_x_triu.T
    return A_y

def perm_graph(A, coords, node_com, comm_sizes):
    N = A.shape[0]
    # Create permutation matrix
    P = np.zeros(A.shape)
    i = np.arange(N)
    j = np.random.permutation(N)
    P[i, j] = 1

    # Permute
    A_p = P.dot(A).dot(P.T)
    assert np.sum(np.diag(A_p)) == 0, 'Diagonal of permutated A is not 0'
    coords_p = P.dot(coords)
    node_com_p = P.dot(node_com)
    G = Graph(A_p)
    G.set_coordinates(coords_p)
    G.info = {'node_com': node_com_p,
              'comm_sizes': comm_sizes,
              'perm_matrix': P}
    return G

def perturbated_graphs(g_params, creat=5, dest=5, pct=True, perm=False, seed=None):
    """
    Create 2 closely related graphs. The first graph is created following the
    indicated model and the second is a perturbated version of the previous
    where links are created or destroid with a small probability.
    Arguments:
        - g_params: a dictionary containing all the parameters for creating
          the desired graph. The options are explained in the documentation
          of the function 'create_graph'
        - eps_c: probability for creating new edges
        - eps_d: probability for destroying existing edges
    """
    Gx = create_graph(g_params, seed)
    if pct:
        Ay = perturbate_percentage(Gx, creat, dest)
    else:
        Ay = perturbate_probability(Gx, creat, dest)
    coords_Gy = Gx.coords
    node_com_Gy = Gx.info['node_com']
    comm_sizes_Gy = Gx.info['comm_sizes']
    assert np.sum(np.diag(Ay)) == 0, 'Diagonal of A is not 0'

    if perm:
        Gy = perm_graph(Ay, coords_Gy, node_com_Gy, comm_sizes_Gy)
    else:
        Gy = Graph(Ay)
        Gy.set_coordinates(coords_Gy)
        Gy.info = {'node_com': node_com_Gy,
                   'comm_sizes': comm_sizes_Gy}
    assert Gy.is_connected(), 'Could not create connected graph Gy'
    return Gx, Gy


def create_filter(S, ps):
    """
    Creates a graph filter from the GSO and a dict of parameters.
    
    Arguments
    ---------
        - S: GSO of the graph
        - ps: dict of parameters. It must contain the keys:
            - type: how to generate the coefficients of the filter. Either:
                -RandH: random filter coefficients. Must contain the key 'K'
                    that indicates the number of coefficients of the filter
                -FixedH: fixed filter coefficients determined in advance.
                    Must contain the key 'hs' with the list of filters
            - neigh: if True, creates an NGF, else a classic graph filter
            - H_norm: whether or not to normalize the filter by its norm
    """
    if ps['type'] == 'RandH':
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
    else:
        H = np.zeros((S.shape))
        for l, h in enumerate(hs):
            H += h*np.linalg.matrix_power(S, l)
        
    if ps['H_norm']:
        H /= np.linalg.norm(H)

    return H


class NeighborhoodGF:
    """
    This class allows to pre-calculate and store the neighborhood graph filters
    associated with real-world datasets' graphs, as the high number of nodes
    means a really high computation time of the eigenvalue decomposition and
    shortest paths matrix.
    """
    def __init__(self, S, K_max, arch_name, device='cpu'):
        self.S = S
        self.N = self.S.shape[0]
        self.K_max = K_max
        self.arch_name = arch_name
        self.device = device

        if os.path.exists(arch_name + '.npy'):
            self.distances = np.load(arch_name + '.npy')
        else:
            self.distances = shortest_path(self.S, unweighted=True, directed=False)
            np.save(arch_name, self.distances)

        self.calc_filters()

    def calc_filters(self):
        self.H = torch.zeros((self.K_max, self.N, self.N), device=self.device)
        self.H[0,:,:] = torch.eye(self.N, device=self.device)
        for k in range(1, self.K_max):
            Hpow = torch.from_numpy((self.distances == k)).type(torch.float).to(self.device)
            d = torch.real(torch.linalg.eigvals(Hpow))
            dmax = torch.max(d)
            self.H[k,:,:] = Hpow / dmax

    def get_filter(self, hs):
        k = hs.size()[0]
        assert k <= self.K_max, "Too much filter coefficients"
        H = torch.zeros((self.N, self.N), device=self.device)
        for i in range(k):
            H += hs[i] * self.H[i,:,:]

        return H

def calc_powers(S, K, device='cpu'):
    N = S.shape[0]
    St = torch.Tensor(S).to(device)
    Spow = torch.zeros((K, N, N), device=device)
    Spow[0,:,:] = torch.eye(N, device=device)

    for k in range(1, K):
        Spow[k,:,:] = Spow[k-1,:,:] @ St
    return Spow
