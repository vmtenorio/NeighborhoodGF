import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi, BarabasiAlbert
from torch import Tensor, LongTensor

# Graph Type Constants
SBM = 1
ER = 2
BA = 3

MAX_RETRIES = 20

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
    VV = np.linalg.inv(V)
    SS = V.dot(D.dot(VV))
    diff = np.absolute(S-SS)
    if diff.max() > 1e-6:
        print("Eigendecomposition not good enough")
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


class DiffusedSparse:
    """
    Class for generating a graph signal X following a diffusion
    process generated over an underlying graph.
    Arguments:
        - G: graph where signal X will be defined
        - n_samples: a list with the number of samples for training, validation
          and test. Alternatively, if only an integer is provided
    """
    def __init__(self, G, n_samples, L, n_delts, min_d=-1, max_d=1,
                 median=True, same_coeffs=False, neg_coeffs=False):

        if isinstance(n_samples, int):
            self.n_train = n_samples
            self.n_val = int(np.floor(0.25*n_samples))
            self.n_test = int(np.floor(0.25*n_samples))
        elif len(n_samples) == 3:
            self.n_train = n_samples[0]
            self.n_val = n_samples[1]
            self.n_test = n_samples[2]
        else:
            raise RuntimeError('n_samples must be an integer or a list with the \
                                samples for training, validation and test')
        self.G = G

        self.median = median
        if neg_coeffs:
            self.h = 2 * np.random.rand(L) - 1
        else:
            self.h = np.random.rand(L)
        self.random_diffusing_filter()
        self.create_samples_S(n_delts, min_d, max_d)
        self.create_samples_X()

    def median_neighbours_nodes(self, X, G):
        X_aux = np.zeros(X.shape)
        for i in range(G.N):
            _, neighbours = np.asarray(G.W.todense()[i, :] != 0).nonzero()
            X_aux[:, i] = np.median(X[:, np.append(neighbours, i)], axis=1)
        return X_aux

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = Tensor(self.train_Y).view([self.n_train, N])
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = Tensor(self.val_Y).view([self.n_val, N])
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = Tensor(self.test_Y).view([self.n_test, N])

    def to_unit_norm(self):
        self.train_X = self._to_unit_norm(self.train_X)
        self.val_X = self._to_unit_norm(self.val_X)
        self.test_X = self._to_unit_norm(self.test_X)

    def _to_unit_norm(self, signals):
        """
        Divide each signal by its norm so all samples have unit norm
        """
        norm = np.sqrt(np.sum(signals**2, axis=1))
        if 0 in norm:
            print("WARNING: signal with norm 0")
            return None
        return (signals.T/norm).T

    def plot_train_signals(self, ids, show=True):
        if not isinstance(ids, list) and not isinstance(ids, range):
            ids = [ids]
        for id in ids:
            Sx = self.train_S[id, :]
            X = self.train_X[id, :]
            Y = self.train_Y[id, :]
            _, axes = plt.subplots(2, 2)
            self.G.plot_signal(S, ax=axes[0, 0])
            self.G.plot_signal(X, ax=axes[0, 1])
            self.G.plot_signal(Y, ax=axes[1, 1])
        if show:
            plt.show()

    def delta_values(self, G, n_samp, n_deltas, min_delta, max_delta):
        n_comms = G.info['comm_sizes'].size
        if n_comms > 1:
            step = (max_delta-min_delta)/(n_comms-1)
        else:
            step = (max_delta-min_delta)/(n_deltas-1)
        ds_per_comm = np.ceil(n_deltas/n_comms).astype(int)
        delta_means = np.arange(min_delta, max_delta+0.1, step)
        delta_means = np.tile(delta_means, ds_per_comm)[:n_deltas]
        delt_val = np.zeros((n_deltas, n_samp))
        for i in range(n_samp):
            delt_val[:, i] = np.random.randn(n_deltas)*step/4 + delta_means
        return delt_val

    def sparse_S(self, G, delta_values):
        """
        Create random sparse signal s composed of different deltas placed in the
        different communities of the graph. If the graph is an ER, then deltas
        are just placed on random nodes
        """
        n_samp = delta_values.shape[1]
        S = np.zeros((G.N, n_samp))
        # Randomly assign delta value to comm nodes
        for i in range(n_samp):
            for j in range(delta_values.shape[0]):
                delta = delta_values[j, i]
                com_j = j % G.info['comm_sizes'].size
                com_nodes, = np.asarray(G.info['node_com'] == com_j).nonzero()
                rand_index = np.random.randint(0, G.info['comm_sizes'][com_j])
                S[com_nodes[rand_index], i] = delta
        return S.T

    def add_noise(self, p_n, test_only=True):
        if p_n == 0:
            return
        self.test_X = self.add_noise_to_X(self.test_X, p_n)

        if test_only:
            return
        self.val_X = self.add_noise_to_X(self.val_X, p_n)
        self.train_X = self.add_noise_to_X(self.train_X, p_n)

    def add_noise_to_X(self, X, p_n):
        p_x = np.sum(X**2, axis=1)
        sigma = np.sqrt(p_n * p_x / X.shape[1])
        noise = (np.random.randn(X.shape[0], X.shape[1]).T * sigma).T
        return X + noise

    def random_diffusing_filter(self):
        """
        Create a linear random diffusing filter with L random coefficients
        using the graphs shift operator from G.
        Arguments:
            - L: number of filter coeffcients
        """
        self.H = np.zeros(self.G.W.shape)
        S = self.G.W.todense()
        for l in range(self.h.size):
            self.H += self.h[l]*np.linalg.matrix_power(S, l)
        self.H = norm_graph(self.H)

    def create_samples_S(self, delts, min_d, max_d):
        train_deltas = self.delta_values(self.G, self.n_train, delts, min_d, max_d)
        val_deltas = self.delta_values(self.G, self.n_val, delts, min_d, max_d)
        test_deltas = self.delta_values(self.G, self.n_test, delts, min_d, max_d)
        self.train_S = self.sparse_S(self.G, train_deltas)
        self.val_S = self.sparse_S(self.G, val_deltas)
        self.test_S = self.sparse_S(self.G, test_deltas)

    def create_samples_X(self):
        self.train_X = self.H.dot(self.train_S.T).T
        self.val_X = self.H.dot(self.val_S.T).T
        self.test_X = self.H.dot(self.test_S.T).T
        if self.median:
            self.train_X = self.median_neighbours_nodes(self.train_X, self.G)
            self.val_X = self.median_neighbours_nodes(self.val_X, self.G)
            self.test_X = self.median_neighbours_nodes(self.test_X, self.G)

class DenoisingSparse(DiffusedSparse):
    '''
    TODO
    '''
    def __init__(self, G, n_samples, L, n_delts, p_n, min_d=-1,
                 max_d=1, median=True, same_coeffs=False, neg_coeffs=False, test_only=False):

        super(DenoisingSparse, self).__init__(G, n_samples, L, n_delts, min_d,
                                          max_d, median, same_coeffs, neg_coeffs)

        self.to_unit_norm()
        self.train_Y = self.train_X.copy()
        self.val_Y = self.val_X.copy()
        self.test_Y = self.test_X.copy()
        self.add_noise(p_n, test_only=test_only)


class SourcelocSynthetic(DiffusedSparse):
    def __init__(self, G, n_samples, L, min_d=-1,
                 max_d=1, median=True, same_coeffs=False, neg_coeffs=False):

        super(SourcelocSynthetic, self).__init__(G, n_samples, L,
                                                1, min_d, max_d,    # Just 1 delt
                                                median, same_coeffs,
                                                neg_coeffs)

    def delta_values(self, n_samp, min_delta, max_delta):
        step = max_delta-min_delta
        delt_val = np.random.randn(n_samp)*step/4# + delta_means
        return delt_val

    def sparse_S(self, G, delta_values):
        """
        Create random sparse signal s composed of different deltas placed in the
        different communities of the graph. If the graph is an ER, then deltas
        are just placed on random nodes
        """
        n_samp = delta_values.shape[0]
        S = np.zeros((G.N, n_samp))
        Y = np.zeros(n_samp)

        # Randomly assign delta value to comm nodes
        for i in range(n_samp):
            com_idx = np.random.randint(0, G.info['comm_sizes'].size)
            node_idx = np.random.randint(0, G.info['comm_sizes'][com_idx])

            com_nodes, = np.asarray(G.info['node_com'] == com_idx).nonzero()
            S[com_nodes[node_idx], i] = delta_values[i]
            Y[i] = com_idx
        return S.T, Y

    def create_samples_S(self, delts, min_d, max_d):
        train_deltas = self.delta_values(self.n_train, min_d, max_d)
        val_deltas = self.delta_values(self.n_val, min_d, max_d)
        test_deltas = self.delta_values(self.n_test, min_d, max_d)
        self.train_S, self.train_Y = self.sparse_S(self.G, train_deltas)
        self.val_S, self.val_Y = self.sparse_S(self.G, val_deltas)
        self.test_S, self.test_Y = self.sparse_S(self.G, test_deltas)

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = LongTensor(self.train_Y)
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = LongTensor(self.val_Y)
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = LongTensor(self.test_Y)

