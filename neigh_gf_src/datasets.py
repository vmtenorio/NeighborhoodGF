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


class BaseGraphDataset:
    """
    Class for generating a graph signal X following a diffusion
    process generated over an underlying graph.
    Arguments:
        - G: graph where signal X will be defined
        - n_samples: a list with the number of samples for training, validation
          and test. Alternatively, if only an integer is provided
    """
    def __init__(self, G, n_samples, median=True):

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

    def to_unit_norm(self, y_norm=False):
        self.train_X = self._to_unit_norm(self.train_X)
        self.val_X = self._to_unit_norm(self.val_X)
        self.test_X = self._to_unit_norm(self.test_X)
        if y_norm:
            self.train_Y = self._to_unit_norm(self.train_Y)
            self.val_Y = self._to_unit_norm(self.val_Y)
            self.test_Y = self._to_unit_norm(self.test_Y)

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

    def to(self, device):
        self.train_X = self.train_X.to(device)
        self.train_Y = self.train_Y.to(device)
        self.val_X = self.val_X.to(device)
        self.val_Y = self.val_Y.to(device)
        self.test_X = self.test_X.to(device)
        self.test_Y = self.test_Y.to(device)

class DenoisingSparse(BaseGraphDataset):
    '''
    Class for the Denoising problem, where we try to recover the original signal
    from a noisy version of it
    '''
    def __init__(self, G, n_samples, L, n_delts, p_n, x_type="random", min_d=-1,
                 max_d=1, median=True, neg_coeffs=False, test_only=False):

        assert x_type in ["random", "deltas"], \
            'Only "random" input or an sparse signal ("deltas") allowed'

        super(DenoisingSparse, self).__init__(G, n_samples, median)

        self.L = L
        self.n_delts = n_delts
        self.x_type = x_type
        self.min_d = min_d
        self.max_d = max_d
        self.neg_coeffs = neg_coeffs

        self.train_X, self.train_Y = self.create_samples(self.n_train)
        self.val_X, self.val_Y = self.create_samples(self.n_val)
        self.test_X, self.test_Y = self.create_samples(self.n_test)

        self.add_noise_to_X(self.train_Y, p_n)

        self.to_unit_norm(y_norm=True)

    def create_samples(self, n_samples):
        if self.x_type == "deltas":
            deltas = self.delta_values(self.G, n_samples, self.n_delts, self.min_d, self.max_d)
            orig_signal = self.sparse_S(self.G, deltas)
        else:
            orig_signal = np.random.randn(n_samples, self.G.N)

        self.random_diffusing_filter(self.L, self.neg_coeffs)

        Y = self.H.dot(orig_signal.T).T
        if self.median:
            Y = self.median_neighbours_nodes(Y, self.G)

        X = np.random.randn(n_samples, self.G.N)

        return X, Y

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

    def random_diffusing_filter(self, L, neg_coeffs):
        """
        Create a linear random diffusing filter with L random coefficients
        using the graphs shift operator from G.
        Arguments:
            - L: number of filter coeffcients
        """

        if neg_coeffs:
            h = 2 * np.random.rand(L) - 1
        else:
            h = np.random.rand(L)
        self.H = np.zeros(self.G.W.shape)
        S = norm_graph(self.G.W.todense())
        for l in range(h.size):
            self.H += h[l]*np.linalg.matrix_power(S, l)


class SourcelocSynthetic(BaseGraphDataset):
    """
    Class for the source localization problem, where 1 delta is diffused over the
    graph and the algorithm has to recover the community of the node where the delta
    was placed.
    """
    def __init__(self, G, n_samples, min_l=10, max_l=25, min_d=-1,
                 max_d=1, median=True, neg_coeffs=False):

        super(SourcelocSynthetic, self).__init__(G, n_samples, median)

        self.train_X, self.train_Y = self.create_samples(self.n_train,
                                                    min_d, max_d,
                                                    min_l, max_l)
        self.val_X, self.val_Y = self.create_samples(self.n_val,
                                                    min_d, max_d,
                                                    min_l, max_l)
        self.test_X, self.test_Y = self.create_samples(self.n_test,
                                                    min_d, max_d,
                                                    min_l, max_l)
        self.to_unit_norm()

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

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = LongTensor(self.train_Y)
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = LongTensor(self.val_Y)
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = LongTensor(self.test_Y)

    def random_diffusing_filter(self, n_samples, min_l, max_l):
        """
        Create a linear random diffusing filter with L random coefficients
        using the graphs shift operator from G.
        Arguments:
            - L: number of filter coeffcients
        """
        h = np.random.randn(max_l, n_samples)
        N = self.G.W.shape[0]
        H = np.zeros((n_samples, max_l, N, N))

        S = norm_graph(self.G.W.todense())
        Spow = np.zeros((max_l,N,N))
        # Calc powers of S
        for l in range(max_l):
            Spow[l,:,:] = np.linalg.matrix_power(S, l)

        for i in range(n_samples):
            n_coefs = np.random.randint(min_l, max_l)
            coefs = h[:n_coefs,i]

            H[i,:n_coefs,:,:] = coefs[:,None,None]*Spow[:n_coefs,:,:]

        H = np.sum(H, axis=1)
        return H

    def create_samples(self, n_samples, min_d, max_d, min_l, max_l):
        deltas = self.delta_values(n_samples, min_d, max_d)
        delta_S, Y = self.sparse_S(self.G, deltas)
        H = self.random_diffusing_filter(n_samples, min_l, max_l)

        # Increase 1 dimension of delta_S to be T x N x 1
        X = np.matmul(H, delta_S[:,:,None]).squeeze()
        if self.median:
            X = self.median_neighbours_nodes(X, self.G)

        return X, Y



