import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, LongTensor
from scipy.sparse.csgraph import shortest_path

from . import graphutils as gu


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
            X_aux[:, i] = np.median(X[:, np.append(neighbours, i)], axis=1).squeeze()
        return X_aux

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        if n_chans != 0:
            shape = [-1, n_chans, N]
        else:
            shape = [-1, N]
        self.train_X = Tensor(self.train_X).view(shape)
        self.train_Y = Tensor(self.train_Y).view([self.n_train, N])
        self.val_X = Tensor(self.val_X).view(shape)
        self.val_Y = Tensor(self.val_Y).view([self.n_val, N])
        self.test_X = Tensor(self.test_X).view(shape)
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

    def plot_train_signals(self, ids, plot_y=False, show=True):
        if not isinstance(ids, list) and not isinstance(ids, range):
            ids = [ids]
        for id in ids:
            X = self.train_X[id, :]
            if plot_y:
                Y = self.train_Y[id, :]
                _, axes = plt.subplots(1, 2)
                self.G.plot_signal(X, ax=axes[0])
                self.G.plot_signal(Y, ax=axes[1])
            else:
                self.G.plot_signal(X)
        if show:
            plt.show()

    def add_noise(self, p_n, test_only=True):
        if p_n == 0:
            return
        self.test_X = self.add_noise_to_X(self.test_X, p_n)

        if not test_only:
            self.val_X = self.add_noise_to_X(self.val_X, p_n)
            self.train_X = self.add_noise_to_X(self.train_X, p_n)

    def add_noise_to_X(self, X, p_n):
        p_x = np.sum(X**2, axis=1)
        sigma = np.sqrt(p_n * p_x / X.shape[1])
        noise = (np.random.randn(X.shape[0], X.shape[1]).T * sigma).T
        return X + noise

    def to(self, device):
        self.train_X = self.train_X.to(device)
        self.train_Y = self.train_Y.to(device)
        self.val_X = self.val_X.to(device)
        self.val_Y = self.val_Y.to(device)
        self.test_X = self.test_X.to(device)
        self.test_Y = self.test_Y.to(device)

class DenoisingBase(BaseGraphDataset):
    '''
    Base class for the signal Denoising problem, where we try to recover the
    original signal from a noisy version of it. This class should not be used,
    use instead the child classes DenoisingSparse or DenoisingWhite
    '''
    def __init__(self, G, n_samples, p_n, H=None, ftype="classic",
                median=True, neg_coeffs=False, L=5, norm=False):

        assert ftype in ["classic", "neighbours"], \
            'Filter type must be either classic or neighbours'

        super(DenoisingBase, self).__init__(G, n_samples, median)

        assert self.n_train >= self.n_val and self.n_train >= self.n_test, \
                "Need more training samples than validation and test"

        if H is not None:
            self.H = H
        else:
            self.random_diffusing_filter(L, neg_coeffs, ftype)
        
        self.norm = norm

        # Validation and test datasets are a subset of the training dataset
        # (we cannot denoise a signal for which we have not trained for)
        self.train_X, self.train_Y = self.create_samples(self.n_train)
        idxs = np.random.permutation(self.n_train)
        self.val_idx = idxs[:self.n_val]
        self.test_idx = idxs[-self.n_test:]

        self.val_X, self.val_Y = self.train_X[self.val_idx].copy(),\
                                    self.train_Y[self.val_idx].copy()
        self.test_X, self.test_Y = self.train_X[self.test_idx].copy(),\
                                    self.train_Y[self.test_idx].copy()

        if self.norm:
            self.to_unit_norm(y_norm=True)
        
        self.train_Y = self.add_noise_to_X(self.train_Y, p_n)

    def create_samples(self, n_samples):
        orig_signal = self.base_signal(n_samples)

        Y = self.H.dot(orig_signal.T).T
        if self.median:
            Y = self.median_neighbours_nodes(Y, self.G)

        X = np.random.randn(n_samples, self.G.N)

        return X, Y

    def random_diffusing_filter(self, L, neg_coeffs, ftype="classic"):
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
        if ftype == "classic":
            S = gu.norm_graph(self.G.W.todense())
            for l in range(h.size):
                self.H += h[l]*np.linalg.matrix_power(S, l)
        elif ftype == "neighbours":
            distances = shortest_path(self.G.W.todense(), directed=False, unweighted=True)
            for i, j in enumerate(h):
                self.H += j*(distances == i).astype(int)
        else:
            print("WARNING: Unknown filter type. Try with 'classic' or 'neighbours'")
    
    def base_signal(self):
        raise NotImplementedError("Please, use one of the child classes\
                                   DenoisingSparse or DenoisingWhite")


class DenoisingSparse(DenoisingBase):
    '''
    Base class for the Denoising problem, where we try to recover the original
    signal from a noisy version of it. The signal is generated diffusing
    deltas over the graph using a graph filter
    '''
    def __init__(self, G, n_samples, n_delts, p_n, H=None, ftype="classic",
                min_d=-1, max_d=1, median=True, neg_coeffs=False, L=5, norm=False):
        
        self.n_delts = n_delts
        self.min_d = min_d
        self.max_d = max_d

        super(DenoisingSparse, self).__init__(G, n_samples, p_n, H, ftype,
                                              median, neg_coeffs, L, norm)

    def delta_values(self, n_samp):
        n_comms = self.G.info['comm_sizes'].size
        if n_comms > 1:
            step = (self.max_d-self.min_d)/(n_comms-1)
        else:
            step = (self.max_d-self.min_d)/(self.n_delts-1)
        ds_per_comm = np.ceil(self.n_delts/n_comms).astype(int)
        delta_means = np.arange(self.min_d, self.max_d+0.1, step)
        delta_means = np.tile(delta_means, ds_per_comm)[:self.n_delts]
        self.delt_val = np.zeros((self.n_delts, n_samp))
        for i in range(n_samp):
            self.delt_val[:, i] = np.random.randn(self.n_delts)*step/4 + delta_means

    def base_signal(self, n_samp):
        """
        Create random sparse signal s composed of different deltas placed in the
        different communities of the graph. If the graph is an ER, then deltas
        are just placed on random nodes
        """
        delta_val = self.delta_values(n_samp)
        S = np.zeros((self.G.N, n_samp))
        # Randomly assign delta value to comm nodes
        for i in range(n_samp):
            for j in range(delta_val.shape[0]):
                delta = delta_val[j, i]
                com_j = j % self.G.info['comm_sizes'].size
                com_nodes, = np.asarray(self.G.info['node_com'] == com_j).nonzero()
                rand_index = np.random.randint(0, self.G.info['comm_sizes'][com_j])
                S[com_nodes[rand_index], i] = delta
        return S.T


class DenoisingWhite(DenoisingBase):
    '''
    Class for the Denoising problem, where we try to recover the original signal
    from a noisy version of it. The signal is generated diffusing either deltas or
    a white signal over the graph using a graph filter
    '''
    def __init__(self, G, n_samples, p_n, H=None, ftype="classic",
                 median=True, neg_coeffs=False, L=5, norm=False):
        super(DenoisingWhite, self).__init__(G, n_samples, p_n, H, ftype,
                                              median, neg_coeffs, L, norm)

    def base_signal(self, n_samp):
        return np.random.randn(n_samp, self.G.N)


class SourcelocSynthetic(BaseGraphDataset):
    """
    Class for the source localization problem, where 1 delta is diffused over the
    graph and the algorithm has to recover the community of the node where the delta
    was placed.
    """
    def __init__(self, G, n_samples, min_l=10, max_l=25, median=True):

        super(SourcelocSynthetic, self).__init__(G, n_samples, median)

        self.N = self.G.W.shape[0]

        self.calc_powers_S(max_l)

        self.calc_highest_degree_node()

        self.train_X, self.train_Y = self.create_samples(self.n_train,
                                                    min_l, max_l)
        self.val_X, self.val_Y = self.create_samples(self.n_val,
                                                    min_l, max_l)
        self.test_X, self.test_Y = self.create_samples(self.n_test,
                                                    min_l, max_l)
        #self.to_unit_norm()

    def calc_powers_S(self, max_l):
        S = gu.norm_graph(self.G.W.todense())
        self.Spow = np.zeros((max_l,self.N,self.N))
        self.Spow[0,:,:] = np.eye(self.N)
        # Calc powers of S
        for l in range(1, max_l):
            self.Spow[l,:,:] = self.Spow[l-1,:,:] @ S

    def calc_highest_degree_node(self):
        node_deg = self.G.W.sum(axis=1)
        k = self.G.info['comm_sizes'].size
        highest_nodes = np.zeros(k)

        for com in range(k):
            com_nodes, = np.asarray(self.G.info['node_com'] == com).nonzero()
            idx_max = np.argmax(node_deg[com_nodes])
            highest_nodes[com] = com_nodes[idx_max]

        self.G.info['nodes_highest'] = highest_nodes.astype(int)

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = LongTensor(self.train_Y)
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = LongTensor(self.val_Y)
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = LongTensor(self.test_Y)

    def create_samples(self, n_samples, min_l, max_l):
        X = np.zeros((n_samples, self.N))
        Y = np.zeros(n_samples)
        S = gu.norm_graph(self.G.W.todense())

        for i in range(n_samples):
            diff = np.random.randint(min_l, max_l)

            com_idx = np.random.randint(0, self.G.info['comm_sizes'].size)
            #node_idx = np.random.randint(0, self.G.info['comm_sizes'][com_idx])
            #com_nodes, = np.asarray(self.G.info['node_com'] == com_idx).nonzero()

            signal = np.zeros(self.N)
            #signal[com_nodes[node_idx]] = 1
            signal[self.G.info['nodes_highest'][com_idx]] = 1
            X[i,:] = self.Spow[diff,:,:] @ signal
            Y[i] = com_idx

        return X, Y


class SourcelocSyntheticGaussian(BaseGraphDataset):
    def __init__(self, G, n_samples, pos_value=0.5, neg_value=-0.1,
                    diffusion=1, median=True):

        super(SourcelocSyntheticGaussian, self).__init__(G, n_samples, median)

        self.N = self.G.W.shape[0]
        self.diffusion = diffusion

        self.train_X, self.train_Y = self.create_samples(self.n_train,
                                                    pos_value, neg_value)
        self.val_X, self.val_Y = self.create_samples(self.n_val,
                                                    pos_value, neg_value)
        self.test_X, self.test_Y = self.create_samples(self.n_test,
                                                    pos_value, neg_value)
        #self.to_unit_norm()

    def create_samples(self, n_samples, pos_value, neg_value):
        X = np.zeros((n_samples, self.N))
        Y = np.zeros(n_samples)
        for i in range(n_samples):
            com_idx = np.random.randint(0, self.G.info['comm_sizes'].size)

            #com_nodes, = np.asarray(self.G.info['node_com'] == com_idx).nonzero()
            pos_gauss = np.random.randn(self.N) + pos_value
            neg_gauss = np.random.randn(self.N) + neg_value
            X[i,:] = np.where(self.G.info['node_com'] == com_idx, pos_gauss, neg_gauss)
            Y[i] = com_idx

        for i in range(self.diffusion):
            X = X @ self.G.W.todense()

        if self.median:
            X = self.median_neighbours_nodes(X, self.G)
        return X, Y

    def to_tensor(self, n_chans=1):
        N = self.train_X.shape[1]
        self.train_X = Tensor(self.train_X).view([self.n_train, n_chans, N])
        self.train_Y = LongTensor(self.train_Y)
        self.val_X = Tensor(self.val_X).view([self.n_val, n_chans, N])
        self.val_Y = LongTensor(self.val_Y)
        self.test_X = Tensor(self.test_X).view([self.n_test, n_chans, N])
        self.test_Y = LongTensor(self.test_Y)
