import numpy as np
import matplotlib.pyplot as plt
from pygsp.graphs import Graph, StochasticBlockModel, ErdosRenyi, BarabasiAlbert
from torch import Tensor, LongTensor
from scipy.sparse.csgraph import shortest_path

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
                median=True, neg_coeffs=False, L=5):

        assert ftype in ["classic", "neighbours"], \
            'Filter type must be either classic or neighbours'

        super(DenoisingBase, self).__init__(G, n_samples, median)

        assert self.n_train >= self.n_val and self.n_train >= self.n_test, \
                "Need more training samples than validation and test"

        if H is not None:
            self.H = H
        else:
            self.random_diffusing_filter(L, neg_coeffs, ftype)

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

        self.train_Y = self.add_noise_to_X(self.train_Y, p_n)

        # self.to_unit_norm(y_norm=True)

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
            S = norm_graph(self.G.W.todense())
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
                min_d=-1, max_d=1, median=True, neg_coeffs=False, L=5):
        
        self.n_delts = n_delts
        self.min_d = min_d
        self.max_d = max_d

        super(DenoisingSparse, self).__init__(G, n_samples, p_n, H, ftype,
                                              median, neg_coeffs, L)

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
                 median=True, neg_coeffs=False, L=5):
        super(DenoisingWhite, self).__init__(G, n_samples, p_n, H, ftype,
                                              median, neg_coeffs, L)

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
        S = norm_graph(self.G.W.todense())
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
        S = norm_graph(self.G.W.todense())

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
