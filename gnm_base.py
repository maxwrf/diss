import numpy as np
import networkx as nx

from seed_network import get_seed_network
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat


class GNM():
    def __init__(self,
                 A: np.ndarray,
                 D: np.ndarray,
                 m: int,
                 model_type: str,
                 params: np.ndarray) -> None:

        self.A = A  # adjacency matrix
        self.D = D  # euclidean distances
        self.n_nodes = len(D)  # number of nodes
        self.m = m  # target number of connections
        self.model_type = model_type  # str indicating generative rule
        self.params = params  # matrix of eta and gamma combinations
        # number of parameter combinations
        self.n_params = self.params.shape[0]
        # connections mat for params combs
        self.b = np.zeros((m, self.n_params))
        self.epsilon = 1e-5

        # construct nx grpah
        self.G = nx.from_numpy_array(A)
        for i, j in self.G.edges:
            self.G.edges[i, j]['weight'] = D[i, j]

    def main(self):
        # TODO: implement all other types of generative rules
        if self.model_type == "clu_avg":
            # compute initial the cost matrix
            self.c = np.fromiter(nx.clustering(self.G).values(), dtype=float)
            K = (self.c[:, np.newaxis] + self.c) / 2

            # start the algorithm using the different paramter combos
            for i_param in range(self.n_params):
                eta = self.params[i_param, 0]
                gamma = self.params[i_param, 1]
                self.b[:, i_param] = self.fcn_clu_avg(eta, gamma, K)
        else:
            pass

    def get_clustering_coeff(self):
        """
        Computes the clustering coefficient for each node in A
        """
        clu_coeff = np.zeros(self.n_nodes)
        for i_node in range(self.n_nodes):
            nbrs = np.where(self.A[i_node, :])[0]
            n_nbrs = len(nbrs)
            if n_nbrs > 1:

                S = self.A[nbrs, :][:, nbrs]
                X = self.A[nbrs[:, np.newaxis], nbrs]
                print(np.sum(S) / (n_nbrs ** 2 - n_nbrs))
                clu_coeff[i_node] = np.sum(S) / (n_nbrs ** 2 - n_nbrs)

    def debug_add_rand_edges(self, n=10):
        """
        Randomly adds edges in A
        """
        nrows, ncols = self.A.shape

        random_indices = np.random.choice(
            nrows*ncols, n, replace=False)

        random_indices = np.unravel_index(random_indices, (nrows, ncols))

        for idx_x, idx_y in zip(random_indices[0], random_indices[1]):
            self.A[idx_x, idx_y] = 1

    def fcn_clu_avg(self, eta, gamma, K):
        # the cluster averages
        K = K + self.epsilon

        # compute cost and value
        Fd = np.power(self.D, eta)
        Fk = np.power(K, gamma)

        # compute the prob for adding connection where there is none
        A = self.A > 0
        Ff = Fd * Fk * (~A)

        # stores the numbe of edges for each node
        k = sum(A, 2)

        # get the indicies of the upper right of the p matrix
        u, v = np.triu_indices(self.n, k=1)
        indx = (v - 1) * self.n + u

        # TODO: Need to check, this can only be one of the two, whats delta
        P = Ff.flatten()[indx]
        P_me = Ff[u, v]

        # number of connections we start with
        m_seed = np.count_nonzero(self.A) / 2

        print(m_seed, self.m)

        for i in range(int(m_seed + 1), self.m):
            # select the element to update, change adjacency matrix
            C = np.concatenate(([0], np.cumsum(P_me)))
            r = np.sum(np.random.rand() * C[-1] >= C)
            uu = u[r]
            vv = v[r]

            # update the adjacency matrix
            A[uu, vv] = 1
            A[vv, uu] = 1

            # update the degree matrix
            k[uu] += 1
            k[uu] += 1

            # get where we need to update the clustering coefficients
            bu = A[uu, :]
            su = A[bu, bu]

            bv = A[vv, :]
            sv = A[bv, bv]

            bth = bu & bv

            # update the clustering coefficient in the respective positions
            self.c[bth] = self.c[bth] + 2. / (k[bth]**2 - k[bth])
            self.c[uu] = np.count_nonzero(su) / (k[uu] * (k[uu] - 1))
            self.c[vv] = np.count_nonzero(sv) / (k[vv] * (k[vv] - 1))
            self.c[k <= 1] = 0

            # compute cost matrix from clustering coefficients
            bth[[uu, vv]] = True
            K[:, bth] = np.maximum(self.c[:, np.newaxis], self.c[bth])[
                bth] + self.epsilon
            K[bth, :] = np.maximum(self.c[:, np.newaxis], self.c[bth])[
                :, bth] + self.epsilon

            # Compute the updated probabilities with the new graph
            Ff[bth, :] = Fd[bth, :] * (K[bth, :] ** gamma)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth] ** gamma)
            Ff = Ff * (~A)
            P_me = Ff[u, v]
            print(np.sum(A), "asd")

        print(np.sum(A))
        raise BaseException


# load a seed network, A = adjacency matrix and D = distance matrix
d_p = "/local/data/mphilcompbio/2022/mw894/diss/fake-seed/"
A = get_seed_network()
coordinates = loadmat(d_p + 'dk_coordinates.mat')['coordinates']
D = squareform(pdist(coordinates))

# load a fake parameter sampling space
etalimits = [-7, 7]
gamlimits = [-7, 7]
nruns = 64

p, q = np.meshgrid(np.linspace(etalimits[0], etalimits[1], int(np.sqrt(nruns))),
                   np.linspace(gamlimits[0], gamlimits[1], int(np.sqrt(nruns))))

params = np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)


# testing the model
gnm = GNM(A, D, 20, "clu_avg", params)
# gnm.main()
