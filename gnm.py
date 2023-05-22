# Reference: MATLAB Brain Connectivity Toolbox
import numpy as np

from seed_network import get_seed_network
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
        self.n_params = self.params.shape[0]  # number of param combos
        self.b = np.zeros((m, 2, self.n_params))  # out, conenctions
        self.epsilon = 1e-5

    def main(self):
        """
        Initiates the network generation leveraging the diffferent rules
        """
        if self.model_type == "clu_avg":
            # compute initial the value matrix
            self.c = self.get_clustering_coeff()
            K = (self.c[:, np.newaxis] + self.c) / 2

            # start the algorithm using the different paramter combos
            for i_param in range(self.n_params):
                eta = self.params[i_param, 0]
                gamma = self.params[i_param, 1]
                self.b[:, :, i_param] = self.gen_clu_avg(eta, gamma, K)
        else:
            pass

    def get_clustering_coeff(self) -> np.array:
        """
        Computes the clustering coefficient for each node in A
        """
        clu_coeff = np.zeros(self.n_nodes)
        for i_node in range(self.n_nodes):
            nbrs = np.where(self.A[i_node, :])[0]
            n_nbrs = len(nbrs)
            if n_nbrs > 1:
                S = self.A[nbrs, :][:, nbrs]
                clu_coeff[i_node] = np.sum(S) / (n_nbrs ** 2 - n_nbrs)

        return clu_coeff

    def debug_add_rand_edges(self, n=10) -> None:
        """
        Randomly adds edges in A
        No loops allowed
        """
        nrows, ncols = self.A.shape

        random_indices = np.random.choice(
            nrows*ncols, n, replace=False)

        random_indices = np.unravel_index(random_indices, (nrows, ncols))

        for idx_x, idx_y in zip(random_indices[0], random_indices[1]):
            if idx_x == idx_y:
                continue
            self.A[idx_x, idx_y] = 1
            self.A[idx_y, idx_x] = 1

    def gen_clu_avg(self, eta, gamma, K) -> None:
        """
        generative nework build using average clustering coefficient
        """

        # maths instability
        K = K + self.epsilon

        # compute cost and value
        Fd = self.D ** eta
        Fk = K ** gamma

        # compute the prob for adding connection where there is none
        Ff = Fd * Fk * (self.A == 0)

        # degree of each node
        k = sum(self.A, 2)

        # get the indicies of the upper right of the p matrix
        u, v = np.triu_indices(self.n_nodes, k=1)
        P = Ff[u, v]

        # number of connections we start with
        m_seed = np.count_nonzero(self.A) / 2

        for i in range(int(m_seed + 1), self.m + 1):
            # select the element to update
            # C = np.concatenate(([0], np.cumsum(P)))
            # r = np.sum(np.random.rand() * C[-1] >= C)
            r = np.argmax(P)
            uu = u[r]
            vv = v[r]

            # update the adjacency matrix
            self.A[uu, vv] = 1
            self.A[vv, uu] = 1

            # update the node degree array
            k[uu] += 1
            k[vv] += 1

            # update the clustering coefficient: at row node)
            bu = np.where(self.A[uu, :])[0]
            su = self.A[bu, :][:, bu]
            self.c[uu] = np.sum(su) / (k[uu] ** 2 - k[uu])

            # update the clustering coefficient: at col node)
            bv = np.where(self.A[vv, :])[0]
            sv = A[bv, :][:, bv]
            self.c[vv] = np.sum(sv) / (k[vv] ** 2 - k[vv])

            # update the clustering coefficient: at common neighbours
            bth = np.intersect1d(bu, bv)
            self.c[bth] = self.c[bth] + 2. / (k[bth]**2 - k[bth])

            # clean up because we have not checked that node degree must > 1
            self.c[k < 2] = 0

            # get all the indices in the value matrix to update
            bth = np.union1d(bth, np.array([uu, vv]))

            # update all rows
            K[:, bth] = (np.repeat(self.c[:, np.newaxis], len(
                bth), axis=1) + self.c[bth]) / 2 + self.epsilon

            # update all cols
            K[bth, :] = (np.repeat(self.c[:, np.newaxis], len(
                bth), axis=1) + self.c[bth]).T / 2 + self.epsilon

            # Compute the updated probabilities with the new graph
            Ff[bth, :] = Fd[bth, :] * (K[bth, :] ** gamma)
            Ff[:, bth] = Fd[:, bth] * (K[:, bth] ** gamma)
            Ff = Ff * (self.A == 0)
            P = Ff[u, v]

        # return indcies of upper triangle adjacent nodes (there is an edge now)
        triu_idx = np.triu_indices(self.A.shape[0], k=1)
        edge_idx = np.where(self.A[triu_idx] == 1)[0]
        out = np.column_stack((triu_idx[0][edge_idx], triu_idx[1][edge_idx]))
        return out


# load a seed network, A = adjacency matrix and D = distance matrix
d_p = "/local/data/mphilcompbio/2022/mw894/diss/fake-seed/"
A, D = get_seed_network()

# load a fake parameter sampling space
etalimits = [-7, 7]
gamlimits = [-7, 7]
nruns = 1

p, q = np.meshgrid(np.linspace(etalimits[0], etalimits[1], int(np.sqrt(nruns))),
                   np.linspace(gamlimits[0], gamlimits[1], int(np.sqrt(nruns))))

params = np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)

params = params*-1
# testing the model
A = A[:10, :10]
D = D[:10, :10]
gnm = GNM(A, D, 10, "clu_avg", params)
gnm.main()
