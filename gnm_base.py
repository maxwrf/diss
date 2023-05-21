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
        self.m = m  # target number of connections
        self.model_type = model_type  # str indicating generative rule
        self.params = params  # matrix of eta and gamma combinations
        # number of parameter combinations
        self.n_params = self.params.shape[0]
        # connections mat for params combs
        self.b = np.zeros((m, self.n_params))

        n = len(D)

        # construct nx grpah
        self.G = nx.from_numpy_array(A)
        for i, j in self.G.edges:
            self.G.edges[i, j]['weight'] = D[i, j]

    def main(self):
        # TODO: implement all other types of generative rules
        if self.model_type == "clu_avg":
            self.clu = nx.clustering(self.G)
            # Kseed = (self.clu[:, np.newaxis] + self.clu) / 2
            for i_param in range(self.n_params):
                eta = params[i_param, 0]
                gam = params[i_param, 1]
                continue
                b[:, i_param] = fcn_clu_avg(
                    A, Kseed, D, m, eta, gam, modelvar, epsilon)
        else:
            pass


# load a seed network and a
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
gnm.main()
