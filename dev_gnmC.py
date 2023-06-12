# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['gnm_build'])  # noqa

from GNMC import hello

import time
import h5py
import numpy as np
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph

gnm_rules = [
    'spatial'
]


def generate_param_space(n_runs: int = 100,
                         eta_limts: np.array = [-7, 7],
                         gamma_limits: np.array = [-7, 7]) -> np.array:
    """
        Createas a linear parameter space defined by the eta and gamma bounds
        for the desired number of runs
        """
    p, q = np.meshgrid(np.linspace(
        eta_limts[0], eta_limts[1], int(np.sqrt(n_runs))),
        np.linspace(
        gamma_limits[0], gamma_limits[1], int(np.sqrt(n_runs))))

    return np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)


def main(A_init: np.ndarray,
         D: np.ndarray,
         A_Ys: np.ndarray,
         eta_limits=[-7, 7],
         gamma_limits=[-7, 7],
         n_runs=2,
         n_samples=None,
         store=False
         ):

    start_time = time.time()

    # For development limit samples
    if n_samples is not None:
        A_Ys = A_Ys[:n_samples, ...]

    # generate the parameter space
    params = generate_param_space(n_runs, eta_limits, gamma_limits)
    n_params = params.shape[0]

    # sample x model x params x N ks statistics
    K_all = np.zeros((A_Ys.shape[0],
                      len(gnm_rules),
                      params.shape[0],
                      4))

    # sample x model x params
    K_max_all = np.zeros((A_Ys.shape[0],
                          len(gnm_rules),
                          params.shape[0]))

    # for each data sample
    for i_sample in range(A_Ys.shape[0]):

        A_Y = A_Ys[i_sample, ...]

        # number of target connections
        m = np.count_nonzero(A_Y) // 2

        # number of nodes
        n = A_Y.shape[0]

        # compute energy stats
        x = np.zeros((n, 4))
        x[:, 0] = np.sum(A_Y, axis=0)
        x[:, 1] = Graph.clustering_coeff(A_Y, n)
        x[:, 2] = Graph.betweenness_centrality(A_Y.astype(float), n)
        x[:, 3] = Graph.euclidean_edge_length(A_Y, D)

        # Run the generative models for this sample
        for j_model in range(len(gnm_rules)):
            b = hello(
                A_init,
                D,
                params,
                int(m),
                int(j_model))
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

    return 0


# get the initalized adjacency matrix where > 20% of the patient samples
# connectomes already had connections (target data set)
config = params_from_json("./config.json")
A, D, A_Ys = get_seed_network(config,
                              prop=.2,
                              get_connections=True
                              )
main(A, D, A_Ys, n_runs=1000, n_samples=2, store=False)
