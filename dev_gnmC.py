# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['gnm_build'])  # noqa

from GNMC import hello

import time
import datetime
import h5py
import numpy as np
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph
from utils.gnm_utils import ks_test

gnm_rules = [
    'spatial',
    'clu-avg',
    'deg-avg',
    'deg-min',
    'deg-max',
    'deg-dist',
    'deg-prod'
]

gnm_loopkup = {
    'spatial': 0,
    'neighbors': 1,
    'matching': 2,
    'clu-avg': 3,
    'clu-min': 4,
    'clu-max': 5,
    'clu-dist': 6,
    'clu-prod': 7,
    'deg-avg': 8,
    'deg-min': 9,
    'deg-max': 10,
    'deg-dist': 11,
    'deg-prod': 12
}


def reconstruct_A(b: np.array, param_idx: int, n_nodes: int):
    A = np.zeros((n_nodes, n_nodes))

    power_ten = 10**(np.ceil(np.log10(b[:, param_idx])+1) // 2)
    idx_x = (b[:, param_idx] // power_ten - 1).astype(int)
    idx_y = (b[:, param_idx] % power_ten).astype(int)
    A[idx_x, idx_y] = 1
    return A+A.T


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
         store=False,
         dset_name=""
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
    for sample_idx in range(A_Ys.shape[0]):

        A_Y = A_Ys[sample_idx, ...]

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
            model_num = gnm_loopkup[gnm_rules[j_model]]
            b = hello(
                A_init,
                D,
                params,
                int(m),
                int(model_num))

            K = np.zeros((params.shape[0], 4))
            n_nodes = A_init.shape[0]

            # over generated graphs from all parameter combinations
            for param_idx in range(params.shape[0]):
                # reconstructs the adjacency matrix from the indices
                A_Y_head = reconstruct_A(b, param_idx, n_nodes)

                # compute energy states
                y = np.zeros((n, 4))
                y[:, 0] = np.sum(A_Y_head, axis=0)
                y[:, 1] = Graph.clustering_coeff(A_Y_head, n)
                y[:, 2] = Graph.betweenness_centrality(A_Y_head, n)
                y[:, 3] = Graph.euclidean_edge_length(A_Y_head, D)

                # compute the states
                K[param_idx, ...] = np.array(
                    [ks_test(x[..., l], y[..., l]) for l in range(4)])

                # store the results
                K_all[sample_idx, j_model, param_idx, :] = K[param_idx, :]
                K_max_all[sample_idx, j_model,
                          param_idx] = np.max(K[param_idx, :])

    if store:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        with h5py.File(
                config['results_path'] + timestamp + "_" + dset_name + ".h5", 'w') as f:
            f.create_dataset('K_all', data=K_all)
            f.create_dataset('K_max_all', data=K_max_all)
            f.attrs['n_samples'] = A_Ys.shape[0]
            f.attrs['gnm_rules'] = gnm_rules
            f.attrs['n_runs'] = n_runs
            f.attrs['eta_limits'] = eta_limits
            f.attrs['gamma_limits'] = gamma_limits
            f.attrs['eta'] = params[:, 0]
            f.attrs['gamma'] = params[:, 1]

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
main(A,
     D,
     A_Ys,
     n_runs=1000,
     n_samples=2,
     store=True,
     dset_name='synthetic')
