# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['gnm_build'])  # noqa

from GNMC import get_gnms

import time
import datetime
import h5py
import numpy as np
from tqdm.auto import tqdm
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph
from utils.gnm_utils import ks_test, reconstruct_A, generate_param_space, gnm_loopkup, gnm_rules


def main(A_init: np.ndarray,
         D: np.ndarray,
         A_Ys: np.ndarray,
         config: dict,
         eta_limits=[-7, 7],
         gamma_limits=[-7, 7],
         n_runs=2,
         n_samples=None,
         store=False,
         dset_name=""
         ):

    start_time = time.time()

    # For development limit samples
    if n_samples != -1:
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
    for sample_idx in tqdm(range(A_Ys.shape[0])):

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
            b, X = get_gnms(
                A_Y,
                A_init[sample_idx, ...],
                D,
                params,
                int(m),
                int(model_num))

            K = np.zeros((params.shape[0], 4))
            n_nodes = A_init.shape[1]

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
            f.create_dataset('eta', data=params[:, 0])
            f.create_dataset('gamma', data=params[:, 1])
            f.attrs['n_samples'] = A_Ys.shape[0]
            f.attrs['gnm_rules'] = gnm_rules
            f.attrs['n_runs'] = n_runs
            f.attrs['eta_limits'] = eta_limits
            f.attrs['gamma_limits'] = gamma_limits

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution Time:", execution_time, "seconds")

    return 0


# TESTING
# config = params_from_json("./config.json")
# A_init, D, A_Ys = get_seed_network(config,
#                               prop=.2,
#                               get_connections=True
#                               )
# main(A_init,
#      D,
#      A_Ys,
#      n_runs=1000,
#      n_samples=2,
#      store=True,
#      dset_name='synthetic')
