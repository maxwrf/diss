# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.dirname(current_path))  # noqa

import time
import h5py
import numpy as np
from tqdm.auto import tqdm
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph
from utils.gnm_utils import ks_test
from gnm.gnm import GNM


def main(A_init: np.ndarray,
         D: np.ndarray,
         A_Ys: np.ndarray,
         eta_limits=[-7, 7],
         gamma_limits=[-7, 7],
         n_runs=64,
         n_samples=None,
         store=False
         ):

    start_time = time.time()

    # For development limit samples
    if n_samples is not None:
        A_Ys = A_Ys[:n_samples, ...]

    # generate the parameter space
    params = GNM.generate_param_space(n_runs, eta_limits, gamma_limits)
    n_params = params.shape[0]

    # sample x model x params x N ks statistics
    K_all = np.zeros((A_Ys.shape[0],
                      len(GNM.gnm_rules),
                      params.shape[0],
                      4))

    # sample x model x params
    K_max_all = np.zeros((A_Ys.shape[0],
                          len(GNM.gnm_rules),
                          params.shape[0]))

    # for each data sample
    pbar_out = tqdm(range(A_Ys.shape[0]))
    for i_sample in pbar_out:
        pbar_out.set_description(f"Processing sample {i_sample}")

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
        pbar = tqdm(range(len(GNM.gnm_rules)), leave=False)
        for j_model in pbar:
            pbar.set_description(f"Processing {GNM.gnm_rules[j_model]}")
            model = GNM(A_init, D, m, GNM.gnm_rules[j_model], params)
            model.generate_models()
            K = np.zeros((params.shape[0], 4))

            # over generated graphs from all parameter combinations
            for k_param in range(params.shape[0]):
                # reconstructs the adjacency matrix from the indices
                A_Y_head = model.reconstruct_A(b_index=k_param)

                # compute energy states
                y = np.zeros((n, 4))
                y[:, 0] = np.sum(A_Y_head, axis=0)
                y[:, 1] = Graph.clustering_coeff(A_Y_head, n)
                y[:, 2] = Graph.betweenness_centrality(A_Y_head, n)
                y[:, 3] = Graph.euclidean_edge_length(A_Y_head, D)

                # compute the states
                K[k_param, ...] = np.array(
                    [ks_test(x[..., l], y[..., l]) for l in range(4)])

                # store the results
                K_all[i_sample, j_model, k_param, :] = K[k_param, :]
                K_max_all[i_sample, j_model, k_param] = np.max(K[k_param, :])

    if store:
        with h5py.File(config['results_path'] + "gnm_results.h5", 'w') as f:
            f.create_dataset('K_all', data=K_all)
            f.create_dataset('K_max_all', data=K_max_all)
            f.attrs['n_samples'] = A_Ys.shape[0]
            f.attrs['gnm_rules'] = GNM.gnm_rules
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
main(A, D, A_Ys, n_runs=64, n_samples=4, store=False)
