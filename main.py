# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.dirname(current_path))  # noqa

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph
from utils.gnm_utils import ks_test
from gnm.gnm import GNM


def main(A, D, A_Ys,
         n_samples=2,
         eta_limits=[-7, 7],
         gamma_limits=[-7, 7],
         n_runs=64):
    # subsample for development
    A_Ys = A_Ys[0:n_samples, ...]

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
            model = GNM(A, D, m, GNM.gnm_rules[j_model], params)
            model.generate_models()
            nb = model.b.shape[0]
            K = np.zeros((nb, 4))

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

    # VIZ etc.

    # make a table with energy statistics for each model type
    idx = np.indices(K_max_all.shape).reshape(K_max_all.ndim, -1).T
    df_results = pd.DataFrame({'sample_idx': idx[:, 0],
                               'model_idx':  idx[:, 1],
                               'param_idx':  idx[:, 2],
                               'model_name': np.array(GNM.gnm_rules)[idx[:, 1]],
                               'eta': params[:, 0][idx[:, 2]],
                               'gamma': params[:, 1][idx[:, 2]],
                               'max_energy': np.ravel(K_max_all)
                               })

    # get the top performing parameter combination each (min energy)
    df_top_performing = df_results.groupby(['sample_idx',
                                            'model_idx']).apply(
        lambda g: g.loc[g['max_energy'].idxmin(), :]).reset_index(drop=True)

    # energy landscape
    model = "spatial"

    for i_sample in range(A_Ys.shape[0]):
        data = df_results.loc[(df_results.sample_idx == i_sample) & (
            df_results.model_name == model), ["eta", "gamma", "max_energy"]].values

        landscape = data[:, -1].reshape(
            (np.unique(data[:, 0]).size, np.unique(data[:, 1]).size))

        # plot
        fig, ax = plt.subplots()
        im = ax.imshow(landscape, cmap=plt.colormaps['viridis'].reversed())
        ax.invert_yaxis()
        ax.set_xticks(
            np.arange(len(np.unique(data[:, 0]))), labels=np.unique(data[:, 0]))
        ax.set_xlabel("Eta")
        ax.set_yticks(
            np.arange(len(np.unique(data[:, 1]))), labels=np.unique(data[:, 1]))
        ax.set_ylabel("Gamma")
        cbar = ax.figure.colorbar(im, ax=ax)
        fig.savefig('./plots/landscape.png')
    return 1


# get the initalized adjacency matrix where > 20% of the patient samples
# connectomes already had connections (target data set)
config = params_from_json("./config.json")
A, D, A_Ys = get_seed_network(config, prop=.2, get_connections=True)

main(A, D, A_Ys, n_samples=2)
