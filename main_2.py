# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m

import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.append(os.path.dirname(current_path))  # noqa


import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from utils.seed_network import get_seed_network
from utils.config import params_from_json
from utils.graph import Graph
from gnm.gnm import GNM


# get the initalized adjacency matrix where > 20% of the patient samples
# connectomes already had connections (target data set)
config = params_from_json("./config.json")
A, D, connections = get_seed_network(config, prop=.2, get_connections=True)

# generate the parameter space
params = GNM.generate_param_space(
    n_runs=64,
    eta_limts=[-7, 7],
    gamma_limits=[-7, 7])
n_params = params.shape[0]

# main generative network model
generativedata = {}
Asynthall = {}
Eall = {}
Kall = {}


for i_sample in range(2):
    # load the sample
    sample = connections[i_sample, ...]

    # number of connections
    m = np.count_nonzero(sample) // 2

    # number of nodes
    n = sample.shape[0]

    # compute energy stats
    x = np.zeros((n, 4))
    x[:, 0] = np.sum(A, axis=0)
    x[:, 1] = Graph.clustering_coeff(A, n)
    x[:, 2] = Graph.betweenness_centrality(A, n)
    x[:, 3] = Graph.euclidean_edge_length(A, D)

    # Run the generative models for this sample
    for model_type in GNM.gnm_rules:
        print(model_type)
