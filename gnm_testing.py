import numpy as np
import matplotlib.pyplot as plt

from gnm.gnm import GNM
from gnm.seed_network import get_seed_network
from utils.utils import params_from_json


# load config
config = params_from_json("./config.json")

# load A = adjacency matrix and D = distance matrix (subsample)
A, D = get_seed_network(config)
A = A[:10, :10].astype(int)
D = D[:10, :10]

# load a fake parameters and number of runs
nruns = 10000
params = np.repeat(np.array([[3,3]]), nruns, axis=0)
m = 10

def edge_hist(gnm: GNM, n_largest: int =20):
    # get all edges across all simulations and position
    edges = gnm.b.copy().reshape(gnm.b.shape[0] * gnm.b.shape[1])

    # generate unique and sorted counts
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    uniq_edges = uniq_edges[sorted_indices]
    counts = counts[sorted_indices]

    # plotting
    fig, ax = plt.subplots(1,1, figsize =(10,7))
    ax.bar(uniq_edges[:n_largest].astype(str), counts[:n_largest])
    ax.tick_params(axis='x', rotation=90)

def matlab_sim(nlargest=20):
    # load the data
    p = config['data_path'] + 'testing/r_clu_avg.csv'
    mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
    edges = mlab_data.copy().reshape(mlab_data.shape[0] * mlab_data.shape[1])

    # generate unique and sorted counts
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    uniq_edges = uniq_edges[sorted_indices]
    counts = counts[sorted_indices]

    # plotting
    fig, ax = plt.subplots(1,1, figsize =(10,7))
    ax.bar(uniq_edges[:nlargest].astype(str), counts[:nlargest])
    ax.tick_params(axis='x', rotation=90)


matlab_sim()


# cluster average
g_clu_avg = GNM(A, D, m, "clu-avg", params)
g_clu_avg.main()
edge_hist(g_clu_avg)


# # cluster product
# g_clu_min = GNM(A, D, 3, "clu-prod", params)
# g_clu_min.main()
# g_clu_min.b[...,-1]

# # cluster minimum
# g_clu_min = GNM(A, D, 3, "clu-min", params)
# g_clu_min.main()
# g_clu_min.b[...,-1]

# # cluster maximum
# g_clu_max = GNM(A, D, 3, "clu-max", params)
# g_clu_max.main()
# g_clu_max.b[...,-1]

# # cluster distance
# g_clu_dist = GNM(A, D, 3, "clu-dist", params)
# g_clu_dist.main()
# g_clu_dist.b[...,-1]

# # degree average
# g_deg_avg = GNM(A, D, 3, "deg-avg", params)
# g_deg_avg.main()
# g_deg_avg.b[...,-1]

# # degree product
# g_deg_min = GNM(A, D, 3, "deg-prod", params)
# g_deg_min.main()
# g_deg_min.b[...,-1]

# # degree minimum
# g_deg_min = GNM(A, D, 3, "deg-min", params)
# g_deg_min.main()
# g_deg_min.b[...,-1]

# # degree maximum
# g_deg_max = GNM(A, D, 3, "deg-max", params)
# g_deg_max.main()
# g_deg_max.b[...,-1]

# # degree distance
# g_deg_dist = GNM(A, D, 3, "deg-dist", params)
# g_deg_dist.main()
# g_deg_dist.b[...,-1]

# neighbors
# g_neigh = GNM(A, D, 3, "neighbors", params)
# g_neigh.main()
# g_neigh.b[...,-1]

# matching
# g_matching = GNM(A, D, 3, "matching", params)
# g_matching.main()
# g_matching.b[...,-1]

# matching

# g_spatial = GNM(A, D, 3, "spatial", params)
# g_spatial.main()
# g_spatial.b[...,-1]