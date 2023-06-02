import numpy as np
import matplotlib.pyplot as plt

from gnm.gnm import GNM
from gnm.seed_network import get_seed_network
from ..utils.utils import params_from_json, plot_edge_freqs


# load config
config = params_from_json("./config.json")

# load A = adjacency matrix and D = distance matrix (subsample)
A, D = get_seed_network(config)
A = A[:10, :10].astype(int)
D = D[:10, :10]

# load a fake parameters and number of runs
nruns = 10000
params = np.repeat(np.array([[3, 3]]), nruns, axis=0)
m = 10


# cluster average
p = config['data_path'] + 'testing/r_clu_avg.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_clu_avg = GNM(A, D, m, "clu-avg", params)
g_clu_avg.main()
plot_edge_freqs(g_clu_avg.b)

# cluster product
p = config['data_path'] + 'testing/r_clu_prod.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_clu_prod = GNM(A, D, m, "clu-prod", params)
g_clu_prod.main()
plot_edge_freqs(g_clu_prod.b)

# cluster minimum
p = config['data_path'] + 'testing/r_clu_min.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_clu_min = GNM(A, D, m, "clu-min", params)
g_clu_min.main()
plot_edge_freqs(g_clu_min.b)

# cluster maximum
p = config['data_path'] + 'testing/r_clu_max.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_clu_max = GNM(A, D, m, "clu-max", params)
g_clu_max.main()
plot_edge_freqs(g_clu_max.b)

# cluster dist
p = config['data_path'] + 'testing/r_clu_diff.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_clu_dist = GNM(A, D, m, "clu-dist", params)
g_clu_dist.main()
plot_edge_freqs(g_clu_dist.b)

# degree average
p = config['data_path'] + 'testing/r_deg_avg.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_deg_avg = GNM(A, D, m, "deg-avg", params)
g_deg_avg.main()
plot_edge_freqs(g_deg_avg.b)

# degree product
p = config['data_path'] + 'testing/r_deg_prod.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_deg_prod = GNM(A, D, m, "deg-prod", params)
g_deg_prod.main()
plot_edge_freqs(g_deg_prod.b)

# degree minimum
p = config['data_path'] + 'testing/r_deg_min.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_deg_min = GNM(A, D, m, "deg-min", params)
g_deg_min.main()
plot_edge_freqs(g_deg_min.b)

# degree minimum
p = config['data_path'] + 'testing/r_deg_max.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_deg_max = GNM(A, D, m, "deg-max", params)
g_deg_max.main()
plot_edge_freqs(g_deg_max.b)

# degree distance
p = config['data_path'] + 'testing/r_deg_diff.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_deg_dist = GNM(A, D, m, "deg-dist", params)
g_deg_dist.main()
plot_edge_freqs(g_deg_dist.b)

# neighbors
p = config['data_path'] + 'testing/r_neighbors.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_neighbors = GNM(A, D, m, "neighbors", params)
g_neighbors.main()
plot_edge_freqs(g_neighbors.b)

# matching
p = config['data_path'] + 'testing/r_matching.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_matching = GNM(A, D, m, "matching", params)
g_matching.main()
plot_edge_freqs(g_matching.b)

# spatial
p = config['data_path'] + 'testing/r_spatial.csv'
mlab_data = np.genfromtxt(p, delimiter=',').astype(int)
plot_edge_freqs(mlab_data)

g_spatial = GNM(A, D, m, "spatial", params)
g_spatial.main()
plot_edge_freqs(g_spatial.b)
