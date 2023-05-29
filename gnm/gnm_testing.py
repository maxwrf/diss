import numpy as np

from gnm.gnm import GNM
from seed_network import get_seed_network
from utils import params_from_json

np.random.seed(123)


# load config
config = params_from_json("./config.json")

# load a seed network, A = adjacency matrix and D = distance matrix
A, D = get_seed_network(config)

# load a fake parameter sampling space
etalimits = [-7, 7]
gamlimits = [-7, 7]
nruns = 10

p, q = np.meshgrid(np.linspace(etalimits[0], etalimits[1], int(np.sqrt(nruns))),
                   np.linspace(gamlimits[0], gamlimits[1], int(np.sqrt(nruns))))

params = np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)

# testing the model
A = A[:5, :5].astype(int)
D = D[:5, :5]




# # cluster average
# g_clu_avg = GNM(A, D, 3, "clu-avg", params)
# g_clu_avg.main()
# g_clu_avg.b[...,-1]

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
params = np.array([[3,3]])
params = np.repeat(params, 100, axis=0)
g_spatial = GNM(A, D, 3, "spatial", params)
g_spatial.main()
g_spatial.b[...,-1]