# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m
import numpy as np
import networkx as nx

from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from gnm.seed_network import get_seed_network


d_p = "/local/data/mphilcompbio/2022/mw894/diss/fake-seed/"

A, D = get_seed_network()

connections = loadmat(
    d_p + 'example_binarised_connectomes.mat')[
        'example_binarised_connectomes']

modeltype = [
    'sptl',
    'neighbors',
    'matching',
    'clu-avg',
    'clu-min',
    'clu-max',
    'clu-diff',
    'clu-prod',
    'deg-avg',
    'deg-min',
    'deg-max',
    'deg-diff',
    'deg-prod'
]

modelvar = [
    'powerlaw',
    'powerlaw'
]

etalimits = [-7, 7]
gamlimits = [-7, 7]
nruns = 64

p, q = np.meshgrid(np.linspace(etalimits[0], etalimits[1], int(np.sqrt(nruns))),
                   np.linspace(gamlimits[0], gamlimits[1], int(np.sqrt(nruns))))

params = np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)
n_params = params.shape[0]

# main generative network model
generativedata = {}
Asynthall = {}
Eall = {}
Kall = {}


for i_sample in range(2):
    # load the sample
    sample = connections[i_sample, ...]
    G_sample = nx.from_numpy_array(sample)  # create undirected nx graph
    for i, j in G_sample.edges:
        G_sample.edges[i, j]['weight'] = D[i, j]
    m = np.count_nonzero(sample) // 2
    n = sample.shape[0]

    # compute energy stats
    x = [None] * 4
    x[0] = dict(G_sample.degree())
    x[1] = nx.clustering(G_sample)
    x[2] = nx.betweenness_centrality(G_sample)
    x[3] = dict(G_sample.degree(weight='weight'))

    # Run the generative models for this sample
    for i_model in range(13):
        pass
