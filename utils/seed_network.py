# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/ii.%20Computing%20the%20seed%20network.m
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform


def get_seed_network(config: dict, prop=0.2, get_connections=False) -> tuple:
    """
    Loads illustrative seed networks as in 'GNMs for neurodevelopmental struct'
    prop = proportion of connections that were already in place
    """

    # connections across the neurons, across all the sixty eight samples
    connectomes = loadmat(
        config['data_path'] + 'example_binarised_connectomes.mat')[
            'example_binarised_connectomes']

    # remove the first dimension to get average
    connections = np.mean(connectomes, axis=0).squeeze()

    # get index where > prop & prepare corresponding matrix
    index = np.where(connections == prop)
    A = np.zeros(connections.shape)
    A[index] = 1

    # laod distances
    coordinates = loadmat(config['data_path'] +
                          'dk_coordinates.mat')['coordinates']
    D = squareform(pdist(coordinates))

    if get_connections:
        return A, D, connectomes

    return A, D
