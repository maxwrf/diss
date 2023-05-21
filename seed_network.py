# Reference: https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/ii.%20Computing%20the%20seed%20network.m
import numpy as np
from scipy.io import loadmat


def get_seed_network(prop=0.2):
    d_p = "/local/data/mphilcompbio/2022/mw894/diss/fake-seed/"

    # connections across the neurons, across all the sixty eight samples
    connections = loadmat(
        d_p + 'example_binarised_connectomes.mat')[
            'example_binarised_connectomes']

    # remove the first dimension to get average
    connections = np.mean(connections, axis=0).squeeze()

    # get index where > prop & prepare corresponding matrix
    index = np.where(connections == prop)
    A = np.zeros(connections.shape)
    A[index] = 1

    return A

# Location of the neurons (only required for viz?)
# dk_coordinates = loadmat(d_p + 'dk_coordinates.mat')['coordinates']
