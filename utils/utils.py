import json
import numpy as np
import matplotlib.pyplot as plt


def params_from_json(p: str) -> dict:
    params = None
    with open(p, 'r') as f:
        params = json.load(f)

    return params


def plot_edge_freqs(b: np.array, n_largest: int =20):
    """
    Given b a np.array of size edges * simulation
    plots a sorted barplot of the frequencies
    """

    # get all edges across all simulations and position
    edges = b.copy().reshape(b.shape[0] * b.shape[1])

    # generate unique and sorted counts
    uniq_edges, counts = np.unique(edges, axis=0, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    uniq_edges = uniq_edges[sorted_indices]
    counts = counts[sorted_indices]

    # plotting
    fig, ax = plt.subplots(1,1, figsize =(10,7))
    ax.bar(uniq_edges[:n_largest].astype(str), counts[:n_largest])
    ax.tick_params(axis='x', rotation=90)
    plt.close(fig)
    plt.close()
    return fig