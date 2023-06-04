import numpy as np
import matplotlib.pyplot as plt


def plot_edge_freqs(b: np.array, n_largest: int = 20):
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.bar(uniq_edges[:n_largest].astype(str), counts[:n_largest])
    ax.tick_params(axis='x', rotation=90)
    plt.close(fig)
    plt.close()
    return fig


def ks_test(x, y):
    """
    Computes the Kolmogorov-Smirnov (KS) between two arrays.
    Is a non-parametric test of equality of continous 1D probability 
    distributions from two samples
    A = adjacency matrix
    n = number of nodes
    """
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_sorted = np.sort(np.concatenate([x_sorted, y_sorted]))

    # Calculate the cumulative distribution functions (CDFs)
    cdf_x = np.searchsorted(x_sorted, np.sort(np.concatenate(
        [x_sorted, y_sorted])), side='right') / x.shape[0]
    cdf_y = np.searchsorted(y_sorted, np.sort(np.concatenate(
        [x_sorted, y_sorted])), side='right') / y.shape[0]

    # ks stastistic is the maximum difference
    diff_cdf = np.abs(cdf_x - cdf_y)
    ks_statistic = np.max(diff_cdf)

    return ks_statistic
