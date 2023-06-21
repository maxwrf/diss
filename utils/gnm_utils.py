import numpy as np
import matplotlib.pyplot as plt

gnm_rules = [
    'spatial',
    'neighbors',
    'matching',
    'clu-avg',
    'clu-min',
    'clu-max',
    'clu-dist',
    'clu-prod',
    'deg-avg',
    'deg-min',
    'deg-max',
    'deg-dist',
    'deg-prod'
]

gnm_loopkup = {
    'spatial': 0,
    'neighbors': 1,
    'matching': 2,
    'clu-avg': 3,
    'clu-min': 4,
    'clu-max': 5,
    'clu-dist': 6,
    'clu-prod': 7,
    'deg-avg': 8,
    'deg-min': 9,
    'deg-max': 10,
    'deg-dist': 11,
    'deg-prod': 12
}


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

    # Calculate the cumulative distribution functions (CDFs)
    cdf_x = np.searchsorted(x_sorted, np.sort(np.concatenate(
        [x_sorted, y_sorted])), side='right') / x.shape[0]
    cdf_y = np.searchsorted(y_sorted, np.sort(np.concatenate(
        [x_sorted, y_sorted])), side='right') / y.shape[0]

    # ks stastistic is the maximum difference
    diff_cdf = np.abs(cdf_x - cdf_y)
    ks_statistic = np.max(diff_cdf)

    return ks_statistic


def reconstruct_A(b: np.array, param_idx: int, n_nodes: int):
    A = np.zeros((n_nodes, n_nodes))

    power_ten = 10**(np.ceil(np.log10(b[:, param_idx])+1) // 2)
    idx_x = (b[:, param_idx] // power_ten - 1).astype(int)
    idx_y = (b[:, param_idx] % power_ten).astype(int)
    A[idx_x, idx_y] = 1
    return A+A.T


def generate_param_space(n_runs: int = 100,
                         eta_limts: np.array = [-7, 7],
                         gamma_limits: np.array = [-7, 7]) -> np.array:
    """
        Createas a linear parameter space defined by the eta and gamma bounds
        for the desired number of runs
        """
    p, q = np.meshgrid(np.linspace(
        eta_limts[0], eta_limts[1], int(np.sqrt(n_runs))),
        np.linspace(
        gamma_limits[0], gamma_limits[1], int(np.sqrt(n_runs))))

    return np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)


x = [1.2, 2.5, 3.7, 4.1, 5.6]
y = [1.0, 2.2, 3.6, 4.4, 5.8]