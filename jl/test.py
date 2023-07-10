import numpy as np


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


data1 = np.array([1.2, 3.4, 2.1, 5.6, 4.2, 8.3])
data2 = np.array([0.5, 2.3, 1.1, 4.9, 3.7, 7.4])

print(ks_test(data1, data2))
