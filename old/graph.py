import numpy as np


class Graph:
    @staticmethod
    def clustering_coeff(A: np.array, n: int) -> np.array:
        """
        Computes the clustering coefficients for each node
        A = adjacency matrix
        n = number of nodes
        """
        clu_coeff = np.zeros(n)
        for i_node in range(n):
            nbrs = np.where(A[i_node, :])[0]
            n_nbrs = len(nbrs)
            if n_nbrs > 1:
                S = A[nbrs, :][:, nbrs]
                clu_coeff[i_node] = np.sum(S) / (n_nbrs ** 2 - n_nbrs)

        return clu_coeff

    @staticmethod
    def betweenness_centrality(A: np.array,  n: int) -> np.array:
        """
        Computes the betweenes centrality for each node
        A = adjacency matrix
        n = number of nodes
        Ref: https://arxiv.org/pdf/0809.1906.pdf
        Ref: https://github.com/fiuneuro/brainconn/blob/c24bd15/brainconn/centrality/centrality.py#L12
        """
        if not (A.dtype == np.float64 or A.dtype == np.float32):
            raise BaseException

        # FORWARD PASS
        n = len(A)  # number of nodes
        eye = np.eye(n)  # identity matrix
        d = 1  # path length
        NPd = A.copy()  # number of paths of length |d|
        NSPd = A.copy()  # number of shortest paths of length |d|
        NSP = A.copy()  # number of shortest paths of any length
        L = A.copy()  # length of shortest paths

        # shortes paths of length 1 are only thos of node with itself
        NSP[np.where(eye)] = 1
        L[np.where(eye)] = 1

        # as long as there are still shortest paths of the current length d
        while np.any(NSPd):
            d += 1
            print(d)
            # Computes the number of paths connecting i & j of length d
            NPd = np.dot(NPd, A)

            # if no connection between edges yet, this is the shortest path
            NSPd = NPd * (L == 0)
            # Add the new shortest path entries (in L multiply with length)
            NSP += NSPd
            L += (d * (NSPd != 0))

            print(NSP[14, :])

        L[L == 0] = np.inf      # L for disconnected vertices is inf
        L[np.where(eye)] = 0    # no loops
        NSP[NSP == 0] = 1  # NSP for disconnected vertices is 1

        print(np.allclose(NSP, NSP.T))

        # BACKWARD PASS
        # dependecy number of shortest paths from i to any other vertex that
        # pass through j
        DP = np.zeros((n, n))  # vertex on vertex dependency
        diam = d - 1  # the maximum distance between any two nodes

        # iterate from longest shortest path to shortest
        for d in range(diam, 1, -1):
            # DPd1 is dependency, shortes paths from i to any other vertex that
            # pas through i with length of d
            DPd1 = np.dot(((L == d) * (1 + DP) / NSP), A.T) * \
                ((L == (d - 1)) * NSP)
            DP += DPd1

        return np.sum(DP, axis=0) / 2

    @staticmethod
    def euclidean_edge_length(A: np.array, W: np.array) -> np.array:
        """
        Computes the summed euclidean edge length for each vertex
        A = adjacency matrix
        W = edge weight matrix
        """
        return (W * A).sum(1)


# Graph.betweenness_centrality(np.array([
#     [0, 0, 1, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 0, 0],
#     [1, 1, 0, 1, 0, 0, 0],
#     [0, 0, 1, 0, 1, 0, 0],
#     [0, 0, 0, 1, 0, 1, 1],
#     [0, 0, 0, 0, 1, 0, 1],
#     [0, 0, 0, 0, 1, 1, 0]
# ], dtype=float), 7)


A = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0]], dtype=float)

Graph.betweenness_centrality(A, 20)
