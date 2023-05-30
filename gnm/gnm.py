# Reference: MATLAB Brain Connectivity Toolbox
import numpy as np

class GNM():
    def __init__(self,
                 A: np.ndarray,
                 D: np.ndarray,
                 m: int,
                 model_type: str,
                 params: np.ndarray) -> None:

        self.A = np.array(A, copy=True)  # adjacency matrix
        self.D = D  # euclidean distances
        self.n_nodes = len(D)  # number of nodes
        self.m = m  # target number of connections
        self.model_type = model_type  # str indicating generative rule
        self.params = params  # matrix of eta and gamma combinations
        self.n_params = self.params.shape[0]  # number of param combos
        self.b = np.zeros((m, self.n_params),dtype=int)  # n_connenctions, runs
        self.epsilon = 1e-5

        self.funcs = {
            'avg': lambda c_1, c_2: np.add(c_1, c_2) / 2,
            'dist': lambda c_1, c_2: np.abs(np.subtract(c_1, c_2)),
            'max': np.maximum,
            'min': np.minimum,
            'prod': np.multiply
        }

    @staticmethod
    def generate_param_space(n_runs: int = 100, 
                             eta_limts: np.array = [-7, 7],
                             gamma_limits: np.array =[-7, 7]) -> np.array:
        """
        Createas a linear parameter space defined by the eta and gamma bounds
        for the desired number of runs
        """
        p, q = np.meshgrid(np.linspace(
            eta_limts[0], eta_limts[1], int(np.sqrt(n_runs))),
                        np.linspace(
            gamma_limits[0], gamma_limits[1], int(np.sqrt(n_runs))))

        return np.unique(np.vstack((p.flatten(), q.flatten())).T, axis=0)

    def main(self):
        """
        Initiates the network generation leveraging the diffferent rules
        """
        print('Model type:', self.model_type)

        # copy of the original adjacency matrix to reset after each param combo
        A_initial = np.array(self.A, copy=True)

        # start the algorithm using the different paramter combos
        for i_param in range(self.n_params):
            # reset adjacency matrix and value matrix
            self.A = np.array(A_initial, copy=True)
            self.K = None

            eta = self.params[i_param, 0]
            gamma = self.params[i_param, 1]
            self.b[:, i_param] = self.__main(eta, gamma)


    def get_clustering_coeff(self) -> np.array:
        """
        Computes the clustering coefficient for each node in A
        """
        clu_coeff = np.zeros(self.n_nodes)
        for i_node in range(self.n_nodes):
            nbrs = np.where(self.A[i_node, :])[0]
            n_nbrs = len(nbrs)
            if n_nbrs > 1:
                S = self.A[nbrs, :][:, nbrs]
                clu_coeff[i_node] = np.sum(S) / (n_nbrs ** 2 - n_nbrs)

        return clu_coeff

    def debug_add_rand_edges(self, n=10) -> None:
        """
        Randomly adds edges in A
        No loops allowed
        """
        nrows, ncols = self.A.shape

        random_indices = np.random.choice(
            nrows*ncols, n, replace=False)

        random_indices = np.unravel_index(random_indices, (nrows, ncols))

        for idx_x, idx_y in zip(random_indices[0], random_indices[1]):
            if idx_x == idx_y:
                continue
            self.A[idx_x, idx_y] = 1
            self.A[idx_y, idx_x] = 1

    def __init_K(self) -> None:
        """
        Initializes the value matrix
        """
        if self.model_type == 'neighbors':
            self.K = self.A.dot(A)*~np.eye(self.A.shape[0], dtype=bool)
        elif self.model_type == 'matching':
            self.K = self.get_matching_indices()
        elif self.model_type == 'spatial':
            self.K = np.ones(self.A.shape)
        else:
            # if cluster or degree models
            if self.model_type[0:3] == 'clu':
                self.stat = self.get_clustering_coeff()
            elif self.model_type[0:3] == 'deg':
                self.stat = np.sum(self.A, axis=0)
            f = self.funcs[self.model_type[4:]]
            self.K = f(self.stat[:,np.newaxis], self.stat)
        
        # minimum prob
        self.K = self.K + self.epsilon
    
    def __update_K(self, bth, k) -> None:
        """
        Updates the value matrix after a new edge has been added
        """
        if self.model_type == 'neighbors':
            # needs to update all edges where one vertex is either uu or vv
            # and the vertex has an edge shared with vv or uu respectively
            uu, vv = bth

            # get all the nodes connected to uu
            x = self.A[uu,:].copy()
            x[vv] = 0

            # get all the nodes connected to vv
            y = self.A[:, vv].copy()
            y[uu] = 0

            # the nodes that are connected to uu (as in x) but also to vv 
            # (check here) will have annother common neighbor
            self.K[vv, x] +=1
            self.K[x, vv] +=1
            
            # those nodes that are connected vv (as in y) but also to uu 
            # (check here) will have annother common neigbor
            self.K[y, uu] +=1
            self.K[uu, y] +=1

        elif self.model_type == 'matching':
            # need to update all nodes that have common neighbors with uu and
            # vv respectively

            uu, vv = bth
            update_uu = np.where(self.A.dot(self.A[:, uu]))[0]
            update_vv = np.where(self.A.dot(self.A[:, vv]))[0]
            update_uu = update_uu[update_uu!= uu]
            update_vv = update_vv[update_vv!= vv]

            # for each of these node combinations we recompute matching score
            # TODO: refactor
            for j in update_uu:
                connects = sum(k[[uu, j]]) - 2* self.A[uu, j]
                union_connects = np.dot(self.A[uu,:], self.A[j,:]) * 2
                score = union_connects / connects if connects > 0 else self.epsilon
                self.K[uu, j] = score
                self.K[j, uu] = score
                
            for j in update_vv:
                connects = sum(k[[vv, j]]) - 2* self.A[vv, j]
                union_connects = np.dot(self.A[vv,:], self.A[j,:]) * 2
                score = union_connects / connects if connects > 0 else self.epsilon
                self.K[vv, j] = score
                self.K[j, vv] = score

        elif self.model_type == 'spatial':
            return
        
        else:
            # for clu an deg models, update all rows & columns for verices in bth
            f = self.funcs[self.model_type[4:]]

            self.K[:, bth] = f(np.repeat(self.stat[:, np.newaxis], len(
                    bth), axis=1), self.stat[bth])

            self.K[bth, :] = f(np.repeat(self.stat[:, np.newaxis], len(
                    bth), axis=1), self.stat[bth]).T

            # numerical stability
            self.K[:, bth] = self.K[:, bth] + self.epsilon
            self.K[bth, :] = self.K[bth, :] + self.epsilon

    def __update_stat(self, uu, vv, k) -> np.array:
        if self.model_type[0:3] == 'clu':
            # update the clustering coefficient: at row node
            bu = np.where(self.A[uu, :])[0]
            su = self.A[bu, :][:, bu]
            self.stat[uu] = np.sum(su) / (k[uu] ** 2 - k[uu])

            # update the clustering coefficient: at col node)
            bv = np.where(self.A[vv, :])[0]
            sv = self.A[bv, :][:, bv]
            self.stat[vv] = np.sum(sv) / (k[vv] ** 2 - k[vv])

            # update the clustering coefficient: at common neighbours
            bth = np.intersect1d(bu, bv)
            self.stat[bth] = self.stat[bth] + 2. / (k[bth]**2 - k[bth])

            # clean up because we have not checked that node degree must > 1
            self.stat[k < 2] = 0

            # get all the indices in the value matrix to update
            bth = np.union1d(bth, np.array([uu, vv]))
        elif self.model_type[0:3] == 'deg':
            self.stat = k
            bth = np.array([uu, vv])
        else: # for neighbours and matching
            bth = np.array([uu, vv])
        return bth

    def __main(self, eta, gamma) -> None:
        """
        generative nework build
        """

        # compute initial value matrix
        self.__init_K()

        # compute cost and value
        Fd = self.D ** eta
        Fk = self.K ** gamma

        # compute the prob for adding connection where there is none
        Ff = Fd * Fk * (self.A == 0)

        # degree of each node
        k = np.sum(self.A, axis=0)

        # get the indicies of the upper right of the p matrix
        u, v = np.triu_indices_from(self.A, k=1)
        P = Ff[u, v]

        # number of connections we start with
        m_seed = np.count_nonzero(self.A) / 2

        for i in range(int(m_seed + 1), self.m + 1):
            # select the element to update (biased dice)
            #C = np.concatenate(([0], np.cumsum(P)))
            #r = np.sum(np.random.rand() * C[-1] >= C)
            r = np.random.choice(range(len(P)), p=P/sum(P))

            uu = u[r]
            vv = v[r]

            # update the node degree array
            k[uu] += 1
            k[vv] += 1

            # update the adjacency matrix
            self.A[uu, vv] = 1
            self.A[vv, uu] = 1

            # update the statistic
            bth = self.__update_stat(uu, vv, k)

            # update value matrix
            self.__update_K(bth, k)

            # Compute the updated probabilities with the new graph
            Ff[bth, :] = Fd[bth, :] * (self.K[bth, :] ** gamma)
            Ff[:, bth] = Fd[:, bth] * (self.K[:, bth] ** gamma)
            Ff = Ff * (self.A == 0)
            P = Ff[u, v]


        # return indcies of upper triangle adjacent nodes (there is an edge now)
        mask = np.triu(self.A, k=1)
        indices = np.where(mask)
        return np.array([(idx_x+1)*10+idx_y for idx_x,idx_y in zip(indices[0], indices[1])], dtype=int)

    def get_matching_indices(self) -> np.array:
        """
        For any two nodes in the adjacency matrix, computes the overlap in the 
        connections.
        Note that if two nodes have a common neigbour, the two edeges are both
        counted as part of the intesection set.
        Any connection between the two nodes is excluded
        """

        # Compute the degree (sum of neighbors)
        degree = np.sum(self.A, axis=0)
        
        # Compute the intersection matrix (common neighbors)
        intersection = np.dot(self.A, self.A)*~np.eye(self.A.shape[0], dtype=bool)

        # Compute the union of connections, exclude connections between nodes
        union = degree[:, np.newaxis] + degree - 2*self.A 

        return np.divide(intersection*2, union, where=union!=0)


