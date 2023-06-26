// GNM rules
// 0: 'spatial',
// 1: 'neighbors',
// 2: 'matching',
// 3: 'clu-avg',
// 4: 'clu-min',
// 5: 'clu-max',
// 6: 'clu-dist',
// 7: 'clu-prod',
// 8: 'deg-avg',
// 9: 'deg-min',
// 10: 'deg-max',
// 11: 'deg-dist',
// 12: 'deg-prod'

#include <math.h>
#include <vector>
#include "gnm.h"
#include <iostream>

double GNM::ksTest(
    const std::vector<double> &x,
    const std::vector<double> &y)
{
    std::vector<double> xSorted = x;
    std::vector<double> ySorted = y;

    std::sort(xSorted.begin(), xSorted.end());
    std::sort(ySorted.begin(), ySorted.end());

    std::vector<double> combinedData;
    combinedData.insert(combinedData.end(), xSorted.begin(), xSorted.end());
    combinedData.insert(combinedData.end(), ySorted.begin(), ySorted.end());

    std::sort(combinedData.begin(), combinedData.end());

    std::vector<double> sortedCombined = combinedData;

    std::sort(sortedCombined.begin(), sortedCombined.end());

    std::vector<double> cdf_x(sortedCombined.size());
    std::vector<double> cdf_y(sortedCombined.size());

    for (size_t i = 0; i < sortedCombined.size(); ++i)
    {
        size_t count_x = std::lower_bound(xSorted.begin(), xSorted.end(), sortedCombined[i]) - xSorted.begin();
        size_t count_y = std::lower_bound(ySorted.begin(), ySorted.end(), sortedCombined[i]) - ySorted.begin();
        cdf_x[i] = static_cast<double>(count_x) / x.size();
        cdf_y[i] = static_cast<double>(count_y) / y.size();
    }

    std::vector<double> diff_cdf(cdf_x.size());
    for (size_t i = 0; i < cdf_x.size(); ++i)
    {
        diff_cdf[i] = std::abs(cdf_x[i] - cdf_y[i]);
    }

    return *std::max_element(diff_cdf.begin(), diff_cdf.end());
}

std::vector<double> GNM::getBetweennesCentrality(
    std::vector<std::vector<double> > &A,
    int n)
{
    // FORWARD PASS
    double d = 1.0;                           // path length
    std::vector<std::vector<double> > NPd = A; // number of paths of length |d|
    std::vector<std::vector<double> > NSP = A; // number of shortest paths of any length
    std::vector<std::vector<double> > L = A;   // length of shortest paths

    // shortest paths of length 1 are only those of node with itself
    for (int i = 0; i < n; i++)
    {
        NSP[i][i] = 1.0;
        L[i][i] = 1.0;
    }

    // as long as there are still shortest paths of the current length d
    // break out of the loop if none of the nodes i has a shortest path of the length d
    bool hasNSPd = true;
    while (hasNSPd)
    {
        hasNSPd = false;
        ++d;

        std::vector<std::vector<double> > temp = NPd;
        for (int i = 0; i < n; ++i)
        {
            for (int j = i; j < n; ++j)
            {
                // Compute the number of paths connecting i & j of length d
                for (int k = 0; k < n; ++k)
                {
                    temp[i][j] = temp[j][i] += NPd[i][k] * A[k][j];
                }

                // If there is such path and no shorter entry, add d to the L matrix
                if (temp[i][j] > 0.0 && L[i][j] == 0.0)
                {
                    NSP[i][j] = NSP[j][i] += temp[i][j];
                    L[i][j] = L[j][i] = d;
                    hasNSPd = true;
                }
            }
        }
        NPd = temp;
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            if (L[i][j] == 0.0)
            {
                L[i][j] = L[j][i] = INFINITY;
            }
            if (NSP[i][j] == 0.0)
            {
                NSP[i][j] = NSP[j][i] = 1.0;
            }
        }
    }

    // BACKWARD PASS
    std::vector<double> result(n, 0.0);
    std::vector<std::vector<double> > DP(n, std::vector<double>(n, 0.0)); // vertex on vertex dependency
    double diam = d - 1.0;                                               // the maximum distance between any two nodes

    // iterate from longest shortest path to shortest
    for (double currentD = diam; currentD > 1.0; currentD--)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < n; k++)
                {
                    sum += ((L[i][k] == currentD) * (1 + DP[i][k]) / NSP[i][k]) * A[j][k];
                }
                DP[i][j] += sum * ((L[i][j] == (currentD - 1)) * NSP[i][j]);
                result[j] += sum * ((L[i][j] == (currentD - 1)) * NSP[i][j]) / 2;
            }
        }
    }

    return result;
}

std::vector<double> GNM::getClusteringCoeff(
    std::vector<std::vector<double> > &A,
    int n_nodes,
    std::vector<double> &k)
{
    std::vector<double> result(n_nodes);
    for (int i_node = 0; i_node < n_nodes; ++i_node)
    {
        if (k[i_node] > 1) // only if there are more than one neighbors
        {
            // get the neighbors
            int neighbors[int(k[i_node])];
            int n_neighbors = 0;
            for (int j = 0; j < n_nodes; ++j)
            {
                if (A[i_node][j])
                {
                    neighbors[n_neighbors] = j;
                    n_neighbors++;
                }
            }

            // get the connections across neighbors
            int sum_S = 0;
            for (int i = 0; i < n_neighbors; ++i)
            {
                for (int j = i + 1; j < n_neighbors; ++j)
                {
                    sum_S += A[neighbors[i]][neighbors[j]];
                }
            }

            result[i_node] = (double)(2 * sum_S) / (double)(k[i_node] * (k[i_node] - 1));
        }
        else
        {
            result[i_node] = 0;
        }
    }
    return result;
}

std::vector<double> GNM::getEdgeLength(
    std::vector<std::vector<double> > &A,
    double **D,
    int n_nodes)
{
    std::vector<double> result(n_nodes, 0.0);

    for (int i = 0; i < n_nodes; ++i)
    {
        for (int j = i + 1; j < n_nodes; ++j)
        {
            result[i] += A[i][j] * D[i][j];
            result[j] += A[i][j] * D[i][j];
        }
    }

    return result;
}

std::vector<int> GNM::updateClusteringCoeff(int uu, int vv)
{
    // get the neighbors at row node (uu)
    int uu_neighbors[(int)k_current[uu]];
    int uu_n_neighbors = 0;

    // get the neighbors at col node (vv)
    int vv_neighbors[(int)k_current[vv]];
    int vv_n_neighbors = 0;

    // get common neighbor at row and col nodes
    int uv_n_neighbors = 0;
    std::vector<int> uv_neighbors;

    for (int i = 0; i < n_nodes; ++i)
    {
        // row neighbor
        if (A_current[uu][i])
        {
            uu_neighbors[uu_n_neighbors] = i;
            uu_n_neighbors++;
        }

        // col neighbor
        if (A_current[vv][i])
        {
            vv_neighbors[vv_n_neighbors] = i;
            vv_n_neighbors++;
        }

        // common neighbors
        if ((A_current[uu][i]) && (A_current[vv][i]))
        {
            uv_n_neighbors++;
            uv_neighbors.push_back(i);
        }
    }

    // get connections across row neighbors and update
    int sum_S_uu = 0;
    for (int i = 0; i < uu_n_neighbors; ++i)
    {
        for (int j = i + 1; j < uu_n_neighbors; ++j)
        {
            sum_S_uu += A_current[uu_neighbors[i]][uu_neighbors[j]];
        }
    }
    clu_coeff_current[uu] = (double)(2 * sum_S_uu) / (double)(k_current[uu] * (k_current[uu] - 1));

    // get connections across col neighbors and update
    int sum_S_vv = 0;
    for (int i = 0; i < vv_n_neighbors; ++i)
    {
        for (int j = i + 1; j < vv_n_neighbors; ++j)
        {
            sum_S_vv += A_current[vv_neighbors[i]][vv_neighbors[j]];
        }
    }
    clu_coeff_current[vv] = (double)(2 * sum_S_vv) / (double)(k_current[vv] * (k_current[vv] - 1));

    // get connections across common neighbors and update (always + 2/possible)
    for (int i = 0; i < uv_neighbors.size(); i++)
    {
        clu_coeff_current[i] = clu_coeff_current[i] + (double)2 / (double)(k_current[i] * (k_current[i] - 1));
    }

    // cleanup
    for (int i = 0; i < n_nodes; i++)
    {
        if (k_current[i] < 3)
        {
            clu_coeff_current[i] = 0;
        }
    }

    return uv_neighbors;
}

void GNM::resetAcurrent()
{
    // Initialize a copy of A_init to work on
    for (int i = 0; i < n_nodes; ++i)
    {
        for (int j = 0; j < n_nodes; ++j)
        {
            A_current[i][j] = A_init[i][j];
        }
    }
}

void GNM::initK()
// Initialize the value matrix
{
    switch (model)
    {
    case 0: // spatial
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = 1.0;
            }
        }
        break;

    case 1: // neighbors
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = 0;

                if (!(i == j))
                {
                    for (int l = 0; l < n_nodes; ++l)
                    {
                        K_current[i][j] = K_current[j][i] += A_current[i][l] * A_current[j][l];
                    }
                }

                K_current[i][j] = K_current[j][i] += epsilon;
            }
        }
        break;

    case 2: // matching
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                // get the common neighbors, the intersection
                double intersect_connects = 0;
                for (int l = 0; l < n_nodes; ++l)
                {
                    intersect_connects += A_current[i][l] * A_current[j][l];
                }

                // get the possible common neighbors
                double union_connects = k_current[i] + k_current[j] - 2 * A_current[i][j];

                K_current[i][j] = K_current[j][i] = intersect_connects > 0 ? ((intersect_connects * 2) / union_connects) + epsilon : epsilon;
            }
        }
        break;

    case 3: // clu-avg
        clu_coeff_current = getClusteringCoeff(A_current, n_nodes, k_current);
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] + clu_coeff_current[j]) / 2) + epsilon;
            }
        }
        break;

    case 4: // clu-min
        clu_coeff_current = getClusteringCoeff(A_current, n_nodes, k_current);
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] > clu_coeff_current[j]) ? clu_coeff_current[j] : clu_coeff_current[i]) + epsilon;
            }
        }
        break;

    case 5: // clu-max
        clu_coeff_current = getClusteringCoeff(A_current, n_nodes, k_current);
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] > clu_coeff_current[j]) ? clu_coeff_current[i] : clu_coeff_current[j]) + epsilon;
            }
        }
        break;

    case 6: // clu-dist
        clu_coeff_current = getClusteringCoeff(A_current, n_nodes, k_current);
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = fabs(clu_coeff_current[i] - clu_coeff_current[j]) + epsilon;
            }
        }
        break;

    case 7: // clu-prod
        clu_coeff_current = getClusteringCoeff(A_current, n_nodes, k_current);
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = clu_coeff_current[i] * clu_coeff_current[j] + epsilon;
            }
        }
        break;

    case 8: // deg-avg
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] + k_current[j]) / 2) + epsilon;
            }
        }
        break;

    case 9: // deg-min
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[j] : k_current[i]) + epsilon;
            }
        }
        break;

    case 10: // deg-max
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[i] : k_current[j]) + epsilon;
            }
        }
        break;

    case 11: // deg-dist
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = fabs(k_current[i] - k_current[j]) + epsilon;
            }
        }
        break;

    case 12: // deg-prod
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = i; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = k_current[i] * k_current[j] + epsilon;
            }
        }
        break;
    }
}

void GNM::updateK(std::vector<int> bth)
{
    switch (model)
    {

    case 1: // neighbors
        for (int i = 0; i < n_nodes; i++)
        {
            if ((i != bth[0]) && (i != bth[1]))
            {
                if (A_current[bth[1]][i])
                {
                    K_current[bth[0]][i] = K_current[i][bth[0]] = K_current[i][bth[0]] + 1;
                }

                if (A_current[bth[0]][i])
                {
                    K_current[bth[1]][i] = K_current[i][bth[1]] = K_current[i][bth[1]] + 1;
                }
            }
        }
        break;

    case 2:
    {
        int uu = bth[0];
        int vv = bth[1];

        std::vector<int> update_uu;
        std::vector<int> update_vv;

        for (int i = 0; i < n_nodes; ++i)
        {
            // find the nodes that have a common neighbor with uu and vv
            bool check_uu = true;
            bool check_vv = true;
            for (int j = 0; j < n_nodes; ++j)
            {
                if (check_uu && (A_current[uu][j] * A_current[i][j]) && (i != uu))
                {
                    update_uu.push_back(i);
                    check_uu = false;
                }
                if (check_vv && (A_current[vv][j] * A_current[vv][j]) && (i != vv))
                {
                    update_vv.push_back(i);
                    check_vv = false;
                }
            }
        }

        // update the matching scores for any node with common neighbor with uu
        for (int j : update_uu)
        {
            double intersect_c = 0;
            for (int l = 0; l < n_nodes; ++l)
            {
                intersect_c += A_current[j][l] * A_current[uu][l];
            }

            double union_c = k_current[uu] + k_current[j] - 2 * A_current[uu][j];

            K_current[j][uu] = K_current[uu][j] = (intersect_c > 0) ? ((intersect_c * 2) / union_c) : epsilon;
        }

        // update the matching scores for any node with common neighbor with vv
        for (int j : update_vv)
        {
            double intersect_c = 0;
            for (int l = 0; l < n_nodes; ++l)
            {
                intersect_c += A_current[j][l] * A_current[vv][l];
            }

            double union_c = k_current[vv] + k_current[j] - 2 * A_current[vv][j];

            K_current[j][vv] = K_current[vv][j] = (intersect_c > 0) ? ((intersect_c * 2) / union_c) : epsilon;
        }
    }
    break;

    case 3: // clu-avg
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] + clu_coeff_current[j]) / 2) + epsilon;
            }
        }
        break;

    case 4: // clu-min
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] > clu_coeff_current[j]) ? clu_coeff_current[j] : clu_coeff_current[i]) + epsilon;
            }
        }
        break;

    case 5: // clu-max
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((clu_coeff_current[i] > clu_coeff_current[j]) ? clu_coeff_current[i] : clu_coeff_current[j]) + epsilon;
            }
        }
        break;

    case 6: // clu-dist
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = fabs(clu_coeff_current[i] - clu_coeff_current[j]) + epsilon;
            }
        }
        break;

    case 7: // clu-prod
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = clu_coeff_current[i] * clu_coeff_current[j] + epsilon;
            }
        }
        break;

    case 8: // deg-avg
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] + k_current[j]) / 2) + epsilon;
            }
        }
        break;

    case 9: // deg-min
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[j] : k_current[i]) + epsilon;
            }
        }
        break;

    case 10: // deg-max
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = ((k_current[i] > k_current[j]) ? k_current[i] : k_current[j]) + epsilon;
            }
        }
        break;

    case 11: // deg-dist
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = fabs(k_current[i] - k_current[j]) + epsilon;
            }
        }
        break;

    case 12: // deg-prod
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                K_current[i][j] = K_current[j][i] = k_current[i] * k_current[j] + epsilon;
            }
        }
        break;

    default: // spatial
        break;
    }
}

void GNM::runParamComb(int i_pcomb)
// main function for the generative network build
{
    // get the params
    double eta = params[i_pcomb][0];
    double gamma = params[i_pcomb][1];

    // initiate the degree of each node
    for (int i = 0; i < n_nodes; ++i)
    {
        k_current[i] = 0;
        for (int j = 0; j < n_nodes; ++j)
        {
            k_current[i] += A_current[i][j];
        }
    }

    // compute the inital value matrix
    initK();

    // initiate cost and value matrices
    std::vector<std::vector<double> > Fd(n_nodes, std::vector<double>(n_nodes));
    std::vector<std::vector<double> > Fk(n_nodes, std::vector<double>(n_nodes));
    std::vector<std::vector<double> > Ff(n_nodes, std::vector<double>(n_nodes));

    for (int i = 0; i < n_nodes; ++i)
    {
        for (int j = 0; j < n_nodes; ++j)
        {
            Fd[i][j] = pow(D[i][j], eta);
            Fk[i][j] = pow(K_current[i][j], gamma);
            Ff[i][j] = Fd[i][j] * Fk[i][j] * (A_current[i][j] == 0);
        }
    }

    // get the indices of the upper triangle of P (denoted u and v)
    std::vector<int> u(n_nodes * (n_nodes - 1) / 2);
    std::vector<int> v(n_nodes * (n_nodes - 1) / 2);
    int upper_tri_index = 0;

    for (int i = 0; i < n_nodes; ++i)
    {
        for (int j = i + 1; j < n_nodes; ++j)
        {
            u[upper_tri_index] = i;
            v[upper_tri_index] = j;
            upper_tri_index++;
        }
    }

    // initiate P
    std::vector<double> P(upper_tri_index);
    for (int i = 0; i < upper_tri_index; ++i)
    {
        P[i] = Ff[u[i]][v[i]];
    }

    // Number of connections we start with
    int m_seed = 0;
    for (int i = 0; i < upper_tri_index; ++i)
    {
        if (A_current[u[i]][v[i]] != 0)
        {
            m_seed++;
        }
    }

    // prep C
    std::vector<double> C(upper_tri_index + 1);
    C[0] = 0;

    // main loop adding new connections to adjacency matrix
    for (int i = m_seed + 1; i <= m; ++i)
    {
        // compute the cumulative sum of the probabilities
        for (int j = 0; j < upper_tri_index; ++j)
        {
            C[j + 1] = C[j] + P[j];
        }

        // select an element
        double r = std::rand() / (RAND_MAX + 1.0) * C[upper_tri_index];
        int selected = 0;
        while (selected < upper_tri_index && r >= C[selected + 1])
        {
            selected++;
        }
        int uu = u[selected];
        int vv = v[selected];

        // update the node degree array
        k_current[uu] += 1;
        k_current[vv] += 1;

        // update the adjacency matrix
        A_current[uu][vv] = A_current[vv][uu] = 1;

        // get the node indices for update
        std::vector<int> bth;
        if ((model > 2) && (model < 8)) // if cluster model
        {
            bth = updateClusteringCoeff(uu, vv);
        }
        bth.push_back(uu);
        bth.push_back(vv);

        // update K matrix
        updateK(bth);

        // update Ff matrix (probabilities)
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i)
        {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j)
            {
                Ff[i][j] = Ff[j][i] = Fd[i][j] * pow(K_current[i][j], gamma) * (A_current[i][j] == 0);
            }
        }

        // update p
        // TODO: This currently just updates every position
        for (int i = 0; i < upper_tri_index; ++i)
        {
            P[i] = Ff[u[i]][v[i]];
        }
    }

    // update b with the result
    int nth_edge = 0;
    for (int i = 0; i < upper_tri_index; ++i)
    {
        if (A_current[u[i]][v[i]])
        {
            b[nth_edge][i_pcomb] = (u[i] + 1) * pow(10, ceil(log10(v[i] + 1))) + v[i];
            nth_edge = nth_edge + 1;
        }
    }
}

void GNM::generateModels()
// Initiates the network generation leveraging the different rules
{
    // Prep scores for A_Y
    std::vector<std::vector<double> > energy_Y(4, std::vector<double>(n_nodes, 0.0));

    // 1. nodal degree
    for (int i = 0; i < n_nodes; ++i)
    {
        for (int j = 0; j < n_nodes; ++j)
        {
            energy_Y[0][i] += A_Y[i][j];
        }
    }

    // 2. clustering coefficient
    energy_Y[1] = getClusteringCoeff(A_Y, n_nodes, energy_Y[0]);

    // 3. betweens centrality
    energy_Y[2] = getBetweennesCentrality(A_Y, n_nodes);

    // 4. edge length
    energy_Y[3] = getEdgeLength(A_Y, D, n_nodes);

    for (int i_pcomb = 0; i_pcomb < n_p_combs; i_pcomb++)
    {
        // compute the adjacency matrix for the param combination
        resetAcurrent();
        runParamComb(i_pcomb);

        // evaluate
        std::vector<std::vector<double> > energy(4, std::vector<double>(n_nodes, 0.0));

        // // 1. nodal degree
        for (int i = 0; i < n_nodes; i++)
        {
            energy[0][i] = k_current[i];
        }

        // 2. clustering coefficient
        energy[1] = getClusteringCoeff(A_current, n_nodes, energy[0]);

        // 3. betweens centrality
        energy[2] = getBetweennesCentrality(A_current, n_nodes);

        // 4. edge length
        energy[3] = getEdgeLength(A_current, D, n_nodes);

        K[0][i_pcomb] = ksTest(energy_Y[0], energy[0]);
        K[1][i_pcomb] = ksTest(energy_Y[1], energy[1]);
        K[2][i_pcomb] = ksTest(energy_Y[2], energy[2]);
        K[3][i_pcomb] = ksTest(energy_Y[3], energy[3]);
    }
}