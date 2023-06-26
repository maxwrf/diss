#include <iostream>
#include <vector>
#include <cmath>

std::vector<double> betweenness_centrality(std::vector<std::vector<double>> &A, int n)
{
    // FORWARD PASS
    double d = 1.0;                           // path length
    std::vector<std::vector<double>> NPd = A; // number of paths of length |d|
    std::vector<std::vector<double>> NSP = A; // number of shortest paths of any length
    std::vector<std::vector<double>> L = A;   // length of shortest paths

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

        std::vector<std::vector<double>> temp = NPd;
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
    std::vector<std::vector<double>> DP(n, std::vector<double>(n, 0.0)); // vertex on vertex dependency
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

int main()
{
    std::vector<std::vector<double>> A =
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0},
         {0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1},
         {0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1},
         {0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0},
         {0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0},
         {0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0},
         {0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0},
         {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1},
         {0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0},
         {0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0},
         {0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1},
         {0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0},
         {1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0},
         {0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1},
         {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0},
         {1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
         {1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0},
         {0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1},
         {0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1},
         {0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0}};

    int n = A.size();

    std::vector<double> centrality = betweenness_centrality(A, n);

    // Print the results
    for (int i = 0; i < n; i++)
    {
        std::cout << "Node " << i << " centrality: " << centrality[i] << std::endl;
    }

    return 0;
}