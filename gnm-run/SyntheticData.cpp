//
// Created by Max Würfel on 29.06.23.
//

#include "SyntheticData.h"
#include "SpikeTrain.h"
#include <vector>

void SyntheticData::getSyntheticData(std::vector<std::vector<std::vector<double>>> &connectomes,
                                     std::vector<double> &connectomesM,
                                     std::vector<std::vector<double>> &A_init,
                                     std::vector<std::vector<double>> &D,
                                     std::string &pData, std::string &pDist,
                                     int n1, int n2, int n3) {
    // Load the connectomes
    std::vector<double> temp = SpikeTrain::readDoubleDataset(pData,
                                                             "connectomes");

    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                connectomes[i][j][k] = temp[(i * n2 + j) * n3 + k];
                A_init[j][k] += connectomes[i][j][k];
                connectomesM[i] += connectomes[i][j][k];
            }
        }
    }

    // Load the distances
    std::vector<double> temp2 = SpikeTrain::readDoubleDataset(pDist, "distances");
    int counter = 0;
    for (int j = 0; j < n2; ++j) {
        for (int k = 0; k < n3; ++k) {
            D[j][k] = temp2[j * n3 + k];
            // Decide whether edge is part of init
            // TODO: This is correct according to the source code but does not make sense to me ">="
            if (A_init[j][k] == (n1 * 0.2)) {
                A_init[j][k] = 1;
                counter++;
            } else {
                A_init[j][k] = 0;
            }
        }
    }
};