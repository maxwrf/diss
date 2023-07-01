//
// Created by Max Würfel on 29.06.23.
//

#include "TestDataWeights.h"
#include "SpikeTrain.h"
#include <vector>

void TestDataWeights::getSyntheticData(std::vector<std::vector<double>> &A,
                                       std::vector<std::vector<double>> &A_init,
                                       std::vector<std::vector<double>> &D,
                                       std::string &pA, std::string &pA_init, std::string &pD,
                                       int n) {
    // Load the connectomes
    std::vector<double> temp3 = SpikeTrain::readDoubleDataset(pD, "D");
    std::vector<double> temp = SpikeTrain::readDoubleDataset(pA, "A");
    std::vector<double> temp2 = SpikeTrain::readDoubleDataset(pA_init, "Ainit");


    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = temp[i * n + j];
            A_init[i][j] = temp2[i * n + j];
            D[i][j] = temp3[i * n + j];
        }
    }
};