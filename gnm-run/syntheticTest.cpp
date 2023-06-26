//
// Created by Max Würfel on 26.06.23.
//

#include "SpikeTrain.h"
#include "GNM.h"
#include <H5Cpp.h>
#include <string>
#include <vector>
#include <iostream>

void getSyntheticData(
        std::vector<std::vector<std::vector<double>>> &connectomes,
        std::vector<double> &connectomesM,
        std::vector<std::vector<double>> &A_init,
        std::vector<std::vector<double>> &D,
        int n1, int n2, int n3
) {

    std::string pData = "/Users/maxwuerfek/code/gnm/testData/syntheticData.h5";
    std::string pDist = "/Users/maxwuerfek/code/gnm/testData/syntheticDataDistances.h5";

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
    };
}

int main() {
    // Load the synthetic data
    int n1 = 270, n2 = 68, n3 = 68;
    std::vector<std::vector<double>> A_init(n2, std::vector<double>(n3, 0));
    std::vector<double> connectomesM(n1, 0);
    std::vector<std::vector<std::vector<double>>> connectomes(
            n1, std::vector<std::vector<double>>(n2, std::vector<double>(n3)));
    std::vector<std::vector<double>> D(n2, std::vector<double>(n3));

    getSyntheticData(
            connectomes,
            connectomesM,
            A_init,
            D,
            n1,
            n2,
            n3);

    // Only generate the models for some number of samples
    int nSamples = 2;
    if (nSamples == -1) {
        nSamples = n1;
    }

    // Generate parameter space
    int nRuns = 1000;
    std::vector<double> etaLimits = {-7, 7};
    std::vector<double> gammaLimits = {-7, 7};
    std::vector<std::vector<double>> paramSpace = GNM::generateParamSpace(nRuns,
                                                                          etaLimits,
                                                                          gammaLimits);
    
    // Initialize Kall
    std::vector<std::string> rules = GNM::getRules();
    std::vector<std::vector<std::vector<std::vector<double>>>> Kall(
            nSamples,
            std::vector<std::vector<std::vector<double>>>(
                    rules.size(),
                    std::vector<std::vector<double>>(
                            paramSpace.size(),
                            std::vector<double>(4))));

    // Run the generative models
    for (int iSample = 0; iSample < nSamples; ++iSample) {
        int m = connectomesM[iSample] / 2;
        for (int jModel = 0; jModel < rules.size(); ++jModel) {

            std::vector<std::vector<int>> b(
                    m,
                    std::vector<int>(paramSpace.size())
            );

            std::vector<std::vector<double>> K(
                    paramSpace.size(),
                    std::vector<double>(4)
            );

            GNM model(
                    connectomes[iSample],
                    A_init,
                    D,
                    paramSpace,
                    b,
                    K,
                    m,
                    jModel,
                    paramSpace.size(),
                    connectomes[iSample].size());

            model.generateModels();
            Kall[iSample][jModel] = K;

            std::cout << iSample << jModel << std::endl;
        }
    }

    // Store the results
    std::string pOut = "/Users/maxwuerfek/code/gnm/testData/testKall.h5";
    H5::H5File file(pOut, H5F_ACC_TRUNC);
    hsize_t dims[4] = {
            static_cast<hsize_t>(nSamples),
            rules.size(),
            paramSpace.size(),
            4
    };
    H5::DataSpace dataspace(4, dims);
    H5::DataSet dataset = file.createDataSet("Kall", H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(Kall.data(), H5::PredType::NATIVE_DOUBLE);
    file.close();
    dataset.close();

    return 0;
}