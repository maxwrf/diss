//
// Created by Max Würfel on 26.06.23.
//

#include "GNM.h"
#include <string>
#include <vector>
#include <iostream>
#include "TestData.h"

int main() {
    std::string pData = "/Users/maxwuerfek/code/diss/gnm-run/testData/syntheticData.h5";
    std::string pDist = "/Users/maxwuerfek/code/diss/gnm-run/testData/syntheticDataDistances.h5";
    std::string pOut = "/Users/maxwuerfek/code/diss/gnm-run/testData/testKall.h5";

    // Load the synthetic data
    int n1 = 270, n2 = 68, n3 = 68;
    std::vector<std::vector<double>> A_init(n2, std::vector<double>(n3, 0));
    std::vector<double> connectomesM(n1, 0);
    std::vector<std::vector<std::vector<double>>> connectomes(
            n1, std::vector<std::vector<double>>(n2, std::vector<double>(n3)));
    std::vector<std::vector<double>> D(n2, std::vector<double>(n3));

    SyntheticData::getSyntheticData(
            connectomes,
            connectomesM,
            A_init,
            D,
            pData,
            pDist,
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
            auto startT = std::chrono::high_resolution_clock::now();

            // Init to store results
            std::vector<std::vector<int>> b(m, std::vector<int>(paramSpace.size()));
            std::vector<std::vector<double>> K(paramSpace.size(), std::vector<double>(4));

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

            auto endT = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT);
            std::cout << "Done | Sample: " << iSample << " Model: " << jModel << " Duration (MS): " << duration.count()
                      << std::endl;
        }
    }

    // Store the results
    GNM::saveResults(pOut, Kall, paramSpace);
    return 0;
}