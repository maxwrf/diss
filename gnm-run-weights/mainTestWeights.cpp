//
// Created by Max Würfel on 26.06.23.
//

#include "GNM.h"
#include <string>
#include <vector>
#include <iostream>
#include "TestDataWeights.h"
#include "WeightedGNM.h"
# include <map>


int main() {
    // Load the test data
    std::string pA = "/Users/maxwuerfek/code/diss/gnm-run-weights/testDataWeights/A.h5";
    std::string pA_init = "/Users/maxwuerfek/code/diss/gnm-run-weights/testDataWeights/Ainit.h5";
    std::string pD = "/Users/maxwuerfek/code/diss/gnm-run-weights/testDataWeights/D.h5";

    // Load the synthetic data
    int n = 68, m = 80;
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0));
    std::vector<std::vector<double>> A_init(n, std::vector<double>(n, 0));
    std::vector<std::vector<double>> D(n, std::vector<double>(n));

    TestDataWeights::getSyntheticData(
            A,
            A_init,
            D,
            pA,
            pA_init,
            pD,
            n);

    // Generate parameter space
    double eta = -3.2, gamma = 0.38, alpha = 0.05, omega = 0.9;
    std::vector<std::vector<double>> paramSpace = {{eta, gamma, alpha, omega}};

    // Weighted model parameters
    int start = 0, optiFunc = 0, optiSamples = 5;
    double optiResolution = 0.05;

    // Initialize Kall (Samples, Models, Parameters, K)
    std::vector<std::string> rules = GNM::getRules();
    std::vector<std::vector<std::vector<std::vector<double>>>> Kall(
            1,
            std::vector<std::vector<std::vector<double>>>(
                    rules.size(),
                    std::vector<std::vector<double>>(
                            paramSpace.size(),
                            std::vector<double>(4))));

    // Initialize AkeepAll and WkeepAll (Sample => Model, Params, Iterations (i.e., added edges), nxn)
    std::map<int, std::vector<std::vector<std::vector<std::vector<double>>>>> AkeepAll, WkeepAll;

    // Run the generative models
    int nSamples = 1;
    for (int iSample = 0; iSample < nSamples; ++iSample) {
        for (int jModel = 0; jModel < rules.size(); ++jModel) {
            auto startT = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<int>> b(
                    m,
                    std::vector<int>(paramSpace.size())
            );

            std::vector<std::vector<double>> K(
                    paramSpace.size(),
                    std::vector<double>(4)
            );

            std::vector<std::vector<std::vector<double>>> Akeep, Wkeep;

            WeightedGNM model(
                    A,
                    A_init,
                    D,
                    paramSpace,
                    b,
                    K,
                    m,
                    jModel,
                    paramSpace.size(),
                    A.size(),
                    start,
                    optiFunc,
                    optiResolution,
                    optiSamples,
                    Akeep,
                    Wkeep
            );

            model.generateModels();
            Kall[iSample][jModel] = K;


            auto endT = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endT - startT);
            std::cout << "Done | Sample: " << iSample << " Model: " << jModel << " Duration (MS): " << duration.count()
                      << std::endl;
        }
    }

    // Store the results
    std::string pOut = "/Users/maxwuerfek/code/diss/gnm-run/testData/testKall.h5";
    GNM::saveResults(pOut, Kall, paramSpace);
    return 0;
}