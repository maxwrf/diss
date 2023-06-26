#include <iostream>
#include "SpikeSet.h"
#include "GNM.h"
#include <H5Cpp.h>

int main() {
    // Read in the spike data
    std::string pIn = "/Users/maxwuerfek/code/diss/data/g2c_data";
    double corrCutoff = 0.2;
    int nSamples = 2;
    SpikeSet spikeData(pIn, nSamples);

    // Generate parameter space
    int nRuns = 10;
    std::vector<double> etaLimits = {-7, 7};
    std::vector<double> gammaLimits = {-7, 7};
    std::vector<std::vector<double>> paramSpace = GNM::generateParamSpace(nRuns,
                                                                          etaLimits,
                                                                          gammaLimits);

    // Initialize outputs
    std::vector<std::string> rules = GNM::getRules();

    // Initialize Kall
    std::vector<std::vector<std::vector<std::vector<double>>>> Kall(
            spikeData.spikeTrains.size(),
            std::vector<std::vector<std::vector<double>>>(
                    rules.size(),
                    std::vector<std::vector<double>>(
                            paramSpace.size(),
                            std::vector<double>(4))));


    // Run generative models
    for (int iSample = 0; iSample < spikeData.spikeTrains.size(); ++iSample) {
        for (int jModel = 0; jModel < rules.size(); ++jModel) {

            std::vector<std::vector<int>> b(
                    spikeData.spikeTrains[iSample].m,
                    std::vector<int>(paramSpace.size())
            );

            std::vector<std::vector<double>> K(
                    paramSpace.size(),
                    std::vector<double>(4)
            );

            GNM model(
                    spikeData.spikeTrains[iSample].A_Y,
                    spikeData.spikeTrains[iSample].A_init,
                    spikeData.D,
                    paramSpace,
                    b,
                    K,
                    spikeData.spikeTrains[iSample].m,
                    jModel,
                    paramSpace.size(),
                    spikeData.spikeTrains[iSample].A_Y.size());

            model.generateModels();
            Kall[iSample][jModel] = K;
        }
    }

    // Store the results
    std::string pOut = "/Users/maxwuerfek/code/diss/results/";
    H5::H5File file(pOut + "Kall.h5", H5F_ACC_TRUNC);
    hsize_t dims[4] = {
            spikeData.spikeTrains.size(),
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
