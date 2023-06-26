//
// Created by Max Würfel on 26.06.23.
//

#include "Slurm.h"
#include "SpikeSet.h"
#include "GNM.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>

void Slurm::generateInputs(std::string &inDirPath,
                           std::string &outDirPath,
                           double corrCutoff,
                           int nSamples,
                           int nRuns) {
    // Load the spikes
    SpikeSet spikeData(inDirPath, nSamples);

    // Generate parameter space
    std::vector<double> etaLimits = {-7, 7};
    std::vector<double> gammaLimits = {-7, 7};
    std::vector<std::vector<double>> paramSpace = GNM::generateParamSpace(nRuns,
                                                                          etaLimits,
                                                                          gammaLimits);

    // Check directory to store results
    if (!std::filesystem::exists(outDirPath)) {
        std::filesystem::create_directories(outDirPath);
    }

    for (int iSample = 0; iSample < spikeData.spikeTrains.size(); ++iSample) {
        // save the results
        std::string n = std::to_string(iSample);
        std::ofstream file(outDirPath + "/sample_" + n + ".dat", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file for saving vectors" << std::endl;
            return;
        }

        // Write the vector sizes
        size_t sizeA_Y = spikeData.spikeTrains[iSample].A_Y.size();
        file.write(reinterpret_cast<const char *>(&sizeA_Y), sizeof(sizeA_Y));

        size_t sizeA_init = spikeData.spikeTrains[iSample].A_init.size();
        file.write(reinterpret_cast<const char *>(&sizeA_init), sizeof(sizeA_init));

        size_t sizeParamSpace = paramSpace.size();
        file.write(reinterpret_cast<const char *>(&sizeParamSpace), sizeof(sizeParamSpace));

        // Write the data
        for (const auto &innerVec: spikeData.spikeTrains[iSample].A_Y) {
            file.write(reinterpret_cast<const char *>(innerVec.data()), innerVec.size() * sizeof(double));
        }

        for (const auto &innerVec: spikeData.spikeTrains[iSample].A_init) {
            file.write(reinterpret_cast<const char *>(innerVec.data()), innerVec.size() * sizeof(double));
        }

        for (const auto &innerVec: paramSpace) {
            file.write(reinterpret_cast<const char *>(innerVec.data()), innerVec.size() * sizeof(double));
        }
    }
};

void Slurm::readDatFile(std::string &inPath,
                        std::vector<std::vector<double>> &A_Y,
                        std::vector<std::vector<double>> &A_init,
                        std::vector<std::vector<double>> &paramSpace

) {
    // Open the file
    std::ifstream file(inPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading vectors." << std::endl;
        return;
    }

    // Read the sizes of the vectors and reshape
    size_t sizeA_Y;
    file.read(reinterpret_cast<char *>(&sizeA_Y), sizeof(sizeA_Y));
    A_Y.resize(sizeA_Y);

    size_t sizeA_init;
    file.read(reinterpret_cast<char *>(&sizeA_init), sizeof(sizeA_init));
    A_init.resize(sizeA_init);

    size_t sizeParamSpace;
    file.read(reinterpret_cast<char *>(&sizeParamSpace), sizeof(sizeParamSpace));
    paramSpace.resize(sizeParamSpace);

    // Read in the data
    for (auto &row: A_Y) {
        size_t sizeRow = sizeA_Y; //square
        row.resize(sizeRow);
        file.read(reinterpret_cast<char *>(row.data()), sizeRow * sizeof(double));
    }

    for (auto &row: A_init) {
        size_t sizeRow = sizeA_init; //square
        row.resize(sizeRow);
        file.read(reinterpret_cast<char *>(row.data()), sizeRow * sizeof(double));
    }

    for (auto &row: paramSpace) {
        size_t sizeRow = 2;
        row.resize(sizeRow);
        file.read(reinterpret_cast<char *>(row.data()), sizeRow * sizeof(double));
    }
};