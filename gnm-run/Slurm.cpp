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
#include <map>
#include <iomanip>
#include <sstream>

void Slurm::generateInputs(std::string &inDirPath,
                           std::string &outDirPath,
                           double corrCutoff,
                           int nSamples,
                           int nRuns,
                           int dSet) {
    // Load the spikes
    SpikeSet spikeData(inDirPath, nSamples, dSet);

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
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << iSample;
        std::string n = oss.str();
        std::ofstream file(outDirPath + "/sample_" + n + ".dat", std::ios::binary);
        if (!file.is_open()) {
            std::cout << "Failed to open file for saving vectors" << std::endl;
            return;
        }

        // Write the group string size
        size_t groupIdSize = spikeData.spikeTrains[iSample].groupId.size();
        file.write(reinterpret_cast<const char *>(&groupIdSize), sizeof(groupIdSize));

        // Write the vector sizes
        size_t sizeA_Y = spikeData.spikeTrains[iSample].A_Y.size();
        file.write(reinterpret_cast<const char *>(&sizeA_Y), sizeof(sizeA_Y));

        size_t sizeA_init = spikeData.spikeTrains[iSample].A_init.size();
        file.write(reinterpret_cast<const char *>(&sizeA_init), sizeof(sizeA_init));

        size_t sizeD = spikeData.D.size();
        file.write(reinterpret_cast<const char *>(&sizeD), sizeof(sizeD));

        size_t sizeParamSpace = paramSpace.size();
        file.write(reinterpret_cast<const char *>(&sizeParamSpace), sizeof(sizeParamSpace));

        // Write the data
        file.write(spikeData.spikeTrains[iSample].groupId.c_str(), groupIdSize);

        for (const auto &innerVec: spikeData.spikeTrains[iSample].A_Y) {
            file.write(reinterpret_cast<const char *>(innerVec.data()),
                       innerVec.size() * sizeof(double));
        }

        for (const auto &innerVec: spikeData.spikeTrains[iSample].A_init) {
            file.write(reinterpret_cast<const char *>(innerVec.data()),
                       innerVec.size() * sizeof(double));
        }

        for (const auto &innerVec: spikeData.D) {
            file.write(reinterpret_cast<const char *>(innerVec.data()),
                       innerVec.size() * sizeof(double));
        }

        for (const auto &innerVec: paramSpace) {
            file.write(reinterpret_cast<const char *>(innerVec.data()),
                       innerVec.size() * sizeof(double));
        }
    }
};

void Slurm::readDatFile(std::string &inPath,
                        std::vector<std::vector<double>> &A_Y,
                        std::vector<std::vector<double>> &A_init,
                        std::vector<std::vector<double>> &D,
                        std::vector<std::vector<double>> &paramSpace,
                        std::string &groupId
) {
    // Open the file
    std::ifstream file(inPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for loading vectors." << std::endl;
        return;
    }

    // Read group ID string size
    size_t groupIdSize;
    file.read(reinterpret_cast<char *>(&groupIdSize), sizeof(groupIdSize));
    groupId.resize(groupIdSize);

    // Read the sizes of the vectors and reshape
    size_t sizeA_Y;
    file.read(reinterpret_cast<char *>(&sizeA_Y), sizeof(sizeA_Y));
    A_Y.resize(sizeA_Y);

    size_t sizeA_init;
    file.read(reinterpret_cast<char *>(&sizeA_init), sizeof(sizeA_init));
    A_init.resize(sizeA_init);

    size_t sizeD;
    file.read(reinterpret_cast<char *>(&sizeD), sizeof(sizeD));
    D.resize(sizeD);

    size_t sizeParamSpace;
    file.read(reinterpret_cast<char *>(&sizeParamSpace), sizeof(sizeParamSpace));
    paramSpace.resize(sizeParamSpace);

    // Read in the data
    file.read(&groupId[0], groupIdSize);

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

    for (auto &row: D) {
        size_t sizeRow = sizeD; //square
        row.resize(sizeRow);
        file.read(reinterpret_cast<char *>(row.data()), sizeRow * sizeof(double));
    }

    for (auto &row: paramSpace) {
        size_t sizeRow = 2; // Fixed eta and gamma
        row.resize(sizeRow);
        file.read(reinterpret_cast<char *>(row.data()), sizeRow * sizeof(double));
    }
};

void Slurm::saveResFile(std::string &outDirPath,
                        std::vector<std::vector<std::vector<double>>> &Kall,
                        std::vector<std::vector<double>> &paramSpace,
                        std::string &groupId) {
    // Open the file
    std::ofstream file(outDirPath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open file for saving vectors" << std::endl;
        return;
    }

    // Write the group string size
    size_t groupIdSize = groupId.size();
    file.write(reinterpret_cast<const char *>(&groupIdSize), sizeof(groupIdSize));

    // Write the K all vector sizes
    size_t size1Kall = Kall.size();
    size_t size2Kall = Kall[0].size();
    size_t size3Kall = Kall[0][0].size();

    file.write(reinterpret_cast<const char *>(&size1Kall), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&size2Kall), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&size3Kall), sizeof(size_t));

    // Write the param Space vector size
    size_t size1paramSpace = paramSpace.size();
    size_t size2paramSpace = paramSpace[0].size();

    file.write(reinterpret_cast<const char *>(&size1paramSpace), sizeof(size_t));
    file.write(reinterpret_cast<const char *>(&size2paramSpace), sizeof(size_t));

    // Write the groupId
    file.write(groupId.c_str(), groupIdSize);

    // Write the KAll data
    for (size_t i = 0; i < size1Kall; i++) {
        for (size_t j = 0; j < size2Kall; j++) {
            file.write(reinterpret_cast<const char *>(Kall[i][j].data()),
                       size3Kall * sizeof(double));
        }
    }

    // Write the paramSpace data
    for (size_t i = 0; i < size1paramSpace; i++) {
        file.write(reinterpret_cast<const char *>(paramSpace[i].data()),
                   size2paramSpace * sizeof(double));
    }

    file.close();
}

void Slurm::combineResFiles(std::string &inDirPath,
                            std::vector<std::vector<std::vector<std::vector<double>>>> &Kall,
                            std::vector<std::vector<double>> &paramSpace,
                            std::vector<std::string> &groupIds) {
    // read in the files
    int filesRead = 0;
    for (const auto &entry: std::filesystem::directory_iterator(inDirPath)) {
        if (std::filesystem::is_regular_file(entry.path())) {
            std::string fName = entry.path().filename().string();
            if (fName.substr(fName.find_last_of(".") + 1) == "res") {
                // Open the file
                std::ifstream file(inDirPath + "/" + fName, std::ios::binary);

                // Prepare
                std::vector<std::vector<std::vector<double>>> KallSample;
                std::string groupId;

                // Read in group Id size and resize
                size_t groupIdSize;
                file.read(reinterpret_cast<char *>(&groupIdSize), sizeof(groupIdSize));
                groupId.resize(groupIdSize);

                // Read in Kall dimensions and resize the vector
                size_t size1Kall, size2Kall, size3Kall;
                file.read(reinterpret_cast<char *>(&size1Kall), sizeof(size_t));
                file.read(reinterpret_cast<char *>(&size2Kall), sizeof(size_t));
                file.read(reinterpret_cast<char *>(&size3Kall), sizeof(size_t));

                KallSample.resize(
                        size1Kall, std::vector<std::vector<double>>(
                                size2Kall, std::vector<double>(size3Kall)));

                // Read in all param dimensions and resize the vector
                size_t size1paramSpace, size2paramSpace;
                file.read(reinterpret_cast<char *>(&size1paramSpace), sizeof(size_t));
                file.read(reinterpret_cast<char *>(&size2paramSpace), sizeof(size_t));

                paramSpace.resize(size1paramSpace, std::vector<double>(size2paramSpace));

                // Read in the group ID
                file.read(&groupId[0], groupIdSize);

                // Read in the Kall data
                for (size_t i = 0; i < size1Kall; i++) {
                    for (size_t j = 0; j < size2Kall; j++) {
                        file.read(reinterpret_cast<char *>(KallSample[i][j].data()),
                                  size3Kall * sizeof(double));
                    }
                }

                // Read in the ParamSpace
                if (filesRead == 0) {
                    for (size_t i = 0; i < size1paramSpace; i++) {
                        file.read(reinterpret_cast<char *>(paramSpace[i].data()),
                                  size2paramSpace * sizeof(double));
                    }
                }

                file.close();
                Kall.push_back(KallSample);
                groupIds.push_back(groupId);
                filesRead++;
            }
        }
    }
};

void Slurm::writeGroupsHDF5(std::vector<std::string> &groupIds,
                            std::vector<std::vector<std::vector<std::vector<double>>>> &Kall,
                            std::vector<std::vector<double>> &paramSpace,
                            std::string &outDirPath) {
    // Get the unique IDs
    std::map<std::string, std::vector<size_t>> idMap;
    for (size_t i = 0; i < groupIds.size(); ++i) {
        idMap[groupIds[i]].push_back(i);
    }

    for (const auto &[id, indices]: idMap) {
        std::cout << "Group: " << id << std::endl;

        // Subset the Kall vector for this group
        std::vector<std::vector<std::vector<std::vector<double>>>> KallGroup;
        for (const auto &index: indices) {
            KallGroup.push_back(Kall[index]);
        }

        // Write the results
        std::string outFile = outDirPath + "/results_" + id + ".h5";
        GNM::saveResults(outFile, KallGroup, paramSpace);
    }
}