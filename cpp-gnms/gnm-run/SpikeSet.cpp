//
// Created by Max Würfel on 25.06.23.
//

#include <string>
#include <filesystem>
#include <cmath>
#include "SpikeSet.h"
#include <algorithm>

SpikeSet::SpikeSet(std::string path_,
                   int nSamples,
                   int dSet,
                   double dt,
                   double corrCutoff,
                   int meaType_) {
    path = path_;
    sttcCutoff = corrCutoff;
    meaType = meaType_;

    // Get the electrode positions for this MEA type
    getElectrodePos();

    // Compute the distance matrix
    getDistanceMatrix();

    // read in the files
    int samplesRead = 0;
    for (const auto &entry: std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry.path())) {
            std::string f = entry.path().filename().string();
            if (f.substr(f.find_last_of(".") + 1) == "h5") {
                // For development the number of samples can be limited
                if ((nSamples != -1) && (samplesRead == nSamples)) {
                    break;
                }

                hd5FileNames.push_back(f);
                spikeTrains.push_back(SpikeTrain(path + "/" + f,
                                                 electrodes,
                                                 numElectrodes,
                                                 dt,
                                                 sttcCutoff,
                                                 dSet,
                                                 meaType));

                samplesRead++;
            }
        }
    }
}

void SpikeSet::getElectrodePos() {
    std::vector<std::vector<int>> excludeElectrodes;
    bool yFromTopRight = false;
    int startDistMultiplier;
    int rowNumElectrodes;
    switch (meaType) {
        case 0: // MCS_8x8_200um
            rowNumElectrodes = 8;
            electrodeDist = 200;
            yFromTopRight = true;
            excludeElectrodes = {{1, 1},
                                 {1, 8},
                                 {8, 1},
                                 {8, 8}}; // 15?
            startDistMultiplier = 1;
            break;

        case 1: // MCS_8x8_100um
            rowNumElectrodes = 8;
            electrodeDist = 100;
            excludeElectrodes = {{1, 1},
                                 {1, 8},
                                 {8, 1},
                                 {8, 8}};
            startDistMultiplier = 1;
            break;

        case 2: // APS_64x64_42um
            rowNumElectrodes = 64;
            electrodeDist = 42;
            startDistMultiplier = 0;
            break;

        default:
            throw std::runtime_error("MEA not implemented");
    }

    // Compute the electrode positions
    numElectrodes = rowNumElectrodes * rowNumElectrodes - excludeElectrodes.size();
    electrodePos = std::vector<std::vector<double>>(numElectrodes, std::vector<double>(2));
    electrodes = std::vector(numElectrodes, std::vector<int>(2));

    int electrodeNum = 0;
    for (int i = startDistMultiplier; i < (rowNumElectrodes + startDistMultiplier); ++i) {
        for (int j = startDistMultiplier; j < (rowNumElectrodes + startDistMultiplier); ++j) {
            std::vector<int> electrode = {i + 1 - startDistMultiplier, j + 1 - startDistMultiplier};
            if (std::find(excludeElectrodes.begin(), excludeElectrodes.end(), electrode) == excludeElectrodes.end()) {
                electrodes[electrodeNum] = electrode;
                electrodePos[electrodeNum][0] = i * electrodeDist;
                if (yFromTopRight) {
                    electrodePos[electrodeNum][1] = ((rowNumElectrodes + 1) - j) * electrodeDist;
                } else {
                    electrodePos[electrodeNum][1] = j * electrodeDist;
                }
                electrodeNum++;
            }
        }
    }
}

void SpikeSet::getDistanceMatrix() {
    D = std::vector<std::vector<double>>(
            electrodes.size(),
            std::vector<double>(electrodes.size())
    );
    // Compute distances across electrodes
    for (int i = 0; i < numElectrodes; ++i) {
        for (int j = 0; j < numElectrodes; ++j) {
            double dx = electrodePos[j][0] - electrodePos[i][0];
            double dy = electrodePos[j][1] - electrodePos[i][1];
            double distance = std::sqrt(dx * dx + dy * dy);
            D[i][j] = distance;
        }
    }
}

//int main() {
//    std::string p = "/Users/maxwuerfek/code/diss/data/g2c_data";
//    SpikeSet spike_data(p, 1);
//    return 0;
//};