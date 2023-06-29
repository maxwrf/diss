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
                                                 electrodePos,
                                                 numElectrodes,
                                                 sttcCutoff,
                                                 dSet));

                samplesRead++;
            }
        }
    }
}

void SpikeSet::getElectrodePos() {
    std::vector<int> excludeElectrodes;
    int rowNumElectrodes;
    bool yFromTopRight = false;
    switch (meaType) {
        case 0: // MCS_8x8_200um
            rowNumElectrodes = 8;
            electrodeDist = 200;
            yFromTopRight = true;
            excludeElectrodes = {11, 18, 81, 88, 15};
            break;

        case 1: // MCS_8x8_100um
            rowNumElectrodes = 8;
            electrodeDist = 100;
            excludeElectrodes = {11, 18, 81, 88};
            break;
    }

    // Compute the electrode positions
    numElectrodes = rowNumElectrodes * rowNumElectrodes - excludeElectrodes.size();
    electrodePos = std::vector<std::vector<double>>(numElectrodes, std::vector<double>(2));
    electrodes = std::vector<int>(numElectrodes);
    
    int electrodeNum = 0;
    for (int i = 1; i < (rowNumElectrodes + 1); ++i) {
        for (int j = 1; j < (rowNumElectrodes + 1); ++j) {
            int electrode = i * 10 + j;
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