//
// Created by Max Würfel on 25.06.23.
//

#include <string>
#include <filesystem>
#include <cmath>
#include "SpikeSet.h"

SpikeSet::SpikeSet(std::string path_, int nSamples) {
    path = path_;

    // set the electrodes and distance
    electrodes = {
            "ch_12", "ch_13", "ch_14", "ch_16", "ch_17", "ch_21", "ch_22",
            "ch_23", "ch_24", "ch_25", "ch_26", "ch_27", "ch_28", "ch_31",
            "ch_32", "ch_33", "ch_34", "ch_35", "ch_36", "ch_37", "ch_38",
            "ch_41", "ch_42", "ch_43", "ch_44", "ch_45", "ch_46", "ch_47",
            "ch_48", "ch_51", "ch_52", "ch_53", "ch_54", "ch_55", "ch_56",
            "ch_57", "ch_58", "ch_61", "ch_62", "ch_63", "ch_64", "ch_65",
            "ch_66", "ch_67", "ch_68", "ch_71", "ch_72", "ch_73", "ch_74",
            "ch_75", "ch_76", "ch_77", "ch_78", "ch_82", "ch_83", "ch_84",
            "ch_85", "ch_86", "ch_87"
    };


    // Compute the distance matrix
    electrodeDist = 200;
    sttcCutoff = 0.2;
    numElectrodes = electrodes.size();

    electrodePos = std::vector<std::vector<double>>(
            numElectrodes,
            std::vector<double>(2)
    );

    D = std::vector<std::vector<double>>(
            electrodes.size(),
            std::vector<double>(electrodes.size())
    );

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
                                                 sttcCutoff));

                samplesRead++;
            }
        }
    }
}

void SpikeSet::getDistanceMatrix() {
    // Compute electrode positions
    for (int i = 0; i < numElectrodes; ++i) {
        electrodePos[i][0] = (electrodes[i][electrodes[i].length() - 2] - '0') * electrodeDist;
        electrodePos[i][1] = (9 - (electrodes[i][electrodes[i].length() - 1] - '0')) * electrodeDist;
    }

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