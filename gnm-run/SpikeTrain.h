//
// Created by Max Würfel on 25.06.23.
//

#ifndef REVOLUTION_SPIKETRAIN_H
#define REVOLUTION_SPIKETRAIN_H

#include <vector>
#include <string>

class SpikeTrain {
public:
    std::string FILE_NAME;
    int div;
    int numActiveElectrodes;
    double sttcCutoff;
    std::vector<int> activeElectrodes;
    // std::string region;

    std::vector<double> spikes;
    std::vector<double> spikeCounts;
    std::vector<double> recordingTime;
    std::vector<std::vector<double>> activeElectrodePos;
    std::vector<std::vector<double>> sttc;
    std::vector<std::vector<double>> A_Y;
    std::vector<std::vector<double>> A_init;
    int m;

    static std::vector<double> readDoubleDataset(std::string file_name,
                                                 std::string dataset_name);

    static std::vector<std::string> readByteStringDataset(std::string file_name,
                                                          std::string dataset_name);

    SpikeTrain(std::string FILE_NAME,
               std::vector<std::vector<double>> &electrodePos,
               int numElectrodes,
               double sttcCutoff
    );
};


#endif //REVOLUTION_SPIKETRAIN_H
