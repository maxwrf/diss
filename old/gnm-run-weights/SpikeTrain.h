//
// Created by Max Würfel on 25.06.23.
//

#ifndef REVOLUTION_SPIKETRAIN_H
#define REVOLUTION_SPIKETRAIN_H

#include <vector>
#include <string>

class SpikeTrain {
private:
    void getGroupId(int dSet);

    void getActiveElectrodeNumbers(std::vector<int> &electrodes);

    void initAdjacencyMatrices(int numElectrodes);

public:
    std::string FILE_NAME;
    std::string groupId;

    int numActiveElectrodes;
    double sttcCutoff;

    // To be read from HD5
    std::vector<std::string> activeElectrodeNames;
    std::vector<double> spikes;
    std::vector<double> spikeCounts;
    std::vector<double> recordingTime;

    // To be computed
    std::vector<int> activeElectrodes;
    std::vector<std::vector<double>> sttc;
    std::vector<std::vector<double>> A_Y;
    std::vector<std::vector<double>> A_init;
    int m;

    static std::vector<double> readDoubleDataset(std::string file_name,
                                                 std::string dataset_name);

    std::vector<std::string> readByteStringDataset(std::string file_name,
                                                   std::string dataset_name);

    SpikeTrain(std::string FILE_NAME,
               std::vector<std::vector<double>> &electrodePos,
               std::vector<int> &electrodes,
               int numElectrodes,
               double dt,
               double sttcCutoff,
               int dSet
    );
};


#endif //REVOLUTION_SPIKETRAIN_H
