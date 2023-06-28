//
// Created by Max Würfel on 25.06.23.
//

#ifndef REVOLUTION_SPIKESET_H
#define REVOLUTION_SPIKESET_H

#include <string>
#include <vector>
#include "SpikeTrain.h"

class SpikeSet {
private:
    void getDistanceMatrix();

public:
    std::string path;
    std::vector<std::string> hd5FileNames;
    std::vector<SpikeTrain> spikeTrains;

    std::vector<std::string> electrodes;
    int electrodeDist;
    int numElectrodes;
    double sttcCutoff;
    std::vector<std::vector<double>> electrodePos;
    std::vector<std::vector<double>> D;

    SpikeSet(std::string path_, int nSamples, int dSet);
};


#endif //REVOLUTION_SPIKESET_H
