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

    void getElectrodePos();

public:
    std::string path;
    std::vector<std::string> hd5FileNames;
    std::vector<SpikeTrain> spikeTrains;

    double sttcCutoff;

    int meaType;
    int electrodeDist;
    int numElectrodes;
    std::vector<int> electrodes;
    std::vector<std::vector<double>> electrodePos;
    std::vector<std::vector<double>> D;

    SpikeSet(std::string path_,
             int nSamples,
             int dSet,
             double corrCutoff,
             int meaType_);


};


#endif //REVOLUTION_SPIKESET_H
