//
// Created by Max Würfel on 26.06.23.
//

#include <iostream>
#include <vector>
#include "Slurm.h"
#include "GNM.h"

int main(int argc, char *argv[]) {
    // Read the path from the command line
    std::string p;
    if (argc == 2) {
        p = argv[1];
    } else {
        std::cout << "Invalid number of command-line arguments. One argument, the path." << std::endl;
    }

    std::vector<std::vector<double>> A_Y, A_init, paramSpace;
    Slurm::readDatFile(p, A_Y, A_init, paramSpace);

    for (int jModel = 0; jModel < GNM::getRules().size(); ++jModel) {
        std::vector<std::vector<int>> b(
                spikeData.spikeTrains[iSample].m,
                std::vector<int>(paramSpace.size())
        );

        std::vector<std::vector<double>> K(
                paramSpace.size(),
                std::vector<double>(4)
        );

        GNM model(
                A_Y,
                A_init,
                D,
                paramSpace,
                b,
                K,
                spikeData.spikeTrains[iSample].m,
                jModel,
                paramSpace.size(),
                A_Y.size());

        model.generateModels();
    }

    return 0;
};