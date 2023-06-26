//
// Created by Max Würfel on 26.06.23.
//

#ifndef GNM_RUN_SLURM_H
#define GNM_RUN_SLURM_H

#include <string>

class Slurm {
public:
    static void generateInputs(std::string &inDirPath,
                               std::string &outDirPath,
                               double corrCutoff,
                               int nSamples,
                               int nRuns);

    static void readDatFile(std::string &inPath,
                            std::vector<std::vector<double>> &A_Y,
                            std::vector<std::vector<double>> &A_init,
                            std::vector<std::vector<double>> &paramSpace);
};


#endif //GNM_RUN_SLURM_H
