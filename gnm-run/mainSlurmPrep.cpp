//
// Created by Max Würfel on 26.06.23.
//
#include "Slurm.h"
# include <string>

int main() {
    std::string inDirPath = "/Users/maxwuerfek/code/diss/data/g2c_data";
    std::string outDirPath = "/Users/maxwuerfek/code/diss/slurm";
    double corrCutoff = 0.2;
    int nSamples = -1;
    int nRuns = 10;

    Slurm::generateInputs(inDirPath, outDirPath, corrCutoff, nSamples, nRuns);

    return 0;
}