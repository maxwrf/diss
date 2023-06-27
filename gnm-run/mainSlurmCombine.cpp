//
// Created by Max Würfel on 27.06.23.
//

#include <string>
#include "Slurm.h"
#include <vector>
#include "GNM.h"

int main() {
    std::string inDirPath = "/Users/maxwuerfek/code/diss/slurm";
    std::string outFile = "/Users/maxwuerfek/code/diss/slurm/results.h5";
    std::vector<std::vector<std::vector<std::vector<double>>>> Kall;
    std::vector<std::vector<double>> paramSpace;
    Slurm::combineResFiles(inDirPath, Kall, paramSpace);
    GNM::saveResults(outFile, Kall, paramSpace);
    return 0;
}