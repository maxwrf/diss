//
// Created by Max Würfel on 27.06.23.
//

#include <string>
#include "Slurm.h"
#include <vector>

int main() {
    bool cluster = false;
    std::string inDirPath;

    if (cluster) {
        inDirPath = "/store/DAMTPEGLEN/mw894/slurm";
    } else {
        inDirPath = "/Users/maxwuerfek/code/diss/gnm-run/slurm";
    }

    std::vector<std::vector<std::vector<std::vector<double>>>> Kall;
    std::vector<std::vector<double>> paramSpace;
    std::vector<std::string> groupIds;
    Slurm::combineResFiles(inDirPath, Kall, paramSpace, groupIds);
    Slurm::writeGroupsHDF5(groupIds, Kall, paramSpace, inDirPath);
    return 0;
}