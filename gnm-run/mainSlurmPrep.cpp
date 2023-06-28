//
// Created by Max Würfel on 26.06.23.
//

#include "Slurm.h"
# include <string>

int main() {
    bool cluster = false;

    std::string inDirPath, outDirPath;
    double corrCutoff;
    int nSamples, nRuns;

    if (cluster) {
        inDirPath = "/store/DAMTPEGLEN/mw894/g2c_data";
        outDirPath = "/store/DAMTPEGLEN/mw894/slurm";
        corrCutoff = 0.2;
        nSamples = -1;
        nRuns = 10000;
    } else {
        inDirPath = "/Users/maxwuerfek/code/diss/data/g2c_data";
        outDirPath = "/Users/maxwuerfek/code/diss/slurm";
        corrCutoff = 0.2;
        nSamples = -1;
        nRuns = 10;
    }

    Slurm::generateInputs(inDirPath, outDirPath, corrCutoff, nSamples, nRuns);
    return 0;
}