//
// Created by Max Würfel on 26.06.23.
//

#include "Slurm.h"
# include <string>

// This is required for the key
std::vector<std::string> dSets = {
        "Charlesworth2015",
        "Hennig2011",
        "Demas2006"
};

// Type of array
std::vector<std::string> meaTypes = {
        "MCS_8x8_200um",
        "MCS_8x8_100um"
};

int main() {
    // User required parameters
    bool cluster = false;
    int dSet = 2;
    int meaType = 1;
    double corrCutoff = 0.2;
    int nSamples = -1;
    int nRuns = 10;

    // Set the paths
    std::string inDirPath, outDirPath;
    if (cluster) {
        inDirPath = "/store/DAMTPEGLEN/mw894/data/" + dSets[dSet];
        outDirPath = "/store/DAMTPEGLEN/mw894/slurm/" + dSets[dSet];

    } else {
        inDirPath = "/Users/maxwuerfek/code/diss/data/" + dSets[dSet];
        outDirPath = "/Users/maxwuerfek/code/diss/slurm/" + dSets[dSet];
    }

    Slurm::generateInputs(inDirPath,
                          outDirPath,
                          corrCutoff,
                          nSamples,
                          nRuns,
                          dSet,
                          meaType

    );
    return 0;
}