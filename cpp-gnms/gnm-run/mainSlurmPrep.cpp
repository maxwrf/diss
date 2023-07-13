//
// Created by Max Würfel on 26.06.23.
//

#include "Slurm.h"
# include <string>

// This is required for the key
std::vector<std::string> dSets = {
        "Charlesworth2015",
        "Hennig2011",
        "Demas2006",
        "Maccione2014"
};

// Type of array
std::vector<std::string> meaTypes = {
        "MCS_8x8_200um",
        "MCS_8x8_100um",
        "APS_64x64_42um"
};

int main() {
    // User required parameters
    bool cluster = false;
    int dSet = 0;
    double corrCutoff = 0.2;
    int nSamples = -1;
    int nRuns = 10000;
    double dt = 0.05;

    // Get the dSet Type
    int meaType;
    switch (dSet) {
        case 0: {
            meaType = 0;
            break;
        }
        case 2: {
            meaType = 1;
            break;
        }
        case 3: {
            meaType = 2;
            break;
        }
    };

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
                          meaType,
                          dt

    );
    return 0;
}