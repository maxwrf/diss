//
// Created by Max Würfel on 26.06.23.
//

#include "Slurm.h"
# include <string>

int main() {
    bool cluster = true;
    int dSet = 1;

    std::vector<std::string> dSets = {
            "Charlesworth2015",
            "Xu2011"
    };

    std::string inDirPath, outDirPath;
    double corrCutoff;
    int nSamples, nRuns;


    if (cluster) {
        inDirPath = "/store/DAMTPEGLEN/mw894/data/" + dSets[dSet];
        outDirPath = "/store/DAMTPEGLEN/mw894/slurm/" + dSets[dSet];
        corrCutoff = 0.2;
        nSamples = -1;
        nRuns = 10;
    } else {
        inDirPath = "/Users/maxwuerfek/code/diss/data/"  + dSets[dSet];
        outDirPath = "/Users/maxwuerfek/code/diss/slurm/" + dSets[dSet];
        corrCutoff = 0.2;
        nSamples = -1;
        nRuns = 10;
    }

    Slurm::generateInputs(inDirPath,
                          outDirPath,
                          corrCutoff,
                          nSamples,
                          nRuns,
                          dSet
                          );
    return 0;
}