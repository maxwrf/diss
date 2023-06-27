//
// Created by Max Würfel on 26.06.23.
//

#ifndef GNM_RUN_SLURM_H
#define GNM_RUN_SLURM_H

#include <string>
#include <vector>

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
                            std::vector<std::vector<double>> &D,
                            std::vector<std::vector<double>> &paramSpace,
                            std::string &groupId);

    static void saveResFile(std::string &outDirPath,
                            std::vector<std::vector<std::vector<double>>> &Kall,
                            std::vector<std::vector<double>> &paramSpace,
                            std::string &groupId);

    static void combineResFiles(std::string &inDirPath,
                                std::vector<std::vector<std::vector<std::vector<double>>>> &Kall,
                                std::vector<std::vector<double>> &paramSpace,
                                std::vector<std::string> &groupIds
    );

    static void writeGroupsHDF5(std::vector<std::string> &groupIds,
                                std::vector<std::vector<std::vector<std::vector<double>>>> &Kall,
                                std::vector<std::vector<double>> &paramSpace,
                                std::string &outDirPath
    );
};


#endif //GNM_RUN_SLURM_H
