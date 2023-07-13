//
// Created by Max Würfel on 26.06.23.
//

#include <iostream>
#include <vector>
#include "Slurm.h"
#include "GNM.h"

int main(int argc, char *argv[]) {
    // Read the path from the command line
    std::string inPath;
    if (argc == 2) {
        inPath = argv[1];
    } else {
        std::cout << "Invalid number of command-line arguments. One argument, the path." << std::endl;
    }

    std::vector<std::vector<double>> A_Y, A_init, D, paramSpace;
    std::string groupId;
    Slurm::readDatFile(inPath, A_Y, A_init, D, paramSpace, (std::string &) groupId);

    // Compute m
    int m = 0;
    for (int i = 0; i < A_Y.size(); ++i) {
        for (int j = i + 1; j < A_Y.size(); ++j) {
            m += (int) A_Y[i][j];
        }
    }

    // Initialize Kall
    std::vector<std::string> rules = GNM::getRules();
    std::vector<std::vector<std::vector<double>>> Kall(
            rules.size(), std::vector<std::vector<double>>(
                    paramSpace.size(), std::vector<double>(4)));

    // Generate the models
    for (int jModel = 0; jModel < GNM::getRules().size(); ++jModel) {
        std::vector<std::vector<int>> b(
                m,
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
                m,
                jModel,
                paramSpace.size(),
                A_Y.size());

        model.generateModels();
        Kall[jModel] = K;
    }

    // Save the result
    size_t dotPos = inPath.find_last_of('.');
    std::string outPath = inPath.substr(0, dotPos) + ".res";
    Slurm::saveResFile(outPath, Kall, paramSpace, (std::string &) groupId);

    return 0;
};