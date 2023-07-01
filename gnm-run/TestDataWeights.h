//
// Created by Max Würfel on 29.06.23.
//

#ifndef GNM_RUN_TESTDATAWEIGHTS_H
#define GNM_RUN_TESTDATAWEIGHTS_H


#include <vector>
#include <string>

class TestDataWeights {
public:
    static void getSyntheticData(std::vector<std::vector<double>> &A,
                                 std::vector<std::vector<double>> &A_init,
                                 std::vector<std::vector<double>> &D,
                                 std::string &pA, std::string &pA_init, std::string &pD,
                                 int n);
};

#endif //GNM_RUN_TESTDATAWEIGHTS_H
