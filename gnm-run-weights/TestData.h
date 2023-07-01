//
// Created by Max Würfel on 29.06.23.
//

#ifndef GNM_RUN_SYNTHETICDATA_H
#define GNM_RUN_SYNTHETICDATA_H


#include <vector>
#include <string>

class SyntheticData {
public:
    static void getSyntheticData(std::vector<std::vector<std::vector<double>>> &connectomes,
                                 std::vector<double> &connectomesM,
                                 std::vector<std::vector<double>> &A_init,
                                 std::vector<std::vector<double>> &D,
                                 std::string &pData, std::string &pDist,
                                 int n1, int n2, int n3);
};


#endif //GNM_RUN_SYNTHETICDATA_H
