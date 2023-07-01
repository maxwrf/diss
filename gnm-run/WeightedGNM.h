//
// Created by Max Würfel on 01.07.23.
//

#ifndef GNM_RUN_WEIGHTEDGNM_H
#define GNM_RUN_WEIGHTEDGNM_H

#include "GNM.h"

class WeightedGNM : public GNM {

private:
    std::vector<std::vector<double>> W_current;

    void runParamComb(int i_pcomb) override;

    // Get the reps (actual differences to current edge value)
    std::vector<double> getReps(double currentEdgeValue) {
        std::vector<double> reps(nReps);
        for (int i = 0; i < nReps; i++) {
            reps[i] = currentEdgeValue + currentEdgeValue * repVec[i];
        }
        return reps;
    }

public:
    int start;
    int optiFunc;
    int nReps;
    std::vector<double> repVec;
    // Parameters, Iterations, nxn
    std::vector<std::vector<std::vector<std::vector<double>>>> &A_keep, &W_keep;

    WeightedGNM(std::vector<std::vector<double>> &A_Y_,
                std::vector<std::vector<double>> &A_init_,
                std::vector<std::vector<double>> &D_,
                std::vector<std::vector<double>> &params_,
                std::vector<std::vector<int>> &b_,
                std::vector<std::vector<double>> &K_,
                int m_,
                int model_,
                int n_p_combs_,
                int n_nodes_,
                int start_,
                int optiFunc_,
                double optiResolution_,
                int optiSamples_,
                std::vector<std::vector<std::vector<std::vector<double>>>> &A_keep_,
                std::vector<std::vector<std::vector<std::vector<double>>>> &W_keep_
    ) : GNM(A_Y_,
            A_init_,
            D_,
            params_,
            b_,
            K_,
            m_,
            model_,
            n_p_combs_,
            n_nodes_),
        start(start_),
        optiFunc(optiFunc_),
        A_keep(A_keep_),
        W_keep(W_keep_) {

        // Resize the private variables
        W_current.resize(n_nodes, std::vector<double>(n_nodes));

        // Initiate the keep arrays (params, iterations (i.e., edges added), nxn)
        A_keep.resize(n_p_combs, std::vector<std::vector<std::vector<double>>>(m - m_seed,
                                                                               std::vector<std::vector<double>>(
                                                                                       n_nodes, std::vector<double>(
                                                                                               n_nodes))));
        W_keep.resize(n_p_combs, std::vector<std::vector<std::vector<double>>>(m - m_seed,
                                                                               std::vector<std::vector<double>>(
                                                                                       n_nodes, std::vector<double>(
                                                                                               n_nodes))));

        // Prepare repVec (relative differences to current edge value)
        double minVal = -optiSamples_ * optiResolution_;
        double maxVal = -minVal;
        nReps = static_cast<int>((maxVal - minVal) / optiResolution_) + 1;
        repVec.resize(nReps);
        for (int i = 0; i < nReps; i++) {
            repVec[i] = minVal + i * optiResolution_;
        }
    };
};


#endif //GNM_RUN_WEIGHTEDGNM_H
