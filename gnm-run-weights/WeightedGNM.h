//
// Created by Max Würfel on 01.07.23.
//

#ifndef GNM_RUN_WEIGHTEDGNM_H
#define GNM_RUN_WEIGHTEDGNM_H

#include "GNM.h"
#include "WeightedModel.h"

class WeightedGNM : public GNM {

private:
    std::vector<std::vector<double>> W_current;

    void runParamComb(int i_pcomb) override;

public:
    WeightedModel &wModel;

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
                WeightedModel &wModel_
    ) : GNM(A_Y_,
            A_init_,
            D_,
            params_,
            b_,
            K_,
            m_,
            model_,
            n_p_combs_,
            n_nodes_), wModel(wModel_) {
        W_current.resize(n_nodes, std::vector<double>(n_nodes));
    };
};


#endif //GNM_RUN_WEIGHTEDGNM_H
