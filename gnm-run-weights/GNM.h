//
// Created by Max Würfel on 26.06.23.
//

#ifndef REVOLUTION_GNM_H
#define REVOLUTION_GNM_H

# include <vector>
# include <string>

class GNM {
private:
    std::vector<std::vector<double>> &A_Y;
    std::vector<double> clu_coeff_current;

    static double ksTest(const std::vector<double> &x,
                         const std::vector<double> &y);

    static std::vector<double> getBetweennesCentrality(
            std::vector<std::vector<double> > &A,
            int n);

    static std::vector<double> getClusteringCoeff(
            std::vector<std::vector<double> > &A,
            int n_nodes,
            std::vector<double> &k);

    static std::vector<double> getEdgeLength(
            std::vector<std::vector<double> > &A,
            std::vector<std::vector<double>> &D,
            int n_nodes);

    void resetAcurrent();

    virtual void runParamComb(int i_pcomb);

protected:
    std::vector<double> k_current;
    std::vector<std::vector<double>> A_current;

    void initK();

    std::vector<std::vector<double>> K_current;

    std::vector<int> updateClusteringCoeff(
            int uu,
            int vv);

    void updateK(std::vector<int> bth);

    double epsilon = 1e-5;
public:
    std::vector<std::vector<double>> &A_init;
    std::vector<std::vector<double>> &D;
    std::vector<std::vector<double>> &params;
    std::vector<std::vector<int>> &b;
    std::vector<std::vector<double>> &K;
    int m;
    int model;
    int n_p_combs;
    int n_nodes;

    void generateModels();

    static std::vector<std::vector<double>> generateParamSpace(int n_runs,
                                                               std::vector<double> eta_limits = {-7, 7},
                                                               std::vector<double> gamma_limits = {-7, 7});

    static void saveResults(std::string &p,
                            std::vector<std::vector<std::vector<std::vector<double>>>> &Kall,
                            std::vector<std::vector<double>> &paramSpace
    );

    static std::vector<std::string> getRules();


    // define the constructor
    GNM(std::vector<std::vector<double>> &A_Y_,
        std::vector<std::vector<double>> &A_init_,
        std::vector<std::vector<double>> &D_,
        std::vector<std::vector<double>> &params_,
        std::vector<std::vector<int>> &b_,
        std::vector<std::vector<double>> &K_,
        int m_,
        int model_,
        int n_p_combs_,
        int n_nodes_) : A_Y(A_Y_),
                        A_init(A_init_),
                        D(D_),
                        params(params_),
                        b(b_),
                        K(K_) {

        m = m_;
        model = model_;
        n_p_combs = n_p_combs_;
        n_nodes = n_nodes_;

        // resize the private variables
        A_current.resize(n_nodes, std::vector<double>(n_nodes));
        K_current.resize(n_nodes, std::vector<double>(n_nodes));
        k_current.resize(n_nodes);
        clu_coeff_current.resize(n_nodes);
    }
};


#endif //REVOLUTION_GNM_H