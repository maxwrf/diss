#ifndef GNM_H
#define GNM_H

class GNM
{
private:
    std::vector<std::vector<double> > A_Y;
    std::vector<std::vector<double> > A_current;
    std::vector<std::vector<double> > K_current;
    std::vector<double> k_current;
    std::vector<double> clu_coeff_current;
    double epsilon = 1e-5;

    double ksTest(const std::vector<double> &x,
                  const std::vector<double> &y);

    std::vector<double> getBetweennesCentrality(
        std::vector<std::vector<double> > &A,
        int n);

    static std::vector<double> getClusteringCoeff(
        std::vector<std::vector<double> > &A,
        int n_nodes,
        std::vector<double> &k);

    std::vector<double> getEdgeLength(
        std::vector<std::vector<double> > &A,
        double **D,
        int n_nodes);

    std::vector<int> updateClusteringCoeff(
        int uu,
        int vv);

    void resetAcurrent();

    void initK();

    void updateK(std::vector<int> bth);

    void runParamComb(int i_pcomb);

public:
    double **A_init;
    double **D;
    double **params;
    int **b;
    double **K;
    int m;
    int model;
    int n_p_combs;
    int n_nodes;

    void generateModels();

    // define the constructor
    GNM(double **A_Y_,
        double **A_init_,
        double **D_,
        double **params_,
        int **b_,
        double **K_,
        int m_,
        int model_,
        int n_p_combs_,
        int n_nodes_)
    {
        A_init = A_init_;
        D = D_;
        params = params_;
        b = b_;
        K = K_;
        m = m_;
        model = model_;
        n_p_combs = n_p_combs_;
        n_nodes = n_nodes_;

        // resize the private variables
        A_Y.resize(n_nodes, std::vector<double>(n_nodes));
        A_current.resize(n_nodes, std::vector<double>(n_nodes));
        K_current.resize(n_nodes, std::vector<double>(n_nodes));
        k_current.resize(n_nodes);
        clu_coeff_current.resize(n_nodes);

        // Initialize a copy of A_Y as cpp vector
        for (int i = 0; i < n_nodes; ++i)
        {
            for (int j = 0; j < n_nodes; ++j)
            {
                A_Y[i][j] = A_Y_[i][j];
            }
        }
    }
};

#endif // GNM_H