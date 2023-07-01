//
// Created by Max Würfel on 01.07.23.
//

#include "WeightedGNM.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>


void WeightedGNM::runParamComb(int i_pcomb)
// main function for the generative network build
{
    // get the params
    double eta = params[i_pcomb][0];
    double gamma = params[i_pcomb][1];
    double alpha = params[i_pcomb][2];
    double omega = params[i_pcomb][3];

    // initiate the degree of each node
    for (int i = 0; i < n_nodes; ++i) {
        k_current[i] = 0;
        for (int j = 0; j < n_nodes; ++j) {
            k_current[i] += A_current[i][j];
        }
    }

    // compute the inital value matrix
    initK();

    // initiate cost and value matrices
    std::vector<std::vector<double>> Fd(n_nodes, std::vector<double>(n_nodes));
    std::vector<std::vector<double>> Fk(n_nodes, std::vector<double>(n_nodes));
    std::vector<std::vector<double>> Ff(n_nodes, std::vector<double>(n_nodes));

    for (int i = 0; i < n_nodes; ++i) {
        for (int j = 0; j < n_nodes; ++j) {
            Fd[i][j] = pow(D[i][j], eta);
            Fk[i][j] = pow(K_current[i][j], gamma);
            Ff[i][j] = Fd[i][j] * Fk[i][j] * (A_current[i][j] == 0);
        }
    }

    // initiate P
    std::vector<double> P(upper_tri_index);
    for (int i = 0; i < upper_tri_index; ++i) {
        P[i] = Ff[u[i]][v[i]];
    }

    // prep C
    std::vector<double> C(upper_tri_index + 1);
    C[0] = 0;

    // main loop adding new connections to adjacency matrix
    for (int i = m_seed + 1; i <= m; ++i) {
        // compute the cumulative sum of the probabilities
        for (int j = 0; j < upper_tri_index; ++j) {
            C[j + 1] = C[j] + P[j];
        }

        // select an element
        double r = std::rand() / (RAND_MAX + 1.0) * C[upper_tri_index];
        int selected = 0;
        while (selected < upper_tri_index && r >= C[selected + 1]) {
            selected++;
        }
        int uu = u[selected];
        int vv = v[selected];

        // update the node degree array
        k_current[uu] += 1;
        k_current[vv] += 1;

        // update the adjacency matrix
        A_current[uu][vv] = A_current[vv][uu] = 1;

        // get the node indices for update
        std::vector<int> bth;
        if ((model > 2) && (model < 8)) // if cluster model
        {
            bth = updateClusteringCoeff(uu, vv);
        }
        bth.push_back(uu);
        bth.push_back(vv);

        // update K matrix
        updateK(bth);

        // update Ff matrix (probabilities)
        for (int bth_i = 0; bth_i < bth.size(); ++bth_i) {
            int i = bth[bth_i];
            for (int j = 0; j < n_nodes; ++j) {
                Ff[i][j] = Ff[j][i] = Fd[i][j] * pow(K_current[i][j], gamma) * (A_current[i][j] == 0);
            }
        }

        // update p, TODO: This currently just updates every position
        for (int i = 0; i < upper_tri_index; ++i) {
            P[i] = Ff[u[i]][v[i]];
        }

        // Update Akeep
        A_keep[i_pcomb][i - m_seed - 1] = A_current;

        // If we use w Model, and we are past the start iteration
        if (i >= (start + m_seed + 1)) {
            // Initalize W
            if (i == (start + m_seed + 1)) {
                W_current = A_current;
            } else {
                W_current = W_keep[i_pcomb][i - m_seed - 2];
                W_current[uu][vv] = W_current[vv][uu] = 1;
            }

            // Find the edges in the W matrix
            std::vector<int> edgeRowIdx, edgeColIdx;
            int nEdges = 0;
            for (int j = 0; j < n_nodes; j++) {
                for (int k = j + 1; k < n_nodes; k++) {
                    if (W_current[j][k] != 0) {
                        edgeRowIdx.push_back(j);
                        edgeColIdx.push_back(k);
                        nEdges++;
                    }
                }
            }

            // Compute Eq. 3, Communicability. Simulate over edges
            std::vector<std::vector<double>> sumComm(nEdges, std::vector<double>(nReps));
            for (int jEdge = 0; jEdge < nEdges; ++jEdge) {
                double currentEdgeValue = W_current[edgeRowIdx[jEdge]][edgeColIdx[jEdge]];
                std::vector<double> reps = getReps(currentEdgeValue);

                // Over reps
                Eigen::MatrixXd W_currentSynthEigen(n_nodes, n_nodes);
                for (int l = 0; l < n_nodes; ++l) {
                    for (int n = l; n < n_nodes; ++n) {
                        W_currentSynthEigen(l, n) = W_currentSynthEigen(n, l) = W_current[l][n];
                    }
                }

                for (int kRep = 0; kRep < nReps; kRep++) {
                    W_currentSynthEigen(edgeRowIdx[jEdge], edgeColIdx[jEdge]) = W_currentSynthEigen(
                            edgeColIdx[jEdge],
                            edgeRowIdx[jEdge]) = reps[kRep];

                    Eigen::MatrixXd comm(n_nodes, n_nodes);
                    switch (optiFunc) {
                        case 0: {
                            // Non-normalized matrix exponential of W
                            comm = W_currentSynthEigen.exp();
                            break;
                        }

                        case 1: {
                            // Normalized matrix exponential of W
                            Eigen::VectorXd s = W_currentSynthEigen.rowwise().sum();
                            for (int n = 0; n < s.size(); n++) {
                                if (s(n) == 0) {
                                    s(n) = epsilon;
                                }
                            }
                            Eigen::MatrixXd S = s.asDiagonal();
                            Eigen::MatrixXd temp = (S.inverse().array().sqrt().matrix());
                            Eigen::MatrixXd adj = temp * W_currentSynthEigen * temp;
                            comm = adj.exp();
                            break;
                        }
                    }
                    sumComm[jEdge][kRep] = comm.sum();
                }
            }

            // Compute Eq. 4, Objective function.
            std::vector<double> curve(nEdges);
            for (int jEdge = 0; jEdge < nEdges; ++jEdge) {
                std::vector<double> x(nReps), y(nReps);
                double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

                for (int kRep = 0; kRep < nReps; kRep++) {
                    x[kRep] = kRep + 1;
                    y[kRep] = std::pow((sumComm[jEdge][kRep] * D[edgeRowIdx[jEdge]][edgeColIdx[jEdge]]), omega);
                    sumX += x[kRep];
                    sumY += y[kRep];
                    sumXY += x[kRep] * y[kRep];
                    sumX2 += x[kRep] * x[kRep];
                }

                curve[jEdge] = (nReps * sumXY - sumX * sumY) / (nReps * sumX2 - sumX * sumX);
            }

            // Compute Eq. 5, Update the connection strengths
            for (int jEdge = 0; jEdge < nEdges; ++jEdge) {
                W_current[edgeRowIdx[jEdge]][edgeColIdx[jEdge]] = W_current[edgeColIdx[jEdge]][edgeRowIdx[jEdge]] =
                        W_current[edgeRowIdx[jEdge]][edgeColIdx[jEdge]] - (alpha * curve[jEdge]);
                if (W_current[edgeRowIdx[jEdge]][edgeColIdx[jEdge]] < 0) {
                    W_current[edgeRowIdx[jEdge]][edgeColIdx[jEdge]] = W_current[edgeColIdx[jEdge]][edgeRowIdx[jEdge]] = 0;
                }
            }
            // Need W for next iteration
            W_keep[i_pcomb][i - m_seed - 1] = W_current;
        }
    }
}
