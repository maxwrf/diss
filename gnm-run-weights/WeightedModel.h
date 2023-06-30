//
// Created by Max Würfel on 30.06.23.
//

#ifndef GNM_RUN_WEIGHTEDMODEL_H
#define GNM_RUN_WEIGHTEDMODEL_H

struct WeightedModel {
    int update;
    int start;
    int optiFunc;
    double optiResolution;
    int optiSamples;
    int nReps;
    std::vector<double> repVec;

    WeightedModel(int update_,
                  int start_,
                  int optiFunc_,
                  double optiResolution_,
                  int optiSamples_) : update(update_),
                                      start(start_),
                                      optiFunc(optiFunc_),
                                      optiResolution(optiResolution_),
                                      optiSamples(optiSamples_) {
        double minVal = -optiSamples * optiResolution;
        double maxVal = -minVal;
        nReps = static_cast<int>((maxVal - minVal) / optiResolution) + 1;
        repVec.resize(nReps);
        for (int i = 0; i < nReps; i++) {
            repVec[i] = minVal + i * optiResolution;
        }
    }

    std::vector<double> getReps(double currentEdgeValue) {
        std::vector<double> reps(nReps);
        for (int i = 0; i < nReps; i++) {
            reps[i] = currentEdgeValue + currentEdgeValue * repVec[i];
        }
        return reps;
    }
};

#endif //GNM_RUN_WEIGHTEDMODEL_H
