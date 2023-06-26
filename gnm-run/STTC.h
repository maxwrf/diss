//
// Created by Max Würfel on 25.06.23.
//

#ifndef REVOLUTION_STTC_H
#define REVOLUTION_STTC_H

#include <vector>

class STTC {
public:
    static double run_P(
            int na,
            int nb,
            double dt,
            std::vector<double>& sta_data,
            std::vector<double>& stb_data
    );

    static double run_T(
            int n,
            double dt,
            double start,
            double end,
            std::vector<double> &spike_times_1);

    static double sttc(std::vector<double>& st1_data,
                       std::vector<double>& st2_data,
                       int n1,
                       int n2,
                       double dt,
                       std::vector<double>& time_data);

    static std::vector<std::vector<double>> tiling(double dt,
            std::vector<double> &time,
            std::vector<double> &spikes,
            std::vector<double> &spike_counts
            );
};


#endif //REVOLUTION_STTC_H
