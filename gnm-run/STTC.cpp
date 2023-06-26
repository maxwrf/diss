//
// Created by Max Würfel on 25.06.23.
//

#include "STTC.h"
#include <cmath>
#include <iostream>
#include <numeric>

double STTC::run_P(int na,
                   int nb,
                   double dt,
                   std::vector<double> &sta_data,
                   std::vector<double> &stb_data) {
    /* Calculate the term P_1. the fraction of spikes from train 1 that
     * are within +/- dt of train 2.
     */

    int i, j, N12;

    N12 = 0;
    j = 0;
    for (i = 0; i <= (na - 1); i++) {
        while (j < nb) {
            // check every spike in train 1 to see if there's a spike in
            //  train 2 within dt  (don't count spike pairs)
            //  don't need to search all j each iteration
            if (fabs(sta_data[i] - stb_data[j]) <= dt) {
                N12 = N12 + 1;
                break;
            } else if (stb_data[j] > sta_data[i]) {
                break;
            } else {
                j = j + 1;
            }
        }
    }
    return N12;
}

double STTC::run_T(int n,
                   double dt,
                   double start,
                   double end,
                   std::vector<double> &spike_times_1) {
    /* Calculate T_A, the fraction of time 'tiled' by spikes with +/- dt.
 *
 * This calculation requires checks to see that (a) you don't count
 * time more than once (when two or more tiles overlap) and checking
 * beg/end of recording.
 */

    double time_A;
    int i = 0;
    double diff;

    // maximum
    time_A = 2 * (double) n * dt;

    // Assume at least one spike in train!

    // if just one spike in train
    if (n == 1) {

        if ((spike_times_1[0] - start) < dt) {
            time_A = time_A - start + spike_times_1[0] - dt;
        } else if ((spike_times_1[0] + dt) > end) {
            time_A = time_A - spike_times_1[0] - dt + end;
        }
    } else { /* more than one spike in train */
        while (i < (n - 1)) {
            diff = spike_times_1[i + 1] - spike_times_1[i];
            if (diff < 2 * dt) {
                // subtract overlap
                time_A = time_A - 2 * dt + diff;
            }

            i++;
        }

        // check if spikes are within dt of the start and/or end, if so
        // just need to subtract overlap of first and/or last spike as all
        // within-train overlaps have been accounted for (in the case that
        // more than one spike is within dt of the start/end

        if ((spike_times_1[0] - start) < dt) {
            time_A = time_A - start + spike_times_1[0] - dt;
        }
        if ((end - spike_times_1[n - 1]) < dt) {
            time_A = time_A - spike_times_1[n - 1] - dt + end;
        }
    }
    return time_A;
}

double STTC::sttc(std::vector<double> &st1_data,
                  std::vector<double> &st2_data,
                  int n1,
                  int n2,
                  double dt,
                  std::vector<double> &time_data) {
    double TA, TB, PA, PB, T;

    T = time_data[1] - time_data[0];

    TA = run_T(n1, dt, time_data[0], time_data[1], st1_data);
    TA = TA / T;

    TB = run_T(n2, dt, time_data[0], time_data[1], st2_data);
    TB = TB / T;

    PA = run_P(n1, n2, dt, st1_data, st2_data);
    PA = PA / (double) n1;

    PB = run_P(n2, n1, dt, st2_data, st1_data);
    PB = PB / (double) n2;

    return (0.5 * (PA - TB) / (1 - TB * PA) + 0.5 * (PB - TA) / (1 - TA * PB));
}

std::vector<std::vector<double>> STTC::tiling(double dt,
                                              std::vector<double> &time,
                                              std::vector<double> &spikes,
                                              std::vector<double> &spike_counts
) {
    int n_electrodes = static_cast<int>(spike_counts.size());
    std::vector<std::vector<double>> result(n_electrodes,
                                            std::vector<double>(n_electrodes));

    // Compute cumulative sum
    std::vector<double> cumsum(spike_counts.size());
    std::partial_sum(spike_counts.begin(), spike_counts.end(), cumsum.begin());
    cumsum.insert(cumsum.begin(), 0.0);

    for (int i = 0; i < n_electrodes; i++) {
        // retrieve spike train 1
        int n1 = ((int) cumsum[i + 1]) - ((int) cumsum[i]);
        std::vector<double> st1(spikes.begin() + (int) cumsum[i],
                                spikes.begin() + (int) cumsum[i + 1]);

        for (int j = i; j < n_electrodes; j++) {
            // retrieve spike train 2 indices
            int n2 = ((int) cumsum[j + 1]) - ((int) cumsum[j]);
            std::vector<double> st2(spikes.begin() + (int) cumsum[j],
                                    spikes.begin() + (int) cumsum[j + 1]);

            // compute sttc
            result[i][j] = result[j][i] = sttc(st1,
                                               st2,
                                               n1,
                                               n2,
                                               dt,
                                               time);
        }
    }
    return result;
}


//int main() {
//    double dt = 0.05;
//    int n1 = 10, n2 = 10;
//    std::vector<double> st1_data(n1), st2_data(n2), time_data(2);
//    time_data[0] = 0.1;
//    time_data[1] = 1;
//
//
//    for (int i = 0; i < 10; i++) {
//        st1_data[i] = (10 - i) / 10;
//        st2_data[i] = i / 10;
//    }
//
//    double res = STTC::sttc(st1_data, st2_data, n1, n2, dt, time_data);
//
//    std::cout << res << std::endl;
//
//
//    return 0;
//}