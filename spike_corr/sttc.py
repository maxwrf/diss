# Reference: https://github.com/sje30/sjemea/tree/master

import numpy as np
from tqdm import tqdm
from multiprocess import Pool


class STTC:
    """
    Pseudo static class for the spike time tiling coefficient
    """
    @staticmethod
    def __get_T(n: int, dt: float, T_start: float, T_stop: float, spike_times: np.array) -> float:
        """
        Computes the time of the spikes +- dt in the total recording time.
        Accounts for overlapping spikes +- dt and start/end of the recoring.
        """
        T_out = 2 * dt * n

        # remove overlap
        delta = np.minimum(spike_times[1:] - spike_times[:-1] - 2*dt, 0)
        T_out += np.sum(delta)

        # start overlap
        if (spike_times[0] - dt) < T_start:
            T_out += (spike_times[0] - dt) - T_start

        # end overlap
        if (spike_times[-1] + dt) > T_stop:
            T_out += T_stop - (spike_times[-1] + dt)

        return T_out

    @staticmethod
    def __get_Ps(n1: int,
                 n2: int,
                 dt: float,
                 spike_times_1: np.array,
                 spike_times_2: np.array) -> tuple:
        """
        Computes the proportion of spikes from spike set 1 (and 2), that lies in
        spike set 2 +- dt.
        TODO: Potentially more efficient if you do not compute all diffs, but need 
        to iterate then.
        """
        diff = np.abs(np.subtract.outer(spike_times_1, spike_times_2)) <= dt
        x1 = np.sum(np.sum(diff, axis=1) > 0)
        x2 = np.sum(np.sum(diff, axis=0) > 0)

        return x1 / n1, x2 / n2

    @staticmethod
    def sttc(spike_times_1: np.array,
             spike_times_2: np.array,
             dt: float, rec_time: np.array) -> float:
        """
        Warning: Spike times 1 & 2 need to be sorted smallest to largest
        """
        n1 = spike_times_1.shape[0]
        n2 = spike_times_2.shape[0]

        T = rec_time[1] - rec_time[0]

        TA = STTC.__get_T(n1, dt, rec_time[0], rec_time[1], spike_times_1) / T
        TB = STTC.__get_T(n2, dt, rec_time[0], rec_time[1], spike_times_2) / T

        PA, PB = STTC.__get_Ps(n1, n2, dt, spike_times_1, spike_times_2)

        print(TA, TB, PA, PB)

        return 0.5 * (PA - TB) / (1 - TB * PA) + 0.5 * (PB - TA) / (1 - TA * PB)

    @staticmethod
    def tiling_all_pairwise(spikes: list, dt: float, rec_time: np.array) -> np.array:
        """
        Computes pairwise sttc for all spikes in list
        """
        """
        Computes pairwise sttc for all spikes in list
        """
        n_spikes = len(spikes)
        m = np.ones((n_spikes, n_spikes))
        idx = np.triu_indices_from(m, k=1)

        for i, j in tqdm(zip(idx[0], idx[1])):
            res = STTC.sttc(spikes[i], spikes[j], dt, rec_time)
            m[i, j] = m[j, i] = res

        return m
