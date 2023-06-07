import numpy as np
import h5py
import sttc.import_sttc_C  # adds path for .so
from STTC import tiling


class SpikeTrain():
    def __init__(self, file_path):
        """
        Given the path to an mea file, returns a Spike Train object
        """
        with h5py.File(file_path, 'r') as file:
            self.file = str(file['meta/age'][()])
            self.age = file['meta/age'][()][0]
            self.region = file['meta/region'][()][0]
            self.recording_time = file['recordingtime'][()]
            self.spike_counts = file['sCount'][()]
            self.electrode_pos = file['epos'][()]
            self.spike_data = file['spikes'][()]
            self.electrodes = file['names'][()]
            self.array = file['array'][()]

    def get_sttc(self, dt=0.05):
        self.sttc = tiling(
            self.spike_data,
            np.cumsum(np.insert(self.spike_counts, 0, 0)),
            self.recording_time,
            0.05
        )

    def get_A(self, mea_electrodes: np.array, sttc_cutoff):
        self.elelectrodes = np.vectorize(
            lambda s: s[:5])(self.electrodes.astype(str))

        # get the indices of the electrodes that are active in the complete MEA
        st_idx = np.where(np.isin(mea_electrodes, self.elelectrodes))[0]

        # construct the adjacency matrix
        self.A = np.zeros((mea_electrodes.shape[0], mea_electrodes.shape[0]))
        A_subset = ((self.sttc > sttc_cutoff) *
                    ~np.eye(self.sttc.shape[0], dtype=bool)).astype(int)
        self.A[np.repeat(st_idx, st_idx.shape[0]),
               np.tile(st_idx, st_idx.shape[0])] = A_subset.flatten()
