# Reference: https://github.com/sje30/g2chvc/tree/master

import h5py
import os
import numpy as np
import pandas as pd
from spike_train import SpikeTrain
from scipy.spatial.distance import cdist


class G2C_data():
    def __init__(self, mea_data_dir: str):
        # load the file info
        self.mea_data_dir = mea_data_dir
        self.__make_meafile_cache()

        # load the meta data
        self.mea_file_df = pd.DataFrame([self.__sort_data(f)
                                         for f in self.mea_data_files])

        # self.mea_file_df['region'] = self.mea_file_df['region'].apply(
        # lambda s: s.decode())

    def __make_meafile_cache(self) -> list:
        """
        Finds all the mea files in the provied dir in h5 format
        return list of path
        """
        mea_data_files = []
        for file_name in os.listdir(self.mea_data_dir):
            if file_name.endswith('.h5'):
                file_path = os.path.join(self.mea_data_dir, file_name)
                mea_data_files.append(file_path)
        self.mea_data_files = mea_data_files

    def __sort_data(self, file_name: str) -> pd.DataFrame:
        """
        For each file, returns a df containing path, age and region
        """
        with h5py.File(file_name, 'r') as file:
            file_age = file['meta/age'][()]
            # file_region = file['meta/region'][()]

        return {'file': str(file_name),
                'age': file_age[0]}
        # 'region': file_region[0]}

    def __get_spikes(self) -> list:
        """
        Given an age and a region, computes reads the corresponding spikes and
        returns a list of the spike data
        """
        mea_file_subset = self.mea_file_df[(
            self.mea_file_df['age'].isin(self.ages))].copy()
#                                           & (self.mea_file_df['region'].isin(self.regions))].copy()

        # add column for easy spike data retrieval
        mea_file_subset['spike_idx'] = range(len(mea_file_subset))

        # collect the data
        spike_l = list()
        for file_path in mea_file_subset['file']:
            spike_l.append(SpikeTrain(file_path))

        return spike_l, mea_file_subset

    def create_spike_data(self,
                          ages: list = [14, 17],
                          regions: list = ['ctx', 'hpc']) -> None:
        """
        Loads the spike data from the provided directory and returns a list of the
        loaded data as well as a df with the meta data
        """

        self.ages = ages
        self.regions = regions

        # collect the data for the ages
        self.spikes, self.spikes_data = self.__get_spikes()

        # get the unique electrodes
        electrodes = [s.electrodes for s in self.spikes]
        electrodes = np.concatenate(electrodes, axis=0)
        electrodes = np.vectorize(lambda s: s[:5])(electrodes)
        self.electrodes = np.unique(electrodes).astype(str)

        # fist number times 200, 9 - second number times 200
        self.electrode_pos = np.array([
            np.array([int(str(e[-2]))*200,
                      (9-int(e[-1]))*200]) for e in self.electrodes])

        self.D = cdist(self.electrode_pos,
                       self.electrode_pos, metric='euclidean')

        return 0


if __name__ == "__main__":
    dir = "/Users/maxwuerfek/code/diss/data/Demas2006"
    g = G2C_data(dir)
    g.create_spike_data()
