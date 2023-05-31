# Reference: https://github.com/sje30/g2chvc/tree/master

import h5py
import os
import numpy as np
import pandas as pd


class G2C_data():
    def __init__(self, mea_data_dir: str):
        self.mea_data_dir = mea_data_dir
        self.__make_meafile_cache()

        self.mea_file_df = pd.DataFrame([self.__sort_data(f)
                                         for f in self.mea_data_files])

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
            file_region = file['meta/region'][()]

        return {'file': str(file_name), 'age': file_age, 'region': file_region}

    def __load_meta_data(self):
        # load all the meta data from the mea directory

        mea_file_df['region'] = mea_file_df['region'].apply(
            lambda s: s.decode())

    def h5_read_spikes(file_path) -> np.array:
        """
        Given the path to an mea file, returns a numpy array of the spike data
        """
        with h5py.File(file_path, 'r') as file:
            # spikes_data = file['spikes'][:]
            spikes_data = file['spikes'][()]

        return spikes_data

    def spikes(ages, regions, mea_file_df) -> list:
        """
        Given an age and a region, computes reads the corresponding spikes and 
        returns a list of the spike data
        """
        mea_file_subset = mea_file_df[(mea_file_df['age'].isin(ages))
                                      & (mea_file_df['region'].isin(regions))]

        # add column for easy spike data retrieval
        mea_file_subset['spike_idx'] = range(len(mea_file_subset))

        # collect the data
        spike_l = list()
        for file_name in mea_file_subset['file']:
            spikes_data = h5_read_spikes(file_name)
            spike_l.append(spikes_data)

        return spike_l, mea_file_subset

    def create_spike_data(self,
                          ages: list = [14, 17],
                          regions: list = ['ctx', 'hpc']):
        """
        Loads the spike data from the provided directory and returns a list of the
        loaded data as well as a df with the meta data
        """

        # collect the data for the ages
        spike_data, meta_data = spikes(ages, regions, mea_file_df)

        return spike_data, meta_data
