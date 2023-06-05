# Reference: https://github.com/sje30/g2chvc/tree/master

import h5py
import os
import numpy as np
import pandas as pd


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


class G2C_data():
    def __init__(self, mea_data_dir: str):
        # load the file info
        self.mea_data_dir = mea_data_dir
        self.__make_meafile_cache()

        # load the meta data
        self.mea_file_df = pd.DataFrame([self.__sort_data(f)
                                         for f in self.mea_data_files])

        self.mea_file_df['region'] = self.mea_file_df['region'].apply(
            lambda s: s.decode())

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

        return {'file': str(file_name),
                'age': file_age[0],
                'region': file_region[0]}

    def __get_spikes(self) -> list:
        """
        Given an age and a region, computes reads the corresponding spikes and 
        returns a list of the spike data
        """
        mea_file_subset = self.mea_file_df[(self.mea_file_df['age'].isin(self.ages))
                                           & (self.mea_file_df['region'].isin(self.regions))].copy()

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
