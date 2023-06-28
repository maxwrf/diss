import os
import h5py
import numpy as np

if __name__ == '__main__':
    dir = "/Users/maxwuerfek/code/diss/data/Xu2011"
    mea_data_files = []
    for file_name in os.listdir(dir):
        if file_name.endswith('.h5'):
            file_path = os.path.join(dir, file_name)
            mea_data_files.append(file_path)

    all_names = np.array([])
    for p in mea_data_files:
        with h5py.File(dir + "/" + file_name, 'r') as file:
            names = file['names'][()]
            names = np.array([n.decode() for n in names])
            all_names = np.append(all_names, names)

    unique_names = np.unique(all_names)
    print(unique_names)
