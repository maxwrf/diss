import h5py
from scipy.io import loadmat

d_p = "/local/data/mphilcompbio/2022/mw894/diss/"


# data 0.01Hz
rodent_50k_001Hz_trajectory_data = loadmat(
    d_p + "data/0.01Hz/rodent_50k_001Hz_trajectory_data.mat")[
    'rodent_50k_001Hz_trajectory_data']

rodent_100k_001Hz_generative_models = loadmat(
    d_p + "data/0.01Hz/rodent_100k_001Hz_generative_models.mat")[
    'rodent_100k_001Hz_generative_models.mat']


arrays = {}
f = h5py.File(d_p + "data/0.01Hz/rodent_100k_001Hz_generative_models.mat")
for k, v in f.items():
    arrays[k] = np.array(v)

# data HD-GNM data


x.keys()
