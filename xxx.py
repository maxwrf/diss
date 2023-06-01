import numpy as np
from utils.config import params_from_json
from utils.g2c_data import G2C_data
from spike_corr.sttc import STTC

# setup
config = params_from_json("./config.json")
mea_data_dir = config['data_path'] + 'g2c_data/'

g2c = G2C_data(mea_data_dir)
g2c.create_spike_data(ages=[14, 17], regions=['ctx', 'hpc'])


# delta ts for spike time tiling coefficient
# times = [0.05, 0.005, 0.001]
# ci_mean_df = pd.DataFrame()

# for time in times:
#     mn_ci = [mean_ci(x, time) for x in s_list]
#     ci_mean = pd.Series(mn_ci)
#     ci_mean_df = pd.concat([ci_mean_df, ci_mean], axis=1)

# ret = ci_mean_df


# def mean_ci(s, time):
#     m = tiling_allpairwise(s, dt=time)
#     upper_triangle = np.triu_indices(len(m), k=1)

#     mean_val = np.mean(m[upper_triangle], axis=None,
#                        keepdims=False, where=~np.isnan(m))

#     return mean_val


spike_times_1 = np.array([2.1, 6, 10])
spike_times_2 = np.array([1, 2, 2.2, 5])
rec_time = np.array([0, 11])
dt = 0.5
s_d = [spike_times_1, spike_times_2]

STTC.sttc(spike_times_1, spike_times_2, dt, rec_time)

# dt = 0.05
# g2c.spike_meta_data['recording_time'].mean(axis=0)
# STTC.tiling_all_pairwise(g2c.spike_data, dt, np.array([0., 911.23162393]))
