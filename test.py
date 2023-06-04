import numpy as np
from utils.config import params_from_json
from g2c_data.g2c_data import G2C_data
import spike_corr.import_sttc_C  # adds path for .so
from STTC import sttc, tiling

# setup
config = params_from_json("./config.json")
mea_data_dir = config['data_path'] + 'g2c_data/'

g2c = G2C_data(mea_data_dir)
g2c.create_spike_data(ages=[7], 
                      regions=['ctx'])


# g2c.spike_meta_data['recording_time'].mean(axis=0)

dt = 0.05
tiling(g2c.spike_data[:10], dt, np.array([0., 911.23162393]))


sttc(g2c.spike_data[0], g2c.spike_data[1], dt, np.array([0., 912]))
