import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.insert(0, os.path.dirname(current_path))  # noqa

import numpy as np
from utils.config import params_from_json
from g2c_data.g2c_data import G2C_data
import spike_corr.import_sttc_C  # adds path for .so
from STTC import sttc, tiling


# setup
config = params_from_json("../config.json")
mea_data_dir = config['data_path'] + 'g2c_data/'

g2c = G2C_data(mea_data_dir)
g2c.create_spike_data(ages=[7],
                      regions=['ctx'])

# compute using tiling
res_tiling = tiling(
    g2c.spikes[0].spike_data,
    np.cumsum(np.insert(g2c.spikes[0].spike_counts, 0, 0)),
    g2c.spikes[0].recording_time,
    0.05
)
