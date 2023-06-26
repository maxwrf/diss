import sys
import os


current_path = os.path.dirname(os.path.abspath(__file__))  # noqa
sys.path.insert(0, os.path.dirname(current_path))  # noqa
from utils.config import params_from_json
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['sttc_build'])  # noqa

import numpy as np

from g2c_data.g2c_data import G2C_data
from STTC import sttc, tiling

# setup
config = params_from_json("./config.json")
mea_data_dir = config['data_path'] + 'g2c_data/'

g2c = G2C_data(mea_data_dir)
g2c.create_spike_data(ages=[28],
                      regions=['ctx'])

# compute using tiling
res_tiling = tiling(
    g2c.spikes[0].spike_data,
    np.cumsum(np.insert(g2c.spikes[0].spike_counts, 0, 0)),
    g2c.spikes[0].recording_time,
    0.05
)

# compute using sttc only
indixes = np.cumsum(np.insert(g2c.spikes[0].spike_counts, 0, 0))
st1 = g2c.spikes[0].spike_data[indixes[0]:(indixes[1]-1)]
st2 = g2c.spikes[0].spike_data[indixes[1]:(indixes[2]-1)]
res_sttc = sttc(st1, st2, 0.05, g2c.spikes[0].recording_time)

res_sttc == res_tiling[0, 1]
