import sys
import os

os.chdir("/Users/maxwuerfek/code/diss")  # noqa

import numpy as np
from utils.config import params_from_json
from g2c_data.g2c_data import G2C_data
import sttc.import_sttc_C  # adds path for .so
from STTC import sttc, tiling


# setup
config = params_from_json("./config.json")
mea_data_dir = config['data_path'] + 'g2c_data/'

g2c = G2C_data(mea_data_dir)
g2c.create_spike_data(ages=[7, 10],
                      regions=['ctx'])

# compute sttc
corr_cuttoff = 0.2
for spike in g2c.spikes:
    spike.get_sttc(dt=0.05)
    spike.get_A(g2c.electrodes, sttc_cutoff=0.2)
