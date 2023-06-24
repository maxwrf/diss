import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['sttc_build'])  # noqa

import numpy as np
from g2c_data.g2c_data import G2C_data
from utils.gnm_utils import generate_param_space
import argparse
import os


def generate_inputs(config,
                    eta_limits,
                    gamma_limits,
                    corr_cuttoff):

    # load the data
    mea_data_dir = config['data_path'] + 'g2c_data/'
    g2c = G2C_data(mea_data_dir)
    g2c.create_spike_data(ages=[
        7, 10, 11, 14, 17, 18, 21, 24, 25, 28
    ], regions=['ctx', 'hpc'])

    # make a directory to store the temp data files
    slurm_dir = config['slurm_dir']
    if not os.path.exists(slurm_dir):
        os.makedirs(slurm_dir)

    for i_sample, spike in enumerate(g2c.spikes):
        spike.get_sttc(dt=0.05)
        spike.get_A(g2c.electrodes, sttc_cutoff=corr_cuttoff)
        spike.get_A_init_rand()

        params = generate_param_space(nruns, eta_limits, gamma_limits)

        np.savez(slurm_dir+f'sample_{i_sample}.dat',
                 div=spike.age,
                 region=spike.region,
                 A_init=spike.A_init,
                 A_Y=spike.A_Y,
                 D=g2c.D,
                 params=params)

    print(len(g2c.spikes), "data files stored in :", slurm_dir)


if __name__ == "__main__":
    """
    ages: 7, 10, 11, 14, 17, 18, 21, 24, 25, 28
    regions: 'ctx', 'hpc'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nruns", "--nruns",
        help="number of runs over param space", type=int, default=64, nargs='?')

    args = parser.parse_args()
    args = vars(args)
    nruns = args['nruns']

    eta_limits = [-7, 7]
    gamma_limits = [-7, 7]
    corr_cuttoff = 0.2
    print("\nNruns:", nruns)

    generate_inputs(config,
                    eta_limits,
                    gamma_limits,
                    corr_cuttoff)
