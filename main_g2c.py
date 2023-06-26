import sys
from utils.config import params_from_json

# setup
config = params_from_json("./config.json")  # noqa # noqa
sys.path.append(config['sttc_build'])  # noqa

import numpy as np
from g2c_data.g2c_data import G2C_data
from run_gnms import main
import argparse

if __name__ == "__main__":
    mea_data_dir = config['data_path'] + 'g2c_data/'

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--region", help="brain region to use", default="hpc", nargs='?')
    parser.add_argument(
        "-div", "--div", help="days in vitro to use", type=int, default=14, nargs='?')
    parser.add_argument(
        "-nruns", "--nruns", help="number of runs over param space", type=int, default=10, nargs='?')
    parser.add_argument(
        "-nsamples", "--nsamples", help="number of samples to use (-1 is all samples)", type=int, default=2, nargs='?')

    args = parser.parse_args()
    args = vars(args)

    region = args["region"]
    div = args["div"]
    nruns = args["nruns"]
    nsamples = args["nsamples"]
    print("Region:", region, "\nDIV:", div,
          "\nNruns:", nruns, "\nNsamples:", nsamples)

    # ages: 7, 10, 11, 14, 17, 18, 21, 24, 25, 28
    # regions: 'ctx', 'hpc'
    g2c = G2C_data(mea_data_dir)
    dset_name = 'g2c' + region + 'div' + str(div)
    g2c.create_spike_data(ages=[div], regions=[region])

    # compute sttc and construct the adjacency matrices
    corr_cuttoff = 0.2

    A_inits = np.zeros((len(g2c.spikes), g2c.D.shape[0], g2c.D.shape[1]))
    A_Ys = np.zeros_like(A_inits)

    for i_sample, spike in enumerate(g2c.spikes):
        spike.get_sttc(dt=0.05)
        spike.get_A(g2c.electrodes, sttc_cutoff=corr_cuttoff)
        spike.get_A_init_rand()

        A_inits[i_sample, ...] = spike.A_init
        A_Ys[i_sample, ...] = spike.A_Y

    # running the generative models
    main(A_inits,
         g2c.D,
         A_Ys,
         config,
         dset_name=dset_name,
         n_runs=nruns,
         n_samples=nsamples,
         store=True)
