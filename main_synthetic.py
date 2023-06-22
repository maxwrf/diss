"""
The purpose of the syntehic data run is to check wether the implementation is 
working correctly as results can be compared to the MATLAB programm at
https://github.com/DanAkarca/generativenetworkmodel/blob/master/Scripts/iii.%20Running%20initial%20generative%20models.m
"""
import numpy as np
from run_gnms import main
from utils.config import params_from_json
from utils.seed_network import get_seed_network

# get the initalized adjacency matrix where > 20% of the patient samples
# connectomes already had connections (target data set)
config = params_from_json("./config.json")
A_init, D, A_Ys = get_seed_network(config,
                                   prop=.2,
                                   get_connections=True)

A_init = np.repeat(A_init[np.newaxis, ...], A_Ys.shape[0], axis=0)
main(A_init,
     D,
     A_Ys,
     config,
     dset_name='synthetic',
     n_runs=1000,
     n_samples=2,
     store=True)
