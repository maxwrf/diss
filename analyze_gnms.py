import h5py
import numpy as np
import pandas as pd
from utils.config import params_from_json
from utils.analysis import *


def analyze(config, f, dset_name):
    """
    Analyzes the results from the generative model runs
    """

    with h5py.File(f, 'r') as f:
        K_all = np.array(f['K_all'])
        K_max_all = np.array(f['K_max_all'])
        metadata = {key: value for key, value in f.attrs.items()}

    # make table with results
    idx = np.indices(K_max_all.shape).reshape(K_max_all.ndim, -1).T
    df_results = pd.DataFrame({'sample_idx': idx[:, 0],
                               'model_idx':  idx[:, 1],
                               'param_idx':  idx[:, 2],
                               'model_name': np.array(metadata['gnm_rules'])[idx[:, 1]],
                               'eta': metadata['eta'][idx[:, 2]],
                               'gamma': metadata['gamma'][idx[:, 2]],
                               'max_energy': np.ravel(K_max_all)
                               })

    summary_table(df_results, config, dset_name)
    plot_landscape(df_results, metadata, config, dset_name)
    plot_boxplots(df_results, metadata, config, dset_name)

    return 0


if __name__ == "__main__":
    dset_names = [
        # "20230611231729_gnm_g2cctxdiv7.h5",
        # "20230612004715_gnm_g2chpcdiv7.h5",
        # "20230612000122_gnm_g2chpcdiv28.h5",
        # "20230612000623_gnm_g2cctxdiv28.h5",
        "20230615231328_synthetic.h5",
    ]

    for dset in dset_names:
        config = params_from_json("./config.json")
        f_p = config['results_path'] + dset
        analyze(config, f_p, dset)
