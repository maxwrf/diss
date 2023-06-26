import h5py
import numpy as np
import pandas as pd
from utils import Utils


def analyze(config, f):
    """
    Analyzes the results from the generative model runs
    """

    with h5py.File(f, 'r') as f:
        K_all = f['Kall'][()]
        K_max_all = np.max(K_all, axis=-1)
        param_space = f['paramSpace'][()]

    # make table with results
    idx = np.indices(K_max_all.shape).reshape(K_max_all.ndim, -1).T
    df_results = pd.DataFrame({'sample_idx': idx[:, 0],
                               'model_idx':  idx[:, 1],
                               'param_idx':  idx[:, 2],
                               'model_name': Utils.gnm_rules[idx[:, 1]],
                               'eta': param_space[:, 0][idx[:, 2]],
                               'gamma': param_space[:, 1][idx[:, 2]],
                               'max_energy': np.ravel(K_max_all)
                               })

    Utils.summary_table(df_results, config)
    Utils.plot_landscape(df_results, Utils.gnm_rules, config)
    Utils.plot_boxplots(df_results, Utils.gnm_rules, config)

    return 0


if __name__ == "__main__":
    config = Utils.params_from_json("./config.json")
    f_p = '/Users/maxwuerfek/code/diss/gnm-run/testData/testKall.h5'
    analyze(config, f_p)
