import h5py
import numpy as np
import pandas as pd
import glob
from utils import Utils


def analyze(config, filePathIn):
    """
    Analyzes the results from the generative model runs
    """

    with h5py.File(filePathIn, 'r') as f:
        K_all = f['Kall'][()]
        K_max_all = np.max(K_all, axis=-1)
        param_space = f['paramSpace'][()]
        #groupId = f['groupId'][()][0]

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

    Utils.summary_table(df_results, filePathIn)
    Utils.plot_landscape(df_results, Utils.gnm_rules, filePathIn)
    Utils.plot_boxplots(df_results, Utils.gnm_rules, filePathIn)

    return 0


if __name__ == "__main__":
    config = Utils.params_from_json(
        "/Users/maxwuerfek/code/diss/gnm-analyze/config.json")

    #dir = "/Users/maxwuerfek/code/diss/gnm-analyze/results/Demas2006"
    #dir = "/Users/maxwuerfek/code/diss/gnm-analyze/results/Charlesworth2015"

    #filesPaths = glob.glob(dir + "/*.h5")

    filesPaths = [
        "/Users/maxwuerfek/code/diss/gnm-run-weights/testData/testKall.h5"]

    for f_p in filesPaths:
        analyze(config, f_p)
