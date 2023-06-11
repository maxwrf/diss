import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.config import params_from_json


def plot_boxplots(df_results, metadata, config):
    """
    For every rule get for every sample the best model (param comb)
    Plot a histogram accordingly
    """

    df_temp = df_results.groupby(['model_name', 'sample_idx']).agg(
        {'max_energy': ['min']}).reset_index()

    data = [df_temp[df_temp.model_name ==
                    name].max_energy.values.flatten() for name in metadata['gnm_rules']]

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))

    axes.set_xticklabels(metadata['gnm_rules'])

    axes.boxplot(data)
    fig.savefig(config['plot_path'] + 'boxplots.png')


def plot_landscape(df_results, metadata, config):
    """
    For every model type draws the energy landscapes for all parameter 
    combinations
    """

    # energy landscape
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 12))
    axes = axes.flatten()

    for i_model, model in enumerate(metadata['gnm_rules']):
        # extreact mean over all samples
        data = df_results.loc[(df_results.model_name == model),
                              ["sample_idx", "eta", "gamma", "max_energy"]]
        data = data.groupby(["eta", "gamma"]).agg({"max_energy": 'mean'})
        data = data.reset_index().values

        # reshape for landscape
        landscape = data[:, -1].reshape(
            (np.unique(data[:, 0]).size, np.unique(data[:, 1]).size)).T

        # plot
        axes[i_model].set_title(model)
        im = axes[i_model].imshow(
            landscape, cmap=plt.colormaps['viridis'], vmin=0, vmax=1)

        axes[i_model].set_xticks(
            np.arange(len(np.unique(data[:, 0]))), labels=np.unique(data[:, 0]))
        axes[i_model].set_yticks(
            np.arange(len(np.unique(data[:, 1]))), labels=np.unique(data[:, 1]))

        axes[i_model].xaxis.set_major_locator(plt.MaxNLocator(3))
        axes[i_model].yaxis.set_major_locator(plt.MaxNLocator(3))

        axes[i_model].set_xlabel("Eta")
        axes[i_model].set_ylabel("Gamma")

    axes[-1].set_visible(False)
    axes[-2].set_visible(False)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    fig.savefig(config['plot_path'] + 'landscape.png')


def summary_table(df_results, config):
    """
    For every sample and model get the best energy value and parameter comb
    Then for every model across all samples compute summary statistics
    """
    idx = df_results.groupby(['model_name', 'sample_idx'])[
        'max_energy'].idxmin().values

    df_temp = df_results.loc[idx]

    df_summary = df_temp.groupby('model_name').agg(
        {
            'max_energy': ['mean', 'min', 'max', 'std'],
            'eta': ['mean', 'min', 'max', 'std'],
            'gamma': ['mean', 'min', 'max', 'std']
        },).reset_index()
    df_summary.to_csv(config['results_path'] + 'results.csv')


def main(config):
    """
    Analyzes the results from the generative model runs
    """
    with h5py.File(config['results_path'] + "gnm_results.h5", 'r') as f:
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

    summary_table(df_results, config)
    plot_landscape(df_results, metadata, config)
    plot_boxplots(df_results, metadata, config)

    return 0


config = params_from_json("./config.json")
main(config)
