import numpy as np
import matplotlib.pyplot as plt


def plot_boxplots(df_results, metadata, config, dset_name):
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
    fig.savefig(config['plot_path'] + 'boxplots_' + dset_name[:-3] + '.png')


def plot_landscape(df_results, metadata, config, dset_name):
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
            np.arange(len(np.unique(data[:, 0]))), labels=np.unique(data[:, 0]).round(0))
        axes[i_model].set_yticks(
            np.arange(len(np.unique(data[:, 1]))), labels=np.unique(data[:, 1]).round(0))

        axes[i_model].xaxis.set_major_locator(plt.MaxNLocator(3))
        axes[i_model].yaxis.set_major_locator(plt.MaxNLocator(3))

        axes[i_model].set_xlabel("Eta")
        axes[i_model].set_ylabel("Gamma")

    axes[-1].set_visible(False)
    axes[-2].set_visible(False)
    fig.tight_layout()
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    fig.savefig(config['plot_path'] + 'landscape_' + dset_name[:-3] + '.png')


def summary_table(df_results, config, dset_name):
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
    df_summary.to_csv(config['results_path'] +
                      'results_' + dset_name[:-3] + '.csv')
