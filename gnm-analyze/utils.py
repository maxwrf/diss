import numpy as np
import matplotlib.pyplot as plt
import json
import os


class Utils:
    gnm_rules = np.array([
        "spatial",
        "neighbors",
        "matching",
        "clu-avg",
        "clu-min",
        "clu-max",
        "clu-dist",
        "clu-prod",
        "deg-avg",
        "deg-min",
        "deg-max",
        "deg-dist",
        "deg-prod"])

    @staticmethod
    def params_from_json(p: str) -> dict:
        params = None
        with open(p, 'r') as f:
            params = json.load(f)

        return params

    @staticmethod
    def plot_boxplots(df_results, gnm_rules, filePathIn):
        """
        For every rule get for every sample the best model (param comb)
        Plot a histogram accordingly
        """

        df_temp = df_results.groupby(['model_name', 'sample_idx']).agg(
            {'max_energy': ['min']}).reset_index()

        data = [df_temp[df_temp.model_name ==
                        name].max_energy.values.flatten() for name in gnm_rules]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))

        axes.set_xticklabels(gnm_rules)

        axes.boxplot(data)

        base_name, _ = os.path.splitext(filePathIn)
        fig.savefig(base_name + '_boxplot' + '.png')

    @staticmethod
    def plot_landscape(df_results, gnm_rules, filePathIn):
        """
        For every model type draws the energy landscapes for all parameter 
        combinations
        """

        # energy landscape
        fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(16, 12))
        axes = axes.flatten()

        for i_model, model in enumerate(gnm_rules):
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

        base_name, _ = os.path.splitext(filePathIn)
        fig.savefig(base_name + '_landscape' + '.png')

    @staticmethod
    def summary_table(df_results, filePathIn):
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

        base_name, _ = os.path.splitext(filePathIn)
        df_summary.to_csv(base_name + '_results' + '.csv')
