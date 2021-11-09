import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import strings
from common import vanilla_ize
from experiments.management.locations_manager import BaseLocationsManager
from experiments.result_parsing._result_parsing_functions import determine_pareto
from metric import Metric

fontsize = 16


def get_sparsity_and_bins(blm, x_feature, metric, binning_parameter=20, binning_type='number_of_bins',
                          sparsity_limits=None):
    sr_x = blm.sparsity_sr(sparsity_mode=x_feature, base_metric=metric)
    if sparsity_limits is not None:
        sr_x = sr_x[(sr_x >= sparsity_limits[0]) & (sr_x <= sparsity_limits[1])]

    if binning_type == 'number_of_bins':
        # Set number of equal width bins
        assert isinstance(binning_parameter, int)
        sr_y = pd.cut(sr_x, bins=binning_parameter, labels=range(binning_parameter))
        n_bins = binning_parameter
    elif binning_type == 'width':
        # Set bin width
        bins = np.arange(sr_x.min(), sr_x.mx() + binning_parameter, binning_parameter)
        n_bins = len(bins) - 1
        sr_y = pd.cut(sr_x, bins=bins, labels=range(n_bins))
    elif binning_type == 'number_in_bins':
        # Each bin (except the final one) has a set number of points
        assert isinstance(binning_parameter, int)
        bins = np.arange(len(sr_x)) // binning_parameter
        n_bins = (len(sr_x) + binning_parameter - 1) // binning_parameter
        sr_y = pd.Series(index=sr_x.sort_values().index, data=bins)
    elif binning_type == 'maximize_n_bins':
        # Maximize the number of bins such that all have a minimum size, possibly skipping datapoints
        assert isinstance(binning_parameter, int)
        n_bins = 1
        best_valid_bins = -1
        while True:
            sr_y = pd.cut(sr_x, bins=n_bins, labels=range(n_bins))
            sr_yc = sr_y.value_counts()
            valid_bins = sr_yc.index[sr_yc >= 30]
            valid_bins_i = len(valid_bins)
            if valid_bins_i < best_valid_bins:
                print(f'Automatic number of bins (maximize, keep most bins, |bin|>={binning_parameter}): {n_bins}')
                break
            else:
                n_bins += 1
                best_valid_bins = valid_bins_i
        # Filter out small bins
        sr_y = sr_y[sr_y.isin(valid_bins)]
    elif binning_type == 'maximize_n_bins_all_points':
        # Maximize the number of bins such that all have a minimum size, not skipping any datapoints
        assert isinstance(binning_parameter, int)
        n_bins = 0
        while True:
            # Try the next n_bins value
            sr_y = pd.cut(sr_x, bins=n_bins + 1, labels=range(n_bins + 1))
            sr_yc = sr_y.value_counts()
            if not all(sr_yc >= binning_parameter):
                # This number of bins would skip part of the data
                print(f'Automatic number of bins (maximize, keep all points, |bin|>={binning_parameter}): {n_bins}')
                return get_sparsity_and_bins(blm, x_feature, metric, binning_type='number_of_bins',
                                             binning_parameter=n_bins, sparsity_limits=sparsity_limits)
            else:
                n_bins += 1
    else:
        raise NotImplementedError(binning_type)
    sr_y.name = 'bin'

    return sr_x, sr_y, n_bins


def makeup_x(ax_, nbins, sparsity_sr):
    ax_.set_xticks([0, nbins - 1])
    ax_.set_xticklabels([f'{sparsity_sr.min():.3f}', f'{sparsity_sr.max():.3f}'], fontsize=fontsize)
    ax_.set_xlabel('Sparsity', fontsize=fontsize)


def sparsity_plot(blm: BaseLocationsManager, ca: str, metric: str, x_feature: str = strings.AVG_4DIST,
                  ax: [None, plt.Axes] = None, time_feature=strings.DURATION, quality_feature=vanilla_ize(strings.ARI),
                  plot_dict=None, **kwargs):
    # def sparsity_plot(blm: BaseLocationsManager, metric, ca, x_feature=strings.AVG_4DIST,
    #                   time_feature=strings.DURATION, quality_feature=vanilla_ize(strings.ARI),
    #                   bins=20, ax=None, sparsity_limits=None, return_datasets=False, **kwargs):
    assert metric in blm.metric, 'Required metric not found in blm'
    mics = list(filter(lambda x: Metric(x).base_metric == metric, blm.mics))

    # Get x data =======================================================================================================
    # Import
    sparsity_sr, sr_bin, number_of_bins = get_sparsity_and_bins(blm=blm, x_feature=x_feature, metric=metric, **kwargs)

    # Get y data =======================================================================================================
    # Import
    df = blm.import_parsed(ca=ca)

    # Filter only used mics
    df = df[df[strings.METRIC_IMPLEMENTATION_CODE].isin(mics)]
    df_pareto = pd.DataFrame(index=df[strings.DATASET].unique(), columns=mics)

    for dsn, df_dataset in df.groupby(strings.DATASET):
        df_pareto.loc[dsn, :] = determine_pareto(df_dataset.set_index(strings.METRIC_IMPLEMENTATION_CODE),
                                                 high_feature=quality_feature, low_feature=time_feature,
                                                 row=None)

    # Combine with x data
    # pd.mean() does not work somehow...
    dfx = df_pareto.join(sr_bin, how='inner')
    df = dfx.groupby('bin').agg(lambda x: np.mean(x)).dropna()

    if plot_dict is None:
        plot_dict = dict(ls='')

    # Make figure ======================================================================================================
    def add(mic_, mic_sr_, ax_, add_avg):
        m = Metric(mic)
        avg = (mic_sr.astype(float)).sum() / number_of_bins
        d = plot_dict.copy()
        d.update(m.marker_dict)
        label = f'{Metric(mic_).legend_name}'
        if add_avg:
            label = f'{label} {avg * 100:.1f}%',
        ax_.plot(df.index, mic_sr_.to_numpy(), label=label, **d)

        return avg

    def makeup_y(ax_):
        ax_.set_yticks([0, 1])
        ax_.set_yticklabels(['0%', '100%'], fontsize=fontsize)
        ax_.set_ylabel('Fraction', fontsize=fontsize)

    def clear_makeup(ax_):
        ax_.set_ylim(-0.1, 1.05)
        ax_.set_xlabel('')
        ax_.set_xticks([])
        ax_.set_ylabel('')
        ax_.set_yticks([0, 1])
        ax_.set_yticklabels(['', ''])

    if ax is False:
        f, ax = plt.subplots(ncols=2, nrows=(len(mics) + 1) // 2)
        for (mic, mic_sr), mic_ax in zip(df.iteritems(), ax.flatten()):
            avg = add(mic, mic_sr, mic_ax, add_avg=True)
            clear_makeup(mic_ax)
            mic_ax.legend(loc='upper center',
                          bbox_to_anchor=(0.5, 1.05) if avg < 0.6 else (0.5, 0.05), ncol=1)

        for axi in ax[:, 0].flatten():
            makeup_y(axi)

        for axi in ax.flatten():
            makeup_x(axi, nbins=number_of_bins, sparsity_sr=sparsity_sr)
        for axi in ax[:-1, :].flatten():
            axi.set_xticklabels(['', ''])

    else:
        if ax is None:
            f, ax = plt.subplots()
        else:
            f = None
            assert isinstance(ax, plt.Axes)
        for mic, mic_sr in df.iteritems():
            add(mic, mic_sr, ax, add_avg=False)
        makeup_x(ax, nbins=number_of_bins, sparsity_sr=sparsity_sr)
        makeup_y(ax)
        ax.set_ylim(-0.1, 1.0)
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=fontsize)

    return f, ax


def sparsity_hist(blm, metric, x_feature=strings.AVG_4DIST, ax=None, plot_dict=None, normalize=False, **kwargs):
    sparsity_sr, sr_bin, n_bins = get_sparsity_and_bins(blm=blm, x_feature=x_feature, metric=metric, **kwargs)

    if plot_dict is None:
        plot_dict = dict(color='b')

    if ax is None:
        f, ax = plt.subplots()
    else:
        f = None

    xy = sr_bin.value_counts()

    x = xy.index
    y = xy.to_numpy()

    ax.bar(x, y, **plot_dict)
    makeup_x(ax, nbins=n_bins, sparsity_sr=sparsity_sr)
    ax.set_ylabel('Subset size', fontsize=fontsize)
    ax.set_yticks([min(y), max(y)])
    ax.set_yticklabels(map(str, [min(y), max(y)]), fontsize=fontsize)

    return f, ax
