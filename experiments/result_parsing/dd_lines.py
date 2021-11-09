from pathlib import Path

from matplotlib import pyplot as plt

import strings as s
from data.summary import add_properties
from experiments.management.locations_manager import BaseLocationsManager
from metric import Metric

fd_out = Path(r'C:\Users\s118344\OneDrive\Conference\Submissions\KDD 2021 Yorick\Submission\figures')


def execute(alm, ca):
    for xf in [s.NUMBER_OF_FEATURES, s.NUMBER_OF_CLASSES, s.NUMBER_OF_DATAPOINTS]:
        for yf in [s.PURITY_WITHOUT_NOISE, s.CLUSTER_DURATION, s.ITERATION_COUNT]:
            for lf in [s.NUMBER_OF_FEATURES, s.NUMBER_OF_CLASSES, s.NUMBER_OF_DATAPOINTS, None]:
                if lf is not None and lf == xf:
                    continue
                _single_file(alm, ca, xf, yf, lf)


def _single_file(alm, ca, xf, yf, lf):
    assert isinstance(alm, BaseLocationsManager)
    results = alm.import_parsed(ca)
    results = add_properties(results)
    if len(Metric.metric_implementations(alm.metric)) == 6:
        f, axarr = plt.subplots(ncols=3, nrows=2)
    else:
        raise NotImplementedError()

    for ax, (mic, df_mic) in zip(axarr.flatten(), results.groupby(s.METRIC_IMPLEMENTATION_CODE)):
        data = df_mic.groupby([xf] + ([] if lf is None else [lf]))[yf].mean().reset_index()

        if lf is None:
            ax.plot(data.set_index(xf)[yf], color=Metric(mic).marker_dict['color'])
        else:
            for lf_value, df_lf in data.groupby(lf):
                # TODO add cmap
                ax.plot(df_lf.set_index(xf)[yf], color=Metric(mic).marker_dict['color'], label=lf)

    nr, nc = f.axes[0].get_subplotspec().get_gridspec().get_geometry()
    f.set_size_inches(h=nr * 5, w=nc * 5)
    alm.fd_lines(ca).mkdir(exist_ok=True, parents=True)
    if lf is None:
        fn = alm.fd_lines(ca) / f'{xf}-{yf}.png'
    else:
        fn = alm.fd_lines(ca) / f'{xf}-{yf} [{lf}].png'
    plt.savefig(fn, bbox_inches='tight')


# def colours_from_cmap(num, c_map_name='inferno'):
#     """
#     Get colour array for a given number of values, based off a cmap, in inverse order.
#
#     Parameters
#     ----------
#     c_map_name: str
#         cmap name for matplotlib.cm.get_cmap
#     num: int or iterable
#         Number of colours in the return array. If not int, the length of num is taken
#
#     Returns
#     -------
#     arr: list of tuple
#         Colours, starting with the 'darkest' (highest in cmap), ending with 'lightest' (lowest in cmap).
#
#     """
#     if not isinstance(num, int):
#         num = len(num)
#     return [cm.get_cmap(c_map_name)(x) for x in np.linspace(start=0, stop=1, num=num + 2, endpoint=True)][-1:1:-1]
#

# def make(fd):
#     """
#     Create all line plots for a given experiment
#
#     Parameters
#     ----------
#     fd: Path or str
#         Folder where results and plots are stored
#     """
#     fd = Path(fd)
#     df_all = import_parsed_results(fd)
#     metric_code = import_settings(fd)[s.METRIC_CODE]
#
#     # 1 folder per dependent feature
#     for dependent_feature in [s.NUMBER_OF_CLUSTERS, s.NUMBER_OF_FEATURES]:
#
#         fd_out = fd / 'line_plots' / dependent_feature
#         fd_out.mkdir(parents=True, exist_ok=True)
#
#         if dependent_feature == s.NUMBER_OF_FEATURES:
#             dfx = df_all
#             line_feature = s.NUMBER_OF_CLUSTERS
#         elif dependent_feature == s.NUMBER_OF_CLUSTERS:
#             dfx = df_all[df_all[s.NUMBER_OF_FEATURES].isin(range(5, 64, 10))].copy()
#             line_feature = s.NUMBER_OF_FEATURES
#         else:
#             raise ValueError(f'dependent_features must be in [{s.NUMBER_OF_FEATURES}, {s.NUMBER_OF_CLUSTERS}]')
#
#         # 1 file per dependent feature
#         for y_feat in [s.PREPROCESSING_DURATION, s.CLUSTER_DURATION, s.DURATION, s.ITERATION_COUNT, s.ARI]:
#             if y_feat not in dfx.columns:
#                 continue
#
#             # 1 file per number of datapoints
#             for n_dp, df in dfx.groupby(s.NUMBER_OF_DATAPOINTS):
#
#                 print(f'Creating line plot for n_dp = {n_dp}, y_feat={y_feat}, dependent_feature={dependent_feature}')
#
#                 n_implementations = len(df[s.METRIC_IMPLEMENTATION_CODE].unique())
#                 f, axarr = plt.subplots(1, n_implementations)
#
#                 for (mic, df_mic), ax in zip(df.groupby(s.METRIC_IMPLEMENTATION_CODE), axarr):
#                     colours = colours_from_cmap('inferno', df_mic[line_feature].unique())
#                     x = zip(df_mic.sort_values(line_feature).groupby(line_feature), colours)
#                     for (lfv, df_lfv), colour in x:
#                         ax.plot(df_lfv[dependent_feature].to_numpy(), df_lfv[y_feat].to_numpy(), '.', color=colour,
#                                 label=lfv)
#                         ax.set_title(mic)
#                         ax.set_ylim(0, max(df.loc[~df[y_feat].isna(), y_feat]))
#
#                 axarr[0].set_ylabel(human_names(y_feat))
#                 for ax in axarr[1:].flatten():
#                     ax.set_yticks([])
#                 for ax in axarr:
#                     ax.set_xlabel(human_names(dependent_feature))
#
#                 f.suptitle(f'{human_names(metric_code)} / {n_dp} (darker = higher {human_names(line_feature)})')
#                 f.set_size_inches(3 * n_implementations, 4)
#                 plt.savefig(fd_out / f'{y_feat}_{n_dp}', bbox_inches='tight')
#                 plt.close()
#
# def add_props(results):
#     results[[s.DATASET, s.NUMBER_OF_DATAPOINTS, s.NUMBER_OF_FEATURES, s.NUMBER_OF_CLUSTERS, s.SEED]] = \
#         results[s.DATASET].str.split('_', expand=True)
#     for x in [s.NUMBER_OF_DATAPOINTS, s.NUMBER_OF_FEATURES, s.NUMBER_OF_CLUSTERS, s.SEED]:
#         results[x] = results[x].astype(int)
#
#
# def make(alm, ca, dp, yfr):
#     assert isinstance(alm, AllLocationsManager)
#     results = alm.import_parsed(ca)
#     results = results[results[s.METRIC_IMPLEMENTATION_CODE] == s.DBB1]
#
#     if not alm.is_real:
#         add_props(results)
#     else:
#         from bitbooster.data.summary import get_summary
#         results = results.merge(get_summary(), left_on=s.DATASET, right_index=True, how='left')
#
#     results = results[results[s.NUMBER_OF_DATAPOINTS] == 10000]
#
#     f, ax = plt.subplots()
#     assert isinstance(ax, plt.Axes)
#     sr = results.groupby(dp)[yfr].mean()
#     # sr = results.set_index(dp)[yfr]
#     if dp == s.NUMBER_OF_DATAPOINTS:
#         ax.semilogx(sr, '.')
#         ax.set_xlabel('log ' + human_names(dp))
#     else:
#         ax.plot(sr, '.')
#         ax.set_xlabel(human_names(dp))
#     ax.set_ylabel(human_names(yfr))
#     plt.show()
#
#
# def run():
#     for x in [s.NUMBER_OF_CLUSTERS, s.NUMBER_OF_DATAPOINTS, s.NUMBER_OF_FEATURES]:
#         # make(AllLocationsManager(False), ca=s.VORONOI, dp=x, yfr=s.RELATIVE_ARI)
#         make(AllLocationsManager(False), ca=s.VORONOI, dp=x, yfr=s.CLUSTER_DURATION)
#         make(AllLocationsManager(False), ca=s.VORONOI, dp=x, yfr=s.ITERATION_COUNT)


if __name__ == '__main__':
    execute(BaseLocationsManager(False), s.VORONOI)
