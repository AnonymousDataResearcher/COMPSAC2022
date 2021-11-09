import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import strings as s
from common import human_names
from bitbooster.common import as_list
from data.summary import add_properties
from experiments.management.locations_manager import BaseLocationsManager
from metric import Metric


def pure_plot(alm, y_axis, ax=None, add_letters=False, average_nc=True, number_of_datapoints=None,
              legend=True, skip_mic=None, logy=False, mult_y=1, err_base=None):
    """
    Parameters
    ----------
    alm: BaseLocationsManager
        BaseLocationsManager for data retrieval
    y_axis: str
        What to plot on the y-axis
    ax: plt.Axes or None
        Where to plot the data. If None, a new ax will be created.
    add_letters: bool
        Whether to add letters for the datasets. Ignored for synthetic
    average_nc: bool
        Whether to average the same experiments with different s.NUMBER_OF_CLASSES. Ignored for real
    number_of_datapoints: int or None
        If not None, only use experiments with this number of datapoints. Otherwise use all. Ignored for synthetic
    legend: bool
        Whether to add a legend to the Axes
    skip_mic: None or (Iterable of) str
        Which MIC's to skip.
    logy: bool
        Plot y as log axis
    mult_y: Number
        Multiply y values by this value (for fixing the axes)
    err_base: str or None
        Which metric to use as true value for the error. If None, the metric of alm is used, or a ValueError will be
        raised if the alm has multiple metrics. If 'respective', then each metric will be compared against its
        respective vanilla implementation.

    Returns
    -------
    ax: plt.Axes
        The Axes where the data has been plot; which is either the input ax or a newly created Axes.
    """
    # Load data
    assert isinstance(alm, BaseLocationsManager)

    if y_axis in [s.PREPROCESSING_DURATION, s.DURATION]:
        df = alm.import_time()
    elif y_axis in [s.MAX_ERROR, s.MEAN_ERROR, s.MEAN_RELATIVE_ERROR, s.MAX_RELATIVE_ERROR]:
        if err_base == 'respective':
            mic_to_vanilla = dict()
            for metric in as_list(alm.metric, str):
                mic_to_vanilla.update({k: Metric.vanilla_implementations(metric)
                                       for k in Metric.metric_implementations(metric)
                                       if k not in skip_mic})
        elif err_base is None:
            if isinstance(alm.metric, str):
                mic_to_vanilla = {k: Metric.vanilla_implementations(alm.metric)
                                  for k in Metric.metric_implementations(alm.metric)
                                  if k not in skip_mic}
            else:
                raise ValueError('err_base is None, but given alm has multiple metrics')
        elif Metric(err_base).is_vanilla:
            mic_to_vanilla = {k: err_base for k in Metric.metric_implementations(alm.metric)
                              if k not in skip_mic}
        else:
            raise NotImplementedError(f'err_base argument {err_base} not valid')

        df = pd.DataFrame()
        for mic, vanilla_mic in mic_to_vanilla.items():
            df_mic = alm.import_error(Metric(vanilla_mic).base_metric)
            df = df.append(df_mic[df_mic[s.METRIC_IMPLEMENTATION_CODE] == mic])
    else:
        raise NotImplementedError(y_axis)
    df = add_properties(df)
    # Use / create new ax
    if ax is None:
        _, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)

    # Filter number of datapoints
    if number_of_datapoints is not None and alm.synthetic:
        # number_of_datapoints is given and we have a synthetic dataset.
        df = df[df[s.NUMBER_OF_DATAPOINTS] == number_of_datapoints]
        if len(df) == 0:
            raise ValueError(f'No results for {number_of_datapoints} datapoints')
    if skip_mic is not None:
        skip_mic = as_list(skip_mic, str)
        df = df[~df[s.METRIC_IMPLEMENTATION_CODE].isin(skip_mic)]

    # Average number of clusters
    df = df[[s.METRIC_IMPLEMENTATION_CODE, y_axis, s.NUMBER_OF_FEATURES]]
    if average_nc and alm.synthetic:
        df = df.groupby([s.METRIC_IMPLEMENTATION_CODE, s.NUMBER_OF_FEATURES])[y_axis].mean().reset_index()
    else:
        pass

    # Make one line per MIC
    lines = []
    labels = []
    for mic, mic_df in df.groupby(s.METRIC_IMPLEMENTATION_CODE):
        _kwargs = {**Metric(mic).marker_dict, **dict(ms=7, ls='-')}

        # Vanilla does not have errors, only durations, so if we are not dealing with duration, then add an invisible
        # point
        labels.append(Metric(mic).title)
        if mic == err_base and y_axis not in [s.DURATION, s.PREPROCESSING_DURATION]:
            lines.append(Line2D([0], [0], **_kwargs))
            continue

        data = mic_df.set_index(s.NUMBER_OF_FEATURES)[y_axis].sort_index() * mult_y

        if logy:
            lines.append(ax.semilogy(data, **_kwargs)[0])
        else:
            lines.append(ax.plot(data, **_kwargs)[0])

    if add_letters and alm.real:
        for _, r in df.groupby([s.ABBREVIATION, s.NUMBER_OF_FEATURES])[y_axis].max().reset_index().iterrows():
            ax.text(x=r[s.NUMBER_OF_FEATURES], y=r[y_axis], s=r[s.ABBREVIATION],
                    ha='center', va='bottom')

    if legend:
        if len(lines) >= 5:
            labels = [lab.replace(s.EUCLIDEAN.capitalize(), s.EUC).replace(s.MANHATTAN.capitalize(), s.MAN)
                      for lab in labels]
            cols = 1
        else:
            cols = 1

        ax.legend(lines, labels, prop={'size': 12}, ncol=cols)
    ax.set_xlabel(human_names(s.NUMBER_OF_FEATURES))
    ax.set_ylabel(human_names(y_axis))
    return ax


def execute(alm):
    assert isinstance(alm, BaseLocationsManager)

    if alm.synthetic:
        res = alm.import_time()
        res = add_properties(res)
        ndp_values = res[s.NUMBER_OF_DATAPOINTS].unique()
        fn_values = [alm.fd_pure / f'{ndp}.png' for ndp in ndp_values]
    else:
        ndp_values = [None]
        fn_values = [alm.fd_pure / f'plots.png']

    for ndp, fn in zip(ndp_values, fn_values):
        f, axarr = plt.subplots(2, 3)
        # TODO centralize and check in AllLocationsManager.export_pure
        all_y_axis = [s.PREPROCESSING_DURATION, s.DURATION, s.MEAN_RELATIVE_ERROR,
                      s.MEAN_ERROR, s.MAX_ERROR, s.MAX_RELATIVE_ERROR]

        # Add plots
        for ax, y_axis in zip(axarr.flatten(), all_y_axis):
            pure_plot(alm=alm, y_axis=y_axis, ax=ax, number_of_datapoints=ndp,
                      legend=y_axis == s.PREPROCESSING_DURATION,
                      logy=(y_axis in [s.DURATION]))

        # axarr.flatten()[-1].set_visible(False)
        f.set_size_inches(w=15, h=10)
        plt.savefig(fn, bbox_inches='tight')
