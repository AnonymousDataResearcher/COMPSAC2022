import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

import strings as s
from common import human_names, result_columns
from data.summary import add_properties
from experiments.management.locations_manager import BaseLocationsManager
from experiments.result_parsing._result_parsing_functions import relatives
from metric import EUCLIDEAN_VANILLA, Metric

abs_x = [s.CLUSTER_DURATION, s.PREPROCESSING_DURATION, s.DURATION, s.ITERATION_COUNT]
rel_x = [relatives[x] for x in abs_x]


# This one does all files
def execute(alm, ca=None):
    if ca is None:
        for cax in s.all_clustering_algorithms:
            execute(alm, cax)
        return

    for yfa in relatives.keys():
        if yfa in abs_x or yfa not in result_columns(ca):
            continue
        else:
            _separate(alm, by=s.METRIC_IMPLEMENTATION_CODE, ca=ca, yfa=yfa)


# This one does one file per xfa
def _separate(alm, by, ca, yfa):
    """
    Make separate plot files.

    Parameters
    ----------
    alm: AllLocationsManager
        All Locations Manager
    by: str
        Each plot contains a single value for this parameter. Must be s.METRIC_IMPLEMENTATION_CODE or s.DATASET
    ca: str
        Clustering algorithm
    yfa: str
        y feature
    """
    assert isinstance(alm, BaseLocationsManager)
    assert by in [s.METRIC_IMPLEMENTATION_CODE, s.DATASET]
    results = _prepped(alm, ca, yfa)
    results = add_properties(results)

    if alm.real:
        results = results[results[s.NUMBER_OF_DATAPOINTS] >= 1000]
        warnings.warn('You should clean up the pareto code, only using nd >=1000 as default')

    xfa_values = [xfa for xfa in abs_x if xfa in result_columns(ca) + [s.DURATION]]

    for xfa in xfa_values:
        make_xfa(xfa, by, yfa, alm, ca, results)

        # Save figure
        alm.fd_pareto(ca).mkdir(exist_ok=True, parents=True)
        fn = alm.fd_pareto(ca) / f'{by}_({xfa},{yfa}).png'
        plt.savefig(fn, bbox_inches='tight')
        plt.close()


def make_xfa(xfa, by, yfa, alm, ca, results=None, single=False, do_mics=None,
             cols=4):
    if results is None:
        results = add_properties(_prepped(alm, ca, yfa))

    if by != s.METRIC_IMPLEMENTATION_CODE:
        raise NotImplementedError(f'Sorry, I broke this for by!={s.METRIC_IMPLEMENTATION_CODE}')

    if do_mics is None:
        do_mics = results[s.METRIC_IMPLEMENTATION_CODE].unique()

    y_lim = (0, max(results[relatives[yfa]]))

    all_by = len(do_mics)

    if not single:
        f, axarr = plt.subplots((all_by + cols - 1) // cols, cols)
    else:
        f, z = plt.subplots()
        axarr = np.array([z] * all_by)

    rename_dict = {relatives[xfa]: s.X, relatives[yfa]: s.Y}
    results = results.rename(columns=rename_dict)
    for ax, by_value in zip(axarr.flatten(), do_mics):
        # Set x/y
        cols = [s.Y, s.X, s.METRIC_IMPLEMENTATION_CODE, s.DATASET]
        df = results.loc[results[s.METRIC_IMPLEMENTATION_CODE] == by_value, cols]

        # Set all datasets to blob for not real
        if alm.synthetic and not single:
            alpha = .4
        elif alm.synthetic and single:
            alpha = .02
        elif alm.real:
            alpha = 1
        else:
            raise NotImplementedError()

        kwargs = dict(alpha=alpha)

        # Basic Pareto Plot
        _base_pareto(ax, df, add_labels=False, **kwargs)

        dfx = df.drop(columns=s.DATASET).groupby(by=by).mean().assign(**{s.DATASET: s.BLOB}).reset_index()

        _add_baseline(ax, add_label=False, alpha=1)

        if single:
            _base_pareto(df=dfx, ax=ax, add_labels=True, ms=10)
        else:

            # Overwrite properties to show black marker
            _base_pareto(df=dfx, ax=ax, add_labels=True, mfc='k', mec='w', fillstyle='full',
                         color='k', marker='o', ms=10)

        ax.set_xlabel(human_names(relatives[xfa]), fontsize=18)
        ax.set_ylabel(human_names(relatives[yfa]), fontsize=18)
        if not single:
            ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

        if ax.get_xlim()[1] >= 2:
            ax.set_xticks(np.arange(0, ax.get_xlim()[1], 1))
        else:
            ax.set_xticks(np.arange(0, ax.get_xlim()[1], .2))

        ax.tick_params(axis='both', which='major', labelsize=15)

        if not single:
            assert len(dfx) == 1
            x = dfx.iloc[0][s.X]
            y = dfx.iloc[0][s.Y]
            ax.set_ylim(*y_lim)
            ax.set_yticks(np.arange(0, y_lim[1], 0.5))
            ax.set_title(f'{Metric(by_value).short_title} [{x:.2f},{y:.2f}]', fontsize=20)
        else:
            ax.legend()

    nr, nc = f.axes[0].get_subplotspec().get_gridspec().get_geometry()
    f.set_size_inches(h=nr * 5, w=nc * 5)

    if single:
        ax = axarr[0]
        ax.set_xlim(0, ax.get_xlim()[1] * 1.1)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

        axarr = axarr[0:1].reshape(1, 1)

    return f, axarr


# TODO add some method here that takes care of creating a single ax for some given input, which can be called
#  externally, and which is called in _separate

# This one does one ax
def _base_pareto(ax, df, add_labels=True, **kwargs):
    """
    Base function for creating a pareto plot.

    Parameters
    ----------
    ax: plt.Axes or None
        If not None, the pareto plot is plot in this. Otherwise a new plt.Figure and plt.Axes is created.
    df: pd.DataFrame
        DataFrame which contains the information. Must contain the following columns: s.METRIC_IMPLEMENTATION_CODE,
        s.X, s.Y. s.METRIC_IMPLEMENTATION_CODE is used to determine the shape and colour of each marker. s.X and s.Y
        will be used to determine the position of each marker. If it also contains a column s.COLOUR_SCALE, that column
        will instead be used to colour the markers; increasing values get a darker colour according to the 'inferno'
        cmap from `matplotlib.cm`.
    add_labels: bool
        If True, each s.METRIC_IMPLEMENTATION_CODE is given a label, usable for for a add_label later.

    Other Parameters
    ----------------
    **kwargs is directly forwarded to plt.plot.

    Returns
    -------
    ax: plt.Axes
        Axes object in which the pareto plot is plot. Returns input ax if given, or a new plt.Axes otherwise
    """
    assert {s.DATASET, s.METRIC_IMPLEMENTATION_CODE, s.X, s.Y}.issubset(df.columns)

    if s.COLOUR_SCALE in df.columns and add_labels:
        # TODO doing both is not yet implemented because scatter doesn't allow labels.
        #  Alternative would be to plot colour-by-colour, labelling the first/last/darkest/lightest point
        #  Another alternative would be to plot a point somewhere at infinity or something, and then pre-scaling the
        #  axis here
        warnings.warn('Labels are not added if colour scale is added (Not implemented yet)')

    # Create Axes if not given
    if ax is None:
        _, ax = plt.subplots()
    assert isinstance(ax, plt.Axes)

    for (dsn, mic), sub_df in df.groupby([s.DATASET, s.METRIC_IMPLEMENTATION_CODE]):
        kw_sub = kwargs.copy()
        for k, v in Metric(mic).marker_dict.items():
            kw_sub.setdefault(k, v)

        if add_labels:
            kw_sub['label'] = Metric(mic).title

        if s.COLOUR_SCALE in df.columns:
            raise NotImplementedError('This no longer works, sorry')
        else:
            if 'edgecolors' in kw_sub:
                # Inconsistency in matplotlib :(
                kw_sub['mec'] = kw_sub.pop('edgecolors')
            ax.plot(sub_df[s.X], sub_df[s.Y], ls='', **kw_sub)

    return ax


# OTHER MAKEUP FUNCTIONS -----------------------------------------------------------------------------------------------
def _add_baseline(ax, add_label=True, **kwargs):
    """
    Adds baseline visual to the given ax.

    Parameters
    ----------
    ax: plt.Axes
        The Axes object to add to
    add_label: bool
        Whether to label the baseline (for use in legend later)
    """
    if add_label:
        kwargs['label'] = human_names(s.EUCLIDEAN)
    else:
        kwargs.pop('label', None)

    kwargs.setdefault('alpha', 0.5)

    ax.plot([0, 1], [1, 1], 'k', lw=3, **kwargs)

    kwargs.pop('label', None)
    ax.plot([1, 1], [0, 1], 'k', lw=3, **kwargs)


def _prepped(alm, ca, yfa):
    assert isinstance(alm, BaseLocationsManager)
    results = alm.import_parsed(ca)

    worked_datasets = list(
        results[(results[s.METRIC_IMPLEMENTATION_CODE] == EUCLIDEAN_VANILLA) & (results[yfa] > 0)][s.DATASET])

    cond_valid = results[s.DATASET].isin(worked_datasets)
    cond_comp = results[s.METRIC_IMPLEMENTATION_CODE] != EUCLIDEAN_VANILLA
    cond_ari = results[yfa] >= 0

    return results[cond_valid & cond_comp & cond_ari]
