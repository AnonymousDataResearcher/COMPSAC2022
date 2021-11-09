import itertools
import subprocess
import warnings

import numpy as np
import pandas as pd

import strings as s
from bitbooster.common import as_list
from common import human_names, is_vanilla_column
from data.summary import get_summary, add_properties
from experiments.management.locations_manager import BaseLocationsManager
from experiments.result_parsing._result_parsing_functions import determine_pareto
from metric import Metric

bold_dominant = True


def highlight_condition(df_dsn, quality_feature, duration_feature, row):
    """
    Determines whether the given row should be highlighted or not, given the df.

    Parameters
    ----------
    df_dsn: pd.DataFrame
        All results
    quality_feature: str
        Which column to use for the quality feature (higher=better)
    duration_feature
        Which column to use for the time feature (higher=better)
    row
        The row which to use

    Returns
    -------
    highlight: bool
        Whether to highlight the given row.
    """
    if is_vanilla_column(quality_feature) and Metric(row.name).is_vanilla:
        return False

    if bold_dominant:
        return determine_pareto(df_dsn, quality_feature, duration_feature, row)
    else:
        ratio = row[quality_feature] / row[duration_feature]
        return ((df_dsn[quality_feature] / df_dsn[duration_feature]) <= ratio).all()


def extended_time_format(df_dsn, row, duration_feature):
    if min(df_dsn[duration_feature]) < 60:
        return f'{(row[duration_feature]):.2f}'
    elif min(df_dsn[duration_feature]) < 3600:
        return f'{(row[duration_feature] / 60):.0f}m'
    else:
        return f'{(row[duration_feature] / 3600):.0f}h'


def dataset_translator(t, max_letters=None):
    z = ' '.join([ti.capitalize() for ti in t.split('_')])
    if t == 'eeg_eye_state':
        z = 'EEG' + z[3:]
    z = z.replace('Localization', 'Loc.').replace('Left', 'L').replace('Right', 'R')
    if max_letters is None:
        return z
    else:
        if len(z) > max_letters:
            short = ''.join([word[0].capitalize() for word in z.split(' ')])
            return [short, z]
        else:
            return [z, None]


def score_type_translator(t):
    if t == s.ARI:
        return 'ARI'
    if t == s.PURITY_WITHOUT_NOISE:
        return 'Purity Excluding Noise'
    if t == s.PURITY_WITH_NOISE:
        return 'Purity Including Noise'

    if t == s.DURATION:
        return 'Total Duration'
    if t == s.CLUSTER_DURATION:
        return 'Clustering Duration'
    if t == s.PREPROCESSING_DURATION:
        return 'Preprocessing Duration'

    if t == s.RATIO:
        return 'Ratio'

    raise NotImplementedError()


def time_format(v):
    if pd.isna(v):
        return '-'
    if v < 60:
        if str(v)[0] == '0':
            return f'{v:.2f}'[1:]
        else:
            return f'{v:.2f}'
    else:
        v /= 60
        return f'{v:.0f}m'


def float_format(v):
    if pd.isna(v):
        return '-'
    else:
        return f'{v:.3f}'


def execute(alm: BaseLocationsManager, ca: [str, None] = None, duration_feature: [str, None] = None,
            quality_feature: [str, None] = None):
    """
    Creates a tex file from the combined experiments file.

    Parameters
    ----------
    alm: BaseLocationsManager
        Locations manager for the experiment
    ca: str or None
        The clustering algorithm for which to compute the table. If None, all clustering algorithms are used
    duration_feature: str or None
        The duration feature to use. If None, all duration features are used
    quality_feature: str or None
        The quality feature to use. If None, all quality features are used

    """

    # Replace None values
    if ca is None:
        ca = s.all_clustering_algorithms
    if quality_feature is None:
        quality_feature = [s.ARI, s.PURITY_WITHOUT_NOISE, s.PURITY_WITH_NOISE]
    if duration_feature is None:
        duration_feature = [s.DURATION, s.CLUSTER_DURATION, s.PREPROCESSING_DURATION]

    # For each clustering algorithm / quality feature / duration feature, make a tex file
    for c, q, d in itertools.product(as_list(ca, str), as_list(quality_feature, str), as_list(duration_feature, str)):
        # Compute the tex string
        latex_string = do_single_alt(alm, c, q, d)

        # Get and create save location
        fn = alm.fd_tex(ca=ca) / f'{quality_feature}-{duration_feature}.tex'
        fn.parent.mkdir(parents=True, exist_ok=True)

        # Save
        with open(fn, 'w+') as wf:
            wf.write('\n'.join(latex_string))


def do_single_alt(alm, ca, quality_feature, duration_feature, add_ratio=False, max_letters=20, to_clip=False,
                  wrap_as_table=True, used_mics=None, sparsity_mode=s.EPS, sparsity_metrics_shown=None,
                  skip_dsn=None):
    # TODO. In hindsight, this has become a big mess of code, but still runs correctly. What would be better is to
    #  index the result per dataset, add columns (info, dsn_abbreviation) and (info, dsn_properties), and then at the
    #  end decide whether things are on separate rows.
    """
    Produces a tex table from the results. For each dataset, the metric with the best ratio between quality and duration
    is highlighted in bold.

    Parameters
    ----------
    alm: BaseLocationsManager
        LocationsManager to retrieve results
    ca: str
        Clustering algorithm for which to get the results
    quality_feature: str
        Feature that defines the quality of the result
    duration_feature: str
        Feature that defines the duration of the result
    add_ratio: bool
        If True, the ratio between the quality_feature and duration_feature is reported instead of the quality_feature
    max_letters: int
        Maximum number of letters used in the dataset names. If dataset names are longer, abbreviations are used, and
        reported as part of the caption.
    to_clip: bool
        If True, also writes the return value to the clip board
    wrap_as_table: bool
        If True, the result is a tex document string that can be used as table. If False, only the tabular part is
        returned.
    used_mics: (Iterable of) str or None
        Which metric implementation(s) to present. If None, all implementations of the metric of alm are used.
    sparsity_mode: str
        Whether to show the sparsity using the eps value ('eps'), the average 4dist ('4dist'), or not (None)
    sparsity_metrics_shown: (Iterable of) str or None
        Which sparsity value(s) to show. If None, the metric(s) of alm are all shown
    skip_dsn: (Iterable of) str or None
        Which datasets are skipped. If None, all datasets > 10000 are used.

    Returns
    -------
    tex: str or None
        A string with the entire table (if wrap_as_table is True), or with the tabular (if wrap_as_table). None if
        to_clip is True, in which case the string is copied to the clip board.
    """
    single_row_per_dataset = True

    # There is a point with a high number of index columns. Pandas complains about this, but this table is relatively
    # short, this ignores the warning
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

    # Importing ========================================================================================================
    results = alm.import_parsed(ca)
    results = add_properties(results)
    status = alm.import_raw(ca).set_index([s.DATASET, s.METRIC_IMPLEMENTATION_CODE])[s.STATUS]

    # Filtering ========================================================================================================
    # Only 10K+ datasets
    if skip_dsn is None:
        skip_dsn = ['iris', 'vicon1']
    skip_dsn = as_list(skip_dsn, str)

    dataset_condition = ~results[s.DATASET].isin(skip_dsn)

    # Filter out results for those datasets
    results = results[dataset_condition].sort_values(s.NUMBER_OF_DATAPOINTS)

    # Which implementations to report in the table =====================================================================
    if used_mics is None:
        used_mics = Metric.metric_implementations(alm.metric)
    used_mics = as_list(used_mics, str)

    used_mcs = set(map(lambda x: Metric(x).base_metric, used_mics))

    if sparsity_metrics_shown is None:
        sparsity_metrics_shown = alm.metric
    sparsity_metrics_shown = as_list(sparsity_metrics_shown, str)

    mc_mics = [(Metric(mic).base_metric_capitalized, Metric(mic).short_tex) for mic in used_mics]

    name = 'name'
    info_col = 'info_col'

    if single_row_per_dataset:
        idx = pd.Index(data=[], name=s.DATASET)
    else:
        idx = pd.MultiIndex.from_product([[], []], names=[s.DATASET, 'foo'])

    df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(mc_mics), index=idx)

    # df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(mc_mics),
    #                   index=pd.MultiIndex.from_product([results[s.DATASET].unique(), [name, info_col]],
    #                                                    names=[s.DATASET, 'foo']))

    reasons = set()

    # Iterate over all datasets and implementations ====================================================================
    for dsn, df_dsn in results.groupby(s.DATASET, sort=False):
        df_dsn = df_dsn.set_index(s.METRIC_IMPLEMENTATION_CODE)
        for mic in used_mics:

            # Tex string for mic
            mic_tex = Metric(mic).short_tex
            mc = Metric(mic).base_metric
            mc_tex = Metric(mic).base_metric_capitalized

            # Invalid result
            if mic not in df_dsn.index:
                quality_value, time_value, ratio_value, quality_ratio_value = \
                    status.loc[dsn, mic], '-', '-', status.loc[dsn, mic]
                reasons.add(quality_value)
            else:

                # Results for this dsn-mic
                row = df_dsn.loc[mic]

                # Result values
                quality_value = f'{row[quality_feature]:.2f}'
                if quality_value.startswith('0.'):
                    quality_value = quality_value[1:]
                if quality_value.startswith('-0.'):
                    quality_value = '-' + quality_value[2:]
                time_value = time_format(row[duration_feature])
                ratio_value = f'{row[quality_feature] / row[duration_feature]:.2f}'
                quality_ratio_value = f'{ratio_value if add_ratio else quality_value}/{time_value}'

                # Boldface result values if they are the best
                for mics_list, highlight in zip([Metric.metric_implementations(mc)],
                                                ['textbf']):
                    # for mics_list, highlight in zip([used_mics, Metric.metric_implementations(mc)],
                    #                                 ['textbf', 'underline']):

                    if highlight_condition(df_dsn.reindex(mics_list).dropna(axis=0, how='all'),
                                           quality_feature, duration_feature, row):
                        quality_value = rf'\{highlight}{{{quality_value}}}'
                        time_value = rf'\{highlight}{{{time_value}}}'
                        ratio_value = rf'\{highlight}{{{ratio_value}}}'
                        quality_ratio_value = rf'\{highlight}{{{quality_ratio_value}}}'

            if single_row_per_dataset:
                df.loc[dsn, (mc_tex, mic_tex)] = quality_ratio_value
            else:
                # Select result value to use
                df.loc[(dsn, name), (mc_tex, mic_tex)] = ratio_value if add_ratio else quality_value
                df.loc[(dsn, info_col), (mc_tex, mic_tex)] = time_value

    # Fix datasets
    df = df.reset_index()
    sparsity_srs = [alm.sparsity_sr(sparsity_mode, m) for m in sparsity_metrics_shown]

    # Caption abbreviations
    if single_row_per_dataset:
        abbreviations = {dataset_translator(dsn, max_letters)[0]: dataset_translator(dsn, max_letters)[1] for dsn in
                         df[(s.DATASET,)].values}
    else:
        abbreviations = {dataset_translator(dsn, max_letters)[0]: dataset_translator(dsn, max_letters)[1] for dsn in
                         df.loc[df[('foo',)] == name, (s.DATASET,)].values}
    abb_str = ', '.join([f'{k}={v}' for k, v in abbreviations.items() if v is not None] + ['Loc. = Localization']) + '.'

    # Get string with description of the dataset
    def info(ds_name):
        ds_info = get_summary().loc[ds_name]

        def fmt(z, bold_max=-1):
            if z < bold_max:
                return rf'\underline{{{z:.3f}}}'.replace('0.', '.')
            else:
                return f'{z:.3f}'.replace('0.', '.')

        sparsity_string = '/'.join([fmt(sparsity_sr.loc[ds_name]) for sparsity_sr in sparsity_srs]) + '$'

        return rf'${ds_info[s.NUMBER_OF_DATAPOINTS] / 1_000:.0f}$K' + \
               f'$/{ds_info[s.NUMBER_OF_FEATURES]}/{ds_info[s.NUMBER_OF_CLASSES]}/' + sparsity_string

    def format_and_mark_as_dense(dataset_name):
        translated_dataset_name = f'{dataset_translator(dataset_name, max_letters)[0]}'
        return translated_dataset_name

    if single_row_per_dataset:
        def _info(ds_name):
            return f'{format_and_mark_as_dense(ds_name)} ({info(ds_name)})'

        df.loc[:, (s.DATASET,)] = df.loc[:, (s.DATASET,)].apply(_info)
    else:
        def _info(ds_name):
            return rf'$\quad {info(ds_name)[1:]}'

        # Convert the info values
        df.loc[df[('foo',)] == info_col, (s.DATASET,)] = df.loc[df[('foo',)] == info_col, (s.DATASET,)].apply(_info)

        # Convert the dataset name values
        df.loc[df[('foo',)] == name, (s.DATASET,)] = \
            df.loc[df[('foo',)] == name, (s.DATASET,)].apply(format_and_mark_as_dense)
        df = df.drop(columns=('foo', ''))

    # Make-up table ====================================================================================================

    # Sort the dataframe such that all mics of the same metric are together, but order is not changed otherwise
    df = df[sorted(df.columns, key=lambda z: z[0])]

    # Put dsn + descriptions on the front
    df = df.set_index((s.DATASET,)).reset_index()

    sparsity_header_string = '/'.join([f's_{{{x[0].capitalize()}}}' for x in sparsity_metrics_shown])

    if sparsity_header_string != '':
        sparsity_header_string = '/' + sparsity_header_string

    if single_row_per_dataset:
        info_string = fr'$|D|/|F|/k{sparsity_header_string}$'
    else:
        info_string = fr'$\quad|D|/|F|/k{sparsity_header_string}$'

    if single_row_per_dataset and len(used_mcs) == 1:
        def f__(column_name):
            if column_name == (s.DATASET, ''):
                return s.DATASET
            return column_name[1]

        df.columns = map(f__, df.columns)

    # Capitalize the dataset name
    if single_row_per_dataset:
        df = df.rename(columns={s.DATASET: f'{s.DATASET.capitalize()} ({info_string})'})
    else:
        df = df.rename(columns={s.DATASET: s.DATASET.capitalize()})

    # Translate results dataframe to tex table, and apply some formatting
    foo = df.to_latex(index=False, escape=False,
                      column_format=('r' if single_row_per_dataset else 'l') + 'r' * len(used_mics)) \
        .replace(r'l}', r'c}').split('\n')

    # Horizontal split lines between metric and quality/duration feature
    cum_sum = 2
    x = ''
    for _, i in zip(*np.unique([Metric(mic).base_metric_capitalized for mic in used_mics], return_counts=True)):
        x += rf'\cmidrule(lr){{{cum_sum}-{cum_sum + i - 1}}}'
        cum_sum += i

    if single_row_per_dataset:
        pass
    else:
        foo = foo[:3] + [x] + [info_string + foo[3]] + [f'\\cmidrule(lr){{1-1}}{x}'] + \
              sum([foo[2 * i + 5:2 * i + 7] + ['\\cmidrule(lr){1-1}'] for i in range(len(results[s.DATASET].unique()))],
                  [])[:-1] + foo[-3:]

    def map_reason(reason):
        if reason == s.IMPOSSIBLE_PARAMETER_COMBINATION:
            return f'{s.IMPOSSIBLE_PARAMETER_COMBINATION} = ' \
                   f'Dataset characteristics make it impossible for the implementation to find $k$ clusters'
        raise NotImplementedError(f'not implemented for {reason}')

    reasons_str = ', '.join(map(map_reason, sorted(reasons)))

    res_str = ' '.join([abb_str, reasons_str]).strip()

    if bold_dominant:
        boldface_explanation = f'The approximations that are pareto optimal are highlighted in bold.'
    else:
        boldface_explanation = f'The approximation with the highest ratio ' \
                               f'between {human_names(quality_feature).lower()} ' \
                               f'(top) and clustering duration ' \
                               f'(bottom) is highlighted in bold.'

    # Put a table wrap around the tabular
    if wrap_as_table:
        foo = ['\\begin{table*}',
               '\\centering',
               '\\caption{Results on the real datasets. '
               + boldface_explanation +
               ' Reported durations are in seconds, '
               'unless stated otherwise. Dataset properties are presented below the dataset name. '
               f'{res_str}.}}',
               '\\label{tab:results:real}'] + foo + ['\\end{table*}']

    # Convert the list to string
    foo = '\n'.join(foo)

    if to_clip:
        # Add result to clipboard
        subprocess.run("clip", universal_newlines=True, input=foo)
        print('Table copied to clipboard')

    warnings.resetwarnings()
    return foo
