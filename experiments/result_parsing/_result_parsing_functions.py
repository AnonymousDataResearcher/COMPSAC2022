import functools

import pandas as pd

import strings as s
from bitbooster.common import as_list
from data.summary import get_summary

relatives = {s.DURATION: s.RELATIVE_DURATION,
             s.CLUSTER_DURATION: s.RELATIVE_CLUSTER_DURATION,
             s.ARI: s.RELATIVE_ARI,
             s.ITERATION_COUNT: s.RELATIVE_ITERATION_COUNT,
             s.NOISE: s.RELATIVE_NOISE,
             s.PURITY_WITH_NOISE: s.RELATIVE_PURITY_WITH_NOISE,
             s.PURITY_WITHOUT_NOISE: s.RELATIVE_PURITY_WITHOUT_NOISE,
             s.NUMBER_OF_CORE_POINTS: s.RELATIVE_NUMBER_OF_CORE_POINTS,
             s.PREPROCESSING_DURATION: s.RELATIVE_PREPROCESSING_DURATION}
absolute_dependent_features = list(relatives.keys()) + [s.PREPROCESSING_DURATION]
relative_dependent_features = list(relatives.values())


def filter_df(df, constant_features=None, separate_features=False):
    """

    Filter a dataset.

    Parameters
    ----------
    df: pd.DataFrame
        source data to be filtered
    constant_features: dict or None
        Input values to be kept constant at a specified value. All experiments not matching these values are
        filtered out before plotting.
    separate_features: (Iterable of) str or Bool
        Feature(s) that are plotted separately (i.e. a separate point for each value of the features). If True, all
        features that are not constant_features are plotted separately (i.e. 1 plot = 1 experiment with given
        constant_features). If False, no feature is plotted separately (i.e. 1 point = 1 metric_code implementation).

    Returns
    -------
    filtered_df: pd.DataFrame
        Filtered DataFrame. Only rows with values specified by constant_features are kept. Only columns that are
        either in separate_features or in RELATIVE_ARI, RELATIVE_DURATION, ALGORITHM are kept.
    """
    assert isinstance(df, pd.DataFrame)
    # CHECK CONSTANT FEATURES
    if constant_features is None:
        constant_features = dict()
    else:
        assert isinstance(constant_features, dict)
    assert set(constant_features.keys()).issubset(s.dataset_features)

    # CHECK SEPARATE FEATURES
    if separate_features is True:
        separate_features = s.dataset_features
    elif separate_features is False:
        separate_features = []
    else:
        separate_features = as_list(separate_features, str)
    assert set(separate_features).issubset(s.dataset_features)

    # SEPARATE CANNOT BE CONSTANT
    assert len(set(separate_features).intersection(constant_features.keys())) == 0

    dfx = df.copy()

    # FILTER DATAFRAME based on constant_features
    for k, v in constant_features.items():
        dfx = dfx[dfx[k] == v]

    # GET PLOT POINTS
    averaged_features = relative_dependent_features + absolute_dependent_features
    averaged_features = [f for f in averaged_features if f in dfx.columns]

    dfx = dfx.groupby([s.METRIC_IMPLEMENTATION_CODE] + separate_features)[averaged_features].mean()
    return dfx.reset_index().dropna(axis=0)


def marker_style(dataset_name):
    if isinstance(dataset_name, str):
        if dataset_name.startswith(s.BLOB):
            return 'o'
        else:
            return f'${get_summary().loc[dataset_name, s.ABBREVIATION]}$'
    elif isinstance(dataset_name, pd.Series):
        all_datasets = dataset_name.unique()
        d = {t: marker_style(t) for t in all_datasets}
        return dataset_name.replace(d)
    raise ValueError(f'Unknown Type: {type(dataset_name)}')


def determine_pareto(df: pd.DataFrame, high_feature, low_feature, row=None):
    """
    Determine whether the given row is pareto optimal. If no row is given, a Series is returned instead, with
    the pareto optimality determined for every row.

    Parameters
    ----------
    df: pd.DataFrame
        All results
    high_feature: str
        The feature that should be higher
    low_feature: str
        The feature that should be lower
    row : pd.Series or None
        Row to determine pareto optimality fow. If None, all rows are computed and returned as Series

    Returns
    -------
    ret: pd.Series or bool
        Whether the given row is pareto optimal wrt to the dataframe. If no row is given, the pareto optimality of each
        row in the df is returned in a Series.

    """
    if row is None:
        return df.apply(lambda r: determine_pareto(df=df, high_feature=high_feature, low_feature=low_feature, row=r),
                        axis=1)

    c1q = (df[high_feature] > row[high_feature])
    c1t = (df[low_feature] <= row[low_feature])
    c2q = (df[high_feature] >= row[high_feature])
    c2t = (df[low_feature] < row[low_feature])
    return not ((c1q & c1t) | (c2q & c2t)).any()
