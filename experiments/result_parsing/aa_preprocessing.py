import pandas as pd

import strings as s
from common import result_columns, vanilla_columns
from experiments.management.locations_manager import BaseLocationsManager
from experiments.result_parsing._result_parsing_functions import relatives
from metric import EUCLIDEAN_VANILLA


def execute(alm, ca=None):
    """
    Process the results from an experiment. Adds duration, relative versions, and cleans up uncompleted seeds and
    invalid experiments.

    Parameters
    ----------
    alm: AllLocationsManager
        AllLocationsManager to use
    ca: string or None
        Clustering algorithm to use. If None, all are used

    """
    assert isinstance(alm, BaseLocationsManager)

    # If all
    if ca is None:
        for cax in s.all_clustering_algorithms:
            execute(alm, cax)
        return

    # This thing here is mostly making it future proof
    index_columns = [s.DATASET, s.METRIC_IMPLEMENTATION_CODE]
    index_columns_per_mic = [s.DATASET]
    absolute_result_columns = result_columns(ca) + [s.DURATION] + vanilla_columns(alm.real, clustering_algorithm=ca)
    absolute_result_columns.remove(s.STATUS)
    relative_dict = {k: v for k, v in relatives.items() if k in absolute_result_columns}
    relative_result_columns = list(relative_dict.values())

    df_raw = alm.import_raw(ca).reset_index()
    if index_columns_per_mic:
        df_raw = df_raw.set_index(index_columns_per_mic)

    # Take out incomplete stuff
    df_raw = df_raw[df_raw[s.STATUS] == s.COMPLETED].drop(columns=[s.STATUS])

    # Total duration of the clustering part
    df_raw[s.DURATION] = df_raw[s.CLUSTER_DURATION] + df_raw[s.PREPROCESSING_DURATION]

    # Get baseline values
    df_baseline = df_raw[df_raw[s.METRIC_IMPLEMENTATION_CODE] == EUCLIDEAN_VANILLA]. \
        drop(columns=[s.METRIC_IMPLEMENTATION_CODE])

    # Create relative values
    df_rel = pd.DataFrame()
    for mic, df_exp in df_raw.groupby([s.METRIC_IMPLEMENTATION_CODE]):
        df_mic = df_exp.drop(columns=[s.METRIC_IMPLEMENTATION_CODE])

        # Divide values by baseline
        df_mic = df_mic / df_baseline

        # Add the mic back
        df_mic = df_mic.assign(**{s.METRIC_IMPLEMENTATION_CODE: mic})

        # Add to total results
        df_rel = df_rel.append(df_mic)

    # Prepare relative for merging
    df_rel = df_rel.reset_index().set_index(index_columns)
    df_rel = df_rel.rename(columns=relative_dict)[relative_result_columns]

    # Prepare absolute for merging
    df_raw = df_raw.reset_index().set_index(index_columns)[absolute_result_columns]

    df = df_rel.merge(df_raw, left_index=True, right_index=True)

    alm.export_parsed(df, ca)
