import time
from pathlib import Path

import numpy as np
import pandas as pd

import strings as s
from experiments.computation.data_flow import get_data
from experiments.management.locations_manager import SingleDatasetManager
from metric_handler import MetricHandler


def fn(mic):
    return Path(f'{mic}.dm')


def get_4dist_from_dm(dm_mic):
    return np.partition(dm_mic, 4, axis=1)[:, 4]


def save_dm(mic, dm):
    dm.tofile(fn(mic))


def load_dm(mic, slm):
    return np.fromfile(fn(mic), dtype=np.float32).reshape(slm.number_of_datapoints, -1)


def remove_dm(mic):
    fn(mic).unlink()


# Get pure and eps -----------------------------------------------------------------------------------------------------
def execute(slm, silent=False):
    """
    Does the pure computation of the distance metric, and saves values for the 4dist graph

    Parameters
    ----------
    slm: SingleLocationsManager
    silent:bool
        Whether to shut up on out

    Notes
    -----
    Creates two files at slm.fn_pure and slm.fn_4dist. The former contains a DataFrame with the available
    implementations as index and [duration, max_err, max_rel_err, avg_err, avg_rel_err] as columns. The latter contains
    a DataFrame without indices, and with each available implementation as column; the values are all 4-dist (the
    distance to the 4-th nearest neighbour) for each point in the dataset.
    """
    if slm.real:
        _real(slm, silent)
    else:
        _synthetic(slm, silent)


def _real(slm, silent):
    """
    Implementation for the real execution (which saves the 4dist as well)

    Parameters
    ----------
    slm: SingleDatasetManager
        Experiment settings
    silent: bool
        If False, updates will be shown to user
    """
    assert isinstance(slm, SingleDatasetManager)
    if not silent:
        print(f'Computing 4dist for dataset {slm.dataset_name}')
    data = dict()
    for mic in slm.mics:
        if not silent:
            print(f'\t{mic} ... ', end='', flush=True)

        # Prep
        original_data, _ = get_data(slm.dataset_name)
        clusterable = MetricHandler(mic).clusterable_from_raw(original_data)
        dist4 = np.empty(slm.number_of_datapoints, dtype=np.float32)
        for i in range(0, slm.number_of_datapoints, 1_000):
            min_idx = i
            max_idx = min(i + 1_000, slm.number_of_datapoints)
            dm = clusterable.get_sub_distance_matrix(range(min_idx, max_idx), None)
            dist4[min_idx:max_idx] = np.partition(dm, 4, axis=1)[:, 4]

        if not silent:
            print(f'\tDone')

        data[mic] = dist4

    # TODO add average and 4dist computations

    slm.export_4dist(pd.DataFrame(data))


def _synthetic(slm: SingleDatasetManager, silent: bool):
    """
    Implementation for the synthetic execution (which saves the 4dist as well)

    Parameters
    ----------
    slm: SingleDatasetManager
        Experiment settings
    silent: bool
        If False, updates will be shown to user
    """

    # Compute distance matrices and record time ========================================================================
    # The computation is slow but easy for large datasets
    n_datapoints = slm.number_of_datapoints
    if n_datapoints > 10_000:
        raise NotImplementedError('More than 10K datapoints pure is currently not efficiently implemented')

    # Storage
    all_implementations_index = pd.Index(slm.mics, name=s.METRIC_IMPLEMENTATION_CODE)
    vanilla_implementations = MetricHandler.vanilla_implementations(slm.metric)
    time_results = pd.DataFrame(index=all_implementations_index, columns=[s.PREPROCESSING_DURATION, s.DURATION])

    # Computation
    for mic_compute in slm.mics:
        if not silent:
            print(f'\r[{slm.dataset_name}] Computing distance matrix {mic_compute}' + ' ' * 10, end='', flush=True)
        res = _get_durations_and_create_dm_file(slm, mic_compute)
        for k, v in res.items():
            time_results.loc[mic_compute, k] = v

    # Export
    slm.export_time(time_results)

    # Compute average distance/4dist ===================================================================================
    # Load vanilla
    vanilla_dms = {k: load_dm(k, slm) for k in vanilla_implementations}

    # Storage
    index = pd.Index(data=vanilla_dms.keys(), name=s.METRIC_IMPLEMENTATION_CODE)
    average_distance_results = pd.Series(index=index, name=s.AVG_DISTANCE, dtype=float)
    average_4dist_results = pd.Series(index=index, name=s.AVG_4DIST, dtype=float)

    # Computing
    for mic_vanilla, dm_vanilla in vanilla_dms.items():
        average_distance_results.loc[mic_vanilla] = np.mean(dm_vanilla)
        average_4dist_results.loc[mic_vanilla] = np.mean(get_4dist_from_dm(dm_vanilla))

    # Exporting
    slm.export_avg_vanilla_distance(average_distance_results)
    slm.export_avg_vanilla_4dist(average_4dist_results)

    # Errors ===========================================================================================================
    # Storage
    error_columns = [s.MAX_ERROR, s.MEAN_ERROR, s.MAX_RELATIVE_ERROR, s.MEAN_RELATIVE_ERROR]
    error_results = {k: pd.DataFrame(index=all_implementations_index, columns=error_columns)
                     for k in vanilla_implementations}

    # Computation
    # For each metric implementation compute the errors
    for mic_compute in all_implementations_index:
        dm_compute = load_dm(mic_compute, slm)

        # Distances of BitBooster are scaled with (2 ** n_bits - 1), instead of normalized
        if MetricHandler(mic_compute).is_bitbooster:
            dm_compute *= (1 / (2 ** MetricHandler(mic_compute).n_bits - 1))

        # Compute errors with respect to each vanilla implementation
        for mic_vanilla, dm_vanilla in vanilla_dms.items():

            # Skip if the exact same implementation (so EVAN is checked with MVAN, but not with EVAN)
            if mic_compute == mic_vanilla:
                continue

            # Print to user if necessary
            if not silent:
                print(f'\r[{slm.dataset_name}] Computing errors {mic_compute} from {mic_vanilla}' + ' ' * 20, end='',
                      flush=True)

            # Get and save the errors
            res = _get_errors(dm_compute=dm_compute, dm_vanilla=dm_vanilla)
            for k, v in res.items():
                error_results[mic_vanilla].loc[mic_compute, k] = v

        # We are done with this mic's dm, so we can remove the file
        remove_dm(mic_compute)

    # Show user we are done with this dataset
    if not silent:
        print(f'\r[{slm.dataset_name}] Done' + ' ' * 50)

    # Export
    for vanilla_metric, df in error_results.items():
        slm.export_error(df, MetricHandler(vanilla_metric).base_metric)


def _get_durations_and_create_dm_file(slm, mic):
    """
    Create a file with the distance matrix, and return the duration

    Parameters
    ----------
    slm: SingleDatasetManager
        Experiment settings
    mic: str
        Metric Implementation name

    Returns
    -------
    t: dict [str -> float]
        Dict with results:
        s.PREPROCESSING_DURATION -> time per datapoint for preparation
        s.DURATION -> time per computation of the distance between two datapoints
    """
    # Prep
    original_data, _ = get_data(slm.dataset_name)
    clusterable, prep_time = MetricHandler(mic).clusterable_from_raw(original_data, return_prep_time=True)
    t0 = time.process_time()
    dm = clusterable.get_sub_distance_matrix(None, None)
    compute_time = time.process_time() - t0

    save_dm(mic, dm)

    # I am not sure if this is needed; but this at least makes sure the memory should not be reserved anymore
    del clusterable
    del dm

    ndp = slm.number_of_datapoints

    res = {
        s.PREPROCESSING_DURATION: prep_time / ndp,
        s.DURATION: compute_time / ndp / ndp,
    }
    return res


def _get_errors(dm_compute, dm_vanilla):
    """
    Computes the errors and 4dist values

    Parameters
    ----------
    dm_compute: np.array
        The distance matrix for metric that is to be computed
    dm_vanilla: np.array
        The distance matrix for the vanilla implementation of that metric

    Returns
    -------
    res: dict[str -> float]
        Dictionary with the results:
        s.MAX_ERROR: maximum error over the entire dm
        s.MAX_RELATIVE_ERROR: maximum relative error over the entire error, skipping values for dm_vanilla=0
        s.MEAN_ERROR: mean error over the entire dm
        s.MEAN_RELATIVE_ERROR: mean relative error over the entire error, skipping values for dm_vanilla=0
    """
    non_zeros = np.where(dm_vanilla != 0)

    dm_err = np.abs(dm_compute - dm_vanilla)
    dm_rel_err = dm_err[non_zeros] / dm_vanilla[non_zeros]

    max_err = dm_err.max()
    max_rel_err = dm_rel_err.max()
    mean_err = dm_err.mean()
    mean_rel_err = dm_rel_err.mean()

    res = {
        s.MAX_ERROR: max_err,
        s.MAX_RELATIVE_ERROR: max_rel_err,
        s.MEAN_ERROR: mean_err,
        s.MEAN_RELATIVE_ERROR: mean_rel_err
    }

    return res
