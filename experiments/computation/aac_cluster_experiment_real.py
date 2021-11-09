import time
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics

import strings as s
from bitbooster.abstract.errors import NotEnoughUniqueDatapointsException, KMedoidsNotEnoughLabelsException
from common import valid_parameter_combination, result_columns, purity, vanilla_ize, evaluation_metrics
from competitors.aqbc import AQBCNegativeError
from competitors.jkkc import JKKCZeroVectorError
from data import summary
from experiments.computation.data_flow import get_data, generate_synthetic_data
from experiments.management.locations_manager import SingleDatasetManager
# Execution ============================================================================================================
from metric import Metric
from metric_handler import MetricHandler


# This executes a single dataset, for each metric implementation in each metric in the SLM
def execute(slm, clustering_algorithm, warm_up=True):
    """
    Run experiment for given SingleLocationManager (Metric + dataset)

    Parameters
    ----------
    slm: SingleLocationsManager
        SingleLocationsManager for access to parameters and locations
    clustering_algorithm: str
        Clustering Algorithm to use. Must be one of s.all_clustering_algorithms
    warm_up: bool
        Whether to compile the njit in advance
    """
    assert isinstance(slm, SingleDatasetManager)
    assert clustering_algorithm in s.all_clustering_algorithms

    # WARM UP ----------------------------------------------------------------------------------------------------------
    optional_heat_up(slm.mics, clustering_algorithm, warm_up)

    # RESULT DATAFRAME GENERATION / LOADING ----------------------------------------------------------------------------
    # Get previous results
    if slm.done_raw(clustering_algorithm):
        df = slm.import_raw(clustering_algorithm)
    else:
        df = pd.DataFrame(index=Metric.metric_implementations(slm.metric),
                          columns=result_columns(clustering_algorithm))
        df.index.name = s.METRIC_IMPLEMENTATION_CODE

    df.reindex(slm.mics)

    # Set all not filled flags to not completed
    df[s.STATUS] = df[s.STATUS].fillna(s.NOT_COMPLETED)

    # Track labels of the vanilla implementation
    vanilla_labels = dict()

    # ACTUAL EXECUTION -------------------------------------------------------------------------------------------------
    for mic, r in df.iterrows():
        if r[s.STATUS] == s.NOT_COMPLETED:
            # Compute results
            print(f'Computing results for dataset={slm.dataset_name}, mic={mic}, algorithm={clustering_algorithm}')
            # noinspection PyTypeChecker
            res = _run_single(slm, clustering_algorithm, mic)

            df.loc[mic, s.STATUS] = res[s.STATUS]

            if res[s.STATUS] == s.COMPLETED:

                if Metric(mic).is_vanilla:
                    vanilla_labels[Metric(mic).base_metric] = res[s.PREDICTED]
                    for k in evaluation_metrics.keys():
                        # Vanilla labels have perfect vanilla scores
                        res[vanilla_ize(k)] = 1.0
                else:
                    metric = Metric(mic).base_metric
                    if metric not in vanilla_labels:
                        vanilla_mic = Metric.vanilla_implementations(metric)
                        if slm.real:
                            # Real labels are saved, and if the vanilla metric is not in vanilla labels,
                            # it is done before
                            # TODO: this is currently an implementation detail, as the vanilla metrics are
                            #  the first in the list in metric. You might want to think about fixing/asserting this
                            #  somehow
                            vanilla_labels[metric] = slm.import_labels(clustering_algorithm, vanilla_mic)
                        else:
                            # Synthetic labels are not saved, so recompute
                            vanilla_labels[metric] = _run_single(slm, clustering_algorithm, vanilla_mic)

                    labels_true = vanilla_labels[metric]
                    labels_pred = res[s.PREDICTED]
                    for k, v in evaluation_metrics.items():
                        res[vanilla_ize(k)] = v(labels_true, labels_pred)

                for rc in result_columns(clustering_algorithm):
                    df.loc[mic, rc] = res.get(rc, pd.NA)

                if slm.real and r[s.STATUS] == s.COMPLETED:
                    slm.export_labels(res[s.PREDICTED], clustering_algorithm, mic)

            # Report results on screen
            print_results(df.loc[mic])

            # Save results
            slm.export_raw(df=df, ca=clustering_algorithm)


# This executes a single dataset and a single metric implementation
def _run_single(slm, clustering_algorithm, mic):
    """
    Runs the experiment for a single implementation of a metric

    Parameters
    ----------
    slm: SingleDatasetManager
        SingleDatasetManager containing info on the dataset
    clustering_algorithm: str
        Which clustering algorithm to use
    mic: str
        Which metric implementation to use

    Returns
    -------
    res: dict
        The results of this run
    """
    assert isinstance(slm, SingleDatasetManager)
    props = summary.get_properties(slm.dataset_name)

    # Verification
    if not valid_parameter_combination(n_features=props[s.NUMBER_OF_FEATURES],
                                       mic=mic,
                                       n_datapoints=props[s.NUMBER_OF_DATAPOINTS],
                                       n_clusters=int(props[s.NUMBER_OF_CLASSES]),
                                       clustering_algorithm=clustering_algorithm):
        return {s.STATUS: s.IMPOSSIBLE_PARAMETER_COMBINATION}

    # Retrieve the data
    original_data, true_labels = get_data(slm.dataset_name)

    # Retrieve additional parameters
    if clustering_algorithm == s.VORONOI:
        kwargs = dict()
    elif clustering_algorithm == s.DBSCAN:
        kwargs = {s.MIN_POINTS: 4, s.EPS: slm.import_eps().loc[mic]}
    else:
        raise NotImplementedError(f'Functionality not implemented for clustering algorithm {clustering_algorithm}')

    # Do the actual experiment
    res = from_original_data(original_data=original_data,
                             true_labels=true_labels,
                             clustering_algorithm=clustering_algorithm,
                             metric_implementation_code=mic,
                             number_of_clusters=int(props[s.NUMBER_OF_CLASSES]),
                             **kwargs)

    if clustering_algorithm == s.VORONOI:
        pass
    elif clustering_algorithm == s.DBSCAN:
        four_dist_values = slm.import_4dist()[mic]
        res[s.NUMBER_OF_CORE_POINTS] = (four_dist_values < kwargs[s.EPS]).sum()
    else:
        raise NotImplementedError(f'Post clustering not implemented for clustering algorithm {clustering_algorithm}')

    return res


# This is the actual clustering from a (numeric) dataset
def from_original_data(original_data, true_labels, metric_implementation_code,
                       number_of_clusters, clustering_algorithm, **kwargs):
    # PREPROCESSING ----------------------------------------------------------------------------------------------------
    try:
        data_object, prep_time = MetricHandler(metric_implementation_code).clusterable_from_raw(original_data,
                                                                                                return_prep_time=True)
        status = s.NOT_COMPLETED
    except JKKCZeroVectorError:
        status = s.JKKC_ZERO_VECTOR
        prep_time = None
        data_object = None
    except AQBCNegativeError:
        status = s.AQBC_NEGATIVE_ERROR
        prep_time = None
        data_object = None

    if status == s.JKKC_ZERO_VECTOR or status == s.AQBC_NEGATIVE_ERROR:
        return {s.STATUS: status}
    elif status == s.NOT_COMPLETED:
        pass
    else:
        raise NotImplementedError(f'Not implemented for status {status}')

    # CLUSTERING -------------------------------------------------------------------------------------------------------
    res = {s.PREPROCESSING_DURATION: prep_time}

    if clustering_algorithm == s.VORONOI:
        t_start = time.process_time()
        try:
            _, pred_labels, res[s.ITERATION_COUNT] = \
                data_object.cluster(number_of_clusters, return_iteration_count=True, **kwargs)
        except NotEnoughUniqueDatapointsException:
            return {s.STATUS: s.UNLUCKY_SEED}
        except KMedoidsNotEnoughLabelsException:
            return {s.STATUS: s.LLOYD_FAIL}
        t_end = time.process_time()
    elif clustering_algorithm == s.DBSCAN:
        t_start = time.process_time()
        pred_labels, stats = data_object.dbscan_cluster(return_stats=True, **kwargs)
        t_end = time.process_time()
        res[s.NOISE] = np.mean(pred_labels == s.NOISE_INT)
        res[s.NUMBER_OF_FOUND_CLUSTERS] = len(set(pred_labels)) - (s.NOISE_INT in pred_labels)
        res.update(stats)
    else:
        raise NotImplementedError(f'Not implemented for clustering algorithm {clustering_algorithm}')

    # RETURNING -------------------------------------------------------------------------------------------------------
    res[s.ARI] = metrics.adjusted_rand_score(labels_true=true_labels, labels_pred=pred_labels)
    res[s.CLUSTER_DURATION] = t_end - t_start
    res[s.STATUS] = s.COMPLETED
    res[s.PREDICTED] = pred_labels
    res[s.PURITY_WITH_NOISE], res[s.PURITY_WITHOUT_NOISE] = purity(true_labels, pred_labels)
    return res


# Helper ===============================================================================================================
# This heats up the jit-compiled engines
def optional_heat_up(mics, clustering_algorithm, warm_up):
    """
    'warms up' all experiments. Some of the experiments use jit-compiled functions, so having them run once is only
    fair

    Parameters
    ----------
    mics: (iterable of) str
        Metric implementations for which to heat up
    clustering_algorithm: str
        Which clustering algorithm to use for the warm-up (this is jit-compiled separately)
    warm_up: bool
        Whether to actually do this heat up, or raise a warning instead

    Notes
    -----
    If warm_up is False, a warning is shown to the user

    """
    if clustering_algorithm == s.VORONOI:
        kwargs = dict()
    elif clustering_algorithm == s.DBSCAN:
        # values don't really matter
        kwargs = {s.EPS: 0.05, s.MIN_POINTS: 4}
    else:
        raise NotImplementedError(f'Not implemented for clustering_algorithm = {clustering_algorithm}')

    if warm_up:
        for mic in mics:
            # Generate the data
            original_data, true_labels = generate_synthetic_data(f'{s.BLOB}_100_2_2_0')
            # Do the actual experiment
            from_original_data(original_data=original_data,
                               true_labels=true_labels,
                               metric_implementation_code=mic,
                               number_of_clusters=2,
                               clustering_algorithm=clustering_algorithm,
                               **kwargs)
    else:
        warnings.warn('Warning: code is not JIT-compiled before experiments.'
                      '\nYou may get unfair or unexpected results.'
                      '\nRecommended that this is only used during debugging.')


# This prints intermediate results
def print_results(sr_res):
    """
    Prints the results on screen

    Parameters
    ----------
    sr_res: pd.Series
        For the info on what to print

    """

    if sr_res.loc[s.STATUS] == s.COMPLETED:
        msg = f'\t{sr_res.loc[s.ARI]:.3f} in {sr_res.loc[s.PREPROCESSING_DURATION]:.2f}' \
              f' + {sr_res.loc[s.CLUSTER_DURATION]:.2f} = ' \
              f'{sr_res[s.CLUSTER_DURATION] + sr_res[s.PREPROCESSING_DURATION]:.2f}s'
        if s.ITERATION_COUNT in sr_res.index:
            msg += f' [{sr_res.loc[s.ITERATION_COUNT]} iter.]'
        if s.NOISE in sr_res.index:
            msg += f' [{sr_res.loc[s.NOISE] * 100:.2f}% noise]'
        print(msg)
    elif sr_res.loc[s.STATUS] == s.IMPOSSIBLE_PARAMETER_COMBINATION:
        print('Dataset cannot be clustered for these parameters')
    elif sr_res.loc[s.STATUS] == s.UNLUCKY_SEED:
        print('Seed caused this dataset to be impossible to cluster for these parameters')
    elif sr_res.loc[s.STATUS] == s.JKKC_ZERO_VECTOR:
        print('Dataset cannot be clustered with JKKC because it contains a zero-vector datapoint')
    elif sr_res.loc[s.STATUS] == s.LLOYD_FAIL:
        print('Dataset cannot be clustered because Lloyd\'s algorithm ran into a problem')
    elif sr_res.loc[s.STATUS] == s.AQBC_NEGATIVE_ERROR:
        print('Dataset cannot be clustered with (BB)AQBC because it contains a negative-value datapoint')
    else:
        print(f'<This is not implemented for status code :{sr_res.loc[s.STATUS]}>')
