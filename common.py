import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import strings
import strings as s
from metric import Metric

results_folder = Path('results')
fd_synthetic = results_folder / 'all_synthetic_experiments'
fd_real = results_folder / 'all_real_experiments'

data_root = Path('data')
raw_data_folder = data_root / 'raw_data'
data_folder = data_root / 'datasets'
summary_file = data_root / 'summary.csv'


def human_names(string_to_be_translated):
    """
    "Translations" for visualization

    Parameters
    ----------
    string_to_be_translated: str
        The string to be translated

    Returns
    -------
    t: str
        The translated string if a translation is available; s otherwise

    Raises
    ------
    UserWarning: if s is not translated yet (s is returned)

    """
    caps_s = [s.PREPROCESSING_DURATION, s.CLUSTER_DURATION, s.DURATION]

    d = {**{x: x.capitalize() + ' (s)' for x in caps_s},
         s.RELATIVE_ARI: 'Relative ' + s.ARI.upper(),
         s.NUMBER_OF_CLUSTERS: '#Clusters',
         s.NUMBER_OF_FEATURES: '#Features',
         s.NUMBER_OF_DATAPOINTS: '#Datapoints',
         s.ITERATION_COUNT: 'Iterations',
         s.METRIC_IMPLEMENTATION_CODE: 'Approximation',
         s.PURITY_WITH_NOISE: 'Purity',
         s.PURITY_WITHOUT_NOISE: 'Purity',
         s.ARI: 'Adjusted Rand Index'}

    t = d.get(string_to_be_translated, string_to_be_translated.capitalize())
    return t.replace('_', ' ')


def valid_parameter_combination(n_features, mic, n_datapoints, n_clusters, clustering_algorithm):
    """
    Verify the combinations of parameters. If inputs are Series, a Series is returned.

    if clustering_algorithm == s.DBSCAN
        valid
    else:
        if n_datapoints < n_clusters
            # There are too few datapoints
            invalid
        elif mic == s.VAN:
            valid
        elif n_feature * 2 ** n_bits > n_clusters
            valid
        else:
            invalid

    Parameters
    ----------
    n_features: pd.Series or int
        Number of features
    mic: pd.Series or str
        MetricHandler implementation code
    n_clusters: pd.Series or int
        Number of clusters
    n_datapoints: pd.Series or int
        Number of datapoints
    clustering_algorithm: pd.Series or str
        Clustering algorithm that is used.

    Returns
    -------
    valid: (pd.Series of) int
        (Series of) whether the given combination is valid.
    """
    if isinstance(mic, str):
        is_bb = Metric(mic).is_bitbooster
        n_bits = Metric(mic).n_bits_or_0
        is_aqbc = Metric(mic).is_aqbc
    elif isinstance(mic, pd.Series):
        is_bb = mic.apply(lambda x: Metric(x).is_bitbooster)
        n_bits = mic.apply(lambda x: Metric(x).n_bits_or_0)
        is_aqbc = mic.apply(lambda x: Metric(x).is_aqbc)
    else:
        raise TypeError(type(mic))
    # Note that:
    # If nf * n >= log2(k)
    # Then nf * log2(2^n) >= log2(k)
    # Then log2((2^n)^nf) >= log2(k)
    # Then (2^n)^nf >= k

    # Note that:
    # If nf  * n >= log2(k)-1

    # Only BitBooster and AQBC have potentially too few datapoints
    condition_voronoi_other = ~is_bb & ~is_aqbc

    # For BitBooster we require (2^n)^nf >= k, or alternatively nf * n >= log2(k)
    condition_voronoi_bitbooster = is_bb & (n_features * n_bits >= np.log2(n_clusters))

    # For AQBC we require 2^nf-1 >= k, or alternatively nf >= log2(k+1)
    condition_voronoi_aqbc = is_aqbc & (n_features >= np.log2(n_clusters + 1))

    # If this condition holds, then the given properties nf and k are actually compatible for the metric
    condition_voronoi_metric = condition_voronoi_other | condition_voronoi_bitbooster | condition_voronoi_aqbc

    # If there are fewer datapoints than required clusters, we cannot cluster this either
    condition_voronoi_datapoints = (n_datapoints >= n_clusters)

    # Both conditions need to hold
    condition_voronoi = condition_voronoi_metric & condition_voronoi_datapoints

    # If the clustering algorithm is DBSCAN, nothing else matters
    return (clustering_algorithm == s.DBSCAN) | ((clustering_algorithm == s.VORONOI) & condition_voronoi)


# Quality Metrics ======================================================================================================
def purity(true_labels, predicted_labels):
    from sklearn.metrics.cluster import contingency_matrix
    cm = contingency_matrix(true_labels, predicted_labels)
    unique_predicted_labels = np.unique(predicted_labels)
    n_points_majority_class = np.sum(np.amax(cm, axis=0)[np.where(unique_predicted_labels != s.NOISE_INT)])
    n_points_clustered = np.sum(predicted_labels != s.NOISE_INT)
    n_points = len(predicted_labels)

    ret1 = np.nan if n_points_clustered == 0 else n_points_majority_class / n_points_clustered
    ret2 = n_points_majority_class / n_points
    return ret1, ret2


def purity_with_noise(true_labels, predicted_labels):
    return purity(true_labels, predicted_labels)[0]


def purity_without_noise(true_labels, predicted_labels):
    return purity(true_labels, predicted_labels)[1]


# Result columns =======================================================================================================
def vanilla_ize(x):
    return f'vanilla_{x}'


def vanilla_columns(is_real, clustering_algorithm):
    if not is_real:
        return []
    else:
        foo = [vanilla_ize(s.ARI), vanilla_ize(s.PURITY_WITHOUT_NOISE)]
        if clustering_algorithm == s.VORONOI:
            return foo
        elif clustering_algorithm == s.DBSCAN:
            return foo + [s.PURITY_WITH_NOISE]
        else:
            raise NotImplementedError()


def is_vanilla_column(feature):
    it = itertools.product([True, False], s.all_clustering_algorithms)
    return feature in sum([vanilla_columns(tf, ca) for tf, ca in it], [])


def quality_columns(clustering_algorithm):
    base = [s.ARI, s.PURITY_WITHOUT_NOISE, vanilla_ize(s.ARI), vanilla_ize(s.PURITY_WITHOUT_NOISE)]
    if clustering_algorithm == s.VORONOI:
        return base
    elif clustering_algorithm == s.DBSCAN:
        return base + [s.PURITY_WITH_NOISE, vanilla_ize(s.PURITY_WITH_NOISE)]
    else:
        raise NotImplementedError()


def time_columns(clustering_algorithm):
    base = [s.CLUSTER_DURATION, s.PREPROCESSING_DURATION]
    if clustering_algorithm == s.VORONOI:
        return base
    elif clustering_algorithm == s.DBSCAN:
        return base
    else:
        raise NotImplementedError()


def result_columns(clustering_algorithm):
    base = [s.STATUS] + quality_columns(clustering_algorithm) + time_columns(clustering_algorithm)

    if clustering_algorithm == s.VORONOI:
        return base + [s.ITERATION_COUNT]
    elif clustering_algorithm == s.DBSCAN:
        return base + [s.NOISE, s.NUMBER_OF_FOUND_CLUSTERS, s.NUMBER_OF_CORE_POINTS,
                       s.INNER_NEIGHBOURHOOD_COMPUTATIONS, s.OUTER_NEIGHBOURHOOD_COMPUTATIONS]
    else:
        raise NotImplementedError(f'Function not implemented for clustering algorithm {clustering_algorithm}')


evaluation_metrics = {
    strings.ARI: adjusted_rand_score,
    strings.PURITY_WITHOUT_NOISE: purity_without_noise,
    strings.PURITY_WITH_NOISE: purity_with_noise
}
