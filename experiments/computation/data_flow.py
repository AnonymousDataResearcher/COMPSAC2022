import pandas as pd
from sklearn.datasets import make_blobs

import strings as s


# ACTUAL LOADING -------------------------------------------------------------------------------------------------------


def get_data(dataset_name):
    from data.summary import is_real_dataset, get_dataset_fn
    if is_real_dataset(dataset_name):
        return import_data_from_file(get_dataset_fn(dataset_name))
    else:
        return generate_synthetic_data(dataset_name)


def import_data_from_file(fn):
    """
    Imports data from file

    Parameters
    ----------
    fn: Path or str
        Location of the dataset

    Returns
    -------
    data: pd.DataFrame
        Data from the location
    labels: pd.Series or None
        Labels of each datapoint, if s.LABEL in the data in the fn. Otherwise None
    """
    df = pd.read_csv(fn)

    if s.LABEL in df.columns:
        true_labels = df[s.LABEL]
        df = df.drop(columns=s.LABEL)
    else:
        true_labels = None
    return df[sorted(df.columns)], true_labels


def generate_synthetic_data(dataset_name):
    """
    Generate synthetic data from string encoding. The string is assumed to be formatted as:
    SHAPE_#DATAPOINTS_#FEATURES_#CLASSES_SEED

    Parameters
    ----------
    dataset_name: str
        The encoded dataset name

    Returns
    -------
    data: pd.DataFrame
        Data values (i.e. #FEATURES-dimensional coordinates)
    classes: pd.Series
        Class values
    """
    shape, n_datapoints, n_features, n_classes, seed, *additional = dataset_name.split('_')
    n_datapoints = int(n_datapoints)
    n_features = int(n_features)
    n_classes = int(n_classes)
    seed = int(seed)

    kwargs = dict()
    if shape == s.BLOB:
        pass
    elif shape == s.SPARSE_BLOB:
        kwargs['cluster_std'] = float(additional[0])
    else:
        raise NotImplementedError(shape)

    x, y = make_blobs(n_samples=n_datapoints, n_features=n_features, centers=n_classes, random_state=seed,
                      **kwargs)

    z = len(str(n_features - 1))

    return pd.DataFrame(data=x, columns=[f'feature{i:0{z}}' for i in range(n_features)]), pd.Series(data=y)
