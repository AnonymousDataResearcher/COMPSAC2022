from itertools import product

import pandas as pd

import strings as s
from data.summary import get_summary
from experiments.management.locations_manager import SingleDatasetManager
from metric import Metric


def compare(old_name=None, new_name=None):
    if (old_name is None and new_name is None) or (old_name == new_name) or (
            old_name is None and new_name == 'real') or (new_name is None and old_name == 'real'):
        raise ValueError('They are the same ....')

    df = get_summary()

    # Labels -----------------------------------------------------------------------------------------------------------
    print('Checking labels ...')
    for (dsn, mic) in product(df.index, Metric.metric_implementations(s.EUCLIDEAN)):
        fn_old = SingleDatasetManager(dsn).fn_labels(s.DBSCAN, mic)
        fn_new = SingleDatasetManager(dsn, 'real_dbscan_adapted').fn_labels(s.DBSCAN, mic)
        if fn_old.exists() and fn_new.exists():
            labels_old = pd.read_csv(fn_old, header=None)
            labels_new = pd.read_csv(fn_new, header=None)
            assert (labels_old == labels_new).all().all()
        else:
            if fn_old.exists():
                print(f'\tNo labels for new/{mic}/{dsn}')
            elif fn_new.exists():
                print(f'\tNo labels for old/{mic}/{dsn}')
            else:  # Both not exist
                pass


if __name__ == '__main__':
    compare(old_name=None, new_name='real_dbscan_adapted')

