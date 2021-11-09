import time

import numpy as np
import pandas as pd

from bitbooster.abstract.clusterable import Clusterable
from experiments.computation.data_flow import generate_synthetic_data
from experiments.management.experiment_manager import SyntheticExperimentManager
from metric_handler import MetricHandler, AQBC, BITBOOSTED_AQBC


def run(sd):
    names = SyntheticExperimentManager(seed=sd, number_of_datapoints=1000,
                                       number_of_features=[10, 20, 30],
                                       number_of_clusters=[10, 20, 30]
                                       ).all_dataset_names

    mh_og = MetricHandler(AQBC)
    mh_bb = MetricHandler(BITBOOSTED_AQBC)
    times = pd.DataFrame(['pp_og', 'c_og', 'pp_bb', 'c_bb'])

    for dsn in names:
        dataset, labs = generate_synthetic_data(dsn)

        c, t = mh_og.clusterable_from_raw(dataset, return_prep_time=True)
        times.loc[dsn, 'pp_og'] = t
        assert isinstance(c, Clusterable)
        t0 = time.process_time()
        _, labs_og = c.voronoi(int(dsn.split('_')[3]))
        t1 = time.process_time()
        times.loc[dsn, 'c_og'] = t1 - t0

        c, t = mh_bb.clusterable_from_raw(dataset, return_prep_time=True)
        times.loc[dsn, 'pp_bb'] = t
        assert isinstance(c, Clusterable)
        t0 = time.process_time()
        _, labs_bb = c.voronoi(int(dsn.split('_')[3]))
        t1 = time.process_time()
        times.loc[dsn, 'c_bb'] = t1 - t0

        assert np.all(labs_bb == labs_og), f'mismatch for {dsn=}'

    print(times.mean())


if __name__ == '__main__':
    run(0)
    run(range(10))
