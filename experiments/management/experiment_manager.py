import functools
import itertools
from abc import ABC

import strings as s
from bitbooster.common import as_list
from data.summary import get_summary
from experiments.management import locations_manager
from experiments.management.locations_manager import BaseLocationsManager


class _BaseExperimentManager(BaseLocationsManager, ABC):
    # TODO merge this with BaseLocationsManager?
    #  I think this is fix to circular imports actually

    def __init__(self, real, metric=None, name=None, mics=None):
        self.name = name
        super().__init__(real, metric=metric, name=name, mics=mics)

    def __run(self, check_method, run_method):
        for dsn in self.all_dataset_names:
            lm = locations_manager.SingleDatasetManager(dsn, metric=self.metric, name=self.name, mics=self.mics)
            b = check_method(lm)
            assert b in [True, False]
            if not b:
                run_method(lm)

    def run_pure_and_4dist(self):
        fns1 = [self.fn_time, self.fn_average_vanilla_4dist, self.fn_average_vanilla_distance]
        fns2 = [self.fn_error(m) for m in self.metric]
        if not all([fn.exists() for fn in fns1 + fns2]):
            def check(x):
                return False
        else:
            gb_time = self.import_time().groupby(s.DATASET)
            gb_avg_vanilla_distance = self.import_average_vanilla_distance().groupby(s.DATASET)
            gb_avg_vanilla_4dist = self.import_average_vanilla_4dist().groupby(s.DATASET)
            gbs = [gb_time, gb_avg_vanilla_4dist, gb_avg_vanilla_distance] + \
                  [self.import_error(bm).groupby(s.DATASET) for bm in self.metric]

            if self.real:
                def check(lm: locations_manager.SingleDatasetManager):
                    return lm.fn_4dist.exists()
            else:
                # Check single value things
                def check(lm: locations_manager.SingleDatasetManager):
                    return all(map(lambda gb: lm.dataset_name in gb.groups, gbs))

        from experiments.computation.aaa_learn_pure_and_4dist import execute
        # Warm-up
        sdm = locations_manager.SingleDatasetManager('iris', metric=s.EUCLIDEAN)
        execute(sdm, silent=True)
        sdm.fn_4dist.unlink()
        sdm.remove_pure()
        sdm.remove_4dist()

        self.__run(check_method=check, run_method=execute)

    def run_eps_estimation(self):
        from experiments.computation.aab_estimate_eps import execute
        self.__run(check_method=lambda lm: lm.done_eps() or not lm.done_4dist(),
                   run_method=execute)

    def run_voronoi(self, warm_up=True):
        from experiments.computation.aac_cluster_experiment_real import execute
        self.__run(check_method=lambda lm: lm.done_raw(s.VORONOI),
                   run_method=functools.partial(execute, clustering_algorithm=s.VORONOI, warm_up=warm_up))

        self.run_vanilla_metrics(s.VORONOI)

    def run_dbscan(self):
        from experiments.computation.aac_cluster_experiment_real import execute
        self.__run(check_method=lambda lm: lm.done_raw(s.DBSCAN) or not lm.done_eps(),
                   run_method=functools.partial(execute, clustering_algorithm=s.DBSCAN))

        self.run_vanilla_metrics(s.DBSCAN)

    def run_vanilla_metrics(self, ca):

        # Vanilla comparison has a build-in verification, so the check_method always evaluates to False
        from experiments.computation.aad_vanilla_metrics import execute
        self.__run(check_method=lambda lm: False,
                   run_method=functools.partial(execute, ca=ca))

    def parse_results(self, ca):
        from experiments.result_parsing.aa_preprocessing import execute
        execute(self, ca)

    def parse_tex(self, ca):
        from experiments.result_parsing.bb_tex import execute
        execute(self, ca)

    def parse_pareto(self, ca):
        from experiments.result_parsing.cc_pareto import execute
        execute(self, ca)

    def parse_pure(self):
        from experiments.result_parsing.ee_pure import execute
        execute(self)


class RealExperimentManager(_BaseExperimentManager):

    def __init__(self, name=None, metric=None, mics=None):
        super().__init__(True, metric=metric, name=name, mics=mics)

    @property
    def all_dataset_names(self):
        # TODO this skips the largest
        return list(get_summary().sort_values(s.NUMBER_OF_DATAPOINTS).index[:-4])


class BaseSyntheticExperimentManager(_BaseExperimentManager):
    def __init__(self, dsn_list, metric=None, name=None, mics=None):
        super().__init__(real=False, metric=metric, name=name, mics=mics)
        self.dsn_list = as_list(dsn_list, str)

    @property
    def all_dataset_names(self):
        return self.dsn_list


class SyntheticExperimentManager(BaseSyntheticExperimentManager):

    def __init__(self, metric=None, number_of_datapoints=None, number_of_clusters=None, seed=None,
                 number_of_features=None, name=None, mics=None):
        if number_of_datapoints is None:
            number_of_datapoints = 10_000
        self.number_of_datapoints = as_list(number_of_datapoints, int)

        if number_of_clusters is None:
            number_of_clusters = range(5, 31, 5)
        self.number_of_clusters = as_list(number_of_clusters, int)

        if seed is None:
            seed = range(10)
        self.seed = as_list(seed, int)

        if number_of_features is None:
            number_of_features = range(4, 65, 4)
        self.number_of_features = as_list(number_of_features, int)

        dsn_list = [f'{shape}_{ndp}_{nf}_{nc}_{seed}' for shape, seed, ndp, nc, nf in
                    itertools.product([s.BLOB], self.seed, self.number_of_datapoints, self.number_of_clusters,
                                      self.number_of_features)]

        super().__init__(dsn_list=dsn_list, metric=metric, name=name, mics=mics)
