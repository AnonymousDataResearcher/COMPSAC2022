import abc
from abc import ABC

import pandas as pd

import strings
import strings as s
from bitbooster.common import as_list
from common import results_folder
from data.summary import is_real_dataset, get_properties
from metric import Metric


class _BaseManager(ABC):
    def __init__(self, real, metric=None, name=None, mics=None):
        assert real in [True, False]
        self.real = real

        # Verify metric/mics input
        if metric is not None:
            metric = as_list(metric, str)
            invalid_metrics = set(filter(lambda x: not Metric.is_valid_metric(x), metric))
            assert not invalid_metrics, f'Invalid metrics: {invalid_metrics}'

        if mics is not None:
            mics = as_list(mics, str)
            invalid_mics = set(filter(lambda x: not Metric.is_valid_metric_implementation(x), mics))
            assert not invalid_mics, f'Invalid mics: {invalid_mics}'

        # Verify combination of metric and mics and save
        if metric is None and mics is None:
            raise ValueError('Must give at least one of "metric" and "mics"')
        elif metric is None and mics is not None:
            self.metric = list(map(lambda x: Metric(x).base_metric, mics))
            self.mics = mics
        elif metric is not None and mics is None:
            self.metric = metric
            self.mics = Metric.metric_implementations(metric)
        elif metric is not None and mics is not None:
            assert set(metric) == set(map(lambda x: Metric(x).base_metric, mics)), \
                'Mismatch between "metric" and "mics"'
            self.metric = metric
            self.mics = mics
        else:
            raise NotImplementedError('This cannot happen')

        # Set name
        if name is None:
            self.fd_base = results_folder / ('real' if self.real else 'synthetic')
        else:
            self.fd_base = results_folder / name

    @property
    def synthetic(self):
        return not self.real

    def fd_clustering_algorithm(self, ca):
        return self.fd_base / ca

    def fn_raw(self, ca):
        return self.fd_clustering_algorithm(ca) / 'raw_results.csv'

    @property
    def fd_pure(self):
        return self.fd_base / 'pure'

    @property
    def fn_time(self):
        return self.fd_pure / 'time.csv'

    def fn_error(self, base_metric):
        return self.fd_pure / f'error_{base_metric}.csv'

    @property
    def fn_eps(self):
        if not self.real:
            raise NotImplementedError()
        return self.fd_4dist / '_eps.csv'

    @property
    def fn_average_vanilla_distance(self):
        return self.fd_4dist / '_avg.csv'

    @property
    def fn_average_vanilla_4dist(self):
        return self.fd_4dist / '_avg_4dist.csv'

    @property
    def fd_4dist(self):
        return self.fd_base / '4dist'

    def fd_labels(self, ca):
        return self.fd_clustering_algorithm(ca) / 'labels'


# Single dataset -------------------------------------------------------------------------------------------------------
class SingleDatasetManager(_BaseManager):

    def __init__(self, dataset_name, metric=None, name=None, mics=None):
        super().__init__(is_real_dataset(dataset_name), metric=metric, name=name, mics=mics)
        self.dataset_name = dataset_name

    # PROPS ------------------------------------------------------------------------------------------------------------

    @property
    def number_of_datapoints(self):
        return get_properties(self.dataset_name)[s.NUMBER_OF_DATAPOINTS]

    def __import_mic_index(self, fn):
        if not fn.exists():
            return pd.DataFrame()
        else:
            df = pd.read_csv(fn)
            return df[df[s.DATASET] == self.dataset_name] \
                .set_index(s.METRIC_IMPLEMENTATION_CODE) \
                .drop(columns=s.DATASET) \
                .dropna(axis='columns', how='all')

    def __export_mic_index(self, df, fn):

        # Get the rest of the results
        if fn.exists():
            # Existing, drop this dataset
            df_all = pd.read_csv(fn)
            df_all = df_all[df_all[s.DATASET] != self.dataset_name]
        else:
            # New
            df_all = None
            fn.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dataframe
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.DataFrame):
            pass
        else:
            raise TypeError('df must be Series of DataFrame')

        # Check this is indeed mic_index
        assert df.index.name == s.METRIC_IMPLEMENTATION_CODE

        # Add dataset name
        df = df.reset_index().assign(**{s.DATASET: self.dataset_name})

        if df_all is None:
            df_all = df
        else:
            # Check this is compatible
            assert set(df.columns) == set(df_all.columns)

            # Add new to existing
            df_all = df_all.append(df)

        try:
            df_all.to_csv(fn, index=False)
        except KeyboardInterrupt:
            # TODO I hope this works...
            print('I AM SAVING... PLEASE DONT INTERRUPT... I\'LL QUIT WHEN I\'M DONE SAVING.\n'
                  'If you interrupt now, you might lose all or some experimental data')
            df_all.to_csv(fn, index=False)
            raise KeyboardInterrupt

    def __remove_mic_index(self, fn):
        if not fn.exists():
            # No results yet
            return
        df = pd.read_csv(fn)
        df = df[df[s.DATASET] != self.dataset_name]
        if len(df) > 0:
            df.to_csv(fn, index=False)
        else:
            fn.unlink()

    # PURE -------------------------------------------------------------------------------------------------------------
    # all
    def remove_pure(self):
        self.remove_time()
        self.remove_avg_vanilla_distance()
        self.remove_avg_vanilla_4dist()
        for m in as_list(self.metric, str):
            self.remove_error(m)

    # Error
    def export_error(self, df, base_metric):
        self.__export_mic_index(df=df, fn=self.fn_error(base_metric))

    def import_error(self, base_metric):
        return self.__import_mic_index(self.fn_error(base_metric))

    def remove_error(self, base_metric):
        self.__remove_mic_index(self.fn_error(base_metric))

    # Time
    def export_time(self, df):
        return self.__export_mic_index(df=df, fn=self.fn_time)

    def import_time(self):
        return self.__import_mic_index(self.fn_time)

    def remove_time(self):
        self.__remove_mic_index(self.fn_time)

    # Average vanilla distance
    def export_avg_vanilla_distance(self, sr):
        assert isinstance(sr, pd.Series)
        assert sr.name == s.AVG_DISTANCE
        return self.__export_mic_index(df=sr, fn=self.fn_average_vanilla_distance)

    def import_avg_vanilla_distance(self):
        return self.__import_mic_index(self.fn_average_vanilla_distance)

    def remove_avg_vanilla_distance(self):
        return self.__remove_mic_index(self.fn_average_vanilla_distance)

    # Average vanilla 4dist
    def export_avg_vanilla_4dist(self, sr):
        assert isinstance(sr, pd.Series)
        assert sr.name == s.AVG_4DIST
        return self.__export_mic_index(df=sr, fn=self.fn_average_vanilla_4dist)

    def import_avg_vanilla_4dist(self):
        return self.__import_mic_index(self.fn_average_vanilla_4dist)

    def remove_avg_vanilla_4dist(self):
        return self.__remove_mic_index(self.fn_average_vanilla_4dist)

    # 4 DIST -----------------------------------------------------------------------------------------------------------
    @property
    def fn_4dist(self):
        if not self.real:
            raise NotImplementedError()
        return self.fd_4dist / f'{self.dataset_name}.csv'

    def export_4dist(self, df):
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == set(Metric.metric_implementations(self.metric))
        self.fn_4dist.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.fn_4dist, index=False)

    def import_4dist(self):
        return pd.read_csv(self.fn_4dist)

    def remove_4dist(self):
        if self.fn_4dist.exists():
            self.fn_4dist.unlink()

    def done_4dist(self):
        return self.fn_4dist.exists()

    # CLUSTERING -------------------------------------------------------------------------------------------------------
    def import_raw(self, ca):
        return self.__import_mic_index(fn=self.fn_raw(ca))

    def export_raw(self, df, ca):
        self.__export_mic_index(df=df, fn=self.fn_raw(ca))

    def remove_raw(self, ca):
        self.__remove_mic_index(fn=self.fn_raw(ca))
        for mic in Metric.metric_implementations(self.metric):
            self.remove_labels(ca, mic)

    def done_raw(self, ca):
        df = self.import_raw(ca)
        if len(df) == 0:
            return False
        return not ((df[s.STATUS] == s.NOT_COMPLETED) | (df[s.STATUS].isna())).any()

    # LABELS -----------------------------------------------------------------------------------------------------------
    def fn_labels(self, ca, mic):
        return self.fd_labels(ca) / f'{self.dataset_name}_{mic}.csv'

    def export_labels(self, labels, ca, mic):
        fn = self.fn_labels(ca, mic)
        fn.parent.mkdir(exist_ok=True, parents=True)
        pd.Series(labels).to_csv(fn, index=False, header=False)

    def import_labels(self, ca, mic):
        fn = self.fn_labels(ca, mic)
        return pd.read_csv(fn, header=None).to_numpy()

    def remove_labels(self, ca, mic):
        if self.fn_labels(ca, mic).exists():
            self.fn_labels(ca, mic).unlink()

    # DBSCAN EPS -------------------------------------------------------------------------------------------------------
    def import_eps(self):
        df = self.__import_mic_index(self.fn_eps)
        if s.EPS in df:
            return df[s.EPS]
        else:
            return pd.Series(name=s.EPS, dtype=float, index=Metric.metric_implementations(self.metric))

    def export_eps(self, sr):
        assert isinstance(sr, pd.Series)
        assert sr.name == s.EPS
        assert sr.index.name == s.METRIC_IMPLEMENTATION_CODE
        self.__export_mic_index(df=sr.to_frame(), fn=self.fn_eps)

    def done_eps(self):
        sr = self.import_eps()
        df = self.import_4dist()
        sr = sr.reindex(df.columns)
        return len(sr) > 0 and not sr.isna().any()

    def remove_eps(self):
        self.__remove_mic_index(self.fn_eps)


# Base multiple datasets -----------------------------------------------------------------------------------------------
class BaseLocationsManager(_BaseManager, abc.ABC):
    def import_raw(self, ca):
        if not self.fn_raw(ca).exists():
            return pd.DataFrame()
        else:
            return pd.read_csv(self.fn_raw(ca))

    def fn_parsed(self, ca):
        return self.fd_clustering_algorithm(ca) / 'results.csv'

    def export_parsed(self, df, ca):
        assert isinstance(df, pd.DataFrame)
        df.to_csv(self.fn_parsed(ca))

    def import_parsed(self, ca):
        return pd.read_csv(self.fn_parsed(ca))

    # TEX --------------------------------------------------------------------------------------------------------------
    def fd_tex(self, ca):
        return self.fd_clustering_algorithm(ca) / 'tex'

    # PARETO -----------------------------------------------------------------------------------------------------------
    def fd_pareto(self, ca):
        return self.fd_clustering_algorithm(ca) / 'pareto'

    def import_time(self):
        return pd.read_csv(self.fn_time)

    def import_error(self, base_metric):
        return pd.read_csv(self.fn_error(base_metric))

    def import_average_vanilla_distance(self):
        return pd.read_csv(self.fn_average_vanilla_distance)

    def import_average_vanilla_4dist(self):
        return pd.read_csv(self.fn_average_vanilla_4dist)

    def fd_lines(self, ca):
        return self.fd_clustering_algorithm(ca) / 'line'

    def sparsity_sr(self, sparsity_mode, base_metric, full=False):
        # Determine exponent of the denominator
        if base_metric == s.EUCLIDEAN:
            exp = 0.5
        elif base_metric in [s.MANHATTAN, s.WEIGHTED_JACCARD]:
            exp = 1
        else:
            raise NotImplementedError(base_metric)

        # Determine the source file
        if sparsity_mode == strings.EPS:
            fn = self.fn_eps
        elif sparsity_mode == strings.AVG_4DIST:
            fn = self.fn_average_vanilla_4dist
        elif sparsity_mode == strings.AVG_DISTANCE:
            fn = self.fn_average_vanilla_distance
        else:
            raise NotImplementedError(f'Not implemented for sparsity mode {sparsity_mode}')

        # Import data, filter
        df = pd.read_csv(fn)
        vanilla_name = Metric.vanilla_implementations(base_metric)
        df = df[df[s.METRIC_IMPLEMENTATION_CODE] == vanilla_name]

        # TODO. This is kind of hacky. The way in which experiments are structured, we never throw away values of
        #  experiments, even if we later decide to use a subset (or different set) of datasets. In principe, this should
        #  never happen, unless you are trying to figure out which datasets should be used.
        if not full:
            df = df.set_index([s.DATASET]).reindex(self.all_dataset_names).reset_index()

        # Add data properties
        from data.summary import add_properties
        df = add_properties(df).set_index(s.DATASET)

        # Compute sparsity
        sr = df[sparsity_mode] / (df[s.NUMBER_OF_FEATURES] ** exp)

        # Wrap and return
        sr.index.name = s.DATASET
        sr.name = s.SPARSITY
        return sr

    @property
    @abc.abstractmethod
    def all_dataset_names(self):
        pass
