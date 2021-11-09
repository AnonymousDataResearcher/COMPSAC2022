import time

import numpy as np
import pandas as pd

from bitbooster.euclidean.bitbooster import EuclideanBinaryObject
from bitbooster.euclidean.vanilla import EuclideanVanillaObject
from bitbooster.manhattan.bitbooster import ManhattanBinaryObject
from bitbooster.manhattan.vanilla import ManhattanVanillaObject
from competitors.amax_bmin import AMaxBMin
from competitors.aqbc import AQBCClusterable
from competitors.bitboosted_aqbc import BitboostedAQBCClusterable
from competitors.jkkc import JKKCClusterable
from competitors.weight_jaccard_min_hash import WeightedJaccardMinHash
from competitors.weighted_jaccard import WeightedJaccard
from experiments.computation.data_flow import get_data
from metric import Metric, JKKC, EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, AMAX_BMIN, AQBC, \
    BITBOOSTED_AQBC


class MetricHandler(Metric):

    @property
    def needs_normalization(self):
        if self.is_bitbooster:
            return False
        if self.s in [AQBC, JKKC, BITBOOSTED_AQBC]:
            return True
        if self.is_wjmh:
            return True
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, AMAX_BMIN]:
            return True

        raise NotImplementedError(self.s)

    def _normalize_if_necessary(self, original_data):
        if self.needs_normalization:
            from bitbooster.preprocessing.normalizer import normalize
            return normalize(original_data)
        else:
            return original_data

    def _preprocess_after_normalization(self, normalized_data):
        if self.is_bitbooster:
            from bitbooster.preprocessing.discretizers import discretize
            from bitbooster.preprocessing.binarizer import binarize
            return binarize(discretize(normalized_data, n_bits=self.n_bits), self.n_bits)
        if self.is_wjmh:
            from competitors.weight_jaccard_min_hash import preprocess_wjmh
            return preprocess_wjmh(normalized_data, sample_size=self.sample_size)
        if self.s == JKKC:
            from competitors.jkkc import preprocess_jkkc
            return preprocess_jkkc(normalized_data)
        if self.s == AQBC:
            from competitors.aqbc import preprocess_aqbc
            return preprocess_aqbc(normalized_data)
        if self.s == BITBOOSTED_AQBC:
            from competitors.bitboosted_aqbc import preprocess_bb_aqbc
            return preprocess_bb_aqbc(normalized_data)
        if self.s in [EUCLIDEAN_VANILLA, MANHATTAN_VANILLA, WEIGHTED_JACCARD_VANILLA, AMAX_BMIN]:
            return normalized_data

        raise NotImplementedError(self.s)

    def _normalize_and_preprocess(self, original_data):
        t0 = time.process_time()
        res = self._preprocess_after_normalization(self._normalize_if_necessary(original_data))
        t1 = time.process_time()

        return res, t1 - t0

    def clusterable_from_raw(self, original_data, index=None, column_names=None, return_prep_time=False):
        if isinstance(original_data, str):
            original_data, _ = get_data(original_data)

        data, t_prep = self._normalize_and_preprocess(original_data)
        c = self.get_clusterable(data, index=index, column_names=column_names, original_data=original_data)

        if return_prep_time:
            return c, t_prep
        else:
            return c

    def get_clusterable(self, data, index, column_names, original_data):
        default = dict(data=data, index=index, column_names=column_names)
        bb_dict = dict(data=data, num_bits=self.n_bits_or_0, num_features=dimensions(original_data)[1],
                       index=index)
        if self.s.startswith('EBB'):
            return EuclideanBinaryObject(**bb_dict)
        if self.s.startswith('MBB'):
            return ManhattanBinaryObject(**bb_dict)
        if self.s == JKKC:
            return JKKCClusterable(**default)
        if self.s == EUCLIDEAN_VANILLA:
            return EuclideanVanillaObject(**default)
        if self.s == MANHATTAN_VANILLA:
            return ManhattanVanillaObject(**default)
        if self.s == AMAX_BMIN:
            return AMaxBMin(**default)
        if self.is_wjmh:
            return WeightedJaccardMinHash(**default)
        if self.s == WEIGHTED_JACCARD_VANILLA:
            return WeightedJaccard(**default)
        if self.s == AQBC:
            return AQBCClusterable(**default)
        if self.s == BITBOOSTED_AQBC:
            return BitboostedAQBCClusterable(data=data, num_features=dimensions(original_data)[1], index=index)

        raise NotImplementedError(self.s)


def dimensions(data):
    if isinstance(data, pd.DataFrame):
        return len(data), len(data.columns)
    elif isinstance(data, np.ndarray):
        return data.shape
    else:
        raise NotImplementedError(type(data))
