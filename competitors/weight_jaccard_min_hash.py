import numpy as np
import pandas as pd
from numba import njit, i4, f4

from bitbooster.abstract.clusterable import Clusterable
from bitbooster.operations.distance_operations import generic_index_with_lowest_sum


def dataframe_or_array(data):
    if isinstance(data, pd.DataFrame):
        index = data.index
        columns = data.columns
        data = data.to_numpy()
    elif isinstance(data, np.ndarray):
        assert data.ndim == 2
        index = None
        columns = None
    else:
        raise NotImplementedError()
    return data, index, columns


def preprocess_wjmh(data, sample_size, seed=0):
    data, index, columns = dataframe_or_array(data)

    dim = data.shape[1]
    generator = np.random.RandomState(seed=seed)

    rs = generator.gamma(2, 1, (sample_size, dim)).astype(np.float32)
    ln_cs = np.log(generator.gamma(2, 1, (sample_size, dim))).astype(np.float32)
    betas = generator.uniform(0, 1, (sample_size, dim)).astype(np.float32)

    return _convert(data.astype(np.float64), rs, ln_cs, betas)


@njit('i4[:](f8[:],f4[:,:],f4[:,:],f4[:,:])')
def _convert_single(data_point, rs, ln_cs, betas):
    sample_size = rs.shape[0]
    hash_values = np.zeros((sample_size * 2), dtype=i4)
    data_zeros = data_point == 0
    data_point[data_zeros] = np.nan
    vlog = np.log(data_point)
    for i in range(sample_size):
        t = np.floor((vlog / rs[i]) + betas[i])
        ln_y = (t - betas[i]) * rs[i]
        ln_a = ln_cs[i] - ln_y - rs[i]
        ln_a[data_zeros] = np.inf
        k = np.argmin(ln_a)
        hash_values[i] = k
        hash_values[i + sample_size] = int(t[k])
    return hash_values


@njit('i4[:,:](f8[:,:],f4[:,:],f4[:,:],f4[:,:])')
def _convert(data, rs, ln_cs, betas):
    n_datapoints = data.shape[0]
    sample_size = rs.shape[0]
    result = np.empty((n_datapoints, sample_size * 2), dtype=i4)
    for i in range(n_datapoints):
        result[i, :] = _convert_single(data[i, :], rs, ln_cs, betas)
    return result


class WeightedJaccardMinHash(Clusterable):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return wjmh_index_with_lowest_sum(vertical_data, horizontal_data)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return wjmh_rectangle_distance_matrix(vertical_data, horizontal_data)


@njit('f4[:,:](i4[:,:],i4[:,:])')
def wjmh_rectangle_distance_matrix(vertical_data, horizontal_data):
    n_vertical = vertical_data.shape[0]
    n_horizontal = horizontal_data.shape[0]
    sample_size = vertical_data.shape[1] // 2
    matrix = np.empty((n_vertical, n_horizontal), dtype=f4)
    for i in range(n_vertical):
        for j in range(n_horizontal):
            sames = vertical_data[i, :] == horizontal_data[j, :]
            matrix[i, j] = 1 - np.logical_and(sames[:sample_size], sames[sample_size:]).sum() / sample_size
    return matrix


@njit(['i8(i4[:,:],i4[:,:])'])
def wjmh_index_with_lowest_sum(vertical_data, horizontal_data):
    return generic_index_with_lowest_sum(vertical_data, horizontal_data, wjmh_rectangle_distance_matrix)
