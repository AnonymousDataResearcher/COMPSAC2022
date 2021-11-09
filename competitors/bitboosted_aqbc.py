import pandas as pd
import numpy as np
from numba import njit, i8

from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.operations.hamming_weight import hamming_weight
from bitbooster.preprocessing import binarize
from competitors.aqbc import preprocess_aqbc


def preprocess_bb_aqbc(data, n_bits=None, epochs=5):
    hashed_data = preprocess_aqbc(data, n_bits=n_bits, epochs=epochs).astype(np.uint64)
    if isinstance(hashed_data, pd.DataFrame):
        return binarize(hashed_data, 1)
    else:
        return binarize(pd.DataFrame(hashed_data), 1).to_numpy()


class BitboostedAQBCClusterable(AbstractBinaryDataObject):

    def __init__(self, data, num_features=None, index=None):
        super().__init__(data=data, num_bits=1, num_features=num_features, index=index)

    # TODO this 0 getting is due to the fact that the ABDO class expects 2d data, but _bb functions below expect 1D
    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return _bb_aqbc_rectangle_distance_matrix(vertical_data[:, 0], horizontal_data[:, 0])

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return _bb_aqbc_lowest_sum_index(vertical_data[:, 0], horizontal_data[:, 0])


@njit('f4(u8,u1,u8,u1)')
def _bb_aqbc_distance_from_data_and_norm(dx, n2x, dy, n2y):
    return np.sqrt(n2x + n2y - 2 * hamming_weight(dx & dy))


@njit(['f4[:,:](u8[:],u8[:],i8[:],i8[:])'])
def _bb_aqbc_rectangle_distance_matrix_from_norms(vertical_data, horizontal_data, vertical_norms, horizontal_norms):
    number_x = vertical_data.shape[0]
    number_y = horizontal_data.shape[0]

    distance_matrix = np.empty((number_x, number_y), dtype=np.float32)
    for i, (dx, n2x) in enumerate(zip(vertical_data, vertical_norms)):
        for j, (dy, n2y) in enumerate(zip(horizontal_data, horizontal_norms)):
            distance_matrix[i, j] = _bb_aqbc_distance_from_data_and_norm(dx, n2x, dy, n2y)

    return distance_matrix


# TODO actual vectorize numba
@njit('i8[:](u8[:])')
def _vectorized_hamming_weight(data):
    res = np.empty(shape=(data.shape[0],), dtype=i8)
    for i, v in enumerate(data):
        res[i] = hamming_weight(v)

    return res


@njit(['f4[:,:](u8[:],u8[:])'])
def _bb_aqbc_rectangle_distance_matrix(vertical_data, horizontal_data):
    vertical_norms = _vectorized_hamming_weight(vertical_data)
    horizontal_norms = _vectorized_hamming_weight(horizontal_data)
    return _bb_aqbc_rectangle_distance_matrix_from_norms(
        vertical_data, horizontal_data, vertical_norms, horizontal_norms)


# This function does not get the generic implementation; as the horizontal norms would need to be recomputed every time,
# which is disadvantageous for this competitor
@njit(['i8(u8[:],u8[:])'])
def _bb_aqbc_lowest_sum_index(vertical_data, horizontal_data):
    vertical_norms = _vectorized_hamming_weight(vertical_data)
    horizontal_norms = _vectorized_hamming_weight(horizontal_data)

    n_vertical = vertical_data.shape[0]
    n_horizontal = horizontal_data.shape[0]

    lowest_sum = np.inf
    lowest_index = vertical_data.shape[0]

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += _bb_aqbc_rectangle_distance_matrix_from_norms(
                vertical_data[i:i + 1],
                horizontal_data[x:x + 1000],
                vertical_norms[i:i + 1],
                horizontal_norms[x:x + 1000]
            ).sum()
            x += 1000

        if i_sum < lowest_sum:
            lowest_sum = i_sum
            lowest_index = i

    return lowest_index
