import pandas as pd
import numpy as np

from numba import njit

from bitbooster.abstract.clusterable import Clusterable
from bitbooster.preprocessing import normalize
from competitors.aqbc_ops import AQBC


# Start of cloned code -------------------------------------------------------------------------------------------------


class AQBCNegativeError(BaseException):
    def __str__(self):
        return f'Encountered one or more negative-valued numbers in the data preprocessing'


def preprocess_aqbc(data, n_bits=None, epochs=5):
    """
    Preprocessing for the AQBC method.

    Parameters
    ----------
    data: np.array of shape (n_samples, n_features)
        Original data
    n_bits: int or None
        Number of bits for the encoding. If None, n_features is used
    epochs: int
        Number of epochs in the

    Returns
    -------

    """
    if not np.all(data >= 0):
        raise AQBCNegativeError()

    if n_bits is None:
        n_bits = data.shape[1]

    if isinstance(data, pd.DataFrame):
        np_data = data.to_numpy()
    else:
        np_data = data

    np_data = normalize(np_data)

    a = AQBC(np.ascontiguousarray(np_data.T), n_bits, epochs, seed=n_bits * epochs)
    a.optimize_all()
    hashed_data = a.B.T

    if isinstance(data, pd.DataFrame):
        z = len(str(n_bits - 1))
        return pd.DataFrame(data=hashed_data, columns=[f'BC{i:0{z}}' for i in range(n_bits)], index=data.index)
    else:
        return hashed_data


class AQBCClusterable(Clusterable):

    def __init__(self, data, index=None, column_names=None):
        assert np.all(np.isin(data, [0, 1]))
        super().__init__(data, index, column_names)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return _aqbc_rectangle_distance_matrix(vertical_data, horizontal_data)

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return _aqbc_lowest_sum_index(vertical_data, horizontal_data)


@njit('f4(b1[:],u1,b1[:],u1)')
def _aqbc_distance_from_data_and_norm(dx, n2x, dy, n2y):
    return np.sqrt(n2x + n2y - 2 * np.sum(np.logical_and(dx, dy)))


@njit(['f4[:,:](b1[:,:],b1[:,:],i8[:],i8[:])'])
def _aqbc_rectangle_distance_matrix_from_norms(vertical_data, horizontal_data, vertical_norms, horizontal_norms):
    number_x = vertical_data.shape[0]
    number_y = horizontal_data.shape[0]

    distance_matrix = np.empty((number_x, number_y), dtype=np.float32)
    for i, (dx, n2x) in enumerate(zip(vertical_data, vertical_norms)):
        for j, (dy, n2y) in enumerate(zip(horizontal_data, horizontal_norms)):
            distance_matrix[i, j] = _aqbc_distance_from_data_and_norm(dx, n2x, dy, n2y)

    return distance_matrix


@njit(['f4[:,:](b1[:,:],b1[:,:])'])
def _aqbc_rectangle_distance_matrix(vertical_data, horizontal_data):
    vertical_norms = np.sum(vertical_data, axis=1)
    horizontal_norms = np.sum(horizontal_data, axis=1)
    return _aqbc_rectangle_distance_matrix_from_norms(vertical_data, horizontal_data, vertical_norms, horizontal_norms)


# This function does not get the generic implementation; as the horizontal norms would need to be recomputed every time,
# which is disadvantageous for this competitor
@njit(['i8(b1[:,:],b1[:,:])'])
def _aqbc_lowest_sum_index(vertical_data, horizontal_data):
    vertical_norms = np.sum(vertical_data, axis=1)
    horizontal_norms = np.sum(horizontal_data, axis=1)

    n_vertical = vertical_data.shape[0]
    n_horizontal = horizontal_data.shape[0]

    lowest_sum = np.inf
    lowest_index = vertical_data.shape[0]

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += _aqbc_rectangle_distance_matrix_from_norms(
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
