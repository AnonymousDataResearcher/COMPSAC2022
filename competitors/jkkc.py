# Implementation of the 2006 paper by Jeong, Kim, Kim and Choi, titled
# An Effective Method for Approximating the Euclidean Distance in High-Dimensional Space
import math
import pandas as pd
import numpy as np
from numba import njit, f8, f4
from sklearn.decomposition import PCA

from bitbooster.abstract.clusterable import Clusterable
from bitbooster.operations.distance_operations import SIGNATURE_INDEX_WITH_LOWEST_SUM, generic_index_with_lowest_sum, \
    SIGNATURE_RECTANGLE_DISTANCE_MATRIX

NORM = 'norm'
REFERENCE_ANGLE = 'reference_angle'


class JKKCZeroVectorError(BaseException):
    """
    Raised when trying to convert a dataset that contains a zero vector to JKKC preprocessed dataset
    """

    def __str__(self):
        return f'Encountered one or more 0-vectors in JKKC preprocessing'


def preprocess_jkkc(data):
    if isinstance(data, pd.DataFrame):
        index = data.index
        data = data.to_numpy()
    else:
        index = None

    if np.any(np.all(data == 0, axis=1)):
        raise JKKCZeroVectorError()

    # Get the norm values
    norms = _get_norms(data)

    # Get the reference vector
    pca = PCA(n_components=1)
    pca.fit(data)
    normalized_reference_vector = pca.components_[0] / np.linalg.norm(pca.components_[0])

    # Get the angles between the data and the reference vector
    angles = _get_angles(data, norms, normalized_reference_vector)

    # Return edited data
    if index is None:
        return np.column_stack([norms, angles])
    else:
        return pd.DataFrame(index=index, data=np.column_stack([norms, angles]), columns=[NORM, REFERENCE_ANGLE])


@njit
def _get_norms(data):
    return np.sum(data * data, axis=1)


@njit
def _get_angles(data, norms, normalized_reference_vector):
    normalized_data = np.empty_like(data, dtype=f8)
    for i in range(len(norms)):
        normalized_data[i, :] = data[i, :] / norms[i]
    return np.arccos(np.minimum(np.maximum(np.dot(normalized_data, normalized_reference_vector), -1.0), 1.0))


class JKKCClusterable(Clusterable):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return lowest_sum_index_euclidean_jkkc(vertical_data, horizontal_data)

    def __init__(self, data, index=None, column_names=None):
        if isinstance(data, np.ndarray) and column_names is None:
            column_names = [NORM, REFERENCE_ANGLE]

        super().__init__(data, index, column_names)
        assert set(self.column_names) == {NORM, REFERENCE_ANGLE}
        ra_index = 0 if self.column_names[0] == REFERENCE_ANGLE else 1

        # Check angle value
        assert all(0 <= self.data[:, ra_index]) and all(self.data[:, ra_index] <= math.pi)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_euclidean_jkkc(vertical_data, horizontal_data)


@njit(['f4(f4,f4,f4,f4)', 'f4(f8,f8,f8,f8)'])
def euclidean_jkkc(nx, rax, ny, rab):
    return np.sqrt(nx * nx + ny * ny - 2 * nx * ny * np.cos(np.abs(rax - rab)))


@njit(SIGNATURE_RECTANGLE_DISTANCE_MATRIX)
def rectangle_distance_matrix_euclidean_jkkc(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty(shape=(number_v, number_h), dtype=f4)
    for i, (nx, rax) in enumerate(zip(vertical_data[:, 0], vertical_data[:, 1])):
        for j, (ny, ray) in enumerate(zip(horizontal_data[:, 0], horizontal_data[:, 1])):
            distance_matrix[i, j] = euclidean_jkkc(nx, rax, ny, ray)
    return distance_matrix


@njit(SIGNATURE_INDEX_WITH_LOWEST_SUM)
def lowest_sum_index_euclidean_jkkc(vertical_data, horizontal_data):
    return generic_index_with_lowest_sum(vertical_data, horizontal_data, rectangle_distance_matrix_euclidean_jkkc)
