import numpy as np
from numba import njit, f4

from bitbooster.abstract.vanilla import BaseVanilla
from bitbooster.operations.distance_operations import SIGNATURE_INDEX_WITH_LOWEST_SUM, generic_index_with_lowest_sum, \
    SIGNATURE_RECTANGLE_DISTANCE_MATRIX


class WeightedJaccard(BaseVanilla):
    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return wj_index_with_lowest_sum(vertical_data, horizontal_data)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return wj_rectangle_distance_matrix(vertical_data, horizontal_data)


@njit(SIGNATURE_RECTANGLE_DISTANCE_MATRIX)
def wj_rectangle_distance_matrix(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty((number_v, number_h), dtype=f4)
    for i, vec_i in enumerate(vertical_data):
        for j, vec_j in enumerate(horizontal_data):
            denominator = np.maximum(vec_i, vec_j).sum()
            if denominator == 0:
                distance_matrix[i, j] = 0
            else:
                distance_matrix[i, j] = 1 - (np.minimum(vec_i, vec_j).sum()) / denominator
    return distance_matrix


@njit(SIGNATURE_INDEX_WITH_LOWEST_SUM)
def wj_index_with_lowest_sum(vertical_data, horizontal_data):
    return generic_index_with_lowest_sum(vertical_data, horizontal_data, wj_rectangle_distance_matrix)
