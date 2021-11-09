import numpy as np
from numba import njit, f4

from bitbooster.abstract.vanilla import BaseVanilla
from bitbooster.operations.distance_operations import generic_index_with_lowest_sum, SIGNATURE_INDEX_WITH_LOWEST_SUM, \
    SIGNATURE_RECTANGLE_DISTANCE_MATRIX


class ManhattanVanillaObject(BaseVanilla):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return manhattan_index_with_lowest_sum(vertical_data, horizontal_data)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_manhattan_vanilla(vertical_data, horizontal_data)


@njit(SIGNATURE_RECTANGLE_DISTANCE_MATRIX)
def rectangle_distance_matrix_manhattan_vanilla(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty((number_v, number_h), dtype=f4)
    for i, vec_i in enumerate(vertical_data):
        for j, vec_j in enumerate(horizontal_data):
            distance_matrix[i, j] = np.sum(np.abs(vec_i - vec_j))
    return distance_matrix


@njit(SIGNATURE_INDEX_WITH_LOWEST_SUM)
def manhattan_index_with_lowest_sum(x_val, y_val):
    return generic_index_with_lowest_sum(x_val, y_val, rectangle_distance_matrix_manhattan_vanilla)
