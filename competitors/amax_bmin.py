import numpy as np
from numba import njit, f4

from bitbooster.abstract.clusterable import Clusterable
from bitbooster.operations.distance_operations import SIGNATURE_INDEX_WITH_LOWEST_SUM, generic_index_with_lowest_sum, \
    SIGNATURE_RECTANGLE_DISTANCE_MATRIX

den = 1 + np.cos(np.pi / 8)
ALPHA = 2 * np.cos(np.pi / 8) / den
BETA = 2 * np.sin(np.pi / 8) / den


class AMaxBMin(Clusterable):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return lowest_sum_index_amax_bmin(vertical_data, horizontal_data)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_amax_bmin(vertical_data, horizontal_data)


@njit(['f4(f8[:],f8[:])', 'f4(i8[:],i8[:])', 'f4(f4[:],f4[:])', 'f4(i4[:],i4[:])'])
def distance_function_amax_bmin(vec_a, vec_b):
    vec_diff = np.abs(vec_a - vec_b)
    hold = vec_diff[0]
    for vec_value in vec_diff[1:]:
        hold = ALPHA * max(hold, vec_value) + BETA * min(hold, vec_value)
    return hold


@njit(SIGNATURE_RECTANGLE_DISTANCE_MATRIX)
def rectangle_distance_matrix_amax_bmin(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty((number_v, number_h), dtype=f4)
    for i, vec_i in enumerate(vertical_data):
        for j, vec_j in enumerate(horizontal_data):
            distance_matrix[i, j] = distance_function_amax_bmin(vec_i, vec_j)
    return distance_matrix


@njit(SIGNATURE_INDEX_WITH_LOWEST_SUM)
def lowest_sum_index_amax_bmin(vertical_data, horizontal_data):
    return generic_index_with_lowest_sum(vertical_data, horizontal_data, rectangle_distance_matrix_amax_bmin)
