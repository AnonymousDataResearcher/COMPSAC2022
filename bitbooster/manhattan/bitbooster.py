from numba import njit

from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.manhattan.bitbooster_functions import manhattan_b1, manhattan_b2, manhattan_b3
from bitbooster.operations.bitbooster_distance_operations import generic_bitbooster_rectangle_distance_matrix, \
    generic_bitbooster_index_with_lowest_sum, SIGNATURE_BITBOOSTER_RECTANGLE_DISTANCE_MATRIX, \
    SIGNATURE_BITBOOSTER_INDEX_WITH_LOWEST_SUM


class ManhattanBinaryObject(AbstractBinaryDataObject):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return index_with_lowest_sum_manhattan_bn(vertical_data, horizontal_data)

    def __init__(self, data, num_bits, num_features=None, index=None):
        if num_bits > 3:
            raise NotImplementedError('Manhattan not implemented for n>3')
        super().__init__(data, num_bits, num_features, index)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_manhattan_bn(vertical_data, horizontal_data)


@njit(SIGNATURE_BITBOOSTER_RECTANGLE_DISTANCE_MATRIX)
def rectangle_distance_matrix_manhattan_bn(x_val, y_val):
    return generic_bitbooster_rectangle_distance_matrix(x_val, y_val, manhattan_b1, manhattan_b2, manhattan_b3)


@njit(SIGNATURE_BITBOOSTER_INDEX_WITH_LOWEST_SUM)
def index_with_lowest_sum_manhattan_bn(x_val, y_val):
    return generic_bitbooster_index_with_lowest_sum(x_val, y_val, manhattan_b1, manhattan_b2, manhattan_b3)
