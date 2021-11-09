from numba import njit

from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.euclidean.bitbooster_functions import euclidean_b1, euclidean_b2, euclidean_b3
from bitbooster.operations.bitbooster_distance_operations import generic_bitbooster_index_with_lowest_sum, \
    generic_bitbooster_rectangle_distance_matrix, SIGNATURE_BITBOOSTER_RECTANGLE_DISTANCE_MATRIX, \
    SIGNATURE_BITBOOSTER_INDEX_WITH_LOWEST_SUM


class EuclideanBinaryObject(AbstractBinaryDataObject):

    def __init__(self, data, num_bits, num_features=None, index=None):
        if num_bits > 3:
            raise NotImplementedError('Euclidean not implemented for n>3')
        super().__init__(data, num_bits, num_features, index)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_euclidean_bn(vertical_data, horizontal_data)

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return index_with_lowest_sum_euclidean_bn(vertical_data, horizontal_data)


@njit(SIGNATURE_BITBOOSTER_INDEX_WITH_LOWEST_SUM)
def index_with_lowest_sum_euclidean_bn(x_val, y_val):
    return generic_bitbooster_index_with_lowest_sum(x_val, y_val, euclidean_b1, euclidean_b2, euclidean_b3)


@njit(SIGNATURE_BITBOOSTER_RECTANGLE_DISTANCE_MATRIX)
def rectangle_distance_matrix_euclidean_bn(x_val, y_val):
    return generic_bitbooster_rectangle_distance_matrix(x_val, y_val, euclidean_b1, euclidean_b2, euclidean_b3)
