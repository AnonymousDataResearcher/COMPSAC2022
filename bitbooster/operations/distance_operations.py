import numpy as np
from numba import njit

SIGNATURE_INDEX_WITH_LOWEST_SUM = [f'i4({x}[:,:],{x}[:,:])' for x in ['i8', 'i4', 'f8', 'f4']]
SIGNATURE_RECTANGLE_DISTANCE_MATRIX = [f'f4[:,:]({x}[:,:],{x}[:,:])' for x in ['i8', 'i4', 'f8', 'f4']]


@njit
def generic_index_with_lowest_sum(x_val, y_val, matrix_distance_function):
    n_vertical = x_val.shape[0]
    n_horizontal = y_val.shape[0]

    lowest_sum = np.inf
    lowest_index = x_val.shape[0]

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += matrix_distance_function(x_val[i: i + 1], y_val[x: x + 1000]).sum()
            x += 1000

        if i_sum < lowest_sum:
            lowest_sum = i_sum
            lowest_index = i

    return lowest_index
