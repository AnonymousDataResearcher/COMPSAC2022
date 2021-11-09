import numpy as np
from numba import njit, f4

SIGNATURE_BITBOOSTER_RECTANGLE_DISTANCE_MATRIX = ['f4[:, :](u8[:, :], u8[:, :])']
SIGNATURE_BITBOOSTER_INDEX_WITH_LOWEST_SUM = ['i8(u8[:, :], u8[:, :])']


@njit
def generic_bitbooster_rectangle_distance_matrix(x_val, y_val, b1f, b2f, b3f):
    len_x = x_val.shape[0]
    len_y = y_val.shape[0]
    n = x_val.shape[1]
    distance_matrix = np.empty((len_x, len_y), dtype=f4)

    if n == 1:
        for i, x0 in enumerate(x_val[:, 0]):
            for j, y0 in enumerate(y_val[:, 0]):
                distance_matrix[i, j] = b1f(x0, y0)
    elif n == 2:
        for i, (x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1])):
            for j, (y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1])):
                distance_matrix[i, j] = b2f(x1, x0, y1, y0)
    elif n == 3:
        for i, (x2, x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1], x_val[:, 2])):
            for j, (y2, y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1], y_val[:, 2])):
                distance_matrix[i, j] = b3f(x2, x1, x0, y2, y1, y0)

    return distance_matrix


@njit
def generic_bitbooster_index_with_lowest_sum(x_val, y_val, b1f, b2f, b3f):
    n_vertical = x_val.shape[0]
    n_horizontal = y_val.shape[0]

    lowest_sum = np.inf
    lowest_index = -1

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += generic_bitbooster_rectangle_distance_matrix(x_val[i: i + 1], y_val[x: x + 1000], b1f, b2f,
                                                                  b3f).sum()
            x += 1000

        if i_sum < lowest_sum:
            lowest_sum = i_sum
            lowest_index = i

    return lowest_index
