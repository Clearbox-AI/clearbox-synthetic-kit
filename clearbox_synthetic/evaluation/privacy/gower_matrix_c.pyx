import numpy
cimport numpy
cimport cython

cdef numpy.ndarray[numpy.float32_t, ndim=1] gower_row(
    numpy.ndarray[numpy.uint8_t, ndim=1] x_categoricals, 
    numpy.ndarray[numpy.float32_t, ndim=1] x_numericals, 
    numpy.ndarray[numpy.uint8_t, ndim=2] Y_categoricals,
    numpy.ndarray[numpy.float32_t, ndim=2] Y_numericals, 
    numpy.ndarray[numpy.float32_t, ndim=1] numericals_ranges,
    numpy.float32_t features_weight_sum
    ):
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] dist_x_Y_categoricals
    cdef numpy.ndarray[numpy.float32_t, ndim=1] dist_x_Y_categoricals_sum
    cdef numpy.ndarray[numpy.float32_t, ndim=2] dist_x_Y_numericals
    cdef numpy.ndarray[numpy.float32_t, ndim=1] dist_x_Y_numericals_sum
    cdef numpy.ndarray[numpy.float32_t, ndim=1] dist_x_Y

    dist_x_Y_categoricals = (x_categoricals!=Y_categoricals).astype(numpy.uint8)
    dist_x_Y_categoricals_sum = dist_x_Y_categoricals.sum(axis=1).astype(numpy.float32)

    dist_x_Y_numericals = numpy.abs((x_numericals-Y_numericals)/numericals_ranges)
    dist_x_Y_numericals = numpy.nan_to_num(dist_x_Y_numericals)
    dist_x_Y_numericals_sum = dist_x_Y_numericals.sum(axis=1).astype(numpy.float32)
    
    dist_x_Y = dist_x_Y_categoricals_sum + dist_x_Y_numericals_sum
    dist_x_Y = dist_x_Y / features_weight_sum

    return dist_x_Y


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def gower_matrix_c(
    numpy.ndarray[numpy.uint8_t, ndim=2] X_categorical,
    numpy.ndarray[numpy.float32_t, ndim=2] X_numerical,
    numpy.ndarray[numpy.uint8_t, ndim=2] Y_categorical,
    numpy.ndarray[numpy.float32_t, ndim=2] Y_numerical,
    numpy.ndarray[numpy.float32_t, ndim=1] numericals_ranges,
    numpy.float32_t features_weight_sum,
    bint fill_diagonal,
    Py_ssize_t first_index
):
    cdef Py_ssize_t i
    cdef numpy.int32_t X_rows
    cdef numpy.ndarray[numpy.float32_t, ndim=1] dist_x_Y
    cdef numpy.ndarray[numpy.float32_t, ndim=1] distance_matrix

    X_rows = X_categorical.shape[0]
    distance_matrix = numpy.zeros(X_rows, dtype=numpy.float32) 

    # for each instance of matrix X
    for i in range(X_rows):
        dist_x_Y = gower_row(X_categorical[i, :], X_numerical[i, :], Y_categorical, Y_numerical, numericals_ranges, features_weight_sum)

        if (fill_diagonal and first_index < 0):
            dist_x_Y[i] = 5.0
        elif (fill_diagonal):
            dist_x_Y[first_index+i] = 5.0
        distance_matrix[i] = numpy.amin(dist_x_Y)

    return distance_matrix