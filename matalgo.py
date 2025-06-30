import numpy as np

# any number whjich is less than this number considred as zero
EPSILON = 1e-12


# this function is used to swap rows in place in a matrix
def swap_rows(matrix, row1, row2):
    if row1 != row2:
        matrix[[row1, row2]] = matrix[[row2, row1]]

        