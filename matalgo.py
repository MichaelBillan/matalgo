import numpy as np

EPSILON = 1e-10

def swap_rows(matrix, row1, row2):
    if row1 != row2:
        matrix[[row1, row2]] = matrix[[row2, row1]]

def rank(mat):
    m, n = len(mat), len(mat[0])
    rank = 0
    row = 0
    for j in range(n):

        if row >= m:
            break

        sub_col = np.abs(mat[row:, j])
        pivot_row_index = row + np.argmax(sub_col)

        if not abs(mat[pivot_row_index][j]) < EPSILON:

            swap_rows(mat, row, pivot_row_index) 

            mat[row, :] /= mat[row, j]

            for r in range(row + 1, m):
                mat[r, :] -= mat[r, j] * mat[row, :]


            row += 1

    for i in range(m):
        if np.any(np.abs(mat[i, :]) > EPSILON):
            rank += 1
    return rank
